"""
Run Photool for this computer *and* the other devices in the house.

    cd backend && python serve.py                 # in a console window
    cd backend && pythonw serve.py --background   # no window: what Windows starts at login

It listens on every network interface at ``port`` (8000 by default), serves the
built page and the API from one address, and refuses everything from the network
unless ``lan_access: true`` is in config.yaml — and then only to devices you have
paired (see docs/guide.md, "Other computers"). With ``tls_cert`` and ``tls_key`` set it
speaks HTTPS.

Before it starts, the page is rebuilt when the app's source is newer than the
built copy (after an update), so a stale page is never served; a build that
fails leaves the old page in place and says why. A second start while one is
already running on the port leaves quietly.

``--background`` (and running under ``pythonw``, which has no console) writes
everything to ``serve.log`` beside the catalog instead, kept to a few megabytes.

On Windows a background server is watched: this process runs the real one
(``--worker``) as a child and starts it again if it dies. Native code can take the
whole process down, and the computer may be somewhere nobody can get at it. Quit
in the tray stays quit, and ``stop.bat`` stops both. If it does die, Python's view
of every thread at that moment is in ``serve-crash.log`` beside ``serve.log``.

For access from outside the house, leave this alone and use Tailscale instead:
keep running the normal way (localhost only) and run ``tailscale serve --bg 8000``.
"""

import blas_threads  # noqa: F401  (before anything loads numpy)

import faulthandler
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from config import load_config

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
# What a build is made from: a change to any of these makes the page stale.
_SOURCES = ("src", "index.html", "package.json", "vite.config.js", "tailwind.config.js",
            "postcss.config.js")
LOG_BYTES = 5 * 1024 * 1024
# npm is a .cmd on Windows: started from a process with no console it would open
# a console window of its own.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# What the worker exits with for the watcher to start it again at once (Restart in
# the tray); any other exit but 0 means it died.
RESTART_EXIT = 75
# How long a stopping server waits for open connections before closing them. A
# connection that never ends, such as a stream, would otherwise hold it up for good.
# Shorter than tray.STOP_WAIT, so the tray's Restart isn't left waiting on one.
GRACEFUL_SECONDS = 8
# A worker that lived this long is not stuck in a crash loop.
SETTLED_SECONDS = 120.0
FIRST_DELAY = 2.0
LONGEST_DELAY = 300.0
CRASH_LOG_BYTES = 1024 * 1024
_crash_log = None     # kept open: faulthandler writes to it at the moment of a crash


def _addresses() -> list[str]:
    out = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            out.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(out)


def _newest(path: Path) -> float:
    if path.is_file():
        return path.stat().st_mtime
    if path.is_dir():
        return max((p.stat().st_mtime for p in path.rglob("*") if p.is_file()), default=0.0)
    return 0.0


def page_is_stale(frontend: Path = FRONTEND) -> bool:
    """Whether the built page is missing or older than what it is built from."""
    built = frontend / "dist" / "index.html"
    if not built.is_file():
        return True
    made = built.stat().st_mtime
    return any(_newest(frontend / name) > made for name in _SOURCES)


def find_npm() -> Optional[str]:
    """npm, as a program this process can start. On Windows Node installs both
    ``npm`` (a shell script for Git Bash) and ``npm.cmd``; the first is found first
    and cannot be started ("%1 is not a valid Win32 application"), which left every
    start since serving the page from before the update."""
    if os.name == "nt":
        found = shutil.which("npm.cmd")
        if found:
            return found
    return shutil.which("npm")


def build_page(frontend: Path = FRONTEND) -> bool:
    """``npm run build``; True when it worked. Without npm, or on a failed build,
    the page already built (if any) is served and the reason is printed."""
    npm = find_npm()
    if not npm:
        print("The page is out of date, and npm (Node.js) is not installed to rebuild it.")
        return False
    print("The app was updated: building the page…")
    try:
        if not (frontend / "node_modules").is_dir():
            subprocess.run([npm, "install"], cwd=frontend, check=True, creationflags=_NO_WINDOW,
                           capture_output=True, text=True, timeout=900)
        done = subprocess.run([npm, "run", "build"], cwd=frontend, creationflags=_NO_WINDOW,
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Building the page failed: {exc}")
        return False
    if done.returncode != 0:
        print("Building the page failed; serving the last page built.")
        print((done.stdout or "")[-2000:] + (done.stderr or "")[-2000:])
        return False
    print("Page built.")
    return True


def already_running(port: int) -> bool:
    """Whether something already answers on this port on this computer."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _log_to_file(config) -> Path:
    """Send print() and the server's own logging to serve.log beside the catalog."""
    path = Path(config.db_path).parent / "serve.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=LOG_BYTES, backupCount=2,
                                                   encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    class _Writer:
        def __init__(self, level):
            self.level = level

        def write(self, text):
            for line in text.rstrip().splitlines():
                if line.strip():
                    logging.getLogger("serve").log(self.level, line)

        def flush(self):
            pass

        def isatty(self):
            return False

    sys.stdout = _Writer(logging.INFO)
    sys.stderr = _Writer(logging.ERROR)
    return path


def _record_crashes(log: Path) -> None:
    """A crash in native code ends the process before Python can log anything, so
    have faulthandler write every thread's Python stack to a file as it happens."""
    global _crash_log
    path = log.with_name("serve-crash.log")
    try:
        if path.is_file() and path.stat().st_size > CRASH_LOG_BYTES:
            path.unlink()
        _crash_log = path.open("a", encoding="utf-8")
        faulthandler.enable(file=_crash_log, all_threads=True)
    except OSError as exc:
        print(f"Cannot keep a crash log at {path}: {exc}")


def supervise(start: Callable[[], int], note: Callable[[str], None],
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> int:
    """Run ``start`` (the worker, to its exit code) until it exits cleanly. It is
    never given up on: a library drive that is not there yet may come back, and a
    wait of minutes costs nothing, but crash after crash waits longer each time."""
    quick = 0
    while True:
        began = clock()
        code = start()
        if code == 0:
            return 0
        if code == RESTART_EXIT:
            quick = 0
            continue
        quick = quick + 1 if clock() - began < SETTLED_SECONDS else 1
        delay = min(LONGEST_DELAY, FIRST_DELAY * 2 ** (quick - 1))
        note(f"The server died (exit code {code & 0xFFFFFFFF:#x}); starting it again in {delay:.0f} s")
        sleep(delay)


def _watch(argv: list[str], log: Path) -> int:
    """Be the watcher: no numpy, no server, and serve.log is only held open for a
    line at a time, since Windows cannot rotate a file another process has open."""
    from tray import python_for_restart

    def note(text: str) -> None:
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} WARNING serve: {text}\n")
        except OSError:
            pass

    command = [python_for_restart(), str(Path(__file__).resolve()), *argv, "--worker"]

    def start() -> int:
        return subprocess.call(command, cwd=os.getcwd(), creationflags=_NO_WINDOW)

    return supervise(start, note)


def _leave(code: int) -> None:
    """Exit now. The server has stopped or had its chance to, and a helper thread
    still running must not keep the process, and so the watcher's restart, waiting."""
    logging.shutdown()
    os._exit(code)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # config.yaml is read from the working folder, and a login item (login_item.py)
    # may start this from anywhere.
    os.chdir(Path(__file__).resolve().parent)
    config = load_config()
    background = "--background" in argv or sys.stdout is None
    worker = "--worker" in argv
    port = int(config.port)
    if background and os.name == "nt" and not worker:
        if already_running(port):
            return 0
        return _watch(argv, Path(config.db_path).parent / "serve.log")
    log = None
    if background:
        log = _log_to_file(config)
        _record_crashes(log)
    if already_running(port):
        print(f"Photool is already running on port {port}; not starting a second one.")
        return 0
    if page_is_stale():
        build_page()
    if not (FRONTEND / "dist" / "index.html").is_file():
        print("The page has not been built: run `npm run build` in frontend/ first.")
        return 1
    if not config.lan_access:
        print("lan_access is off in config.yaml: other devices will be refused.")
        print("Set `lan_access: true` to let paired devices in.")
    scheme = "https" if (config.tls_cert and config.tls_key) else "http"
    for address in _addresses():
        print(f"  On another device:  {scheme}://{address}:{port}")
    print(f"  On this computer:   {scheme}://localhost:{port}")
    import uvicorn

    options = dict(host="0.0.0.0", port=port, proxy_headers=False, timeout_graceful_shutdown=GRACEFUL_SECONDS)
    if scheme == "https":
        options.update(ssl_certfile=config.tls_cert, ssl_keyfile=config.tls_key)
    if log is not None:
        # uvicorn's own console formatting has no console to go to, and a line
        # per thumbnail request would fill the log with nothing worth reading.
        options.update(log_config=None, access_log=False)
        print(f"Running in the background; this log: {log}")
    if background and "--no-tray" not in argv:
        import tray

        if tray.available():
            # In the background there is no window to close: the tray icon is how
            # the server is opened, restarted and quit.
            server = uvicorn.Server(uvicorn.Config("main:app", **options))
            codes = {"quit": 0, "restart": RESTART_EXIT}
            ended = tray.run(server, f"{scheme}://localhost:{port}",
                             [f"{scheme}://{a}:{port}" for a in _addresses()], log, argv,
                             watched=worker,
                             force_exit=(lambda how: os._exit(codes[how])) if worker else None)
            if worker:
                _leave(codes.get(ended, 1))
            return 0
        print("No tray icon (pystray is not installed); stop it with stop.bat.")
    uvicorn.run("main:app", **options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
