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

For access from outside the house, leave this alone and use Tailscale instead:
keep running the normal way (localhost only) and run ``tailscale serve --bg 8000``.
"""

import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from config import load_config

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
# What a build is made from: a change to any of these makes the page stale.
_SOURCES = ("src", "index.html", "package.json", "vite.config.js", "tailwind.config.js",
            "postcss.config.js")
LOG_BYTES = 5 * 1024 * 1024
# npm is a .cmd on Windows: started from a process with no console it would open
# a console window of its own.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # config.yaml is read from the working folder, and a login item (login_item.py)
    # may start this from anywhere.
    os.chdir(Path(__file__).resolve().parent)
    config = load_config()
    background = "--background" in argv or sys.stdout is None
    log = None
    if background:
        log = _log_to_file(config)
    port = int(config.port)
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
    options = dict(host="0.0.0.0", port=port, proxy_headers=False)
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
            tray.run(server, f"{scheme}://localhost:{port}",
                     [f"{scheme}://{a}:{port}" for a in _addresses()], log, argv)
            return 0
        print("No tray icon (pystray is not installed); stop it with stop.bat.")
    uvicorn.run("main:app", **options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
