"""
The server's icon in the system tray: open the album, see where it is reachable,
read its log, restart it or quit it, without a console window or stop.bat.

``serve.py --background`` (how it starts at login) runs the server on a thread
and this icon on the main one. Restart stops the server, starts a new
``serve.py --background`` (which picks up new code, and rebuilds the page when
its sources changed) and exits; Quit stops the server and exits.

Without ``pystray`` the server runs as it always did, with no icon: the icon is a
convenience, and a missing package is no reason not to serve the album.
"""

import logging
import math
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("serve")

# How long a server that was asked to stop may take to let the port go.
STOP_WAIT = 20.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)


def available() -> bool:
    try:
        import pystray  # noqa: F401
    except Exception:
        return False
    return True


def icon_image(size: int = 64):
    """The tray picture: a camera's aperture in Photool's amber, drawn so it needs
    no file."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = size // 16
    d.ellipse((pad, pad, size - pad, size - pad), fill=(251, 191, 36, 255))
    inner = size // 4
    d.ellipse((inner, inner, size - inner, size - inner), fill=(23, 23, 23, 255))
    c = size // 2
    for k in range(6):
        a = math.radians(60 * k + 15)
        r1, r2 = size * 0.12, size * 0.24
        d.line((c + r1 * math.cos(a), c + r1 * math.sin(a), c + r2 * math.cos(a + 0.9), c + r2 * math.sin(a + 0.9)),
               fill=(251, 191, 36, 255), width=max(1, size // 24))
    return img


def python_for_restart() -> str:
    """The interpreter a restart starts, without a console: the venv's pythonw.exe
    when this runs in a venv (sys.executable may be the base Python the venv's
    launcher handed over to, which does not see the venv's packages)."""
    scripts = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    for name in ("pythonw.exe", "python.exe", "python3", "python"):
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    return sys.executable


class Controls:
    """What the menu does, kept apart from pystray so it can be tested: ``stop``
    asks the server to stop and waits for it; ``spawn`` starts a new server."""

    def __init__(self, url: str, log: Optional[Path], stop: Callable[[], bool],
                 spawn: Callable[[], None], quit_icon: Callable[[], None]):
        self.url = url
        self.log = log
        self._stop = stop
        self._spawn = spawn
        self._quit_icon = quit_icon
        self.restarting = False
        self.quitting = False

    def open(self, *_):
        webbrowser.open(self.url)

    def open_log(self, *_):
        if self.log and self.log.is_file():
            if os.name == "nt":
                os.startfile(str(self.log))  # noqa: S606 - our own log file
            else:
                subprocess.Popen(["xdg-open", str(self.log)])

    def restart(self, *_):
        """Stop, start a new server once the port is free, and go."""
        if self.restarting:
            return
        self.restarting = True
        logger.info("Restarting from the tray")
        if self._stop():
            self._spawn()
        else:
            logger.error("The server did not stop in time; not starting a second one")
        self._quit_icon()

    def quit(self, *_):
        logger.info("Quitting from the tray")
        self.quitting = True
        self._stop()
        self._quit_icon()


def run(server, url: str, addresses: list[str], log: Optional[Path], argv: list[str]) -> None:
    """Serve with ``server`` (a ``uvicorn.Server``) on a thread and show the tray
    icon until Quit or Restart. ``argv`` is how this process was started, for the
    restart to start the same way."""
    import pystray

    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    def stop() -> bool:
        server.should_exit = True
        thread.join(STOP_WAIT)
        return not thread.is_alive()

    serve_py = Path(__file__).resolve().parent / "serve.py"

    def spawn():
        args = [python_for_restart(), str(serve_py), *[a for a in argv if a != "--restarted"], "--restarted"]
        subprocess.Popen(args, cwd=os.getcwd(), close_fds=True,
                         creationflags=_NO_WINDOW | _DETACHED)

    icon = pystray.Icon("Photool", icon_image(), "Photool")
    controls = Controls(url, log, stop, spawn, icon.stop)

    reach = [pystray.MenuItem(f"On another device: {a}", None, enabled=False) for a in addresses]
    icon.menu = pystray.Menu(
        pystray.MenuItem("Open Photool", controls.open, default=True),
        pystray.MenuItem(f"On this computer: {url}", None, enabled=False),
        *reach,
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open log", controls.open_log, enabled=bool(log)),
        pystray.MenuItem("Restart", controls.restart),
        pystray.MenuItem("Quit", controls.quit),
    )
    icon.title = f"Photool — {url}"

    def watch(icon_):
        # The icon goes when the server does, whatever stopped it.
        icon_.visible = True
        while thread.is_alive() and not (controls.restarting or controls.quitting):
            time.sleep(1.0)
        if not (controls.restarting or controls.quitting):
            logger.info("The server stopped; removing the tray icon")
            icon_.stop()

    icon.run(setup=watch)
