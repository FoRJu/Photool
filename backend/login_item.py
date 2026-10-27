"""
Start Photool when you log in, in the background, with no window: installed and
removed by one command, the same on Windows and a Mac.

    startup.bat install | remove | status      (Windows)
    ./startup.sh install | remove | status     (macOS)

which run ``python login_item.py <action>`` with the venv's Python.

  Windows  a value under HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run that
           starts ``pythonw serve.py --background``: no console, and the tray icon
           to open, restart or quit it. No administrator rights needed, and it
           shows under Task Manager's Startup apps, where it can be turned off too.
  macOS    a launchd agent, ~/Library/LaunchAgents/<LABEL>.plist, that starts
           ``python serve.py --background --no-tray``. Started at once and at
           every login; stop.sh stops it until the next login, and launchd does
           not start it again in between (no KeepAlive).

Either way the server writes serve.log beside the catalog, a second start while
one runs leaves quietly, and the page is rebuilt at start when the code changed.
The command that is installed names this checkout's own serve.py and the Python
running this script, so run it again after moving the project or the venv.
"""

import os
import plistlib
import platform
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
SERVE = BACKEND / "serve.py"
NAME = "Photool"
LABEL = "io.github.forju.photool"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def windows_command() -> str:
    """What Windows runs at login: the venv's pythonw (no console) on serve.py."""
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    return f'"{pythonw if pythonw.exists() else python}" "{SERVE}" --background'


def mac_agent() -> dict:
    """The launchd agent. Its PATH holds Homebrew's folders, where exiftool, node
    (for rebuilding the page) and ffmpeg are, since launchd starts with almost none."""
    logs = Path.home() / "Library" / "Logs" / NAME
    return {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, str(SERVE), "--background", "--no-tray"],
        "WorkingDirectory": str(BACKEND),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Interactive",
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        # Only what goes wrong before serve.py opens its own log ends up here.
        "StandardOutPath": str(logs / "launchd.log"),
        "StandardErrorPath": str(logs / "launchd.log"),
    }


# ── Windows ──────────────────────────────────────────────────────────

def _old_startup_shortcuts() -> list[Path]:
    """Shortcuts to serve.py in the Startup folder, the way this used to be set up
    by hand: with the Run value as well, Photool would be started twice."""
    folder = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    found = []
    for link in folder.glob("*.lnk") if folder.is_dir() else []:
        try:
            data = link.read_bytes()
        except OSError:
            continue
        if b"serve.py" in data or "serve.py".encode("utf-16-le") in data:
            found.append(link)
    return found


def _windows(action: str) -> int:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                        winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            current, _ = winreg.QueryValueEx(key, NAME)
        except FileNotFoundError:
            current = None
        if action == "install":
            command = windows_command()
            winreg.SetValueEx(key, NAME, 0, winreg.REG_SZ, command)
            print(f"Photool will start at login:\n  {command}")
            for link in _old_startup_shortcuts():
                print(f"[NOTE] A shortcut in the Startup folder starts it too: {link}\n"
                      f"       Delete it, or Photool is started twice (the second one leaves at once).")
            print("Start it now with: serve.bat background")
        elif action == "remove":
            if current is None:
                print("Photool was not set to start at login.")
            else:
                winreg.DeleteValue(key, NAME)
                print("Photool will no longer start at login. stop.bat stops the one running now.")
        else:
            print(f"Starts at login: {current}" if current else "Not set to start at login.")
            for link in _old_startup_shortcuts():
                print(f"Also started by the shortcut {link}")
    return 0


# ── macOS ────────────────────────────────────────────────────────────

def _launchctl(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _mac(action: str) -> int:
    plist = launch_agent_path()
    domain = f"gui/{os.getuid()}"
    if action == "install":
        agent = mac_agent()
        Path(agent["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
        plist.parent.mkdir(parents=True, exist_ok=True)
        # Reinstalling replaces it: take the old one out of launchd first.
        _launchctl("bootout", f"{domain}/{LABEL}")
        with plist.open("wb") as f:
            plistlib.dump(agent, f)
        done = _launchctl("bootstrap", domain, str(plist))
        if done.returncode != 0:
            print(f"[ERROR] launchctl could not load {plist}: {done.stderr.strip()}")
            return 1
        print(f"Photool starts at login, and is starting now: http://localhost:8000\n  {plist}")
        print("./stop.sh stops it until the next login; ./startup.sh remove stops it starting.")
    elif action == "remove":
        _launchctl("bootout", f"{domain}/{LABEL}")
        if plist.exists():
            plist.unlink()
            print("Photool will no longer start at login.")
        else:
            print("Photool was not set to start at login.")
    else:
        loaded = _launchctl("print", f"{domain}/{LABEL}").returncode == 0
        print(f"Starts at login: {plist}" if plist.exists() else "Not set to start at login.")
        if plist.exists() and not loaded:
            print("It is installed but not loaded; ./startup.sh install loads it again.")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    action = argv[0] if argv else "status"
    if action not in ("install", "remove", "status"):
        print("Usage: login_item.py install | remove | status")
        return 2
    system = platform.system()
    if system == "Windows":
        return _windows(action)
    if system == "Darwin":
        return _mac(action)
    print("Starting at login is set up for Windows and macOS only. On Linux, a systemd user "
          f"service that runs `{sys.executable} serve.py --background --no-tray` in {BACKEND} does it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
