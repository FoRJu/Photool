"""
Starting Photool at login (login_item.py): what is installed on each system, and
on Windows, installing and removing it for real under a test name.
"""

import plistlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import login_item  # noqa: E402


def test_the_mac_agent_starts_this_checkouts_server_in_the_background():
    agent = login_item.mac_agent()
    assert agent["ProgramArguments"] == [sys.executable, str(login_item.SERVE), "--background", "--no-tray"]
    assert agent["WorkingDirectory"] == str(login_item.BACKEND)
    assert agent["RunAtLoad"] and not agent["KeepAlive"]
    # Homebrew's folders, for exiftool, node and ffmpeg.
    assert "/opt/homebrew/bin" in agent["EnvironmentVariables"]["PATH"].split(":")
    assert plistlib.loads(plistlib.dumps(agent)) == agent


def test_windows_runs_serve_py_with_no_console():
    command = login_item.windows_command()
    assert command.endswith(f'"{login_item.SERVE}" --background')
    if sys.platform == "win32":
        assert "pythonw.exe" in command


@pytest.mark.skipif(sys.platform != "win32", reason="the Run key is Windows'")
def test_install_and_remove_on_windows(monkeypatch, capsys):
    import winreg

    monkeypatch.setattr(login_item, "NAME", "Photool test (safe to delete)")
    monkeypatch.setattr(login_item, "_old_startup_shortcuts", lambda: [])

    def value():
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, login_item.RUN_KEY) as key:
            try:
                return winreg.QueryValueEx(key, login_item.NAME)[0]
            except FileNotFoundError:
                return None

    try:
        assert login_item.main(["install"]) == 0
        assert value() == login_item.windows_command()
        assert login_item.main(["status"]) == 0 and "Starts at login" in capsys.readouterr().out
    finally:
        assert login_item.main(["remove"]) == 0
    assert value() is None
    assert login_item.main(["remove"]) == 0 and "was not set" in capsys.readouterr().out


def test_an_unknown_action_is_refused():
    assert login_item.main(["now"]) == 2
