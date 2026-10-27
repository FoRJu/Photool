"""
no_console: started hidden at login, the server must not flash a console
window for every helper program it runs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import no_console  # noqa: E402


@pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows matter")
def test_every_subprocess_is_started_without_a_window(monkeypatch):
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    no_console.install()
    monkeypatch.setattr(no_console._HiddenPopen.__bases__[0], "__init__", fake_init)
    with pytest.raises(RuntimeError):
        subprocess.run(["exiftool", "-ver"], capture_output=True)
    assert seen["creationflags"] & no_console.CREATE_NO_WINDOW


def test_the_app_installs_it():
    import main  # noqa: F401

    if sys.platform == "win32":
        assert issubclass(subprocess.Popen, no_console._HiddenPopen)
