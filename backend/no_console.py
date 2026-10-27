"""
Helper programs run without a window of their own.

Started at login, the server runs under pythonw, with no console. Windows then
gives every console program it starts (exiftool for each photo measured, the
denoiser, ffmpeg) a fresh console window of its own, which flashes up and is
gone, again and again while background work runs. Importing this module makes
every subprocess the server starts a hidden one; under a console it changes
nothing that shows.
"""

import subprocess

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _HiddenPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | CREATE_NO_WINDOW
        super().__init__(*args, **kwargs)


def install() -> None:
    """Make every later subprocess.run / Popen in this process windowless.
    Does nothing off Windows."""
    if CREATE_NO_WINDOW and not issubclass(subprocess.Popen, _HiddenPopen):
        subprocess.Popen = _HiddenPopen


install()
