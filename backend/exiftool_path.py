"""
Where exiftool is.

Every caller runs plain ``exiftool`` and relies on the PATH. On a Mac, Homebrew
puts it in /opt/homebrew/bin (Apple Silicon) or /usr/local/bin (Intel), and a
server started by something other than a login shell (a launchd agent, an app
launcher, a terminal whose profile never ran ``brew shellenv``) does not have
those on its PATH. So it is looked for there too, once at start, and its folder
is put on this process's PATH, where every caller and every helper it starts
finds it.

Without exiftool much still works, but quietly less: no capture settings or
lens in the info panel, no Picture Style to pool the camera's look by, slower
RAW thumbnails. ``available()`` is what the health check reports, so the page
can say so instead of showing empty fields.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Where package managers put it, beyond the PATH: Homebrew on Apple Silicon and
# on Intel, MacPorts, and Debian's package.
_PLACES = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin")

_found: Optional[str] = None
_looked = False


def ensure_on_path() -> Optional[str]:
    """The exiftool this process will run, its folder added to PATH when it was
    found outside it; ``None`` when there is none."""
    global _found, _looked
    if _looked:
        return _found
    _looked = True
    _found = shutil.which("exiftool")
    if _found is None:
        for place in _PLACES:
            candidate = Path(place) / "exiftool"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                os.environ["PATH"] = place + os.pathsep + os.environ.get("PATH", "")
                _found = str(candidate)
                break
    if _found:
        logger.info(f"exiftool: {_found}")
    else:
        logger.warning("exiftool not found: capture settings, lens data and the camera's "
                       "Picture Style will be missing. Install it (macOS: brew install "
                       "exiftool; Windows: winget install OliverBetz.ExifTool) and restart.")
    return _found


def available() -> bool:
    return ensure_on_path() is not None
