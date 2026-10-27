"""
Coordinates out of the files.

One exiftool process per batch, like the capture-date reader, with ``-fast2``
so it reads the header and stops: GPS lives in the EXIF block at the front of
every format this library holds, and reading a 30 MB CR3 to its end to find two
numbers at the start of it is what keeps a hard drive busy.

The composite tags are asked for because they carry the sign. The raw EXIF
latitude is always positive, with N or S in a separate reference tag, and a
photograph from Sydney filed in the northern hemisphere is the kind of mistake
nobody notices until the map is drawn.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from exif_batches import batches


def _key(path) -> str:
    s = str(path).replace("\\", "/")
    return s.casefold() if os.name == "nt" else s


def parse(stdout: str) -> dict[str, tuple[float, float]]:
    """exiftool ``-j -n`` output to ``{path key: (lat, lon)}``.

    A file with no fix is left out. So is 0,0: it is in the Gulf of Guinea, and
    a camera that writes it means "no position", not "at sea".
    """
    try:
        rows = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        lat, lon = row.get("GPSLatitude"), row.get("GPSLongitude")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
            continue
        out[_key(row.get("SourceFile", ""))] = (float(lat), float(lon))
    return out


def read(paths: list[Path], stop=None) -> Optional[dict[Path, Optional[tuple[float, float]]]]:
    """Every path's coordinates, or ``None`` for a path without them.

    Returns ``None`` altogether when exiftool could not run at all, so the
    caller does not record thousands of photographs as "no GPS" when the truth
    is "not read".
    """
    result: dict[Path, Optional[tuple[float, float]]] = {}
    ran = False
    for batch in batches(paths):
        if stop is not None and stop.is_set():
            break
        try:
            proc = subprocess.run(
                ["exiftool", "-j", "-n", "-fast2",
                 "-Composite:GPSLatitude", "-Composite:GPSLongitude",
                 *[str(p) for p in batch]],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=max(60, len(batch)),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        if not proc.stdout.strip():
            continue
        ran = True
        found = parse(proc.stdout)
        for p in batch:
            result[p] = found.get(_key(p))
    return result if ran else None
