"""
Batching for exiftool command lines.

exiftool receives the files it should read as command-line arguments, and
Windows caps a process command line at roughly 32k characters. A folder of a
few hundred RAWs — one 5D Mark II CF card's DCIM directory, or a busy shoot
day — goes past that, so exiftool never starts at all. The failure is quiet:
each caller catches the error and returns its "nothing found" default, which
means missing capture dates and, worse, orientation 1 on every frame. A
sideways thumbnail still looks like a thumbnail, so this can go unnoticed for
a long time.

Chunking by both a file count and a total argument length keeps every command
line inside the limit. The count cap is separate because process startup, not
the length limit, is what makes a batch expensive.
"""

from pathlib import Path
from typing import Iterable, Iterator

# One exiftool process per this many files. Large enough to amortise process
# startup (~500 ms on Windows), small enough to stay well inside the
# command-line limit.
EXIFTOOL_BATCH_SIZE = 200

# Secondary cap on the total length of the file arguments. Windows allows
# ~32k characters for the entire command line; 12000 leaves ample room for the
# executable path and the tag switches.
EXIFTOOL_ARG_CHARS = 12000


def batches(
    paths: Iterable[Path],
    max_files: int = EXIFTOOL_BATCH_SIZE,
    max_chars: int = EXIFTOOL_ARG_CHARS,
) -> Iterator[list[Path]]:
    """Chunk paths by count and by total argument length (Windows cmdline limit)."""
    batch: list[Path] = []
    chars = 0
    for p in paths:
        n = len(str(p)) + 1
        if batch and (len(batch) >= max_files or chars + n > max_chars):
            yield batch
            batch, chars = [], 0
        batch.append(p)
        chars += n
    if batch:
        yield batch
