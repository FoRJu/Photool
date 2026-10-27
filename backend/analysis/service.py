"""
Runs the measurements over a folder and stores them in the catalog.

The pass reads the small grid thumbnail rather than opening the original file.
For a CR3 that thumbnail is the camera's own embedded JPEG, so a folder costs a
few hundred small JPEG decodes and no demosaicing at all; if the thumbnail does
not exist yet the normal thumbnail path builds it, which is work the queue would
have done anyway.

It follows the catalog rather than the directory, so measurements belong to
indexed photos and an unindexed stray file is not measured. Rows are keyed by the
file's mtime and size, so re-running over a folder that is already measured does
almost nothing.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

from analysis.image_stats import stats_are_current, stats_from_jpeg
from catalog.database import CatalogDB
from thumbnails import ThumbnailCache, ThumbnailSize, get_thumbnail

logger = logging.getLogger(__name__)


def analyze_folder(
    db: CatalogDB,
    cache: ThumbnailCache,
    library_root: str,
    folder: str,
    should_stop: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Measure every indexed photo in one folder.

    Returns counts, never raises for an individual file: a photo that cannot be
    read is counted as failed and the pass continues, because a folder where one
    CR3 is unreadable should still get statistics for the other four hundred.
    """
    folder_dir = Path(library_root) / folder
    if not folder_dir.is_dir():
        return {
            "folder": folder,
            "total": 0,
            "analyzed": 0,
            "skipped": 0,
            "failed": 0,
            "error": "Folder not found",
        }

    photos = db.get_photos_in_folder(folder, pair_mode="all")
    stored = db.list_photo_stats_for_folder(folder)
    result = {
        "folder": folder,
        "total": len(photos),
        "analyzed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for photo in photos:
        if should_stop and should_stop():
            break

        source = folder_dir / photo.filename
        try:
            stat = source.stat()
        except OSError:
            result["failed"] += 1
            continue

        if stats_are_current(stored.get(photo.filepath), stat.st_mtime, stat.st_size):
            result["skipped"] += 1
            continue

        jpeg = get_thumbnail(source, cache, ThumbnailSize.GRID)
        if not jpeg:
            result["failed"] += 1
            continue

        try:
            stats = stats_from_jpeg(jpeg)
        except Exception as e:
            logger.warning(f"Could not measure {photo.filename}: {e}")
            result["failed"] += 1
            continue

        db.upsert_photo_stats(
            photo.filepath,
            stats,
            source_mtime=int(stat.st_mtime),
            source_size=int(stat.st_size),
            source_kind="grid_thumbnail",
        )
        result["analyzed"] += 1

        if progress:
            done = result["analyzed"] + result["skipped"] + result["failed"]
            progress(done, len(photos))

    return result
