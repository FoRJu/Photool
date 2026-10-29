"""
Filling in the camera data a catalog row is missing.

Indexing a folder reads each photograph's EXIF (camera, lens, exposure, the date
it was taken), but not every road into the catalog did: a library-wide rebuild
indexed folders without it, and a folder once indexed is answered from the
catalog afterwards ("file count matches"), so a row written blank stayed blank.
On one library that was 58% of the photographs, and every search by camera, ISO
or date, and every sort by time, quietly missed them.

This reads them in the background instead of on the request that indexes: a
folder at a time, newest first, one exiftool run per few hundred files. Each
file read is recorded with its size (``photo_exif_reads``), so a photograph that
truly carries no EXIF — a scan, a picture saved from a messaging app — is looked
at once and not again, and one that is replaced on disk is read again.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The columns of `photos` this fills, from get_exif_batch's names.
_COLUMNS = {
    "camera": "camera", "lens": "lens", "focal_length": "focal_length",
    "aperture": "aperture", "shutter_speed": "shutter_speed", "iso": "iso",
    "capture_date": "date", "dimensions": "dimensions",
}

_lock = threading.Lock()
_stop = threading.Event()
_job: dict = {"status": "idle", "done": 0, "total": 0, "filled": 0, "folder": "", "error": ""}
_wanted = False
_thread = None


def job() -> dict:
    return dict(_job)


def pending(db, folder=None) -> list[dict]:
    """Photographs with no camera recorded that have not been read as they are."""
    sql = ("SELECT p.filepath, p.folder, p.file_size FROM photos p "
           "LEFT JOIN photo_exif_reads r ON r.filepath = p.filepath "
           "WHERE p.camera = '' AND (r.filepath IS NULL OR r.source_size != p.file_size)")
    args: tuple = ()
    if folder is not None:
        sql += " AND p.folder = ?"
        args = (folder,)
    sql += " ORDER BY p.folder DESC, p.filepath"
    with db._connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def counts(db) -> dict:
    """How many photographs have their camera data, or have been looked at for it."""
    with db._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN p.camera != '' THEN 1 ELSE 0 END) AS known, "
            "SUM(CASE WHEN p.camera != '' OR (r.filepath IS NOT NULL AND r.source_size = p.file_size) "
            "THEN 1 ELSE 0 END) AS read "
            "FROM photos p LEFT JOIN photo_exif_reads r ON r.filepath = p.filepath"
        ).fetchone()
    return {"photos": int(row["total"] or 0), "known": int(row["known"] or 0),
            "read": int(row["read"] or 0)}


def record(db, rows: list[tuple]) -> int:
    """Store ``(filepath, size, exif dict)`` rows: the EXIF into the photograph's
    row when there is any, and the read itself either way. Returns how many
    photographs gained a camera."""
    if not rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    filled = [(fp, exif) for fp, _size, exif in rows if exif.get("camera")]
    sets = ", ".join(f"{col} = ?" for col in _COLUMNS)
    with db._connect() as conn:
        conn.executemany(
            f"UPDATE photos SET {sets} WHERE filepath = ? AND camera = ''",
            [tuple(str(exif.get(key) or "") for key in _COLUMNS.values()) + (fp,)
             for fp, exif in filled],
        )
        conn.executemany(
            "INSERT INTO photo_exif_reads (filepath, source_size, read_at) VALUES (?, ?, ?) "
            "ON CONFLICT(filepath) DO UPDATE SET source_size = excluded.source_size, "
            "read_at = excluded.read_at",
            [(fp, int(size or 0), now) for fp, size, _exif in rows],
        )
    return len(filled)


def run(library_root, db, folder=None, stop_event: Optional[threading.Event] = None,
        background: bool = False) -> dict:
    """Read every pending photograph (of ``folder``, or of the library), on this
    thread. Stops between folders when ``stop_event`` is set (the background
    worker passes the one :func:`stop` sets). In the ``background`` it steps
    aside, a folder at a time, while someone is using the app."""
    import background as gate
    from thumbnails.extract import get_exif_batch

    rows = pending(db, folder)
    groups: dict = {}
    for row in rows:
        groups.setdefault(row["folder"], []).append(row)
    _job.update(status="running", total=len(rows), done=0, filled=0, error="",
                started_at=time.time())
    root = Path(library_root)
    try:
        for name, items in groups.items():
            if stop_event is not None and stop_event.is_set():
                break
            if background and not gate.wait_turn("Camera data", stop_event and stop_event.is_set):
                break
            _job["folder"] = name
            paths = {root / r["filepath"]: r for r in items}
            present = [p for p in paths if p.is_file()]
            found = get_exif_batch(present, fast=True) if present else {}
            _job["filled"] += record(db, [
                (paths[p]["filepath"], paths[p]["file_size"], found.get(str(p), {}))
                for p in present
            ])
            # A date just read may be on a clock in another zone.
            from catalog import clock

            clock.refresh(db, name)
            _job["done"] += len(items)
        _job["status"] = "stopped" if stop_event is not None and stop_event.is_set() else "completed"
    except Exception as exc:
        logger.exception("Filling in camera data failed")
        _job.update(status="failed", error=str(exc))
    finally:
        _job.update(folder="", finished_at=time.time())
    return job()


def wake(library_root, db) -> None:
    """Fill in whatever is pending, in the background. Returns at once; a call
    while a pass is running asks for one more pass after it, so photographs
    indexed meanwhile are not left for the next start."""
    global _thread, _wanted
    if not getattr(db, "writable", True):
        return
    with _lock:
        _wanted = True
        if _thread is not None:
            return
        _stop.clear()
        _thread = threading.Thread(target=_worker, args=(library_root, db),
                                   name="exif-fill", daemon=True)
        _thread.start()


def _worker(library_root, db) -> None:
    global _thread, _wanted
    while True:
        with _lock:
            if not _wanted or _stop.is_set():
                _thread = None
                return
            _wanted = False
        run(library_root, db, stop_event=_stop, background=True)


def stop() -> None:
    _stop.set()
