"""
Learning camera looks before anyone opens the photographs.

A photograph's look takes several seconds to learn, the first time. Develop no longer
waits on a black screen for it (``engine.starter_base``), but a photograph whose look
is already kept opens finished. So while nobody is using the app, this learns the
looks most likely to be wanted: frames the person has marked first, then the newest
folders. Not the whole library: at several seconds a frame, tens of thousands of
photographs would keep the machine busy for days for frames nobody may open.

It steps aside while the app is in use (background.py), and does not keep the computer
awake (keep_awake.py counts only what someone asked for). A photograph that is edited
keeps the look it has (see ``engine.look_is_kept``).
"""

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# How many of the newest folders are learned ahead, besides marked frames anywhere.
NEWEST_FOLDERS = 3
# Marked frames at most, newest first, in one pass.
MARKED_LIMIT = 2000
# How often it looks again for frames newly marked or imported.
RECHECK_SECONDS = 300.0
# A look is learned from at least this size of decode, as the develop view's would be.
LEARN_SIZE = 1000

_lock = threading.Lock()
_stop = threading.Event()
_again = threading.Event()         # new photographs: look now, not at the next check
_thread: Optional[threading.Thread] = None
_job: dict = {"status": "idle", "done": 0, "total": 0, "learned": 0, "folder": "", "error": ""}
_seen: set = set()


def job() -> dict:
    return dict(_job)


def wanted(db) -> list[str]:
    """Catalog paths to learn, most wanted first: marked frames, then the newest folders."""
    with db._connect() as conn:
        marked = conn.execute(
            "SELECT filepath FROM photos WHERE rating > 0 OR flag <> 0 OR color_label <> 0 "
            "ORDER BY folder DESC, filepath LIMIT ?", (MARKED_LIMIT,)).fetchall()
        folders = conn.execute(
            "SELECT DISTINCT folder FROM photos ORDER BY folder DESC LIMIT ?", (NEWEST_FOLDERS,)).fetchall()
        newest = []
        for row in folders:
            newest += conn.execute("SELECT filepath FROM photos WHERE folder = ? ORDER BY filepath",
                                   (row["folder"],)).fetchall()
    out, seen = [], set()
    for row in list(marked) + list(newest):
        if row["filepath"] not in seen:
            seen.add(row["filepath"])
            out.append(row["filepath"])
    return out


def run_once(library_root, db, should_stop=lambda: False) -> dict:
    """One pass over what is wanted, on this thread."""
    import background as gate
    from export import engine

    root = Path(library_root)
    sources, known = [], set()
    for filepath in wanted(db):
        # A RAW+JPEG pair is one photograph, and its look is the RAW's.
        source = engine.resolve_export_source(root / filepath)
        if (source.suffix.lower() in engine._RAW_EXTENSIONS and str(source) not in known
                and str(source) not in _seen):
            known.add(str(source))
            sources.append(source)
    _job.update(status="running", done=0, total=len(sources), learned=0, folder="", error="")
    try:
        for source in sources:
            if should_stop() or not gate.wait_turn("Camera looks", should_stop):
                break
            _job["folder"] = source.parent.name
            try:
                if source.is_file() and engine.learn_look_ahead(source, LEARN_SIZE):
                    _job["learned"] += 1
            except Exception as exc:
                logger.warning(f"Learning the look of {source.name} ahead failed: {exc}")
            _seen.add(str(source))
            _job["done"] += 1
        _job["status"] = "stopped" if should_stop() else "completed"
    except Exception as exc:
        logger.exception("Learning camera looks ahead failed")
        _job.update(status="failed", error=str(exc))
    finally:
        _job["folder"] = ""
    if _job["learned"]:
        logger.info(f"Learned {_job['learned']} camera look(s) ahead")
    return job()


def _worker(library_root, db) -> None:
    global _thread
    while not _stop.is_set():
        _again.clear()
        run_once(library_root, db, _stop.is_set)
        _again.wait(RECHECK_SECONDS)
    with _lock:
        _thread = None


def start(library_root, db) -> None:
    global _thread
    if not getattr(db, "writable", True):
        return
    with _lock:
        if _thread is not None:
            return
        _stop.clear()
        _thread = threading.Thread(target=_worker, args=(library_root, db), name="look-ahead", daemon=True)
        _thread.start()


def wake() -> None:
    """New photographs are in the catalog: look at what is wanted again now."""
    _again.set()


def stop() -> None:
    _stop.set()
    _again.set()
