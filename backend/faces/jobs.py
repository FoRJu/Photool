"""
The one face scan that runs at a time, and the sweep that starts it on its own.

A scan is started by the People panel ("this folder", "whole library") or by
the sweep, and both go through ``start``: one job, one set of counters the panel
shows, one Stop button. A second start while one runs is refused, because two
passes would fight over the same models and the same thumbnail cache.

The sweep exists because a scan that only runs when someone clicks leaves gaps
nobody sees: a folder imported and never opened, a scan cut off by a restart
half way through a folder. It looks for photographs no scan has looked at with
the current model and scans just their folders — after an import, when a folder
is indexed, and once at startup. It never downloads the models: a library that
has not set faces up stays as it is.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Callable, Optional

logger = logging.getLogger(__name__)

JOB = {
    "status": "idle",      # idle | running | completed | failed | stopped
    "phase": "",           # models | rendering | scanning | grouping
    "scope": "",           # the folder being scanned, or "the whole library"
    "folder": "",
    "total": 0,            # the work in the folder being scanned — the phase's
    "done": 0,             # own denominator, so the bar never jumps backwards
    "folder_index": 0,     # which folder of how many, for a run over the library
    "folders": 0,
    "scanned": 0,
    "skipped": 0,
    "failed": 0,
    "faces": 0,
    "people": 0,
    "dropped": 0,
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
    "detail": "",
    "automatic": False,    # started by the sweep, not by a click
    "closer": False,       # a look at full size (service.look_closer)
    "found": 0,            # faces a closer look found that were not there before
}
STOP = threading.Event()
# Guards the "is a scan already running" check and the setting of the flag, so
# two starts cannot both find the slot free.
GUARD = threading.Lock()

# What the sweep calls a run over the folders it found.
SWEEP_SCOPE = "new photos"
# How long after startup the sweep first looks: the server has other things to
# do in its first minute, and the person opening the page should not wait on it.
STARTUP_DELAY_SECONDS = 60


def recognizer(config) -> str:
    from faces.models import recognizer_name

    return recognizer_name(getattr(config, "face_model", "arcface"))


def defaults(config) -> tuple[float, float]:
    """The join threshold and the smallest face worth grouping."""
    return config.effective_face_threshold(), float(config.face_min_px)


def _claim(scope: str, folders: int, automatic: bool = False, closer: bool = False) -> bool:
    """Take the one job slot, or say it is taken."""
    with GUARD:
        if JOB["status"] == "running":
            return False
        STOP.clear()
        JOB.update(
            status="running", phase="models", scope=scope, folder="", total=0, done=0,
            folder_index=0, folders=folders, scanned=0, skipped=0, failed=0,
            faces=0, people=0, dropped=0, found=0, started_at=time.time(), finished_at=0.0,
            error="", detail="", automatic=automatic, closer=closer,
        )
    return True


def start(app, folders: list[str], scope: str, threshold: Optional[float] = None,
          automatic: bool = False, run_here: bool = False) -> bool:
    """Start a scan of ``folders``. False when one is already running.

    The scan runs on a thread of its own, or on the caller's when ``run_here``
    (the sweep, which is already a background thread).
    """
    if not _claim(scope, len(folders), automatic=automatic):
        return False
    join, min_px = defaults(app.state.config)
    args = (app, folders, join if threshold is None else float(threshold), min_px)
    if run_here:
        _run(*args)
    else:
        threading.Thread(target=_run, args=args, name="faces-scan", daemon=True).start()
    return True


def start_closer(app, filepaths: list[str], scope: str) -> bool:
    """Look again at these photographs at full size (see ``service.look_closer``).
    False when a scan is already running."""
    if not _claim(scope, 1, closer=True):
        return False
    threading.Thread(target=_run_closer, args=(app, list(filepaths)),
                     name="faces-closer", daemon=True).start()
    return True


def _run_closer(app, filepaths: list[str]) -> None:
    from faces.models import ModelsUnavailable, ensure_models
    from faces.service import look_closer, rebuild_people

    config = app.state.config
    threshold, min_px = defaults(config)
    try:
        try:
            ensure_models(config.models_dir, recognizer=recognizer(config))
        except ModelsUnavailable as exc:
            JOB.update(status="failed", error=str(exc))
            return
        JOB.update(phase="scanning", folder_index=1, detail="looking closer")

        def _progress(phase, done, total):
            JOB["phase"] = phase
            JOB["done"] = done
            JOB["total"] = total

        result = look_closer(app.state.catalog.db, config.library_root, filepaths,
                             config.models_dir, min_group_px=min_px,
                             recognizer=recognizer(config), should_stop=STOP.is_set,
                             progress=_progress)
        for key in ("scanned", "failed", "faces", "found"):
            JOB[key] = result[key]
        if result["scanned"]:
            JOB.update(phase="grouping", detail="grouping the faces", total=0, done=0)
            summary = rebuild_people(app.state.catalog.db, threshold, min_group_px=min_px,
                                     model=recognizer(config))
            JOB["people"] = summary["people"]
        JOB.update(status="stopped" if STOP.is_set() else "completed",
                   phase="", detail="", total=0, done=0)
    except Exception as exc:
        logger.exception("Looking closer failed")
        JOB.update(status="failed", error=str(exc), phase="", detail="")
    finally:
        JOB["finished_at"] = time.time()


def stop() -> bool:
    """Ask the running scan to stop. It stops between frames, not mid-frame."""
    if JOB["status"] != "running":
        return False
    STOP.set()
    return True


def _scan_one_folder(app, folder: str, min_px: float, index: int, count: int) -> dict:
    """Scan one folder, reporting the bar in that folder's own terms.

    ``done`` and ``total`` are the work in this folder — frames still to read or
    to look at — because a denominator that changes half way through is a bar
    that jumps backwards. How far the whole request has got is carried by the
    folder's position instead, which only ever counts up.
    """
    from faces.service import scan_folder

    config = app.state.config
    JOB["folder_index"] = index
    JOB["folders"] = count

    def _progress(phase, done, total):
        JOB["phase"] = phase
        JOB["done"] = done
        JOB["total"] = total

    return scan_folder(
        app.state.catalog.db,
        app.state.thumb_cache,
        config.library_root,
        folder,
        config.models_dir,
        min_group_px=min_px,
        should_stop=STOP.is_set,
        progress=_progress,
        recognizer=recognizer(config),
        scan_size=getattr(config, "face_scan_size", "preview"),
        background=bool(JOB.get("automatic")),
    )


def _run(app, folders: list[str], threshold: float, min_px: float) -> None:
    """The scan, reporting as it goes."""
    from faces.models import ModelsUnavailable, ensure_models
    from faces.service import rebuild_people

    config = app.state.config
    try:
        JOB["phase"] = "models"
        JOB["detail"] = "checking the face models"
        try:
            ensure_models(config.models_dir, recognizer=recognizer(config))
        except ModelsUnavailable as exc:
            JOB.update(status="failed", error=str(exc), finished_at=time.time())
            return

        for index, folder in enumerate(folders, start=1):
            if STOP.is_set():
                break
            JOB["folder"] = folder
            JOB["detail"] = folder
            JOB["phase"] = "rendering"
            try:
                result = _scan_one_folder(app, folder, min_px, index, len(folders))
            except Exception as exc:  # a folder that cannot be read at all
                logger.warning("Face scan failed for %s: %s", folder, exc)
                JOB["failed"] += 1
                continue
            JOB["scanned"] += result["scanned"]
            JOB["skipped"] += result["skipped"]
            JOB["failed"] += result["failed"]
            JOB["faces"] += result["faces"]
            JOB["dropped"] += result.get("dropped", 0)

        if not STOP.is_set():
            JOB["phase"] = "grouping"
            JOB["detail"] = "grouping the faces"
            summary = rebuild_people(app.state.catalog.db, threshold, min_group_px=min_px,
                                     model=recognizer(config))
            JOB["people"] = summary["people"]

        JOB["status"] = "stopped" if STOP.is_set() else "completed"
        JOB["phase"] = ""
        JOB["detail"] = ""
        JOB["total"] = 0
        JOB["done"] = 0
    except Exception as exc:
        logger.exception("Face scan failed")
        JOB.update(status="failed", error=str(exc), phase="", detail="")
    finally:
        JOB["finished_at"] = time.time()


# ── The sweep ────────────────────────────────────────────────────────

def folders_needing_scan(db, model: str) -> dict[str, int]:
    """Folders with photographs no scan has looked at with ``model``, and how many.

    A photograph, not a file: a CR3 and its JPG are one, and the scan looks at
    only one half, so a pair counts as looked at when either half was.
    """
    from catalog.service import is_export_folder

    with db._connect() as conn:
        photos = conn.execute("SELECT folder, base_name, filepath FROM photos").fetchall()
        scanned = {
            row[0] for row in conn.execute(
                "SELECT filepath FROM face_scans "
                "WHERE (CASE WHEN model = '' THEN 'sface' ELSE model END) = ?", (model,))
        }
    looked_at: set[tuple[str, str]] = set()
    units: dict[str, set[str]] = defaultdict(set)
    for folder, base, filepath in photos:
        units[folder].add(base)
        if filepath in scanned:
            looked_at.add((folder, base))
    need = {}
    for folder, bases in units.items():
        missing = sum(1 for base in bases if (folder, base) not in looked_at)
        if missing and not is_export_folder(folder):
            need[folder] = missing
    return need


_sweep_lock = threading.Lock()
_sweep_wanted = False
_sweep_thread: Optional[threading.Thread] = None


def wake(app, delay: float = 0.0) -> None:
    """Scan whatever has not been looked at, in the background. Returns at once.

    A call while the sweep is running asks for one more look after it, so
    photographs indexed meanwhile are not left for the next start.
    """
    global _sweep_thread, _sweep_wanted
    if not getattr(app.state.config, "face_auto_scan", True):
        return
    if not getattr(app.state.catalog.db, "writable", True):
        return
    with _sweep_lock:
        _sweep_wanted = True
        if _sweep_thread is not None:
            return
        _sweep_thread = threading.Thread(target=_sweep_worker, args=(app, delay),
                                         name="faces-sweep", daemon=True)
        _sweep_thread.start()


def _sweep_worker(app, delay: float) -> None:
    global _sweep_thread, _sweep_wanted
    if delay:
        time.sleep(delay)
    while True:
        with _sweep_lock:
            if not _sweep_wanted:
                _sweep_thread = None
                return
            _sweep_wanted = False
        try:
            sweep_once(app)
        except Exception:
            logger.exception("Looking for photos to scan for faces failed")


def sweep_once(app, wait: Callable[[float], None] = time.sleep) -> dict:
    """One look: scan the folders with photographs not yet looked at.

    Waits for a scan already running (someone clicked) to finish first, since
    it may well cover the same folders. Returns what it found.
    """
    from faces.detect import available

    config = app.state.config
    usable, _reason = available(config.models_dir, False, recognizer(config))
    if not usable:
        return {"started": False, "reason": "models"}
    while JOB["status"] == "running":
        wait(5.0)
    need = folders_needing_scan(app.state.catalog.db, recognizer(config))
    if not need:
        return {"started": False, "folders": 0}
    folders = sorted(need, reverse=True)       # the newest first
    logger.info("Looking for faces in %d photo(s) not yet scanned, in %d folder(s)",
                sum(need.values()), len(folders))
    started = start(app, folders, SWEEP_SCOPE, automatic=True, run_here=True)
    return {"started": started, "folders": len(folders), "photos": sum(need.values())}
