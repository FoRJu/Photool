"""
The Library's folder cards and map, and the job that reads where photos were taken.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from catalog import clock
from places import gazetteer, gps, store

logger = logging.getLogger(__name__)

router = APIRouter()

_job_lock = threading.Lock()
_stop = threading.Event()
_job: dict = {
    "status": "idle", "phase": "", "folder": "", "folders": 0, "folder_index": 0,
    "total": 0, "done": 0, "found": 0, "error": "", "started_at": 0.0, "finished_at": 0.0,
}
_adopted: set[str] = set()


def _catalog(request: Request):
    return request.app.state.catalog


def _models_dir(request: Request) -> str:
    return request.app.state.config.models_dir


async def _blocking(fn, *args, **kwargs):
    from api.routes import _run_blocking

    return await _run_blocking(fn, *args, **kwargs)


# ── Cards ────────────────────────────────────────────────────────────

def _cards(request: Request) -> dict:
    from access import owner_of

    config = request.app.state.config
    db = _catalog(request).db
    with db._connect() as conn:
        folders = [r[0] for r in conn.execute("SELECT DISTINCT folder FROM photos").fetchall()]
    # Notes kept beside the photographs come back with a rebuilt catalog. Looked
    # for once per folder per run: it is a file check per folder on the photo
    # drive, and the answer does not change while the app is up.
    fresh = [f for f in folders if f not in _adopted]
    if fresh and db.writable:
        if not _adopted:
            store.adopt_home(db, config.library_root)
        store.adopt_sidecars(db, config.library_root, fresh)
        _adopted.update(fresh)
    names = gazetteer.get(_models_dir(request))
    result = store.build_cards(db, names, frozenset(db.demoted_shoots(owner_of(request)[0])))
    result["gazetteer"] = names is not None
    result["gps"] = store.gps_counts(db)
    return result


@router.get("/library/cards")
async def library_cards(request: Request):
    """One card per catalogued folder: cover, dates, place, people, tags, note."""
    return await _blocking(_cards, request)


class FolderNote(BaseModel):
    title: str = ""
    subtitle: str = ""
    cover: str = ""
    place: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    taken_on: str = ""      # YYYY-MM-DD, for the photos whose clock gave no date
    kind: str = ""          # what was shot: "school" (an event, culled to share) or "family"


@router.put("/library/cards/{folder:path}")
async def save_folder_note(folder: str, note: FolderNote, request: Request):
    """A folder's title, subtitle, cover and place, as a person wrote them.

    A folder opened for the first time is still being indexed, which for a few
    hundred RAW files takes minutes; its note can be written meanwhile; the
    card shows it once the folder is in the catalog. So a folder of the library
    on disk is enough, not only one the catalog already has.
    """
    from api.routes import _resolve_folder

    db = _catalog(request).db
    if not db.writable:
        raise HTTPException(status_code=503, detail="The catalog cannot be written")
    with db._connect() as conn:
        known = conn.execute("SELECT 1 FROM photos WHERE folder = ? LIMIT 1", (folder,)).fetchone()
    if not known:
        _resolve_folder(request.app.state.config, folder)   # 400 outside the library, 404 if missing
    saved = await _blocking(store.save_note, db, folder, note.model_dump(),
                            request.app.state.config.library_root)
    # Where it was decides what time it was there (catalog/clock.py).
    await _blocking(clock.refresh, db, folder)
    return {"ok": True, "folder": folder, "note": saved}


class Place(BaseModel):
    place: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


class ManyPlaces(Place):
    folders: list[str]


@router.put("/library/places")
async def set_many_places(req: ManyPlaces, request: Request):
    """One place for several folders at once; no coordinates clears it."""
    db = _catalog(request).db
    if not db.writable:
        raise HTTPException(status_code=503, detail="The catalog cannot be written")
    folders = [f for f in dict.fromkeys(req.folders) if f]
    with db._connect() as conn:
        known = {r[0] for r in conn.execute("SELECT DISTINCT folder FROM photos").fetchall()}
    folders = [f for f in folders if f in known]
    place = {"place": req.place, "lat": req.lat, "lon": req.lon} if req.lat is not None else None
    changed = await _blocking(store.set_place_many, db, folders, place,
                              request.app.state.config.library_root)
    for folder in folders:
        await _blocking(clock.refresh, db, folder)
    return {"ok": True, "changed": changed}


@router.get("/places/home")
async def get_home(request: Request):
    return {"home": await _blocking(store.get_home, _catalog(request).db)}


@router.put("/places/home")
async def put_home(req: Place, request: Request):
    """Where home is. Folders with no place count as home; no coordinates forgets it."""
    db = _catalog(request).db
    if not db.writable:
        raise HTTPException(status_code=503, detail="The catalog cannot be written")
    home = await _blocking(store.set_home, db, req.model_dump(), request.app.state.config.library_root)
    await _blocking(clock.refresh, db)
    return {"ok": True, "home": home}


@router.get("/library/cards/{folder:path}/covers")
async def folder_covers(folder: str, request: Request):
    """Frames worth offering as the folder's cover."""
    db = _catalog(request).db
    return {"folder": folder, "filepaths": await _blocking(store.cover_candidates, db, folder)}


# ── Place names ──────────────────────────────────────────────────────

@router.get("/places/search")
async def places_search(request: Request, q: str = "", limit: int = 8):
    """Towns by name, for giving a folder a place. Fetches the list the first time."""
    if not q.strip():
        return {"places": []}
    try:
        names = await _blocking(gazetteer.get, _models_dir(request), True)
    except OSError as exc:
        raise HTTPException(status_code=503,
                            detail=f"The list of place names could not be fetched: {exc}")
    home = request.query_params.get("home", "")
    home = home.upper()[:2]
    return {"places": [p.as_dict(home) for p in names.search(q, min(20, max(1, limit)), prefer=home)]}


# ── Reading GPS ──────────────────────────────────────────────────────

def _scan(models_dir: str, library_root: str, db, folder: Optional[str]) -> None:
    try:
        _job.update(phase="names")
        try:
            gazetteer.get(models_dir, fetch=True)
        except OSError as exc:
            # The coordinates are worth reading without names: the map works
            # with them, and the names arrive when the list can be fetched.
            logger.warning("Place names unavailable: %s", exc)

        pending = store.gps_pending(db, folder)
        groups: dict[str, list] = {}
        for row in pending:
            groups.setdefault(row["folder"], []).append(row)
        _job.update(phase="reading", total=len(pending), folders=len(groups))
        root = Path(library_root)
        for index, (name, rows) in enumerate(groups.items(), start=1):
            if _stop.is_set():
                break
            _job.update(folder=name, folder_index=index)
            paths = {root / r["filepath"]: r for r in rows}
            found = gps.read(list(paths), stop=_stop)
            if found is None:
                if index == 1 and not _stop.is_set():
                    raise RuntimeError("exiftool could not be run, so no location was read")
                continue
            batch = []
            for path, coords in found.items():
                r = paths[path]
                lat, lon = coords if coords else (None, None)
                batch.append((r["filepath"], lat, lon, r["file_size"]))
                if coords:
                    _job["found"] += 1
            store.record_gps(db, batch)
            _job["done"] += len(rows)
        _job.update(status="stopped" if _stop.is_set() else "completed")
    except Exception as exc:
        logger.exception("Reading locations failed")
        _job.update(status="failed", error=str(exc))
    finally:
        _job.update(phase="", folder="", finished_at=time.time())


# ── Reading new photos on their own ─────────────────────────────────
#
# A whole-library pass is a button, because on a library whose cameras record
# no position it is twenty minutes of disk for nothing. New photographs are
# different: they are few, they arrive a folder at a time, and if a camera does
# start recording position (Canon's phone app can lend it the phone's), it
# should simply show up. So a folder that indexing has just added photos to is
# read in the background, one folder at a time.

_auto_lock = threading.Lock()
_auto_queue: list[tuple] = []
_auto_running = False


def _auto_worker() -> None:
    global _auto_running
    while True:
        with _auto_lock:
            if not _auto_queue:
                _auto_running = False
                return
            models_dir, library_root, db, folder = _auto_queue.pop(0)
        if _job["status"] == "running":
            continue                    # the whole-library pass will cover it
        import background as gate

        gate.wait_turn("Places")
        try:
            rows = store.gps_pending(db, folder)
            if not rows:
                continue
            try:
                gazetteer.get(models_dir, fetch=True)
            except OSError:
                pass
            root = Path(library_root)
            paths = {root / r["filepath"]: r for r in rows}
            found = gps.read(list(paths))
            if found is None:
                continue
            store.record_gps(db, [
                (paths[p]["filepath"], *(c if c else (None, None)), paths[p]["file_size"])
                for p, c in found.items()
            ])
        except Exception:
            logger.exception("Reading locations for %s failed", folder)


def read_new_photos(app, folder: str) -> None:
    """Queue a folder whose photos were just indexed. Returns at once."""
    global _auto_running
    config, catalog = app.state.config, app.state.catalog
    if not catalog.db.writable:
        return
    with _auto_lock:
        if any(item[3] == folder for item in _auto_queue):
            return
        _auto_queue.append((config.models_dir, config.library_root, catalog.db, folder))
        if _auto_running:
            return
        _auto_running = True
    threading.Thread(target=_auto_worker, name="places-auto", daemon=True).start()


def auto_idle() -> bool:
    with _auto_lock:
        return not _auto_running and not _auto_queue


class ScanRequest(BaseModel):
    folder: Optional[str] = None


@router.post("/places/scan")
async def places_scan(req: ScanRequest, request: Request):
    """Read the coordinates of every photo not read yet (or of one folder)."""
    db = _catalog(request).db
    if not db.writable:
        raise HTTPException(status_code=503, detail="The catalog cannot be written")
    with _job_lock:
        if _job["status"] == "running":
            return {"ok": True, "started": False, "job": dict(_job)}
        _stop.clear()
        _job.update(status="running", phase="", folder="", folders=0, folder_index=0,
                    total=0, done=0, found=0, error="", started_at=time.time(), finished_at=0.0)
        config = request.app.state.config
        threading.Thread(
            target=_scan, args=(config.models_dir, config.library_root, db, req.folder),
            name="places-scan", daemon=True,
        ).start()
    return {"ok": True, "started": True, "job": dict(_job)}


@router.post("/places/scan/stop")
async def places_scan_stop():
    if _job["status"] != "running":
        return {"ok": True, "stopped": False}
    _stop.set()
    return {"ok": True, "stopped": True}


@router.get("/places/status")
async def places_status(request: Request):
    db = _catalog(request).db
    counts = await _blocking(store.gps_counts, db)
    return {"gazetteer": gazetteer.is_ready(_models_dir(request)), "gps": counts, "job": dict(_job)}
