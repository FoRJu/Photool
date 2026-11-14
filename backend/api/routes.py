"""
API routes for the photo manager.

All folder paths are generic — no assumption about depth or naming.
A "folder" is any relative path under library_root that contains image files.
e.g. "2025/2025-03-06" or "2024/06/15" or "vacation/hawaii"
"""

import asyncio
import json
import logging
import os
import threading
import time
import urllib.parse
import uuid
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from importer.sd_detect import detect_sd_cards, find_importable_files
from importer.copy import import_files
from thumbnails import (
    get_thumbnail,
    get_preview,
    ThumbnailCache,
    ThumbnailSize,
    CacheNotWritable,
    generate_thumbnails_for_folder,
    generate_thumbnails_batch,
    ThumbnailQueueManager,
)
from thumbnails.extract import get_exif_summary
from catalog import CatalogService
from config import is_video
from access import is_local, owner_of
from storage_health import probe_writable
import exiftool_path

router = APIRouter()


# ── Models ───────────────────────────────────────────────────────────

class SDCardResponse(BaseModel):
    mount_point: str
    total_files: int
    total_size_mb: float
    device: str = ""
    videos: int = 0

    class Config:
        from_attributes = True


class ImportRequest(BaseModel):
    mount_point: str
    verify_checksums: bool = True
    parallel_workers: int = 4


class ImportResultResponse(BaseModel):
    total_files: int
    copied: int
    skipped: int
    failed: int
    elapsed_seconds: float


class ImportJobResponse(BaseModel):
    job_id: str
    total_files: int


class RatingRequest(BaseModel):
    filepath: str               # relative path e.g. "2025/2025-03-06/IMG_0001.CR3"
    rating: Optional[int] = None
    flag: Optional[int] = None
    color_label: Optional[int] = None


class EnqueueRequest(BaseModel):
    folder: str


class ReorderRequest(BaseModel):
    direction: str  # "up" | "down"


class BasketAddRequest(BaseModel):
    filepath: str
    folder: str
    filename: str


class BasketBatchAddRequest(BaseModel):
    items: list[BasketAddRequest]


class PhotoFilterParams(BaseModel):
    """The grid's filter state, reused by bulk operations."""
    min_rating: int = 0
    flag: Optional[int] = None
    color: Optional[int] = None
    tag: Optional[str] = None
    unreviewed: bool = False
    pair_mode: str = "paired"
    camera: Optional[str] = None     # "" for the frames no camera is known for
    hide_cameras: Optional[list[str]] = None   # demoted in this folder, so not on the grid
    bases: Optional[list[str]] = None   # only these pairs: what a search in the folder found


class BasketFilterAddRequest(PhotoFilterParams):
    folder: str


class AdjustmentParams(BaseModel):
    """One person's own edit for a single photo, in the terms the UI shows.

    The ranges are the sliders'; the meaning — stops, channel gains, and gains
    over part of the tonal range — lives in :mod:`adjust.develop`, which is also
    what turns them into pixels. The tone sliders are on Lightroom's -100..+100
    scale so that a value written to a sidecar means the same thing there.

    Only the long-standing controls are spelled out. The rest — the colour
    mixer's ``hue_red``…``lum_magenta`` and the tone curves ``curve``,
    ``curve_r``, ``curve_g``, ``curve_b`` as ``[[x, y], …]`` on 0..255 — are
    accepted as extra fields and cleaned by :func:`adjust.develop.normalize`,
    which drops anything it does not know. A new control then needs no change
    here.
    """
    model_config = ConfigDict(extra="allow")

    exposure: float = 0.0      # stops: -5.0 darker .. +5.0 brighter
    temperature: int = 0       # -100 cool .. +100 warm
    tint: int = 0              # -100 green .. +100 magenta
    # White balance, absolute (adjust.whitebalance): absent is as shot. A tint is
    # read only beside a colour temperature.
    wb_kelvin: Optional[int] = None   # 2000 .. 50000, higher warmer
    wb_tint: Optional[int] = None     # -150 green .. +150 magenta
    contrast: int = 0          # -100 flat .. +100 punchy
    highlights: int = 0        # -100 recover .. +100 brighten
    shadows: int = 0           # -100 deepen .. +100 open up
    whites: int = 0            # -100 .. +100, the top of the range only
    blacks: int = 0            # -100 .. +100, the bottom of the range only
    flat: int = 0              # 0 .. 100, how far to leave the camera's own look behind
    # A learned style (adjust.styles) by id, rendered instead of the camera's own
    # look, and how far to fade it back towards that look.
    style: str = ""
    style_fade: int = 0        # 0 .. 100
    compress: int = 0          # 0 .. 100, squeeze the scene's range (HDR to a normal picture)
    clarity: int = 0           # -100 .. +100, mid-tone local contrast
    vibrance: int = 0          # -100 .. +100, favours the flat colours
    saturation: int = 0        # -100 .. +100
    fade: int = 0              # 0 .. 100, lifted blacks, the matte look
    shadow_hue: int = 0        # 0 .. 360, split toning
    shadow_sat: int = 0        # 0 .. 100
    highlight_hue: int = 0     # 0 .. 360
    highlight_sat: int = 0     # 0 .. 100
    # Framing. A crop is the kept rectangle as fractions of the whole frame after
    # the turn; all four or none. `angle` is degrees clockwise (-45..45).
    angle: float = 0.0
    crop_x: Optional[float] = None
    crop_y: Optional[float] = None
    crop_w: Optional[float] = None
    crop_h: Optional[float] = None


class TagCreateRequest(BaseModel):
    name: str
    kind: str = "event"


class TagRenameRequest(BaseModel):
    old_name: str
    new_name: str


class PhotoTagsRequest(BaseModel):
    filepath: str
    tags: list[str] = []


class TagBulkRequest(PhotoFilterParams):
    folder: str
    tag: str
    remove: bool = False


class BurstFilterRequest(PhotoFilterParams):
    """Folder plus the grid's filters — the scope a burst clean-up applies to."""
    folder: str


class BestRequest(BurstFilterRequest):
    """How many of a folder's frames to suggest."""
    count: int = 20


class BurstRejectRequest(BaseModel):
    """Reject the frames of one burst, leaving ``keep`` untouched."""
    keep: str
    others: list[str] = []


class FolderStateRequest(BaseModel):
    """Where the user stopped in a folder; empty clears it."""
    last_filepath: Optional[str] = None


class ExportRunRequest(BaseModel):
    # A folder name under <library>/export ("" = now's date and time). Only this
    # computer writes there; a paired device always gets a download instead.
    folder: str = ""
    photographer: str = ""
    max_resolution: int = 2048
    quality: int = 85
    strip_exif: bool = True
    # Each photograph is written as it stands (export.engine.export_photo with
    # resize_only): as developed, or the camera's own picture. The batch styles
    # and set matching this once took are gone; a field naming them is ignored.


# ── Helpers ──────────────────────────────────────────────────────────

async def _run_blocking(fn, *args, **kwargs):
    """
    Run a blocking filesystem/CPU call off the event loop.

    Directory walks and EXIF reads are synchronous; calling them directly from
    an async handler stalls every other request — including import progress
    polling while an import is running.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


def _get_cache(request: Request) -> ThumbnailCache:
    if not hasattr(request.app.state, "thumb_cache"):
        config = request.app.state.config
        max_bytes = max(0, int(config.thumbnail_cache_max_mb or 0)) * 1024 * 1024
        request.app.state.thumb_cache = ThumbnailCache(
            config.thumbnail_cache, max_bytes=max_bytes
        )
    return request.app.state.thumb_cache


def _storage_status(request: Request) -> dict:
    """
    Current write access for the two databases, re-probed on demand.

    Startup reports it once, but a drive can drop out while the app is running;
    /health re-checks so the UI banner can go red without a restart. Transitions
    are logged, not every poll.
    """
    config = request.app.state.config
    previous = getattr(request.app.state, "storage", None)
    cache = getattr(request.app.state, "thumb_cache", None)

    def block(label: str, path: str) -> dict:
        writable, error = probe_writable(path)
        return {"label": label, "path": str(path), "writable": writable, "error": error}

    storage = {
        "catalog": block("catalog database", config.db_path),
        "thumbnail_cache": block(
            "thumbnail cache", str(Path(config.thumbnail_cache) / "thumbnails.db")
        ),
    }
    # The store knows about failures a filesystem probe cannot see, e.g. a write
    # that SQLite refused mid-session. Trust the store.
    if cache is not None and not cache.writable:
        storage["thumbnail_cache"]["writable"] = False
        storage["thumbnail_cache"]["error"] = cache.write_error or storage["thumbnail_cache"]["error"]

    blocks = [storage["catalog"], storage["thumbnail_cache"]]
    ok = all(b["writable"] for b in blocks)
    storage["ok"] = ok
    request.app.state.storage = storage

    log = logging.getLogger(__name__)
    was_ok = True if previous is None else bool(previous.get("ok", True))
    if was_ok and not ok:
        for b in blocks:
            if not b["writable"]:
                log.error("[storage] %s is not writable: %s", b["label"], b["error"])
    elif not was_ok and ok:
        log.info("[storage] write access restored")
    return storage


def _get_catalog(request: Request) -> CatalogService:
    return request.app.state.catalog


def _get_queue_manager(request: Request) -> ThumbnailQueueManager:
    return request.app.state.queue_manager


_PAIR_MODES = {"all", "raw", "jpg", "paired", "raw_only", "jpg_only"}


def _filter_kwargs(params: PhotoFilterParams) -> dict:
    """Translate grid filter state into CatalogDB.get_photos_in_folder kwargs."""
    pair_mode = params.pair_mode if params.pair_mode in _PAIR_MODES else "paired"
    return {
        "min_rating": max(0, min(5, int(params.min_rating or 0))),
        "flag_filter": params.flag,
        "color_filter": params.color,
        "tag_filter": (params.tag or "").strip() or None,
        "unreviewed_only": bool(params.unreviewed),
        "pair_mode": pair_mode,
        "camera_filter": params.camera,
        "hide_cameras": params.hide_cameras,
        "base_names": params.bases,
    }


def _require_writable(catalog) -> None:
    """Fail loudly when a write cannot land, instead of reporting success.

    The catalog and the sidecars usually live on a secondary volume that can go
    read-only while reads keep working. A bulk action that quietly did nothing
    is worse than a 503.
    """
    if not catalog.db.writable:
        raise HTTPException(
            status_code=503,
            detail=f"Catalog is not writable: {catalog.db.write_error}",
        )


def _resolve_folder(config, folder_path: str) -> Path:
    """Resolve a relative folder path under library_root, raising 404 if missing."""
    folder = Path(config.library_root) / folder_path
    # Security: prevent path traversal
    try:
        folder.resolve().relative_to(Path(config.library_root).resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid folder path")
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    return folder


def _resolve_file(config, folder_path: str, filename: str) -> Path:
    """Resolve a file within a library folder."""
    folder = _resolve_folder(config, folder_path)
    filepath = folder / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return filepath


def _discover_image_folders(root: Path, extensions: set[str]) -> list[dict]:
    """
    Recursively find all folders under root that contain image files.
    Works with any folder structure / depth.
    """
    folders = []
    root = root.resolve()

    from catalog.service import is_export_folder

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories, and the export folder: what is written there
        # is a copy of photographs already in the library.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and not (Path(dirpath).resolve() == root and is_export_folder(d))]

        image_count = sum(
            1 for f in filenames
            if not f.startswith(".") and Path(f).suffix.lower() in extensions
        )

        if image_count > 0:
            rel_path = str(Path(dirpath).resolve().relative_to(root))
            # Normalize path separators to forward slash
            rel_path = rel_path.replace("\\", "/")
            if rel_path == ".":
                continue  # skip root itself

            folders.append({
                "path": rel_path,
                "file_count": image_count,
            })

    return sorted(folders, key=lambda f: f["path"], reverse=True)


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/health")
async def health(request: Request):
    """Liveness plus write access for the catalog and thumbnail cache.

    "ok" means the app can persist what it is asked to persist. "degraded"
    means it is serving reads from a volume that will not accept writes —
    browsing works, ratings and thumbnails do not stick.
    """
    storage = await _run_blocking(_storage_status, request)
    return {
        "status": "ok" if storage["ok"] else "degraded",
        "version": request.app.version,
        "storage": storage,
        # Helper programs the app reads with; one that is missing is said on the page.
        "tools": {"exiftool": exiftool_path.available()},
    }


@router.get("/sd-cards", response_model=list[SDCardResponse])
async def list_sd_cards(request: Request):
    config = request.app.state.config
    cards = await _run_blocking(detect_sd_cards, config)
    return [
        SDCardResponse(
            mount_point=str(card.mount_point),
            total_files=card.total_files,
            total_size_mb=round(card.total_size_mb, 1),
            device=card.device,
            videos=card.videos,
        )
        for card in cards
    ]


# ── Import (background job + progress) ───────────────────────────────
#
# An import can run for minutes on a full card, so it is not a request/response
# call any more: POST /import starts it, GET /import/progress/{job_id} reports
# where it is. Without that, a large import looks like a hang.

_import_jobs: dict[str, dict] = {}
_import_jobs_lock = threading.Lock()
_IMPORT_JOBS_KEEP = 10  # finished jobs retained for the UI


def _prune_import_jobs():
    """Drop the oldest finished jobs so the table cannot grow without bound."""
    if len(_import_jobs) <= _IMPORT_JOBS_KEEP:
        return
    finished = [
        (j["started_at"], jid) for jid, j in _import_jobs.items()
        if j["status"] != "running"
    ]
    for _, jid in sorted(finished)[: len(_import_jobs) - _IMPORT_JOBS_KEEP]:
        _import_jobs.pop(jid, None)


@router.post("/import", response_model=ImportJobResponse)
async def run_import(req: ImportRequest, request: Request):
    config = request.app.state.config
    cards = await _run_blocking(detect_sd_cards, config)
    card = next(
        (c for c in cards if str(c.mount_point) == req.mount_point), None
    )
    if not card:
        return ImportJobResponse(job_id="", total_files=0)

    files = await _run_blocking(find_importable_files, card, config.import_extensions)
    catalog = _get_catalog(request)

    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id": job_id,
        "status": "running",
        "stage": "starting",
        "total": len(files),
        "completed": 0,
        "copied": 0,
        "skipped": 0,
        "failed": 0,
        "current_file": "",
        "error": "",
        "started_at": time.time(),
        "finished_at": None,
        "result": None,
    }
    with _import_jobs_lock:
        _import_jobs[job_id] = job
        _prune_import_jobs()

    def _progress(current: int, total: int, filename: str):
        job["completed"] = current
        job["total"] = total
        job["current_file"] = filename

    def _stage(stage: str):
        job["stage"] = stage

    def _do_import():
        try:
            summary = import_files(
                files,
                config,
                progress_cb=_progress,
                verify_checksums=req.verify_checksums,
                parallel_workers=req.parallel_workers,
                stage_cb=_stage,
            )
            job["result"] = {
                "total_files": summary.total_files,
                "copied": summary.copied,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "elapsed_seconds": round(summary.elapsed_seconds, 2),
            }
            job["copied"] = summary.copied
            job["skipped"] = summary.skipped
            job["failed"] = summary.failed
            if summary.copied and catalog.db.writable:
                job["stage"] = "indexing"
                _index_imported(request.app, summary)
            job["status"] = "completed"
        except Exception as e:  # pragma: no cover - defensive
            logging.getLogger(__name__).exception("Import failed")
            job["status"] = "failed"
            job["error"] = str(e)
        finally:
            job["current_file"] = ""
            job["finished_at"] = time.time()

    threading.Thread(target=_do_import, daemon=True, name="import-job").start()
    return ImportJobResponse(job_id=job_id, total_files=len(files))


@router.get("/import/progress/{job_id}")
async def import_progress(job_id: str):
    job = _import_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Import job not found")
    payload = dict(job)
    payload["elapsed_seconds"] = round(
        (payload.pop("finished_at") or time.time()) - payload.pop("started_at"), 1
    )
    return payload


# ── Library browsing (filesystem-based) ──────────────────────────────

@router.get("/library/folders")
async def list_library_folders(request: Request):
    """Discover all folders containing images. Works with any folder structure."""
    config = request.app.state.config
    root = Path(config.library_root)
    if not root.exists():
        return {"folders": []}

    extensions = {ext.lower() for ext in config.photo_extensions}
    folders = await _run_blocking(_discover_image_folders, root, extensions)
    return {"folders": folders}


def _list_folder_files(folder: Path, extensions: Optional[set] = None) -> list[dict]:
    """List image files in one library folder (blocking).

    Only the file types the library holds: a folder's XMP sidecars sit beside
    its photographs, and listing them made the grid ask for a thumbnail of each
    one — hundreds of failed requests on every visit to an edited folder.
    """
    files = []
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        try:
            if not entry.is_file(follow_symlinks=False) or entry.name.startswith("."):
                continue
            if extensions is not None and Path(entry.name).suffix.lower() not in extensions:
                continue
            files.append({
                "name": entry.name,
                "path": entry.path,
                "size_mb": round(entry.stat().st_size / (1024 * 1024), 2),
                "extension": Path(entry.name).suffix.lower(),
            })
        except OSError:
            continue
    return files


@router.get("/library/files/{folder_path:path}")
async def list_files_in_folder(folder_path: str, request: Request):
    """List all image files in a folder."""
    config = request.app.state.config
    folder = _resolve_folder(config, folder_path)
    extensions = {ext.lower() for ext in config.photo_extensions}
    return {"files": await _run_blocking(_list_folder_files, folder, extensions)}


# ── Thumbnail endpoints ──────────────────────────────────────────────

@router.get("/thumb/{folder_path:path}")
async def get_thumb(
    folder_path: str,
    request: Request,
    size: str = Query("grid", pattern="^(grid|preview|full)$"),
    v: str = Query("", max_length=64),
    camera: bool = Query(False),
    ver: int = Query(-1, ge=-1),
):
    """
    Get a thumbnail. folder_path includes the filename.
    e.g. /api/thumb/2025/2025-03-06/IMG_0001.CR3?size=grid

    ``camera`` asks for the camera's own picture even of an edited frame: what Develop
    shows at once while it opens, since drawing the edit may have to learn its look.
    ``ver`` draws one version (0 the photo's own edit, 1.. a virtual copy); without
    it the photo's cover is drawn, the version chosen to be seen while browsing.
    """
    config = request.app.state.config
    full_path = Path(config.library_root) / folder_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    cache = _get_cache(request)
    size_map = {
        "grid": ThumbnailSize.GRID,
        "preview": ThumbnailSize.PREVIEW,
        "full": ThumbnailSize.FULL,
    }
    thumb_size = size_map[size]

    loop = asyncio.get_event_loop()
    try:
        thumb_data = None
        edited = False
        if size != "full" and not camera:
            # A frame with an edit shows it. The client's ``v`` only busts the
            # browser's copy; whether to draw the edit is the catalog's call.
            from export.engine import resolve_export_source
            from thumbnails.edited import get_edited_thumbnail

            catalog = _get_catalog(request)
            settings = await _run_blocking(catalog.edit_of, folder_path, None if ver < 0 else ver)
            if settings:
                develop_source = resolve_export_source(full_path, config.export_prefer_raw)
                thumb_data = await loop.run_in_executor(
                    None,
                    lambda: get_edited_thumbnail(develop_source, settings, cache, thumb_size),
                )
                edited = thumb_data is not None
        if thumb_data is None:
            thumb_data = await loop.run_in_executor(
                None, lambda: get_thumbnail(full_path, cache, thumb_size)
            )
    except CacheNotWritable as exc:
        # A generated thumbnail that cannot be stored is not a 500 with an
        # sqlite traceback — it means the cache volume is read-only, which the
        # caller can act on.
        raise HTTPException(status_code=503, detail=f"Thumbnail cache not writable: {exc}")
    if not thumb_data:
        # A file with no picture (an empty recording, a damaged file) shows as a
        # tile that says so, wherever a picture of it is asked for.
        from config import is_video
        from thumbnails.placeholder import no_picture_jpeg

        what = "No video in this file" if is_video(full_path.suffix.lower()) else "No picture in this file"
        return Response(content=no_picture_jpeg(what, thumb_size), media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache", "X-Photool-No-Picture": "1"})

    # An edited frame can change under the same URL, so it may be kept only when
    # the URL carries the edit's version; without one, the browser must ask.
    cacheable = not edited or bool(v)
    return Response(
        content=thumb_data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400, immutable" if cacheable else "no-cache"},
    )


@router.get("/exif/{folder_path:path}")
async def get_exif(folder_path: str, request: Request):
    """Get EXIF for a file. folder_path includes the filename."""
    config = request.app.state.config
    full_path = Path(config.library_root) / folder_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    loop = asyncio.get_event_loop()
    exif = await loop.run_in_executor(None, lambda: get_exif_summary(full_path))
    return exif


@router.post("/thumbnails/generate/{folder_path:path}")
async def pregenerate_thumbnails(folder_path: str, request: Request):
    config = request.app.state.config
    folder = _resolve_folder(config, folder_path)
    cache = _get_cache(request)

    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None,
        lambda: generate_thumbnails_for_folder(
            folder, cache, config.photo_extensions
        ),
    )
    return stats


@router.get("/thumbnails/cache-stats")
async def thumbnail_cache_stats(request: Request):
    cache = _get_cache(request)
    return cache.stats()


@router.post("/thumbnails/prune")
async def prune_thumbnails(request: Request):
    """Drop cached thumbnails whose source file no longer exists."""
    cache = _get_cache(request)
    try:
        return await _run_blocking(cache.prune)
    except CacheNotWritable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/thumbnails/vacuum")
async def vacuum_thumbnails(request: Request):
    """Rewrite the cache file so freed pages go back to the disk."""
    cache = _get_cache(request)
    try:
        return await _run_blocking(cache.vacuum)
    except CacheNotWritable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


class PrefetchRequest(BaseModel):
    filepaths: list[str]  # relative paths to images
    size: str = "grid"    # grid | preview


@router.post("/thumbnails/prefetch")
async def prefetch_thumbnails(req: PrefetchRequest, request: Request):
    """
    Pre-generate thumbnails for a specific list of files.
    YouTube-style buffering: the frontend requests ahead of scroll/cull position.
    Uses all CPU cores for parallel extraction.
    """
    config = request.app.state.config
    cache = _get_cache(request)

    size_map = {
        "grid": ThumbnailSize.GRID,
        "preview": ThumbnailSize.PREVIEW,
        "full": ThumbnailSize.FULL,
    }
    thumb_size = size_map.get(req.size, ThumbnailSize.GRID)

    files = []
    for fp in req.filepaths:
        full = Path(config.library_root) / fp
        if full.exists():
            files.append(full)

    if not files:
        return {"total": 0, "generated": 0, "cached": 0, "failed": 0}

    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None,
        lambda: generate_thumbnails_batch(files, cache, thumb_size),
    )
    return stats


# ── Catalog / rating endpoints ───────────────────────────────────────

@router.post("/catalog/index/{folder_path:path}")
async def index_folder(folder_path: str, request: Request):
    """Index a folder into the catalog."""
    config = request.app.state.config
    catalog = _get_catalog(request)

    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None,
        lambda: catalog.index_folder(
            folder_path,
            config.photo_extensions,
            exif_extractor=get_exif_summary,
        ),
    )
    # A folder already indexed answers "cached" and costs nothing here.
    if stats.get("indexed") and not stats.get("cached"):
        _new_photos(request.app, [folder_path])
    return stats


def _new_photos(app, folders: list[str]) -> None:
    """Photographs that were not in the catalog before: in the background, read
    where they were taken and what they show, fill in their camera data, and
    look for faces in them. Returns at once."""
    from analysis import clip
    from catalog import clock, exif_fill
    from export import look_ahead
    from places.routes import read_new_photos

    config, db = app.state.config, app.state.catalog.db
    for folder in folders:
        read_new_photos(app, folder)
    exif_fill.wake(config.library_root, db)
    look_ahead.wake()
    clock.wake(config.library_root, db)
    clip.wake(config.library_root, db, app.state.thumb_cache)
    face_jobs.wake(app)


def _index_imported(app, summary) -> list[str]:
    """Index the folders an import wrote to, so the photographs are in the
    library without anyone opening each folder first."""
    root = Path(app.state.config.library_root).resolve()
    folders = set()
    for result in summary.results:
        if result.status not in ("copied", "renamed"):
            continue
        try:
            rel = Path(result.destination).resolve().parent.relative_to(root)
        except ValueError:
            continue
        folders.add(str(rel).replace("\\", "/"))
    catalog = app.state.catalog
    indexed = []
    for folder in sorted(folders):
        try:
            stats = catalog.index_folder(folder, app.state.config.photo_extensions,
                                         exif_extractor=get_exif_summary)
        except Exception:
            logging.getLogger(__name__).exception("Indexing %s after import failed", folder)
            continue
        if stats.get("indexed"):
            indexed.append(folder)
    if indexed:
        _new_photos(app, indexed)
    return indexed


@router.post("/catalog/rebuild")
async def rebuild_catalog(request: Request):
    config = request.app.state.config
    catalog = _get_catalog(request)
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None, lambda: catalog.rebuild_from_xmp(config.photo_extensions)
    )
    # The rebuild indexes without camera data, to be quick; it is read after.
    from catalog import exif_fill

    exif_fill.wake(config.library_root, catalog.db)
    return stats


@router.get("/catalog/folders")
async def catalog_folders(request: Request):
    catalog = _get_catalog(request)
    return {"folders": catalog.get_all_folders()}


@router.get("/catalog/photos/{folder_path:path}")
async def catalog_photos(
    folder_path: str,
    request: Request,
    min_rating: int = Query(0, ge=0, le=5),
    flag: Optional[int] = Query(None, ge=-1, le=1),
    color: Optional[int] = Query(None, ge=0, le=5),
    tag: Optional[str] = Query(None),
    unreviewed: bool = Query(False),
    pair_mode: str = Query("all", pattern="^(all|raw|jpg|paired|raw_only|jpg_only)$"),
    camera: Optional[str] = Query(None, max_length=200),
    hide_camera: list[str] = Query([]),
    sort: str = Query("capture_date"),
    desc: bool = Query(False),
):
    catalog = _get_catalog(request)
    photos = catalog.get_photos_in_folder(
        folder_path,
        min_rating=min_rating,
        flag_filter=flag,
        color_filter=color,
        tag_filter=(tag or "").strip() or None,
        unreviewed_only=unreviewed,
        pair_mode=pair_mode,
        camera_filter=camera,
        hide_cameras=hide_camera,
        sort_by=sort,
        sort_desc=desc,
    )
    tag_map = catalog.db.get_tags_for_filepaths([p.filepath for p in photos])
    # One query for the folder, so the grid can mark the edited frames without
    # asking about each one. The settings travel with the flag: the develop view
    # opens on a frame the user just clicked, and it should not need a second
    # round trip to know what the sliders were.
    edits = await _run_blocking(catalog.photo_edits, folder_path)
    edited = await _run_blocking(catalog.edited_filepaths, folder_path)
    versions = await _run_blocking(catalog.version_info, [p.filepath for p in photos])
    return {
        "photos": [
            _photo_row(p, tag_map.get(p.filepath, []), edits.get(p.filepath, {}), p.filepath in edited,
                       versions.get(p.filepath))
            for p in photos
        ],
    }


def _photo_row(p, tags: list, edit: dict, edited: bool, versions: Optional[dict] = None) -> dict:
    """One photo as the grid draws it. ``versions`` is its entry of
    ``catalog.version_info``: how many virtual copies, and which is its cover."""
    versions = versions or {}
    return {
        "filepath": p.filepath,
        "filename": p.filename,
        "folder": p.folder,
        "extension": p.extension,
        "base_name": p.base_name,
        "file_size": p.file_size,
        "rating": p.rating,
        "flag": p.flag,
        "color_label": p.color_label,
        "tags": tags,
        "camera": p.camera,
        "lens": p.lens,
        "focal_length": p.focal_length,
        "aperture": p.aperture,
        "shutter_speed": p.shutter_speed,
        "iso": p.iso,
        "capture_date": p.capture_date,
        "dimensions": p.dimensions,
        "video": is_video(p.extension),
        "duration": p.duration,
        "edit": edit,
        "edited": edited,
        "copies": versions.get("copies", 0),
        "cover": versions.get("cover", 0),
        "cover_edit": versions.get("cover_edit", {}),
    }


@router.get("/catalog/search")
async def search_library(
    request: Request,
    q: str = Query("", max_length=1000),
    sort: str = Query("", pattern="^(|relevance|date|rating|filename|folder)$"),
    desc: bool = Query(True),
    limit: int = Query(1000, ge=1, le=5000),
    describe_words: bool = Query(False, alias="describe"),
    folder: str = Query("", max_length=1000),
):
    """Photos anywhere in the library matching a query (see catalog/search.py),
    or in one ``folder`` only: a search typed while a folder is open.

    The rows are the grid's own, so a result set is drawn like a folder. A pair
    is one row, carrying the pair's rating, flag and label whichever file they
    were set on. ``understood`` is the query as it was read, for the search box
    to show; a query that cannot be read is a 400 saying which term and why.
    With ``describe``, plain words say what the pictures show (smart search).
    """
    from catalog.search import SearchError, describe, unclear

    catalog = _get_catalog(request)
    try:
        found = await _run_blocking(catalog.search, q, sort, desc, owner_of(request)[0],
                                    describe_words, folder)
    except SearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Demoted shoots come after the rest, in their own order: nothing is
    # hidden, but what this person finds less important is not what they see first.
    demoted = catalog.demoted(owner_of(request)[0])
    kept = [p for p in found if not demoted(p)]
    found = kept + [p for p in found if demoted(p)]
    page = found[:limit]
    rows = await _run_blocking(_year_rows, catalog, page)
    return {"query": q, "understood": describe(q, describe_words), "total": len(found), "photos": rows,
            "unclear": unclear(found), "demoted": len(found) - len(kept)}


@router.get("/library/smart-search")
async def smart_search_status(request: Request):
    """Whether searching by what the pictures show is set up, how far the
    library is described, and the download's progress while it is fetched."""
    from analysis import clip

    catalog = _get_catalog(request)
    return {"available": clip.available(), "setup": clip.setup_state(), "job": clip.job(),
            "download_mb": clip.DOWNLOAD_MB, **(await _run_blocking(clip.counts, catalog.db))}


@router.post("/library/smart-search/setup")
async def smart_search_setup(request: Request):
    """Fetch the models (once) and start describing the library in the
    background. Returns at once; the status says how it goes."""
    from analysis import clip

    config = request.app.state.config
    catalog = _get_catalog(request)
    cache = request.app.state.thumb_cache
    clip.setup(on_ready=lambda: clip.wake(config.library_root, catalog.db, cache))
    return {"ok": True, "setup": clip.setup_state()}


@router.get("/library/memories")
async def library_memories(
    request: Request,
    day: str = Query("", pattern=r"^(\d{4}-\d\d-\d\d)?$"),
):
    """On this day: earlier years' best photographs from today's date (or
    ``day``), each year with its frames as the grid draws them."""
    from datetime import date as _date

    catalog = _get_catalog(request)
    today = _date.fromisoformat(day) if day else _date.today()
    # Copies: the years are kept for the next visit, and their photos become rows here.
    years = [dict(y) for y in await _run_blocking(catalog.on_this_day, today, owner_of(request)[0])]
    for year in years:
        year["photos"] = await _run_blocking(_year_rows, catalog, year["photos"])
    return {"day": today.isoformat(), "years": years}


def _year_rows(catalog, photos: list) -> list:
    """Search Photos (pairs merged) as the grid draws them, edits and covers included."""
    edits = catalog.photo_edits() if any(p.edited for p in photos) else {}
    versions = catalog.version_info([p.record.filepath for p in photos]) if photos else {}
    rows = []
    for photo in photos:
        row = _photo_row(photo.record, sorted(photo.tags), edits.get(photo.record.filepath, {}), photo.edited,
                         versions.get(photo.record.filepath))
        row.update(rating=photo.rating, flag=photo.flag, color_label=photo.label)
        rows.append(row)
    return rows


def _slim_rows(catalog, photos: list, edits: Optional[dict] = None, versions: Optional[dict] = None) -> list:
    """Search Photos as little as a thumbnail, a slideshow and a click need: the
    timeline and the shelves send thousands of them. ``edits`` and ``versions``
    when one caller makes many of these at once, so the library's edits are
    read once and not once a call."""
    if edits is None:
        edits = catalog.photo_edits() if any(p.edited for p in photos) else {}
    if versions is None:
        versions = catalog.version_info([p.record.filepath for p in photos]) if photos else {}
    rows = []
    for photo in photos:
        r = photo.record
        v = versions.get(r.filepath) or {}
        rows.append({
            "filepath": r.filepath, "filename": r.filename, "folder": r.folder,
            "extension": r.extension, "capture_date": r.capture_date, "date": photo.date,
            "rating": photo.rating, "flag": photo.flag, "edited": photo.edited,
            "edit": edits.get(r.filepath, {}), "copies": v.get("copies", 0),
            "cover": v.get("cover", 0), "cover_edit": v.get("cover_edit", {}),
            "video": is_video(r.extension),
        })
    return rows


@router.get("/library/collections")
async def library_collections(request: Request):
    """Scenes the library clearly holds — birthdays, the beach, snow — found by
    smart search, each with a few covers from different years. ``available`` is
    false while smart search is not set up (catalog/explore.py)."""
    catalog = _get_catalog(request)
    found = await _run_blocking(catalog.collections, owner_of(request)[0])
    if found is None:
        return {"available": False, "collections": []}
    out = []
    for c in found:
        out.append({**c, "covers": await _run_blocking(_slim_rows, catalog, c["covers"])})
    return {"available": True, "collections": out}


@router.get("/library/collections/{key}")
async def library_collection(key: str, request: Request):
    """One collection a year at a time, the most recent first, each year its
    best frames."""
    from catalog import explore

    catalog = _get_catalog(request)
    years = await _run_blocking(catalog.collection, key, owner_of(request)[0])
    if years is None:
        raise HTTPException(status_code=404, detail="No such collection, or smart search is not set up")
    _, title, text = explore.BY_KEY[key]
    out = []
    for y in years:
        out.append({**y, "photos": await _run_blocking(_slim_rows, catalog, y["photos"])})
    return {"key": key, "title": title, "query": f'shows:"{text}"',
            "total": sum(y["total"] for y in years), "years": out}


class CollectionHideRequest(BaseModel):
    filepath: str


@router.post("/library/collections/{key}/hide")
async def library_collection_hide(key: str, req: CollectionHideRequest, request: Request):
    """Say a photograph does not belong in a collection. Smart search measures
    resemblance, so this is how a wrong match is corrected, and it stays so."""
    catalog = _get_catalog(request)
    if not await _run_blocking(catalog.hide_from_collection, key, req.filepath, owner_of(request)[0]):
        raise HTTPException(status_code=404, detail="No such collection")
    return {"ok": True}


def timeline_rows(catalog, owner: str) -> list:
    """The timeline as the page reads it, kept with the library's other picks:
    making the rows reads every edit in the library, so they are kept too."""
    from catalog import explore

    def build():
        years = explore.timeline(catalog._photos_for(owner))
        every = [p for y in years for p in y["photos"]]
        for y in years:
            for m in y["months"]:
                every += m["photos"] + [s["cover"] for s in m["shoots"]]
        edits = catalog.photo_edits() if any(p.edited for p in every) else {}
        versions = catalog.version_info({p.record.filepath for p in every})

        def slim(photos):
            return _slim_rows(catalog, photos, edits, versions)

        out = []
        for y in years:
            months = [{**m, "photos": slim(m["photos"]),
                       "shoots": [{"folder": s["folder"], "total": s["total"], "cover": slim([s["cover"]])[0]}
                                  for s in m["shoots"]]} for m in y["months"]]
            out.append({**y, "photos": slim(y["photos"]), "months": months})
        return out
    return catalog._picked("timeline_rows", owner, None, build)


@router.get("/library/timeline")
async def library_timeline(request: Request):
    """Every year and month with photographs, the most recent first: each year
    its best frames, each month its best few and its biggest shoots."""
    catalog = _get_catalog(request)
    return {"years": await _run_blocking(timeline_rows, catalog, owner_of(request)[0])}


@router.get("/library/rediscover")
async def library_rediscover(
    request: Request,
    shuffle: str = Query("", max_length=32),
    day: str = Query("", pattern=r"^(\d{4}-\d\d-\d\d)?$"),
):
    """Good frames from more than a year ago not looked at in half a year, and
    one good day at random. The same all day; ``shuffle`` deals again."""
    from datetime import date as _date

    catalog = _get_catalog(request)
    today = _date.fromisoformat(day) if day else _date.today()
    found = await _run_blocking(catalog.rediscover, today, owner_of(request)[0], shuffle)
    good_day = found["day"]
    if good_day:
        good_day = {**good_day, "photos": await _run_blocking(_slim_rows, catalog, good_day["photos"])}
    return {"photos": await _run_blocking(_slim_rows, catalog, found["photos"]), "day": good_day}


class ViewedRequest(BaseModel):
    filepath: str


@router.post("/library/viewed")
async def library_viewed(req: ViewedRequest, request: Request):
    """The person looked at this photograph up close (the viewer or Develop):
    Rediscover leaves it out for a while."""
    catalog = _get_catalog(request)
    await _run_blocking(catalog.mark_viewed, req.filepath, owner_of(request)[0])
    return {"ok": True}


class DemoteRequest(BaseModel):
    folder: str
    camera: str = "*"        # a camera in the folder, "" for none recorded, "*" for all of it
    demoted: bool = True


@router.get("/library/demoted")
async def library_demoted(request: Request):
    """The shoots this device has marked less important: ``[{folder, camera}]``."""
    catalog = _get_catalog(request)
    shoots = await _run_blocking(catalog.db.demoted_shoots, owner_of(request)[0])
    return {"shoots": [{"folder": f, "camera": c} for f, c in sorted(shoots)]}


@router.put("/library/demoted")
async def set_library_demoted(req: DemoteRequest, request: Request):
    """Mark a shoot (a folder, or one camera's photos in it) less important on
    this device, or not. Each device keeps its own: one person's side event is
    another's main one."""
    catalog = _get_catalog(request)
    if not req.folder.strip():
        raise HTTPException(status_code=400, detail="Which folder?")
    _require_writable(catalog)
    owner = owner_of(request)[0]
    await _run_blocking(catalog.db.set_demoted, owner, req.folder, req.camera, req.demoted)
    shoots = await _run_blocking(catalog.db.demoted_shoots, owner)
    return {"ok": True, "shoots": [{"folder": f, "camera": c} for f, c in sorted(shoots)]}


# ── Camera clocks (catalog/clock.py) ────────────────────────────────

def _clocks_state(catalog) -> dict:
    from catalog import clock

    db = catalog.db
    home, source = clock.home_zone(db, clock._gazetteer())
    return {"home_zone": home, "home_source": source, "rules": clock.rules(db),
            "cameras": clock.cameras(db), "moved": clock.moved_days(db), "job": clock.job(),
            "available": bool(clock.zone_names())}


@router.get("/library/clocks")
async def library_clocks(request: Request):
    """Each camera's clock as its files tell it, the rules a person gave, home's
    zone, and the folders whose photographs were taken on another day than
    their camera said."""
    return await _run_blocking(_clocks_state, _get_catalog(request))


@router.get("/library/clocks/zones")
async def clock_zones():
    from catalog import clock

    return {"zones": clock.zone_names()}


class HomeZoneRequest(BaseModel):
    zone: str = ""


@router.put("/library/clocks/home")
async def set_clock_home(req: HomeZoneRequest, request: Request):
    """The zone home is in: where a photograph was taken when nothing else says."""
    from catalog import clock

    catalog = _get_catalog(request)
    _require_writable(catalog)
    if req.zone and clock.zone(req.zone) is None:
        raise HTTPException(status_code=400, detail=f"Not a time zone: {req.zone}")
    await _run_blocking(clock.set_home_zone, catalog.db, req.zone)
    await _run_blocking(clock.refresh, catalog.db)
    catalog.forget_picks()
    return await _run_blocking(_clocks_state, catalog)


class ClockRuleRequest(BaseModel):
    camera: str
    date_from: str = Field("", pattern=r"^(\d{4}-\d\d-\d\d)?$")
    date_to: str = Field("", pattern=r"^(\d{4}-\d\d-\d\d)?$")
    clock_zone: str = ""     # the zone its clock was on; "" for what the files say
    where_zone: str = ""     # the zone it was in; "" for the folder's place, or home
    body: str = ""           # one body's serial; "" for every body of the camera
    shift_seconds: int = Field(0, ge=-366 * 86400, le=366 * 86400)  # how far to move its clock


@router.post("/library/clocks/rules")
async def add_clock_rule(req: ClockRuleRequest, request: Request):
    """What a person knows about one camera's clock over some days (the days its
    clock said)."""
    from catalog import clock

    catalog = _get_catalog(request)
    _require_writable(catalog)
    if not req.camera.strip():
        raise HTTPException(status_code=400, detail="Which camera?")
    for name in (req.clock_zone, req.where_zone):
        if name and clock.zone(name) is None:
            raise HTTPException(status_code=400, detail=f"Not a time zone: {name}")
    if not (req.clock_zone or req.where_zone or req.shift_seconds):
        raise HTTPException(status_code=400,
                            detail="Say which zone the clock was on, where it was, or how far it was off")
    await _run_blocking(clock.add_rule, catalog.db, req.camera.strip(), req.date_from, req.date_to,
                        req.clock_zone, req.where_zone, req.body.strip(), req.shift_seconds)
    await _run_blocking(clock.refresh, catalog.db)
    catalog.forget_picks()
    return await _run_blocking(_clocks_state, catalog)


class ClockAlignRequest(BaseModel):
    reference: str           # the photo whose clock was right
    other: str               # a photo taken at the same moment on the camera to move
    span: str = Field("day", pattern="^(day|onward|always)$")   # that day, from then on, every day


@router.post("/library/clocks/align")
async def align_clocks(req: ClockAlignRequest, request: Request):
    """Two photos taken at the same moment on two cameras: move the second
    camera's clock (that body's alone) to read the first's time: on the second
    photo's day, from then on, or always."""
    from catalog import clock

    catalog = _get_catalog(request)
    _require_writable(catalog)
    try:
        done = await _run_blocking(clock.align, catalog.db, req.reference, req.other, req.span)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    await _run_blocking(clock.refresh, catalog.db)
    catalog.forget_picks()
    return {"aligned": done, **(await _run_blocking(_clocks_state, catalog))}


@router.delete("/library/clocks/rules/{rule_id}")
async def remove_clock_rule(rule_id: int, request: Request):
    from catalog import clock

    catalog = _get_catalog(request)
    _require_writable(catalog)
    if not await _run_blocking(clock.remove_rule, catalog.db, rule_id):
        raise HTTPException(status_code=404, detail="No such rule")
    await _run_blocking(clock.refresh, catalog.db)
    catalog.forget_picks()
    return await _run_blocking(_clocks_state, catalog)


@router.get("/library/years")
async def library_years(request: Request):
    """The years with photographs, the most recent first, for Year in review."""
    catalog = _get_catalog(request)
    return {"years": await _run_blocking(catalog.photo_years, owner_of(request)[0])}


class PerspectiveRequest(BaseModel):
    name: str
    people: list[str] = []


class DefaultPerspectiveRequest(BaseModel):
    id: int = 0              # 0: everything


def _default_perspective(catalog, owner: str) -> int:
    value = catalog.db.owner_setting(owner, "perspective", "0")
    pid = int(value) if value.isdigit() else 0
    return pid if any(p["id"] == pid for p in catalog.db.list_perspectives()) else 0


@router.get("/library/perspectives")
async def library_perspectives(request: Request):
    """The household's perspectives, the one this device opens Year in review
    on, the named people to build one from, and perspectives worth offering."""
    from catalog import perspectives

    catalog = _get_catalog(request)
    every = await _run_blocking(catalog.db.list_perspectives)
    names = await _run_blocking(catalog.db.named_people)
    return {"perspectives": every, "default": _default_perspective(catalog, owner_of(request)[0]),
            "people": names, "suggestions": perspectives.suggestions(names, every)}


@router.post("/library/perspectives")
async def create_perspective(req: PerspectiveRequest, request: Request):
    catalog = _get_catalog(request)
    if not req.name.strip() or not req.people:
        raise HTTPException(status_code=400, detail="A perspective needs a name and at least one person")
    _require_writable(catalog)
    pid = await _run_blocking(catalog.db.save_perspective, req.name.strip(), req.people)
    return {"ok": True, "id": pid}


@router.put("/library/perspectives/{pid}")
async def update_perspective(pid: int, req: PerspectiveRequest, request: Request):
    catalog = _get_catalog(request)
    if not any(p["id"] == pid for p in catalog.db.list_perspectives()):
        raise HTTPException(status_code=404, detail="No such perspective")
    if not req.name.strip() or not req.people:
        raise HTTPException(status_code=400, detail="A perspective needs a name and at least one person")
    _require_writable(catalog)
    await _run_blocking(catalog.db.save_perspective, req.name.strip(), req.people, pid)
    return {"ok": True, "id": pid}


@router.delete("/library/perspectives/{pid}")
async def delete_perspective(pid: int, request: Request):
    catalog = _get_catalog(request)
    _require_writable(catalog)
    if not await _run_blocking(catalog.db.delete_perspective, pid):
        raise HTTPException(status_code=404, detail="No such perspective")
    return {"ok": True}


@router.put("/library/perspectives-default")
async def set_default_perspective(req: DefaultPerspectiveRequest, request: Request):
    """Which perspective this device opens Year in review on (0: everything)."""
    catalog = _get_catalog(request)
    _require_writable(catalog)
    await _run_blocking(catalog.db.set_owner_setting, owner_of(request)[0], "perspective", str(int(req.id)))
    return {"ok": True, "default": _default_perspective(catalog, owner_of(request)[0])}


@router.get("/library/year/{year}")
async def library_year(year: int, request: Request, perspective: int = Query(-1, ge=-1)):
    """Year in review (:mod:`catalog.review`): the year in numbers, its best
    frames and the collage's, its people, biggest shoots and places away.
    ``perspective`` narrows it to one perspective's shoots: 0 is everything,
    and left out it is the one this device opens on."""
    from places.routes import _cards

    catalog = _get_catalog(request)
    owner = owner_of(request)[0]
    if perspective < 0:
        perspective = _default_perspective(catalog, owner)
    cards = (await _run_blocking(_cards, request))["cards"]
    found = await _run_blocking(catalog.year_review, year, cards, owner, perspective)
    found["perspective"] = perspective
    found["top"] = await _run_blocking(_year_rows, catalog, found["top"])
    found["collage"] = [p.record.filepath for p in found["collage"]]
    return found


class CollageRequest(BaseModel):
    filepaths: list[str]
    title: str = ""
    subtitle: str = ""
    width: int = 3000
    # A fixed canvas (export/collage.py FORMATS) instead of free rows: a story,
    # a square, a print. ``portrait`` turns a print upright; ``layout`` is the
    # number of one of /library/collage/layouts' options.
    format: str = ""
    portrait: bool = False
    layout: int = 0
    gap: int = 0                  # in thousandths of the canvas's shorter side
    margin: int = 0               # the same
    background: str = "white"     # one of _BACKGROUNDS, or a colour as #rrggbb
    pattern: str = ""             # drawn in that colour (export.collage_looks.PATTERNS)
    background_photo: str = ""    # or a photograph as the surface, instead
    background_blur: bool = True  # blurred, so what lies on it stands out
    style: str = "clean"          # export.collage.STYLES
    preview: bool = False         # small, quick, for the page to show


_BACKGROUNDS = {
    "white": (255, 255, 255), "black": (0, 0, 0), "cream": (250, 249, 246),
    # For prints scattered on a table, mostly.
    "kraft": (196, 164, 122), "sky": (214, 230, 245), "blush": (246, 218, 214),
    "sage": (214, 226, 205), "navy": (30, 41, 59),
}


def _canvas(req: "CollageRequest") -> tuple[int, int, int, int]:
    """(width, height, gap, margin) in pixels for a fixed-canvas collage."""
    from export.collage import FORMATS

    w, h = FORMATS[req.format]
    # Prints and the screen size are landscape until turned; a story is upright.
    if req.portrait and w > h:
        w, h = h, w
    if req.preview:
        scale = 900 / max(w, h)
        w, h = round(w * scale), round(h * scale)
    short = min(w, h)
    gap = round(short * max(0, min(100, req.gap)) / 1000)
    margin = round(short * max(0, min(150, req.margin)) / 1000)
    return w, h, gap, margin


def _focus(request: Request, filepath: str) -> Optional[tuple]:
    """Where to keep a photo's crop: the middle of its people, as fractions of
    the frame, when their faces are known and the frame is not cropped by an
    edit (the boxes are of the unedited frame). None for its middle."""
    catalog = _get_catalog(request)
    if catalog.edit_of(filepath):
        return None
    record = catalog.db.get_photo(filepath)
    faces = catalog.db.list_faces(filepath)
    if not faces and record:
        # Faces are kept on one file of a RAW+JPEG pair: look at its twin.
        twins = [r for r in catalog.db.get_photos_in_folder(record.folder)
                 if r.base_name == record.base_name and r.filepath != filepath]
        for twin in twins:
            faces = catalog.db.list_faces(twin.filepath)
            if faces:
                break
    faces = [f for f in faces if float(f.get("w") or 0) > 0]
    if not faces:
        return None
    # The bigger faces decide: a stranger small in the background does not.
    biggest = max(float(f["w"]) for f in faces)
    faces = [f for f in faces if float(f["w"]) >= biggest * 0.4]
    left = min(float(f["x"]) for f in faces)
    right = max(float(f["x"]) + float(f["w"]) for f in faces)
    top = min(float(f["y"]) for f in faces)
    bottom = max(float(f["y"]) + float(f["h"]) for f in faces)
    # A little below the faces' middle, so heads are not cut at the chin.
    return ((left + right) / 2, min(1.0, (top + bottom) / 2 + (bottom - top) * 0.3))


def _open_previews(request: Request, filepaths: list[str]) -> tuple[list, list[str]]:
    import io as _io

    from PIL import Image

    images, kept = [], []
    for filepath in filepaths:
        data = _shown_preview(request, filepath)
        if not data:
            continue
        try:
            image = Image.open(_io.BytesIO(data))
            image.load()
        except OSError:
            continue
        images.append(image)
        kept.append(filepath)
    return images, kept


def _shown_preview(request: Request, filepath: str) -> Optional[bytes]:
    """A photograph's preview as the grid shows it: its cover's edit drawn when it has one."""
    from export.engine import resolve_export_source
    from thumbnails.edited import get_edited_thumbnail

    config = request.app.state.config
    cache = _get_cache(request)
    full_path = Path(config.library_root) / filepath
    if not full_path.exists():
        return None
    settings = _get_catalog(request).edit_of(filepath)
    if settings:
        data = get_edited_thumbnail(resolve_export_source(full_path, config.export_prefer_raw),
                                    settings, cache, ThumbnailSize.PREVIEW)
        if data:
            return data
    return get_thumbnail(full_path, cache, ThumbnailSize.PREVIEW)


def _collage(request: Request, req: CollageRequest) -> bytes:
    from export.collage import FORMATS, STYLES, render, render_canvas
    from export.collage_looks import PATTERNS, parse_colour

    if req.format:
        if req.format not in FORMATS:
            raise HTTPException(status_code=400, detail=f"No format '{req.format}'")
        if req.style not in STYLES:
            raise HTTPException(status_code=400, detail=f"No style '{req.style}'")
        if req.pattern not in PATTERNS:
            raise HTTPException(status_code=400, detail=f"No pattern '{req.pattern}'")
        surface = None
        if req.background_photo:
            found, _ = _open_previews(request, [req.background_photo])
            if not found:
                raise HTTPException(status_code=404, detail="The background photo could not be read")
            surface = found[0]
        images, kept = _open_previews(request, req.filepaths[:9])
        if not images:
            raise HTTPException(status_code=404, detail="None of those photos could be read")
        w, h, gap, margin = _canvas(req)
        colour = _BACKGROUNDS.get(req.background) or parse_colour(req.background)
        return render_canvas(images, w, h, req.layout, gap, margin, colour,
                             [_focus(request, fp) for fp in kept], quality=85 if req.preview else 92,
                             style=req.style, pattern=req.pattern, backdrop_photo=surface,
                             backdrop_blur=req.background_blur)
    images, _kept = _open_previews(request, req.filepaths[:30])
    if not images:
        raise HTTPException(status_code=404, detail="None of those photos could be read")
    return render(images, req.title, req.subtitle, width=max(800, min(6000, int(req.width))))


@router.post("/library/collage")
async def library_collage(req: CollageRequest, request: Request):
    """Several photographs as one JPEG (:mod:`export.collage`), titled when
    asked, to download."""
    catalog = _get_catalog(request)
    known = [fp for fp in req.filepaths if catalog.db.get_photo(fp)]
    if not known:
        raise HTTPException(status_code=404, detail="No such photos")
    if req.background_photo and not catalog.db.get_photo(req.background_photo):
        raise HTTPException(status_code=404, detail="No such background photo")
    jpeg = await _run_blocking(_collage, request, req.model_copy(update={"filepaths": known}))
    name = "".join(ch for ch in (req.title or "collage") if ch.isalnum() or ch in " -_").strip() or "collage"
    if req.format and not req.title:
        name = f"collage {req.format}"
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Content-Disposition": f'attachment; filename="Photool {name}.jpg"',
                             "Cache-Control": "no-store"})


@router.post("/library/collage/layouts")
async def library_collage_layouts(req: CollageRequest, request: Request):
    """The layouts on offer for these photos on ``req.format``, least cropped
    first: cells as fractions of the canvas, for the page to draw as choices."""
    from export.collage import FORMATS, layouts_for

    if req.format not in FORMATS:
        raise HTTPException(status_code=400, detail=f"No format '{req.format}'")
    catalog = _get_catalog(request)
    known = [fp for fp in req.filepaths[:9] if catalog.db.get_photo(fp)]
    images, kept = await _run_blocking(_open_previews, request, known)
    if not images:
        raise HTTPException(status_code=404, detail="None of those photos could be read")
    w, h, gap, margin = _canvas(req.model_copy(update={"preview": True}))
    aspects = [im.width / im.height for im in images]
    # Prints keep their own shapes and have arrangements of their own; film its strips.
    found = layouts_for(req.style, aspects, w - 2 * margin, h - 2 * margin, gap)
    return {
        "filepaths": kept,
        "width": w, "height": h,
        "layouts": [{"crop": f["crop"], "by": f["by"], "lines": f["lines"], "angles": f.get("angles"), "order": f.get("order"),
                     "cells": [[(margin + x) / w, (margin + y) / h, cw / w, ch / h] for x, y, cw, ch in f["cells"]]}
                    for f in found],
    }


@router.get("/catalog/stats/{folder_path:path}")
async def folder_stats(folder_path: str, request: Request):
    catalog = _get_catalog(request)
    return catalog.get_folder_stats(folder_path)


@router.get("/catalog/folder-state/{folder_path:path}")
async def get_folder_state(folder_path: str, request: Request):
    """Where the user last stopped reviewing this folder, if anywhere."""
    catalog = _get_catalog(request)
    return catalog.get_folder_state(folder_path)


@router.put("/catalog/folder-state/{folder_path:path}")
async def set_folder_state(folder_path: str, req: FolderStateRequest, request: Request):
    """
    Remember where the user stopped.

    This is browsing position, not a cull decision, so it is not written to XMP
    and a read-only catalog is not a reason to fail the request — the client
    simply keeps its last known position.
    """
    catalog = _get_catalog(request)
    if not catalog.db.writable:
        return {"folder": folder_path, "last_filepath": None, "updated_at": 0}
    return await _run_blocking(
        catalog.set_folder_state, folder_path, (req.last_filepath or "").strip() or None
    )


@router.put("/catalog/rate")
async def rate_photo(req: RatingRequest, request: Request):
    catalog = _get_catalog(request)
    results = {}

    if req.rating is not None:
        results["rating"] = catalog.set_rating(req.filepath, req.rating)
    if req.flag is not None:
        results["flag"] = catalog.set_flag(req.filepath, req.flag)
    if req.color_label is not None:
        results["color_label"] = catalog.set_color_label(req.filepath, req.color_label)

    if not any(results.values()):
        # Photo not in catalog — index its folder first
        parts = req.filepath.rsplit("/", 1)  # split off filename
        if len(parts) == 2:
            folder = parts[0]
            config = request.app.state.config
            await _run_blocking(catalog.index_folder, folder, config.photo_extensions)
            if req.rating is not None:
                results["rating"] = catalog.set_rating(req.filepath, req.rating)
            if req.flag is not None:
                results["flag"] = catalog.set_flag(req.filepath, req.flag)
            if req.color_label is not None:
                results["color_label"] = catalog.set_color_label(
                    req.filepath, req.color_label
                )

    photo = catalog.get_photo(req.filepath)
    if photo:
        return {
            "filepath": photo.filepath,
            "rating": photo.rating,
            "flag": photo.flag,
            "color_label": photo.color_label,
            "synced": True,
        }
    raise HTTPException(status_code=404, detail="Photo not found in catalog")


# ── Burst grouping endpoints ─────────────────────────────────────────
#
# A burst is a set of frames that look the same and were shot seconds apart.
# The grouping runs on the measurements the thumbnail queue already collects
# (difference hash and the file's write time), so it costs a folder query and
# nothing else. It honours the grid's filters, which is also what keeps a CR3
# from grouping with its own JPG: the grid has already collapsed that pair.

@router.get("/catalog/bursts/{folder_path:path}")
async def catalog_bursts(
    folder_path: str,
    request: Request,
    min_rating: int = Query(0, ge=0, le=5),
    flag: Optional[int] = Query(None, ge=-1, le=1),
    color: Optional[int] = Query(None, ge=0, le=5),
    tag: Optional[str] = Query(None),
    unreviewed: bool = Query(False),
    pair_mode: str = Query("all", pattern="^(all|raw|jpg|paired|raw_only|jpg_only)$"),
    camera: Optional[str] = Query(None, max_length=200),
    hide_camera: list[str] = Query([]),
    max_gap_seconds: Optional[int] = Query(None, ge=1, le=600),
):
    """Near-identical frames in one folder, grouped, with a suggested keeper.

    ``max_gap_seconds`` overrides how far apart two identical-looking frames may
    be and still count as one burst. It is a query parameter so the effect of
    widening it can be seen on a real folder; left out, the configured value is
    used.
    """
    catalog = _get_catalog(request)
    return await _run_blocking(
        catalog.find_bursts,
        folder_path,
        max_gap_seconds=max_gap_seconds,
        min_rating=min_rating,
        flag_filter=flag,
        color_filter=color,
        tag_filter=(tag or "").strip() or None,
        unreviewed_only=unreviewed,
        pair_mode=pair_mode,
        camera_filter=camera,
        hide_cameras=hide_camera,
    )


@router.post("/catalog/bursts/reject-rest")
async def reject_burst_rest(req: BurstRejectRequest, request: Request):
    """Reject every frame of one burst except the one being kept."""
    catalog = _get_catalog(request)
    _require_writable(catalog)
    return await _run_blocking(catalog.reject_burst_rest, req.keep, req.others)


@router.post("/catalog/bursts/reject-duplicates")
async def reject_burst_duplicates(req: BurstFilterRequest, request: Request):
    """Keep the sharpest frame of every burst in a folder, reject the rest.

    Reversible one photo at a time with ``U``, and only ever touches frames the
    grouping is confident are the same shot.
    """
    catalog = _get_catalog(request)
    _require_writable(catalog)
    filters = _filter_kwargs(req)
    return await _run_blocking(catalog.reject_burst_duplicates, req.folder, **filters)


@router.post("/catalog/bursts/check-faces")
async def check_burst_faces(req: BurstFilterRequest, request: Request):
    """Look at the faces in this folder's bursts (analysis/eyes.py), so each burst
    keeps the frame with its people at their best and says whose eyes closed.
    Frames already looked at are not looked at again; answers as the bursts do."""
    config = request.app.state.config
    catalog = _get_catalog(request)
    _require_writable(catalog)
    return await _run_blocking(catalog.check_burst_faces, req.folder, config.models_dir,
                               _get_cache(request), **_filter_kwargs(req))


@router.post("/catalog/best")
async def suggest_best(req: BestRequest, request: Request):
    """The frames of a folder most worth keeping, one per moment, to be selected
    for the person to act on. Nothing is flagged or moved."""
    catalog = _get_catalog(request)
    count = max(1, min(500, int(req.count or 20)))
    return await _run_blocking(catalog.suggest_best, req.folder, count, **_filter_kwargs(req))


# ── Tag (event label) endpoints ──────────────────────────────────────

@router.get("/tags")
async def list_tags(request: Request):
    """Every tag in the library with how many photos carry it."""
    catalog = _get_catalog(request)
    return {"tags": await _run_blocking(catalog.db.list_tags)}


@router.post("/tags")
async def create_tag(req: TagCreateRequest, request: Request):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tag name is required")
    catalog = _get_catalog(request)
    tag_id = await _run_blocking(catalog.db.create_tag, name, req.kind)
    if tag_id is None:
        raise HTTPException(status_code=400, detail="Invalid tag name")
    return {"ok": True, "id": tag_id, "name": name}


@router.put("/tags/rename")
async def rename_tag(req: TagRenameRequest, request: Request):
    new_name = (req.new_name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Tag name is required")
    catalog = _get_catalog(request)
    ok = await _run_blocking(catalog.db.rename_tag, req.old_name, new_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Tag not found: {req.old_name}")
    return {"ok": True, "name": new_name}


@router.delete("/tags/{name}")
async def delete_tag(name: str, request: Request):
    catalog = _get_catalog(request)
    ok = await _run_blocking(catalog.db.delete_tag, name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Tag not found: {name}")
    return {"ok": True}


@router.put("/catalog/tags")
async def set_photo_tags(req: PhotoTagsRequest, request: Request):
    """Replace one photo's tags. Writes dc:subject to the sidecar."""
    catalog = _get_catalog(request)
    ok = await _run_blocking(catalog.set_photo_tags, req.filepath, req.tags)
    if not ok:
        raise HTTPException(status_code=404, detail="Photo not found in catalog")
    return {
        "ok": True,
        "filepath": req.filepath,
        "tags": await _run_blocking(catalog.db.get_photo_tags, req.filepath),
    }


class BatchEditRequest(BaseModel):
    filepaths: list[str]
    rating: Optional[int] = None
    flag: Optional[int] = None
    color_label: Optional[int] = None
    add_tags: list[str] = []
    remove_tags: list[str] = []


@router.post("/catalog/batch")
async def batch_edit(req: BatchEditRequest, request: Request):
    """One change to the photos picked in the grid: stars, flag, colour, tags.

    Removing a person's name also takes their face out of those photos.
    """
    catalog = _get_catalog(request)
    if not catalog.db.writable:
        raise HTTPException(status_code=503, detail="The catalog cannot be written")
    result = await _run_blocking(
        catalog.edit_photos, req.filepaths, req.rating, req.flag, req.color_label,
        req.add_tags, req.remove_tags)
    return {"ok": True, **result}


@router.post("/catalog/tags/apply")
async def apply_tag_to_filtered(req: TagBulkRequest, request: Request):
    """
    Tag or untag every photo in a folder that matches the grid's filters —
    the same selection the "add to basket" action uses.
    """
    tag = (req.tag or "").strip()
    if not tag:
        raise HTTPException(status_code=400, detail="Tag name is required")
    catalog = _get_catalog(request)
    filters = _filter_kwargs(req)
    if req.remove:
        result = await _run_blocking(
            catalog.untag_matching_photos, req.folder, tag, **filters
        )
    else:
        result = await _run_blocking(
            catalog.tag_matching_photos, req.folder, tag, **filters
        )
    return {"ok": True, "tag": tag, **result}


# ── Thumbnail queue endpoints ────────────────────────────────────────

@router.get("/thumbnails/queue/jobs")
async def list_queue_jobs(request: Request):
    qm = _get_queue_manager(request)
    return {"jobs": qm.list_jobs()}


@router.get("/thumbnails/queue/status")
async def queue_status(request: Request):
    """
    What the queue worker is doing right now.

    The job list says what is queued. It cannot say whether a worker is alive,
    whether it is measuring a folder, or whether it paused because the cache
    refused writes — and an empty queue with no explanation is indistinguishable
    from a finished one. This reads process state only, so it does not need to
    leave the event loop.
    """
    qm = _get_queue_manager(request)
    return qm.status()


# Library coverage changes slowly and costs a few seconds on a large cache, so
# it is worked out at most every COVERAGE_SECONDS, by one request at a time;
# the page asks every couple of seconds for what is running.
COVERAGE_SECONDS = 30
_coverage_cache: dict = {"at": 0.0, "value": None}
_coverage_lock = threading.Lock()


@router.get("/activity")
async def activity(request: Request):
    """Everything the app is doing in the background, and how far the library
    has got, in one answer.

    The thumbnail queue used to be the only background work, so its job list
    was the activity page. Imports, measuring, face and place scans and exports
    have since joined it, and a page that shows one of six is worse than none:
    it says "idle" while the machine is busy. Each piece is the same state its
    own panel reads; nothing here is kept separately.
    """
    from catalog import exif_fill
    from export.jobs import recent
    from places import routes as places_routes
    from places import store as places_store

    catalog = _get_catalog(request)
    qm = _get_queue_manager(request)

    def _coverage():
        with _coverage_lock:
            if (_coverage_cache["value"] is not None
                    and time.monotonic() - _coverage_cache["at"] < COVERAGE_SECONDS):
                return _coverage_cache["value"]
            faces = catalog.db.face_scan_status()
            gps = places_store.gps_counts(catalog.db)
            exif = exif_fill.counts(catalog.db)
            try:
                # Not stats(): that sums every stored JPEG's size.
                thumbs = _get_cache(request).count("grid")
            except Exception:
                thumbs = 0
            value = {
                "photos": faces["photos"],
                "measured": catalog.db.photo_stats_count(),
                "faces_scanned": faces["scanned"],
                "places_read": gps["read"],
                "places_found": gps["found"],
                "exif_read": exif["read"],
                "exif_known": exif["known"],
                "thumbnails": thumbs,
            }
            _coverage_cache.update(at=time.monotonic(), value=value)
            return value

    coverage = await _run_blocking(_coverage)
    jobs = await _run_blocking(qm.list_jobs)
    with _import_jobs_lock:
        imports = sorted((dict(j) for j in _import_jobs.values()),
                         key=lambda j: j["started_at"], reverse=True)[:3]
    import background
    from analysis import clip

    return {
        "coverage": coverage,
        "background": background.status(),
        "smart_search": clip.job(),
        "queue": qm.status(),
        "thumbnail_jobs": jobs,
        "imports": imports,
        "faces": dict(_face_job),
        "places": dict(places_routes._job),
        "places_auto": not places_routes.auto_idle(),
        "exif": exif_fill.job(),
        # A paired device sees its own exports; this computer sees everyone's,
        # with who made each.
        "exports": recent(_job_owner(request)),
    }


def requested_work_running() -> bool:
    """Whether work someone asked for and is waiting on is running: an import, an
    export, a face or place scan they started, learning a style. Automatic work
    is left out, since it should not keep the computer from going to sleep."""
    from export import engine, jobs as export_jobs
    from places import routes as places_routes

    with _import_jobs_lock:
        importing = any(j["status"] == "running" for j in _import_jobs.values())
    style = engine.style_job()
    return (importing
            or any(j.get("status") == "running" for j in list(_match_jobs.values()))
            or export_jobs.running()
            or (_face_job.get("status") == "running" and not _face_job.get("automatic"))
            or places_routes._job.get("status") == "running"
            or (style.get("status") == "running" and style.get("kind") != "improve"))


def _background_work(request: Request) -> list[dict]:
    """The automatic work running now, one entry per kind, for the page's
    header: what it is, where it has got, and how far."""
    from analysis import clip
    from catalog import clock, exif_fill
    from places import routes as places_routes

    work = []
    face = _face_job
    if face.get("status") == "running":
        work.append({"kind": "Faces", "where": face.get("folder") or face.get("scope", ""),
                     "done": face.get("done", 0), "total": face.get("total", 0),
                     "automatic": bool(face.get("automatic"))})
    for kind, job in (("Smart search", clip.job()), ("Camera data", exif_fill.job()),
                      ("Camera clocks", clock.job()), ("Places", dict(places_routes._job))):
        if job.get("status") == "running":
            work.append({"kind": kind, "where": job.get("folder", ""), "done": job.get("done", 0),
                         "total": job.get("total", 0), "automatic": kind != "Places"})
    if not places_routes.auto_idle():
        work.append({"kind": "Places", "where": "", "done": 0, "total": 0, "automatic": True})
    from export import engine, look_ahead

    ahead = look_ahead.job()
    if ahead.get("status") == "running":
        work.append({"kind": "Camera looks", "where": ahead.get("folder", ""), "done": ahead.get("done", 0),
                     "total": ahead.get("total", 0), "automatic": True})
    style = engine.style_job()
    if style.get("status") == "running":
        work.append({"kind": "Improving a style" if style.get("kind") == "improve" else "Learning a style",
                     "where": style.get("name", ""), "done": style.get("done", 0),
                     "total": style.get("total", 0), "automatic": style.get("kind") == "improve"})
    queue = _get_queue_manager(request).status()
    if queue.get("activity") in ("thumbnails", "measuring"):
        work.append({"kind": "Thumbnails" if queue["activity"] == "thumbnails" else "Measuring",
                     "where": queue.get("activity_folder", ""), "done": 0, "total": 0,
                     "automatic": True})
    return work


@router.get("/background")
async def background_status(request: Request):
    """Whether background work is paused by hand, or stepping aside because
    someone is using the app, which kinds are waiting, and what is running.
    Reads process state only, so it answers at once even while busy."""
    import background

    return {**background.status(), "work": _background_work(request)}


@router.post("/background/pause")
async def background_pause(request: Request):
    """Hold all background work (faces, smart search, camera data, places,
    measuring, thumbnails for unopened folders) until resumed. Work someone
    started by hand, imports and exports carry on."""
    import background

    background.pause()
    return {**background.status(), "work": _background_work(request)}


@router.post("/background/resume")
async def background_resume(request: Request):
    import background

    background.resume()
    return {**background.status(), "work": _background_work(request)}


@router.post("/thumbnails/queue/enqueue")
async def enqueue_folder(req: EnqueueRequest, request: Request):
    qm = _get_queue_manager(request)
    job = qm.enqueue_folder(req.folder)
    return {"job": qm._serialize_job(job)}


@router.post("/thumbnails/queue/{job_id}/pause")
async def pause_queue_job(job_id: int, request: Request):
    qm = _get_queue_manager(request)
    ok = qm.pause_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot pause this job")
    return {"ok": True}


@router.post("/thumbnails/queue/{job_id}/resume")
async def resume_queue_job(job_id: int, request: Request):
    qm = _get_queue_manager(request)
    ok = qm.resume_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot resume this job")
    return {"ok": True}


@router.post("/thumbnails/queue/{job_id}/cancel")
async def cancel_queue_job(job_id: int, request: Request):
    qm = _get_queue_manager(request)
    ok = qm.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot cancel this job")
    return {"ok": True}


@router.post("/thumbnails/queue/{job_id}/reorder")
async def reorder_queue_job(job_id: int, req: ReorderRequest, request: Request):
    qm = _get_queue_manager(request)
    ok = qm.reorder_job(job_id, req.direction)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot reorder this job")
    return {"ok": True}


@router.get("/thumbnails/queue/folder-status/{folder_path:path}")
async def queue_folder_status(folder_path: str, request: Request):
    qm = _get_queue_manager(request)
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, lambda: qm.discover_folder(folder_path))
    return status


# ── Export basket endpoints ──────────────────────────────────────────

def _develop_payload(params: AdjustmentParams) -> dict:
    """The edit set a request carries, as the plain dict the engine expects.

    Everything the request holds goes to ``normalize``, which keeps the edit's
    own keys and drops the rest (a request's filepath, version and so on).
    """
    from adjust.develop import normalize

    return normalize(params.model_dump())


@router.get("/video/{filepath:path}")
async def get_video(filepath: str, request: Request, compatible: bool = Query(False)):
    """A video to play, as it is: the browser asks for the parts it needs (a
    range request) as it plays and seeks.

    ``compatible``: a copy any browser can play, for one that cannot play the
    file itself (a DJI Pocket's HEVC on a phone, say). It is made once, in the
    background: until it is ready this answers 202 with its state ("working",
    "failed" or "unavailable"), and the viewer asks again.
    """
    config = request.app.state.config
    source = _library_file(config, filepath)
    if not is_video(source.suffix):
        raise HTTPException(status_code=404, detail="Not a video")
    if not compatible:
        return FileResponse(source, media_type="video/mp4")
    from thumbnails.video import compatible_copy

    state, path = await _run_blocking(compatible_copy, source)
    if path is None:
        return Response(content=json.dumps({"state": state}), status_code=202,
                        media_type="application/json")
    return FileResponse(path, media_type="video/mp4")


def _library_file(config, filepath: str) -> Path:
    """Resolve a catalog-relative path inside the library.

    The traversal check belongs next to the resolution rather than at each call
    site: a path that arrived from the client is only ever safe joined to the
    library root, and a route that forgets the check is a route that will read
    any file on the machine.
    """
    root = Path(config.library_root)
    path = root / filepath
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return path


# Each device has a basket of its own (access.owner_of): two people choosing
# photos at the same time must not export each other's choices.

@router.get("/export/basket")
async def list_basket(request: Request):
    catalog = _get_catalog(request)
    items = catalog.db.list_basket(owner_of(request)[0])
    edits = await _run_blocking(catalog.photo_edits)
    counts = await _run_blocking(catalog.db.version_counts)
    return {"items": [
        {
            "copies": counts.get(i.filepath, 0),
            "filepath": i.filepath,
            "folder": i.folder,
            "filename": i.filename,
            "added_at": i.added_at,
            "adjustment": edits.get(i.filepath, {}),
        }
        for i in items
    ]}


@router.post("/export/basket/add")
async def add_to_basket(req: BasketAddRequest, request: Request):
    catalog = _get_catalog(request)
    owner = owner_of(request)[0]
    catalog.db.add_to_basket(req.filepath, req.folder, req.filename, owner)
    return {"ok": True, "count": catalog.db.basket_count(owner)}


@router.post("/export/basket/add-batch")
async def add_to_basket_batch(req: BasketBatchAddRequest, request: Request):
    catalog = _get_catalog(request)
    owner = owner_of(request)[0]
    for item in req.items:
        catalog.db.add_to_basket(item.filepath, item.folder, item.filename, owner)
    return {"ok": True, "count": catalog.db.basket_count(owner)}


@router.post("/export/basket/add-filtered")
async def add_filtered_to_basket(req: BasketFilterAddRequest, request: Request):
    """
    One-shot bulk add: copy every photo in a folder that matches the grid's
    current filters (rating / flag / colour label / tag / pair mode) into the
    export basket. Runs the same query the grid uses, so the basket gets
    exactly what the user was looking at.
    """
    catalog = _get_catalog(request)
    filters = _filter_kwargs(req)
    result = await _run_blocking(
        catalog.add_matching_to_basket, req.folder, owner_of(request)[0], **filters)
    return {"ok": True, **result}


# ── One photograph's own edit ────────────────────────────────────────
#
# Non-destructive and keyed on the photograph, not on a basket row. The edit
# outlives the export it was made for: clearing the basket does not clear it,
# adding the frame to a different export finds it already there, and it is
# written to the XMP sidecar on every save so it travels with the file.

class PhotoEditRequest(AdjustmentParams):
    filepath: str
    # 0 is the photograph's own edit; 1, 2, … are its virtual copies.
    version: int = 0


class PhotoEditBatchRequest(AdjustmentParams):
    filepaths: list[str] = []
    # The edit keys to carry (the develop view's Copy checklist); none means the
    # tone, leaving each photo its own crop, level and spots.
    keys: Optional[list[str]] = None
    # The frame the edit was copied from, so a crop is not put on a frame turned
    # the other way.
    source: Optional[str] = None
    # How each frame is adjusted after the paste (export.paste_match.PASTE_AS).
    paste_as: str = "copied"


class PhotoEditBurstRequest(PhotoEditRequest):
    keys: Optional[list[str]] = None
    paste_as: str = "copied"


class PasteAdjustRequest(BaseModel):
    filepath: str                 # the frame pasted onto
    source: str                   # the frame the edit was copied from
    copied: dict = {}             # that frame's whole edit, as copied
    edit: dict = {}               # this frame's edit with the copied parts put on
    own: dict = {}                # this frame's edit before the paste
    paste_as: str = "copied"


def _paste_adjuster(request: Request, source: Optional[str], copied: dict, paste_as: str):
    """The per-frame step of a paste and match (export.paste_match.adjuster), or
    ``None`` for a paste as copied; a request for an unknown one is refused."""
    from export.paste_match import adjuster, faces_from

    config = request.app.state.config
    try:
        return adjuster(Path(config.library_root), source, copied, paste_as, config.export_prefer_raw,
                        faces_of=faces_from(_get_catalog(request).db))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class PhotoEditRestoreRequest(BaseModel):
    # {filepath: edit}, an empty edit for a photo that had none: what a paste
    # replaced, given back.
    edits: dict[str, dict] = {}


@router.get("/photos/edit")
async def get_photo_edit(filepath: str, request: Request, version: int = 0):
    catalog = _get_catalog(request)
    if version:
        found = await _run_blocking(catalog.db.get_version, filepath, version)
        if found is None:
            raise HTTPException(status_code=404, detail="No such copy")
        from adjust.develop import normalize
        return {"filepath": filepath, "version": version, "settings": normalize(found["settings"])}
    return {"filepath": filepath, "version": 0,
            "settings": await _run_blocking(catalog.photo_edit, filepath)}


@router.put("/photos/edit")
async def set_photo_edit(req: PhotoEditRequest, request: Request):
    """Store this photograph's edit, in the catalog and in its sidecar.

    Walking every control back to zero clears the edit rather than filing one
    that does nothing, so "reset" and "never edited" are the same state.
    """
    catalog = _get_catalog(request)
    if not catalog.db.get_photo(req.filepath):
        raise HTTPException(status_code=404, detail="File not found")
    if is_video(req.filepath):
        raise HTTPException(status_code=400, detail="A video is not developed")
    settings = _develop_payload(req)
    if req.version:
        stored = await _run_blocking(catalog.set_copy_edit, req.filepath, req.version, settings)
        if stored is None:
            raise HTTPException(status_code=404, detail="No such copy")
        return {"ok": True, "filepath": req.filepath, "version": req.version, "settings": stored}
    stored = await _run_blocking(catalog.set_photo_edit, req.filepath, settings)
    return {"ok": True, "filepath": req.filepath, "version": 0, "settings": stored}


@router.delete("/photos/edit")
async def delete_photo_edit(filepath: str, request: Request, version: int = 0):
    catalog = _get_catalog(request)
    if version:
        await _run_blocking(catalog.set_copy_edit, filepath, version, {})
        return {"ok": True, "filepath": filepath, "version": version, "settings": {}}
    await _run_blocking(catalog.clear_photo_edit, filepath)
    return {"ok": True, "filepath": filepath, "settings": {}}


# ── Virtual copies ───────────────────────────────────────────────────
#
# More than one edit of the same file. The photograph's own edit is version 0 and
# keeps every route above unchanged; copies are 1, 2, … and carry a name and
# whether an export writes them.

class CopyCreateRequest(BaseModel):
    filepath: str
    from_version: int = 0
    name: str = ""


class CopyUpdateRequest(BaseModel):
    filepath: str
    version: int
    name: Optional[str] = None
    export: Optional[bool] = None


class CoverRequest(BaseModel):
    filepath: str
    version: int = 0


@router.get("/photos/versions")
async def list_photo_versions(filepath: str, request: Request):
    catalog = _get_catalog(request)
    return {"filepath": filepath,
            "copies": await _run_blocking(catalog.photo_versions, filepath),
            "cover": await _run_blocking(catalog.cover, filepath)}


@router.put("/photos/versions/cover")
async def set_photo_cover(req: CoverRequest, request: Request):
    """Choose which version of a photograph is seen while browsing: 0 its own
    edit, or one of its copies. Export is unchanged by it."""
    catalog = _get_catalog(request)
    if not catalog.db.get_photo(req.filepath):
        raise HTTPException(status_code=404, detail="File not found")
    if not await _run_blocking(catalog.set_cover, req.filepath, req.version):
        raise HTTPException(status_code=404, detail="No such copy")
    return {"ok": True, "cover": req.version,
            "copies": await _run_blocking(catalog.photo_versions, req.filepath)}


@router.get("/photos/versions/counts")
async def photo_version_counts(request: Request, folder: Optional[str] = None):
    catalog = _get_catalog(request)
    return {"counts": await _run_blocking(catalog.db.version_counts, folder)}


@router.post("/photos/versions")
async def create_photo_version(req: CopyCreateRequest, request: Request):
    catalog = _get_catalog(request)
    if not catalog.db.get_photo(req.filepath):
        raise HTTPException(status_code=404, detail="File not found")
    made = await _run_blocking(catalog.create_copy, req.filepath, req.from_version, req.name)
    return {"ok": True, "filepath": req.filepath, **made,
            "copies": await _run_blocking(catalog.photo_versions, req.filepath)}


@router.put("/photos/versions")
async def update_photo_version(req: CopyUpdateRequest, request: Request):
    catalog = _get_catalog(request)
    ok = await _run_blocking(catalog.update_copy, req.filepath, req.version, req.name, req.export)
    if not ok:
        raise HTTPException(status_code=404, detail="No such copy")
    return {"ok": True, "copies": await _run_blocking(catalog.photo_versions, req.filepath)}


@router.delete("/photos/versions")
async def delete_photo_version(filepath: str, version: int, request: Request):
    catalog = _get_catalog(request)
    ok = await _run_blocking(catalog.delete_copy, filepath, version)
    if not ok:
        raise HTTPException(status_code=404, detail="No such copy")
    return {"ok": True, "copies": await _run_blocking(catalog.photo_versions, filepath),
            "cover": await _run_blocking(catalog.cover, filepath)}


@router.post("/photos/edit/batch")
async def set_photo_edits(req: PhotoEditBatchRequest, request: Request):
    """Put one edit on a whole selection — the grid's answer to copy/paste.

    Each photo's edit from before comes back as ``previous``, for an undo
    (``/photos/edit/restore``).
    """
    catalog = _get_catalog(request)
    settings = _develop_payload(req)

    def paste():
        adjust = _paste_adjuster(request, req.source, settings, req.paste_as)
        return catalog.apply_edit_to_many(req.filepaths, settings, req.keys, req.source, adjust)

    result = await _run_blocking(paste)
    return {"ok": True, **result}


# ── Matching many photos to one, from the folder view ───────────────
#
# Each frame can take seconds (its look may have to be learned), so a batch runs as a job
# the view follows, and one at a time. dry_run answers at once, for a few frames to look
# at, and saves nothing.

class MatchExamplesRequest(BaseModel):
    filepaths: list[str] = []
    reference: str = ""


class BatchMatchRequest(BaseModel):
    filepaths: list[str] = []
    reference: str
    kind: str = "look"            # export.paste_match.MATCH_KINDS
    amount: float = 75            # 0..100: how far the look goes
    dry_run: bool = False


_match_jobs: dict = {}
_MATCH_JOBS_KEEP = 8
_match_lock = threading.Lock()


# The latest match, kept beside the catalog: what each frame had before and what the match
# gave it. So its undo outlives the selection, the page and the server, and only takes
# back a frame still as the match left it: later work on a frame is never wiped.
_LAST_MATCH_FILE = "last_match.json"


def _last_match_path(request: Request) -> Path:
    return Path(request.app.state.config.db_path).parent / _LAST_MATCH_FILE


def _read_last_match(request: Request) -> Optional[dict]:
    try:
        return json.loads(_last_match_path(request).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_last_match(request: Request, record: dict) -> None:
    path = _last_match_path(request)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not keep the last match for undo: %s", exc)


def _same_edit(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _prepare_match(req: BatchMatchRequest, request: Request):
    from config import is_video
    from export.paste_match import MATCH_KINDS

    config = request.app.state.config
    catalog = _get_catalog(request)
    if req.kind not in MATCH_KINDS:
        raise HTTPException(status_code=400, detail=f"Nothing to match by {req.kind}")
    _library_file(config, req.reference)
    if catalog.db.get_photo(req.reference) is None:
        raise HTTPException(status_code=404, detail="The reference is not in the library")
    targets = list(dict.fromkeys(f for f in req.filepaths if f != req.reference))
    targets = [f for f in targets if catalog.db.get_photo(f) is not None
               and not is_video(Path(f).suffix.lower())]
    if not targets:
        raise HTTPException(status_code=400, detail="No other photos to match")
    # Looked up here, so a test or a later version can stand in for it.
    from export import paste_match

    step = paste_match.matcher(Path(config.library_root), req.reference, catalog.photo_edit(req.reference),
                               req.kind, max(0.0, min(100.0, req.amount)) / 100.0, config.export_prefer_raw,
                               faces_of=paste_match.faces_from(catalog.db))
    return targets, step


@router.post("/photos/edit/match")
async def match_photos(req: BatchMatchRequest, request: Request):
    """Bring many photos to one reference: its look, its light or both, and nothing else of
    its edit. With ``dry_run`` the edits come back unsaved; otherwise a job starts."""
    from adjust.develop import normalize

    catalog = _get_catalog(request)
    targets, step = await _run_blocking(_prepare_match, req, request)

    if req.dry_run:
        def look():
            results = {}
            for filepath in targets:
                edit, note = step(filepath, catalog.photo_edit(filepath))
                results[filepath] = {"settings": normalize(edit), "note": note}
            return results

        return {"results": await _run_blocking(look)}

    with _match_lock:
        if any(j["status"] == "running" for j in _match_jobs.values()):
            raise HTTPException(status_code=409, detail="A match is already running")
        job_id = str(uuid.uuid4())[:8]
        job = {"status": "running", "done": 0, "total": len(targets), "edits": {}, "previous": {},
               "notes": {}, "failed": [], "stop": False}
        _match_jobs[job_id] = job
        for old in [k for k, j in _match_jobs.items() if j["status"] != "running"][:-_MATCH_JOBS_KEEP]:
            del _match_jobs[old]

    record = {"reference": req.reference, "kind": req.kind, "at": time.time(), "previous": job["previous"],
              "after": job["edits"]}

    def run():
        logger = logging.getLogger(__name__)
        try:
            for filepath in targets:
                if job["stop"]:
                    break
                try:
                    own = catalog.photo_edit(filepath)
                    edit, note = step(filepath, own)
                    edit = normalize(edit)
                    catalog.set_photo_edit(filepath, edit)
                    job["previous"][filepath] = own
                    job["edits"][filepath] = catalog.photo_edit(filepath)
                    job["notes"][filepath] = note
                    # Kept as it goes: a crash part way must still be undoable.
                    if job["done"] % 10 == 0:
                        _write_last_match(request, record)
                except Exception:
                    logger.exception("Matching %s failed", filepath)
                    job["failed"].append(filepath)
                job["done"] += 1
            job["status"] = "stopped" if job["stop"] else "completed"
            if job["edits"]:
                _write_last_match(request, record)
        except Exception:  # pragma: no cover - defensive: a job must always end
            logger.exception("The match stopped")
            job["status"] = "failed"

    threading.Thread(target=run, name="batch-match", daemon=True).start()
    return {"job_id": job_id, "total": len(targets)}


@router.post("/photos/edit/match/examples")
async def match_examples(req: MatchExamplesRequest, request: Request):
    """Which of a selection to show a match on first: the frames most likely to go wrong."""
    from export.paste_match import choose_examples

    catalog = _get_catalog(request)

    def choose():
        stats: dict = {}
        wanted = set(req.filepaths) | ({req.reference} if req.reference else set())
        for folder in {f.rsplit("/", 1)[0] for f in wanted if "/" in f}:
            stats.update(catalog.db.list_photo_stats_for_folder(folder))
        return choose_examples(req.filepaths, stats, stats.get(req.reference))

    return {"examples": await _run_blocking(choose)}


@router.get("/photos/edit/match/last")
async def last_match(request: Request):
    """The latest match, while it can still be taken back; ``null`` when there is none."""
    record = await _run_blocking(_read_last_match, request)
    if not record or not record.get("after"):
        return {"last": None}
    return {"last": {"reference": record.get("reference", ""), "kind": record.get("kind", ""),
                     "at": record.get("at", 0), "photos": len(record["after"])}}


@router.post("/photos/edit/match/undo")
async def undo_last_match(request: Request):
    """Take the latest match back from every frame still as it left them. A frame edited
    since keeps that work, and is named."""
    catalog = _get_catalog(request)

    def undo():
        record = _read_last_match(request)
        if not record or not record.get("after"):
            return None
        restored, kept = {}, []
        for filepath, after in record["after"].items():
            if _same_edit(catalog.photo_edit(filepath), after):
                restored[filepath] = record["previous"].get(filepath, {})
            else:
                kept.append(filepath)
        catalog.restore_edits(restored)
        _last_match_path(request).unlink(missing_ok=True)
        return {"restored": restored, "kept": kept}

    result = await _run_blocking(undo)
    if result is None:
        raise HTTPException(status_code=404, detail="No match to take back")
    return result


@router.delete("/photos/edit/match/last")
async def forget_last_match(request: Request):
    """Keep the latest match: stop offering to take it back."""
    _last_match_path(request).unlink(missing_ok=True)
    return {"ok": True}


@router.get("/photos/edit/match/progress/{job_id}")
async def match_progress(job_id: str):
    job = _match_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such match")
    return {k: v for k, v in job.items() if k != "stop"}


@router.post("/photos/edit/match/cancel/{job_id}")
async def match_cancel(job_id: str):
    """Stop after the frame under way. What was done stays, and can be undone."""
    job = _match_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such match")
    job["stop"] = True
    return {"ok": True}


@router.post("/photos/edit/paste-adjust")
async def paste_adjust(req: PasteAdjustRequest, request: Request):
    """The develop view's paste onto one frame, adjusted as ``paste_as`` asks
    (export.paste_match). Nothing is saved: the view puts the result on the frame
    as one step it can undo, and saves it as it saves any edit."""
    from adjust.develop import normalize

    def adjust():
        step = _paste_adjuster(request, req.source, normalize(req.copied), req.paste_as)
        edit = normalize(req.edit)
        if step is None:
            return {"settings": edit, "note": None}
        out, note = step(req.filepath, edit, normalize(req.own))
        return {"settings": normalize(out), "note": note}

    return await _run_blocking(adjust)


@router.post("/photos/edit/restore")
async def restore_photo_edits(req: PhotoEditRestoreRequest, request: Request):
    """Put back the edits a paste replaced: the undo of ``/photos/edit/batch``."""
    catalog = _get_catalog(request)
    restored = await _run_blocking(catalog.restore_edits, req.edits)
    return {"ok": True, "restored": restored}


@router.post("/photos/edit/burst")
async def set_burst_edit(req: PhotoEditBurstRequest, request: Request):
    """Put one frame's edit on that frame and on the rest of its burst.

    Every frame of the burst is written, basket or not: the edit belongs to the
    photograph, so choosing what to export and choosing how are separate
    decisions and only the first one is the basket's.
    """
    catalog = _get_catalog(request)
    if not catalog.db.get_photo(req.filepath):
        raise HTTPException(status_code=404, detail="File not found")
    settings = _develop_payload(req)

    def paste():
        adjust = _paste_adjuster(request, req.filepath, settings, req.paste_as)
        return catalog.apply_edit_to_burst(req.filepath, settings, req.keys, adjust)

    result = await _run_blocking(paste)
    return {"ok": True, **result}


@router.delete("/export/basket/clear")
async def clear_basket(request: Request):
    catalog = _get_catalog(request)
    catalog.db.clear_basket(owner_of(request)[0])
    return {"ok": True}


@router.delete("/export/basket/{filepath:path}")
async def remove_from_basket(filepath: str, request: Request):
    catalog = _get_catalog(request)
    owner = owner_of(request)[0]
    catalog.db.remove_from_basket(filepath, owner)
    return {"ok": True, "count": catalog.db.basket_count(owner)}


@router.get("/export/basket/count")
async def basket_count(request: Request):
    catalog = _get_catalog(request)
    return {"count": catalog.db.basket_count(owner_of(request)[0])}


# ── Photo measurement endpoints ──────────────────────────────────────
#
# Culling and export-time matching both need to know how bright a frame is, what
# colour the camera was treating as white, whether it is softer than its
# neighbours, and whether it is the same frame as its neighbour. Those numbers
# are derived from the file, so the thumbnail queue collects them in the
# background as folders are warmed. These endpoints measure one folder on demand
# and report how much of it still needs it.

@router.get("/analysis/status/{folder_path:path}")
async def analysis_status(folder_path: str, request: Request):
    """How much of a folder has been measured."""
    catalog = _get_catalog(request)
    return catalog.db.photo_stats_status(folder_path)


@router.post("/analysis/run/{folder_path:path}")
async def analysis_run(folder_path: str, request: Request):
    """Measure every photo in one folder.

    Cheap to run again: a photo whose stored measurements still match the file's
    mtime and size is skipped, so this only costs anything the first time.
    """
    config = request.app.state.config
    catalog = _get_catalog(request)
    _require_writable(catalog)
    # Deferred like the other heavy subsystems: this pulls in numpy, which is
    # only needed when a folder is actually being measured.
    from analysis.service import analyze_folder

    result = await _run_blocking(
        analyze_folder,
        catalog.db,
        _get_cache(request),
        config.library_root,
        folder_path,
    )
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/analysis/hints/{folder_path:path}")
async def analysis_hints(
    folder_path: str,
    request: Request,
    min_rating: int = Query(0, ge=0, le=5),
    flag: Optional[int] = Query(None, ge=-1, le=1),
    color: Optional[int] = Query(None, ge=0, le=5),
    tag: Optional[str] = Query(None),
    unreviewed: bool = Query(False),
    pair_mode: str = Query("all", pattern="^(all|raw|jpg|paired|raw_only|jpg_only)$"),
    camera: Optional[str] = Query(None, max_length=200),
    hide_camera: list[str] = Query([]),
):
    """Soft-frame and blown-highlight warnings for one folder.

    Read-only and derived: the measurements are already stored, and what they
    mean is worked out here, every time, so the thresholds can be retuned
    without re-measuring anything. A frame that has not been measured is simply
    absent — "no warnings" and "not looked at yet" are different answers, and
    the caller needs to be able to tell them apart.
    """
    catalog = _get_catalog(request)
    return await _run_blocking(
        catalog.folder_hints,
        folder_path,
        min_rating=min_rating,
        flag_filter=flag,
        color_filter=color,
        tag_filter=(tag or "").strip() or None,
        unreviewed_only=unreviewed,
        pair_mode=pair_mode,
        camera_filter=camera,
        hide_cameras=hide_camera,
    )


# ── Export execution endpoints ───────────────────────────────────────

@router.post("/export/run")
async def run_export(req: ExportRunRequest, request: Request):
    """Write the caller's basket. This computer writes into the library's export
    folder; any other device gets its JPEGs as a download, and nothing it sends
    can name a place on this computer's disks."""
    from export.jobs import start_export

    owner, name = owner_of(request)
    try:
        return start_export(request.app.state.config, _get_catalog(request), req,
                            owner=owner, owner_name=name, download=not is_local(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _job_owner(request: Request):
    """Whose jobs a caller may see: this computer sees everyone's."""
    return None if is_local(request) else owner_of(request)[0]


@router.get("/export/progress/{job_id}")
async def export_progress(job_id: str, request: Request):
    from export.jobs import progress

    state = progress(job_id, _job_owner(request))
    if state is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    return state


@router.post("/export/open/{job_id}")
async def export_open_folder(job_id: str, request: Request):
    """Show a finished export's folder in Explorer, for dragging the photos into
    Google Photos or anywhere else. This computer only: it opens a window on the
    desktop of the computer the server runs on."""
    from export.jobs import written_folder

    if not is_local(request):
        raise HTTPException(status_code=403, detail="Only on the library computer")
    try:
        folder = written_folder(request.app.state.config, job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    import os
    import subprocess as _subprocess
    import sys as _sys

    if _sys.platform == "win32":
        os.startfile(str(folder))  # noqa: S606 - a folder under the export root
    else:
        _subprocess.Popen(["open" if _sys.platform == "darwin" else "xdg-open", str(folder)])
    return {"ok": True, "folder": str(folder)}


# A download is the caller's own and nobody else's, this computer included: the
# person at this computer has the export folder.

@router.get("/export/download/{job_id}")
async def export_download_names(job_id: str, request: Request):
    from export.jobs import download_names

    try:
        return {"files": download_names(job_id, owner_of(request)[0])}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/export/download/{job_id}/photos.zip")
async def export_download_zip(job_id: str, request: Request):
    from export.jobs import download_zip

    try:
        path = await _run_blocking(download_zip, job_id, owner_of(request)[0])
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    stamp = time.strftime("%Y-%m-%d %H%M")
    return FileResponse(path, media_type="application/zip",
                        filename=f"Photool {stamp}.zip")


@router.get("/export/download/{job_id}/file/{name}")
async def export_download_file(job_id: str, name: str, request: Request):
    from export.jobs import download_file

    try:
        path = download_file(job_id, owner_of(request)[0], name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(path, media_type="image/jpeg", filename=name)


# ── Export default output + presets + preview ────────────────────────

@router.get("/export/default-output")
async def default_export_output(request: Request):
    """Where this caller's export goes: a folder under ``path`` on this
    computer, or a download (``download``) on any other device."""
    from export.jobs import export_root

    if not is_local(request):
        return {"download": True, "path": "", "limit": request.app.state.config.remote_export_limit}
    return {"download": False, "path": str(export_root(request.app.state.config))}


# ── Saved edit sets ──────────────────────────────────────────────────

class SavePresetRequest(BaseModel):
    name: str
    settings: dict
    sections: list = []          # the develop panel's sections it carries ("light", "curve", …)


@router.get("/export/develop/presets")
async def list_develop_presets(request: Request):
    """The saved edit sets, for the panel's preset menu."""
    catalog = _get_catalog(request)
    return {"presets": catalog.db.list_develop_presets()}


@router.post("/export/develop/presets")
async def save_develop_preset(req: SavePresetRequest, request: Request):
    """Save the edit on screen under a name (a look of your own), so it can be
    put on other photos, with the panel sections it carries.

    The settings are clamped into range on the way in. A preset is read straight
    back into the sliders, so one holding a value no slider can reach would show
    up as a control the user cannot drag back to where it was. What belongs to
    the frame it was made on is left out: the crop and level, the spots, and the
    masks, which are drawn on that frame (an object mask is its very picture).
    """
    from adjust.develop import MASKS_KEY, OWN_KEYS, normalize

    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="A preset needs a name")
    settings = {k: v for k, v in normalize(req.settings).items() if k not in OWN_KEYS and k != MASKS_KEY}
    if not settings:
        raise HTTPException(status_code=400, detail="There is no edit to save")
    sections = [s for s in req.sections if isinstance(s, str) and 0 < len(s) <= 32][:24]
    catalog = _get_catalog(request)
    catalog.db.save_develop_preset(name[:80], settings, sections)
    return {"ok": True, "name": name, "presets": catalog.db.list_develop_presets()}


@router.delete("/export/develop/presets/{name}")
async def delete_develop_preset(name: str, request: Request):
    """Forget a saved edit set. The photos it was applied to keep their edit."""
    catalog = _get_catalog(request)
    if not catalog.db.delete_develop_preset(name):
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"ok": True, "presets": catalog.db.list_develop_presets()}


def _full_size_of(frame) -> str:
    """A decoded frame's whole photograph size, "WxH" in its own pixels."""
    w, h = frame.size
    scale = frame.scale or 1.0
    return f"{round(w / scale)}x{round(h / scale)}"


class PreviewRequest(BaseModel):
    filepath: str                # relative path e.g. "2025/2025-03-06/IMG_0001.CR3"
    size: int = 512              # preview max dimension
    adjustment: AdjustmentParams = AdjustmentParams()
    # Render through the develop pipeline even with no edit. The develop view asks
    # for this so its "original" is the same picture as an edit sitting at zero,
    # and the first slider moved does not change how the whole frame is drawn.
    linear: bool = False
    # The photograph as the export writes it (export_photo with resize_only): a
    # frame with an edit as developed, one without as the camera's own picture.
    resize_only: bool = False
    # Drawn while a slider is held: the same edit on a smaller copy of the frame
    # (adjust.render.draft_of), quick enough to follow the slider. The view asks
    # for the full preview when the slider is let go.
    draft: bool = False
    # Do not wait for this frame's look to be learned: draw it with its pool's look
    # meanwhile (X-Camera-Look-Pending says so). See export.engine.starter_base.
    starter: bool = False


@router.post("/export/preview")
async def style_preview(req: PreviewRequest, request: Request):
    """
    Render a small preview JPEG of a photo with its edit and return it as a
    streaming response: the develop view's picture, and the export panel's.
    Uses resize-first for speed — sub-second even on CPU.
    """
    import io as _io
    from export.engine import (camera_image, linear_base, look_status, preview_base,
                               resolve_export_source, starter_base)

    config = request.app.state.config
    catalog = _get_catalog(request)
    # Render the file the export will actually develop. A basked JPG with a CR3
    # beside it is one frame with two renderings of it, and showing the camera's
    # JPG while the export writes the RAW is a preview that promises the wrong
    # thing — most visibly for tone, where the camera has already applied a curve
    # the RAW decode has not.
    source = resolve_export_source(
        _library_file(config, req.filepath), config.export_prefer_raw
    )

    loop = asyncio.get_event_loop()
    pending = [False]

    def _render():
        from PIL import Image

        from adjust.render import develop_frame

        settings = _develop_payload(req.adjustment)

        if req.resize_only:
            # The photograph as it stands, at preview size — the same picture the
            # export writes, so the panel promises nothing it will not do. A frame
            # that has been developed is shown as it was developed; one that has
            # not is the camera's own rendering.
            img = None
            has_look = [False]
            look_note = [("none", 0, "")]
            full_size = [""]
            if settings:
                frame = linear_base(source, req.size)
                if frame is not None:
                    full_size[0] = _full_size_of(frame)
                    has_look[0] = frame.look is not None
                    look_note[0] = look_status(source, frame.look)
                    try:
                        img = develop_frame(frame, settings, noise_wait=False)
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "develop preview failed for %s", req.filepath)
            if img is None:
                img = camera_image(source, req.size)
            if img is None:
                return None
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            return buf, has_look[0], look_note[0], full_size[0]

        img = None
        has_look = [False]
        look_note = [("none", 0, "")]
        full_size = [""]
        if settings or req.linear:
            # A hand edit is made in linear light from a decode with headroom,
            # exactly as the export does it; the decoded frame and its
            # base/detail split are kept, so only the curve is redone per drag.
            if req.starter:
                frame, pending[0] = starter_base(source, req.size)
            else:
                frame = linear_base(source, req.size)
            if frame is not None:
                full_size[0] = _full_size_of(frame)
                has_look[0] = frame.look is not None
                look_note[0] = (("starter", frame.look.frames, "") if pending[0]
                                else look_status(source, frame.look))
                whole = frame
                if req.draft:
                    from adjust.render import draft_of

                    frame = draft_of(frame)
                try:
                    # Noise reduction is not waited for: the frame is drawn
                    # without it until it is made, and the view asks again.
                    img = develop_frame(frame, settings, noise_wait=False)
                    if frame is not whole:
                        # Back at the preview's size, so the view lays it out,
                        # and measures it, as the picture it stands in for.
                        k = whole.size[0] / frame.size[0]
                        size = (whole.size if img.size == frame.size
                                else (round(img.width * k), round(img.height * k)))
                        img = img.resize(size, Image.BILINEAR)
                except Exception:
                    logging.getLogger(__name__).exception("develop preview failed for %s", req.filepath)
        if img is None:
            # No edit (or the linear path could not read the file): the plain
            # 8-bit frame.
            img = preview_base(source, req.size)
            if img is None:
                return None
            if settings:
                from adjust.geometry import apply_geometry

                img = apply_geometry(img, settings)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return buf, has_look[0], look_note[0], full_size[0]

    rendered = await loop.run_in_executor(None, _render)
    if rendered is None:
        raise HTTPException(status_code=500, detail="Could not render preview")
    buf, camera_look, (look_state, look_frames, look_hint), full_size = rendered

    # Whether this frame has a camera look to dial: the develop view shows its
    # "Camera look" control only when it does. And where its noise reduction
    # stands: "working" tells the view to ask again.
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                             headers={"Cache-Control": "no-cache",
                                      "X-Camera-Look": "1" if camera_look else "0",
                                      "X-Camera-Look-State": look_state,
                                      "X-Camera-Look-Frames": str(look_frames),
                                      "X-Camera-Look-Hint": look_hint,
                                      "X-Camera-Look-Pending": "1" if pending[0] else "0",
                                      "X-Noise-Reduction": _noise_state(source, req.adjustment),
                                      # The whole photograph's size, uncropped: what the
                                      # develop view's pixel readout counts in.
                                      "X-Photo-Size": full_size})


def _noise_state(source: Path, adjustment: "AdjustmentParams") -> str:
    """Where a frame's noise reduction stands, for the develop view: ``off`` (not
    asked for), ``raw-only`` (not a RAW), ``unavailable`` (no model runtime),
    ``working``, ``ready`` or ``failed``."""
    from adjust import noise
    from export.engine import _RAW_EXTENSIONS

    if not noise.wanted(_develop_payload(adjustment)):
        return "off"
    if source.suffix.lower() not in _RAW_EXTENSIONS:
        return "raw-only"
    if not noise.available():
        return "unavailable"
    state = noise.status(source)
    if state == "ready":
        from export.engine import warm_denoised

        warm_denoised(source)
    return "working" if state == "none" else state


class LearnLookRequest(BaseModel):
    folder: str


@router.post("/camera-look/learn")
async def learn_camera_look(req: LearnLookRequest, request: Request):
    """Teach the camera look from every RAW of a folder, in the background. The
    develop view offers it when a frame's look rests on too few frames or none like
    its scene; the job opens each RAW as the develop view does."""
    from export import engine

    folder = _resolve_folder(request.app.state.config, req.folder)
    files = engine.raw_files_in(folder)
    if not files:
        return {"status": "idle", "done": 0, "total": 0}
    state = engine.learn_status(req.folder)
    if state["status"] != "running":
        threading.Thread(target=engine.learn_folder, args=(req.folder, files), daemon=True,
                         name="camera-look-learn").start()
        state = {"status": "running", "done": 0, "total": len(files)}
    return state


class RelearnLookRequest(BaseModel):
    filepath: str


@router.post("/camera-look/relearn")
async def relearn_camera_look(req: RelearnLookRequest, request: Request):
    """Learn one photograph's look afresh: the develop view offers it on an edited
    photograph still on a look learned the old way, which changes how it is drawn."""
    from export import engine

    source = engine.resolve_export_source(_library_file(request.app.state.config, req.filepath),
                                          request.app.state.config.export_prefer_raw)
    look = await _run_blocking(engine.relearn_look, source)
    state, frames, hint = engine.look_status(source, look)
    return {"state": state, "frames": frames, "hint": hint}


@router.get("/camera-look/learn")
async def camera_look_learning(folder: str):
    from export import engine

    return engine.learn_status(folder)


# ── Styles ───────────────────────────────────────────────────────────
#
# A look learned from many frames and kept under a name, to be rendered on any
# RAW file from any camera (adjust.styles). Learning runs in the background.

class StyleLearnRequest(BaseModel):
    name: str
    folder: str
    # Keep only frames shot with this file's camera and rendering settings.
    like: str = ""


class StyleRenameRequest(BaseModel):
    name: str


@router.get("/styles")
async def list_styles_route():
    from adjust import styles
    from export import engine

    return {"styles": await _run_blocking(styles.list_styles), "job": engine.style_job()}


@router.post("/styles/learn")
async def learn_style_route(req: StyleLearnRequest, request: Request):
    """Learn a style from the RAW frames of a folder (each against the JPEG
    beside it, or the one inside it), in the background."""
    from export import engine

    config = request.app.state.config
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="A style needs a name")
    folder = _resolve_folder(config, req.folder)
    files = engine.raw_files_in(folder)
    if not files:
        raise HTTPException(status_code=400, detail="No RAW files in this folder")
    like = _library_file(config, req.like) if req.like else None
    job = engine.style_job()
    if job.get("status") == "running":
        raise HTTPException(status_code=409, detail=f"Already learning {job.get('name')!r}")
    threading.Thread(target=engine.learn_style, args=(name, files, like), daemon=True,
                     name="style-learn").start()
    return {"status": "running", "name": name, "done": 0, "total": 0}


@router.get("/styles/learn/preview")
async def learn_style_preview(request: Request, folder: str, like: str = ""):
    """What learning a style from this folder like this frame would use: the
    frames that match, the settings they share, and how the rest differ. The
    first look at a folder reads every RAW's settings; learning then reuses them."""
    from export import engine

    config = request.app.state.config
    files = engine.raw_files_in(_resolve_folder(config, folder))
    like_path = _library_file(config, like) if like else None
    return await _run_blocking(engine.style_preview, files, like_path)


@router.get("/styles/learn")
async def learn_style_status():
    from export import engine

    return engine.style_job()


def _style_folders(request: Request, settings: str) -> list[Path]:
    """The library folders with RAW files from the camera a style was learned
    on (its settings string starts with the model), newest first; every folder
    with RAW files when the model is not known or matches none."""
    from export.engine import _RAW_EXTENSIONS

    model = (settings or "").split("|", 1)[0].strip("-")
    raw = tuple(sorted(_RAW_EXTENSIONS))
    marks = ",".join("?" * len(raw))
    db = _get_catalog(request).db
    with db._connect() as conn:
        rows = []
        if model:
            rows = conn.execute(
                f"SELECT DISTINCT folder FROM photos WHERE camera = ? AND lower(extension) IN ({marks})",
                (model, *raw)).fetchall()
        if not rows:
            rows = conn.execute(
                f"SELECT DISTINCT folder FROM photos WHERE lower(extension) IN ({marks})", raw).fetchall()
    root = Path(request.app.state.config.library_root)
    return [root / r[0] for r in sorted(rows, key=lambda r: r[0], reverse=True)]


@router.post("/styles/{style_id}/improve")
async def improve_style_route(style_id: str, request: Request):
    """Look in other folders for frames shot with this style's camera and
    settings, and keep the style refitted with them only if it does no worse on
    its own frames (adjust.styles.improve). In the background; the job reads
    as /styles/learn does, with ``kind`` "improve" and a ``report`` at the end."""
    from adjust import styles
    from export import engine

    style = await _run_blocking(styles.load, style_id)
    if style is None:
        raise HTTPException(status_code=404, detail="No such style")
    if not style.key:
        raise HTTPException(status_code=400, detail=(
            "This style was learned before styles remembered their camera settings; "
            "learn it again to improve it from other folders."))
    job = engine.style_job()
    if job.get("status") == "running":
        raise HTTPException(status_code=409, detail=f"Already working on {job.get('name')!r}")
    folders = await _run_blocking(_style_folders, request, style.settings)
    threading.Thread(target=engine.improve_style, args=(style_id, folders), daemon=True,
                     name="style-improve").start()
    return {"status": "running", "kind": "improve", "name": style.name, "style_id": style_id,
            "phase": "looking", "done": 0, "total": len(folders)}


@router.post("/styles/{style_id}/restore")
async def restore_style_route(style_id: str):
    """Undo the last improvement: put back the version kept before it."""
    from adjust import styles
    from export import engine

    if not await _run_blocking(styles.restore_previous, style_id):
        raise HTTPException(status_code=404, detail="No earlier version of this style")
    engine.clear_preview_cache()
    return {"ok": True, "style": styles.load(style_id).summary()}


@router.patch("/styles/{style_id}")
async def rename_style_route(style_id: str, req: StyleRenameRequest):
    from adjust import styles

    if not await _run_blocking(styles.rename, style_id, req.name):
        raise HTTPException(status_code=404, detail="No such style")
    return {"ok": True}


@router.delete("/styles/{style_id}")
async def delete_style_route(style_id: str):
    """Forget a style. Frames whose edit names it go back to their camera's own
    look; their edits are not rewritten."""
    from adjust import styles

    if not await _run_blocking(styles.delete, style_id):
        raise HTTPException(status_code=404, detail="No such style")
    return {"ok": True}


class LoupeRequest(BaseModel):
    filepath: str
    # Where on the picture the develop view shows, as fractions of it.
    u: float = 0.5
    v: float = 0.5
    size: int = 480              # side of the 1:1 patch, in source pixels
    # The size the develop view draws its preview at: the patch is toned with that
    # frame's base, which is then already decoded rather than decoded again.
    preview_size: int = 1600
    adjustment: AdjustmentParams = AdjustmentParams()


@router.post("/export/develop/loupe")
async def develop_loupe(req: LoupeRequest, request: Request):
    """A 1:1 patch of the edited frame around a point, for checking detail.

    Faces, highlights and shadows are judged at full size; the 1600 px preview
    cannot show whether skin has texture or a sky has a seam.
    """
    import io as _io
    from export.engine import loupe_image, resolve_export_source

    config = request.app.state.config
    source = resolve_export_source(
        _library_file(config, req.filepath), config.export_prefer_raw
    )
    settings = _develop_payload(req.adjustment)
    loop = asyncio.get_event_loop()

    def _render():
        img = loupe_image(source, settings, req.u, req.v, req.size, req.preview_size)
        if img is None:
            return None
        # Lossless: the loupe is where fine detail is judged. A JPEG of the patch,
        # even at quality 95 with full colour, took a sixth of its fine detail away
        # (and at 88 with half-resolution colour, as it was sent, more than a third).
        buf = _io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        buf.seek(0)
        return buf

    buf = await loop.run_in_executor(None, _render)
    if buf is None:
        raise HTTPException(status_code=500, detail="Could not render the loupe")
    return Response(content=buf.getvalue(), media_type="image/png",
                             headers={"Cache-Control": "no-cache"})


class PrefetchRequest(BaseModel):
    # The photographs the develop view may open next, nearest first.
    filepaths: list[str] = []
    size: int = 1600             # the develop view's preview size
    # The photograph on screen, decoded at full resolution for the loupe.
    full: Optional[str] = None


@router.post("/export/develop/prefetch")
async def develop_prefetch(req: PrefetchRequest, request: Request):
    """Get the next photographs ready in the background (export.engine.prefetch).

    Returns at once; a path that cannot be found is skipped rather than failing
    the rest.
    """
    from export.engine import prefetch, resolve_export_source

    config = request.app.state.config

    def source_of(filepath):
        try:
            return resolve_export_source(_library_file(config, filepath), config.export_prefer_raw)
        except HTTPException:
            return None

    sources = [s for s in (source_of(f) for f in req.filepaths[:4]) if s is not None]
    full = source_of(req.full) if req.full else None
    prefetch(sources, req.size, full)
    return {"queued": len(sources) + (1 if full is not None else 0)}


class MaskModelRequest(BaseModel):
    kind: str                    # "subject", "sky", "people", "faces"; "parts" for people's parts


@router.post("/export/develop/mask-model")
async def develop_mask_model(req: MaskModelRequest):
    """Make the model a subject or sky mask needs ready, fetching it the first
    time (a few MB), so the mask is added only once it can select anything."""
    from adjust import depth, parts, segment
    from faces.models import ModelsUnavailable

    if req.kind == "depth":
        try:
            await _run_blocking(depth.ensure)
        except ModelsUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {"ok": True}
    if req.kind == "parts":
        # The parts of people: the people model and the face finder, then the
        # two parsers (hundreds of MB, the first time).
        def _ready():
            segment.ensure("faces")
            parts.ensure()
        try:
            await _run_blocking(_ready)
        except ModelsUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {"ok": True}
    if req.kind not in segment.KINDS:
        raise HTTPException(status_code=400, detail=f"No model selects {req.kind!r}")
    try:
        await _run_blocking(segment.ensure, req.kind)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"ok": True}


@router.get("/export/develop/parts-model")
async def develop_parts_model():
    """Whether the parsers for people's parts are here, and how much a first
    use fetches."""
    from adjust import parts

    return {"available": await _run_blocking(parts.available), "download_mb": parts.DOWNLOAD_MB}


class FrameGeometry(BaseModel):
    """The edit's lens correction and transform, so a selection, a range, a colour
    or the faces are found on the picture as the mask view shows it (adjust.lens)."""
    lens_profile: int = 0
    distortion: int = 0
    tf_vertical: int = 0
    tf_horizontal: int = 0
    tf_aspect: int = 0


class MaskImageRequest(FrameGeometry):
    filepath: str
    kind: str
    size: int = 1600             # the develop preview's size, whose frame is kept
    person: int = 0              # people and faces: one person (1 = leftmost), 0 everyone
    near: int = 100              # depth: the band of distance (adjust.depth.band)
    far: int = 50
    soft: int = 20
    areas: list = []             # people: the parts of them (adjust.parts.AREAS), none for all


def _mask_frame(config, filepath: str, size: int, geometry: FrameGeometry):
    """The frame a mask is drawn on in the develop view: the preview's decode,
    with its lens corrected and its perspective transformed as the edit asks.
    ``None`` if it cannot be read."""
    from export.engine import linear_base, resolve_export_source

    source = resolve_export_source(_library_file(config, filepath), config.export_prefer_raw)
    frame = linear_base(source, size)
    transform = {k: getattr(geometry, k) for k in ("tf_vertical", "tf_horizontal", "tf_aspect")
                 if getattr(geometry, k)}
    if frame is not None and (geometry.lens_profile or geometry.distortion or transform):
        from adjust.render import lens_corrected

        frame = lens_corrected(frame, geometry.lens_profile, geometry.distortion, transform)
    return frame


class DepthAtRequest(FrameGeometry):
    filepath: str
    u: float                     # the point, fractions of the picture as shown: cropped and levelled
    v: float
    settings: dict = {}          # the edit, whose crop and level say where that is
    size: int = 1600


class WhiteBalanceRequest(BaseModel):
    filepath: str


@router.post("/export/develop/white-balance")
async def develop_white_balance(req: WhiteBalanceRequest, request: Request):
    """What a file knows of its own white balance: whether it is a RAW whose Kelvin
    is known, and the Kelvin and tint it was shot at (D65 when it is not)."""
    from adjust import whitebalance
    from export.engine import resolve_export_source

    config = request.app.state.config

    def read():
        source = resolve_export_source(_library_file(config, req.filepath), config.export_prefer_raw)
        return whitebalance.of_source(source)

    found = await _run_blocking(read)
    kelvin, tint = whitebalance.as_shot(found)
    return {"raw": bool(found and found.raw), "as_shot": {"kelvin": int(round(kelvin)), "tint": int(round(tint))}}


class WhitePointRequest(DepthAtRequest):
    pass


@router.post("/export/develop/white-point")
async def develop_white_point(req: WhitePointRequest, request: Request):
    """The Kelvin and tint that make the patch at a point of the frame neutral: the
    grey picker. 422 with a sentence when that patch cannot be read as grey."""
    from adjust import whitebalance
    from adjust.develop import normalize
    from adjust.geometry import source_point

    config = request.app.state.config

    def read():
        frame = _mask_frame(config, req.filepath, req.size, req)
        if frame is None:
            return None
        u, v = source_point(normalize(req.settings), req.u, req.v, *frame.size)
        return whitebalance.white_point(whitebalance.of_frame(frame), whitebalance.sample(frame.rgb, u, v))

    try:
        found = await _run_blocking(read)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return {"kelvin": found[0], "tint": found[1]}


@router.get("/export/develop/second")
async def develop_second_status(path: str):
    """Whether a second exposure's photograph is still there, for the panel to say so."""
    from adjust.double import missing, normalize_second

    second = normalize_second({"path": path})
    if second is None:
        raise HTTPException(status_code=400, detail="Not a path under the library")
    return {"exists": not missing(second), "name": second["path"].rsplit("/", 1)[-1]}


@router.post("/export/develop/depth-at")
async def develop_depth_at(req: DepthAtRequest, request: Request):
    """The distance at a point of the frame on the lens blur's scale: 1 far to 100 near."""
    from adjust import depth
    from adjust.develop import normalize
    from adjust.geometry import source_point
    from faces.models import ModelsUnavailable

    config = request.app.state.config

    def read():
        frame = _mask_frame(config, req.filepath, req.size, req)
        if frame is None:
            return None
        u, v = source_point(normalize(req.settings), req.u, req.v, *frame.size)
        return depth.depth_at(frame, u, v)

    try:
        found = await _run_blocking(read)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return {"depth": found}


@router.post("/export/develop/mask-image")
async def develop_mask_image(req: MaskImageRequest, request: Request):
    """Where a subject, sky, people or faces mask reaches on this frame, as a grey
    PNG of the whole frame (white is all of it), for the develop view to shade."""
    import io as _io

    import numpy as np
    from PIL import Image

    from adjust import depth, segment
    from faces.models import ModelsUnavailable

    if req.kind != "depth" and req.kind not in segment.KINDS:
        raise HTTPException(status_code=400, detail=f"No model selects {req.kind!r}")
    config = request.app.state.config

    def _render():
        frame = _mask_frame(config, req.filepath, req.size, req)
        if frame is None:
            return None
        from adjust.masks import PEOPLE_AREAS

        areas = [a for a in PEOPLE_AREAS if a in req.areas] if req.kind == "people" else []
        if req.kind == "depth":
            matte = depth.band(depth.whole_map(frame), req.near, req.far, req.soft)
        else:
            matte = segment.matte(frame, req.kind, req.person or None, areas)
        buf = _io.BytesIO()
        Image.fromarray((matte * 255.0 + 0.5).astype(np.uint8), "L").save(buf, format="PNG")
        buf.seek(0)
        return buf

    try:
        buf = await _run_blocking(_render)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if buf is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-cache"})


class UprightRequest(FrameGeometry):
    filepath: str
    mode: str = "vertical"       # vertical: tilt and level from the verticals; level: the horizon
    size: int = 1600


@router.post("/export/develop/upright")
async def develop_upright(req: UprightRequest, request: Request):
    """The transform and level that straighten this frame's own lines (adjust.upright),
    for the develop view's Auto buttons to put on the sliders."""
    from export.engine import resolve_export_source, upright_for

    if req.mode not in ("vertical", "level"):
        raise HTTPException(status_code=400, detail="mode is vertical or level")
    config = request.app.state.config
    source = resolve_export_source(_library_file(config, req.filepath), config.export_prefer_raw)
    found = await _run_blocking(upright_for, source,
                                {"lens_profile": req.lens_profile, "distortion": req.distortion},
                                req.mode, req.size)
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return found


class RangeImageRequest(FrameGeometry):
    filepath: str
    range: dict                  # a mask's range (adjust.masks.normalize_range)
    size: int = 1600


@router.post("/export/develop/range-image")
async def develop_range_image(req: RangeImageRequest, request: Request):
    """Where a mask's range reaches on this frame, as a grey PNG of the whole
    frame: the develop view multiplies its shading of the mask by it."""
    import io as _io

    import numpy as np
    from PIL import Image

    from adjust.masks import normalize_range, range_weight

    rng = normalize_range(req.range)
    if rng is None:
        raise HTTPException(status_code=400, detail="That range narrows nothing")
    config = request.app.state.config

    def _render():
        frame = _mask_frame(config, req.filepath, req.size, req)
        if frame is None:
            return None
        w = range_weight(frame.shown(), rng)
        buf = _io.BytesIO()
        Image.fromarray((w * 255.0 + 0.5).astype(np.uint8), "L").save(buf, format="PNG")
        buf.seek(0)
        return buf

    buf = await _run_blocking(_render)
    if buf is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-cache"})


class ColourSampleRequest(FrameGeometry):
    filepath: str
    x: float                     # the point, fractions of the whole frame as shown
    y: float
    size: int = 1600


@router.post("/export/develop/sample")
async def develop_sample_colour(req: ColourSampleRequest, request: Request):
    """The colour of the picture at a point, before the edit, as a colour range
    takes it: its blue and red differences and its brightness (0..100), averaged
    over a few pixels so one noisy pixel does not decide it."""
    from adjust.masks import colour_of

    config = request.app.state.config

    def _sample():
        frame = _mask_frame(config, req.filepath, req.size, req)
        if frame is None:
            return None
        enc = frame.shown()
        h, w = enc.shape[:2]
        cx = min(w - 1, max(0, int(req.x * w)))
        cy = min(h - 1, max(0, int(req.y * h)))
        r = max(1, round(max(h, w) / 400))
        patch = enc[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].reshape(-1, 3).mean(axis=0)
        y, cb, cr = colour_of(patch[None, None, :])
        return {"cb": round(float(cb[0, 0]), 4), "cr": round(float(cr[0, 0]), 4),
                "lum": round(float(y[0, 0]) * 100.0, 1),
                "rgb": [int(round(float(v) * 255)) for v in patch]}

    sample = await _run_blocking(_sample)
    if sample is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return sample


class PeopleRequest(FrameGeometry):
    filepath: str
    size: int = 1600


@router.post("/export/develop/people")
async def develop_people(req: PeopleRequest, request: Request):
    """The people a people or faces mask can choose one of: each face found, left
    to right, as fractions of the whole frame. Person 1 is the first."""
    from adjust import segment
    from faces.models import ModelsUnavailable

    config = request.app.state.config

    def _find():
        frame = _mask_frame(config, req.filepath, req.size, req)
        return None if frame is None else segment.face_boxes(frame)

    try:
        boxes = await _run_blocking(_find)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if boxes is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return {"faces": [{"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}
                      for x, y, w, h in boxes]}


class ObjectModelRequest(BaseModel):
    model: str = "sam2.1"


@router.get("/export/develop/object-models")
async def develop_object_models():
    """The models an object mask can be chosen with: whether each is here yet,
    how much a first use fetches, and whether it finds things by name."""
    from adjust import objects

    def _list():
        return [{"model": name, "label": spec["label"], "available": objects.available(name),
                 "download_mb": round(objects.download_size(name) / 1e6), "finds": objects.finds(name),
                 "license": spec.get("license")}
                for name, spec in objects.MODELS.items()]

    return {"models": await _run_blocking(_list), "default": objects.DEFAULT_MODEL}


@router.post("/export/develop/object-model")
async def develop_object_model(req: ObjectModelRequest):
    """Make an object model ready, fetching it the first time (hundreds of MB),
    so pointing at something answers at once afterwards."""
    from adjust import objects
    from faces.models import ModelsUnavailable

    try:
        await _run_blocking(objects._session_pair, req.model)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"ok": True, "model": objects.model_name(req.model)}


class ObjectPredictRequest(FrameGeometry):
    filepath: str
    model: str = "sam2.1"
    part: dict                   # a mask part: view, points, box (adjust.masks)


def _object_key(req, size: int) -> tuple:
    return (req.filepath, size, req.lens_profile, req.distortion, req.tf_vertical, req.tf_horizontal,
            req.tf_aspect)


@router.post("/export/develop/object-predict")
async def develop_object_predict(req: ObjectPredictRequest, request: Request):
    """What one prompt chooses on its view: the model's three candidates, each a
    grey PNG over the view (as data URLs) with its score, and which is best. Fast
    once the view has been read; the first prompt on a view reads it."""
    import base64

    import numpy as np

    from adjust import objects
    from adjust.masks import _normalize_part
    from faces.models import ModelsUnavailable

    part = _normalize_part(req.part)
    if part is None:
        raise HTTPException(status_code=400, detail="Point at something, or drag a box round it")
    config = request.app.state.config

    def _predict():
        size = objects.picture_size([part])
        frame = _mask_frame(config, req.filepath, size, req)
        if frame is None:
            return None
        logits, scores = objects.predict(objects.picture(frame), part, req.model, _object_key(req, size))
        soft = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return {
            "view": part["view"],
            "best": objects.chosen(scores, None),
            "candidates": [{"score": round(float(sc), 3),
                            "image": "data:image/png;base64," + base64.b64encode(objects.png(m)).decode()}
                           for m, sc in zip(soft, scores)],
        }

    try:
        found = await _run_blocking(_predict)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return found


@router.post("/export/develop/object-find")
async def develop_object_find(req: ObjectPredictRequest, request: Request):
    """Every instance of what a part names (its ``text``) on its view, best
    first: each a grey PNG over the view (a data URL), its score, and the box it
    fills as fractions of the whole frame. SAM 3 only; the first use loads its
    text model."""
    import base64

    import numpy as np

    from adjust import objects
    from adjust.masks import _normalize_part
    from faces.models import ModelsUnavailable

    part = _normalize_part(req.part)
    if part is None or not part.get("text"):
        raise HTTPException(status_code=400, detail="Name what to find")
    if not objects.finds(req.model):
        raise HTTPException(status_code=400, detail="Finding by name needs SAM 3")
    config = request.app.state.config

    def _find():
        size = objects.picture_size([part])
        frame = _mask_frame(config, req.filepath, size, req)
        if frame is None:
            return None
        logits, scores = objects.find(objects.picture(frame), part, req.model, _object_key(req, size))
        left, top, right, bottom = part["view"]
        found = []
        for logit, score in zip(logits, scores):
            soft = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
            ys, xs = np.nonzero(soft > 0.5)
            mh, mw = soft.shape
            box = [left + (right - left) * xs.min() / mw, top + (bottom - top) * ys.min() / mh,
                   left + (right - left) * (xs.max() + 1) / mw, top + (bottom - top) * (ys.max() + 1) / mh]
            found.append({"score": round(float(score), 3), "box": [round(v, 4) for v in box],
                          "image": "data:image/png;base64," + base64.b64encode(objects.png(soft)).decode()})
        return {"view": part["view"], "text": part["text"], "found": found}

    try:
        found = await _run_blocking(_find)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return found


class ObjectCommitRequest(FrameGeometry):
    filepath: str
    model: str = "sam2.1"
    parts: list


@router.post("/export/develop/object-commit")
async def develop_object_commit(req: ObjectCommitRequest, request: Request):
    """The selection an object mask's parts make, worked out on the whole frame,
    brought to its edges and kept: its name goes in the mask as ``matte``, and
    the preview and the export both read that picture."""
    from adjust import objects
    from adjust.masks import _normalize_parts
    from faces.models import ModelsUnavailable

    parts = _normalize_parts(req.parts)
    if not parts:
        raise HTTPException(status_code=400, detail="An object mask needs something pointed at")
    config = request.app.state.config

    def _commit():
        size = objects.picture_size(parts)
        frame = _mask_frame(config, req.filepath, size, req)
        if frame is None:
            return None
        matte = objects.selection(objects.picture(frame), parts, req.model, _object_key(req, size))
        return objects.save(matte)

    try:
        matte_id = await _run_blocking(_commit)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if matte_id is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return {"matte": matte_id, "parts": parts, "model": objects.model_name(req.model)}


class BlurOutlineRequest(FrameGeometry):
    filepath: str
    # What to outline: "point", the thing at the click (u, v); "people", everyone in
    # the frame; "text", every instance of what ``text`` names (SAM 3 only).
    kind: str = "point"
    u: float = 0.5               # the click, fractions of the picture as shown: cropped and levelled
    v: float = 0.5
    text: str = ""
    settings: dict = {}          # the edit, whose crop and level say where that is
    outline: dict = {}           # the outline so far (its blur_outline)
    parts: list = []             # older clients: the outline so far's parts alone
    model: str = "sam2.1"        # the object mask's model: SAM 3 when it was chosen and is here
    size: int = 1600             # the preview's size, whose frame already holds the depth map


@router.post("/export/develop/blur-outline")
async def develop_blur_outline(req: BlurOutlineRequest, request: Request):
    """Outline what lens blur ends at (adjust.outline), added to the outline so far,
    worked out on the whole frame and kept; with the distance there, for the focus.
    A click takes SAM's candidate that holds together at one distance; "people" takes
    everyone at once (with SAM 2.1 each face found, clicked for as a click would be, on
    the one reading of the picture; with SAM 3 its "person"; the people model's selection
    when no face is found); "text" every instance of a word, with SAM 3.

    The model is the one object masks use, so nothing else is loaded for it. The depth
    map is read from the preview's frame, which lens blur has already worked it out on:
    the depth model and SAM cannot share an 8 GB card (adjust.gpu), and reading it on
    a frame of its own opened SAM, then the depth model (letting SAM go), then SAM again."""
    import time

    import cv2
    import numpy as np

    from adjust import depth, objects, outline, segment
    from adjust.develop import normalize, normalize_blur_outline
    from adjust.geometry import source_point
    from adjust.masks import MAX_PARTS, _normalize_parts
    from faces.models import ModelsUnavailable

    config = request.app.state.config
    log = logging.getLogger(__name__)
    model = objects.model_name(req.model)
    if model != "sam2.1" and not objects.available(model):
        model = "sam2.1"             # a model not fetched here (its licence agreed to elsewhere)
    kind = req.kind if req.kind in ("point", "people", "text") else "point"
    text = " ".join(req.text.split())
    if kind == "text" and not text:
        raise HTTPException(status_code=400, detail="Name what to outline")
    if kind == "text" and not objects.finds(model):
        raise HTTPException(status_code=400, detail="Outlining by name needs SAM 3 (choose it in an object mask)")
    held = normalize_blur_outline(req.outline) if req.outline else None
    before = held["parts"] if held else _normalize_parts(req.parts)
    people = bool(held and held.get("people"))

    def _outline():
        times = {}
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            times[name] = round(now - t, 2)
            t = now

        # The depth model first, and on the preview's frame, where it is kept: SAM
        # opens after it, once, rather than either side of it.
        view = _mask_frame(config, req.filepath, max(256, min(int(req.size), 4096)), req)
        if view is None:
            return None
        whole = depth.whole_map(view)
        lap("depth")
        size = objects.PICTURE_SIZE
        frame = _mask_frame(config, req.filepath, size, req)
        if frame is None:
            return None
        rgb8 = objects.picture(frame)
        key = _object_key(req, size)
        lap("decode")
        parts = [p for p in before if objects.finds(model) or not p.get("text")]
        full = [0.0, 0.0, 1.0, 1.0]
        near = None

        def clicked(points, at_point):
            """A part for positive clicks, of SAM's candidates the largest that holds
            together at the distance there (adjust.outline.pick)."""
            nonlocal near
            part = {"view": full, "points": [[round(x, 5), round(y, 5), 1] for x, y in points]}
            logits, scores = objects.predict(rgb8, part, model, key)
            if near is None or near.shape != logits.shape[1:]:
                near = cv2.resize(whole, (logits.shape[2], logits.shape[1]), interpolation=cv2.INTER_AREA)
            part["pick"] = outline.pick(logits, scores, near, depth.depth_at(view, *at_point) / 100.0)
            return part

        at, added, with_people = None, [], people
        if kind == "point":
            u, v = source_point(normalize(req.settings), req.u, req.v, *frame.size)
            u, v = float(min(max(u, 0.0), 1.0)), float(min(max(v, 0.0), 1.0))
            at = depth.depth_at(view, u, v)
            added = [clicked([(u, v)], (u, v))]
        elif kind == "text":
            added = [{"view": full, "text": text}]
        elif objects.finds(model):
            # SAM 3 finds everyone itself, in one pass of the model it already has.
            added = [{"view": full, "text": "person"}]
        else:
            # SAM 2.1 cannot be told "people": each face found is clicked for it, on the
            # face and on the body below it, all on the one reading of the picture.
            for x, y, w, h in segment.face_boxes(view):
                cx, cy = x + w / 2, y + h / 2
                points = [(cx, cy)] + ([(cx, cy + 1.8 * h)] if cy + 1.8 * h < 0.98 else [])
                added.append(clicked(points, (cx, cy)))
        shape = rgb8.shape[:2]

        def everyone():
            got = segment.whole_matte(view, "people")
            if got is None:
                return np.zeros(shape, np.float32)
            return cv2.resize(got, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

        matte = objects.selection(rgb8, _normalize_parts(parts), model, key) if parts else np.zeros(shape, np.float32)
        new = objects.selection(rgb8, _normalize_parts(added), model, key) if added else None
        lap("sam")
        if kind == "people" and not added:
            new = everyone()             # no face found: the people model's selection of everyone
            with_people = with_people or bool((new > 0.5).any())
            lap("people")
        elif people:
            np.maximum(matte, everyone(), out=matte)
            lap("people")
        if kind != "point" and not (new > 0.5).any():
            return {"empty": True}
        np.maximum(matte, new, out=matte)
        # The same thing asked for twice ("Outline the people" again) is kept once.
        unique = {}
        for p in _normalize_parts(parts + added):
            unique.setdefault(json.dumps({k: p.get(k) for k in ("view", "points", "box", "text")}, sort_keys=True), p)
        parts = list(unique.values())[-MAX_PARTS:]
        if at is None:
            # The middle of what was found, for the focus: a group stands at several distances.
            inside = cv2.resize(new, (whole.shape[1], whole.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
            at = int(np.clip(round(float(np.median(whole[inside])) * 100.0), 1, 100)) if inside.any() else 50
        out = {"matte": objects.save(matte), "parts": parts, "model": model}
        if with_people:
            out["people"] = True
        log.info("[blur-outline] %s with %s took %s", kind, model, times)
        return {"outline": out, "depth": at, "times": times}

    try:
        found = await _run_blocking(_outline)
    except ModelsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    if found.get("empty"):
        raise HTTPException(status_code=404, detail="Nothing like that was found here"
                            if kind != "people" else "No one was found here")
    return found


@router.get("/export/develop/object-matte/{matte_id}")
async def develop_object_matte(matte_id: str):
    """A kept object selection, as the grey PNG it is kept as."""
    from adjust import objects

    path = objects._matte_path(matte_id) if all(c in "0123456789abcdef" for c in matte_id) else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="That selection is not here")
    return Response(content=path.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=31536000, immutable"})


class SpotSourceRequest(BaseModel):
    filepath: str
    x: float                     # the spot, fractions of the whole frame
    y: float
    r: float                     # its radius, a share of the longer side
    spots: list = []             # the spots already removed, healed first
    size: int = 1600


@router.post("/export/develop/spot-source")
async def develop_spot_source(req: SpotSourceRequest, request: Request):
    """Where to paint a new spot from: a circle nearby whose surroundings look
    most like the spot's (adjust.spots.find_source), on the frame the develop
    view shows, with its spots so far removed."""
    from adjust.render import healed
    from adjust.spots import find_source, normalize_spots
    from export.engine import linear_base, resolve_export_source

    config = request.app.state.config
    source = resolve_export_source(_library_file(config, req.filepath), config.export_prefer_raw)

    def _find():
        frame = linear_base(source, req.size)
        if frame is None:
            return None
        spots = normalize_spots(req.spots)
        if spots:
            frame = healed(frame, spots)
        return find_source(frame.rgb, min(1.0, max(0.0, req.x)), min(1.0, max(0.0, req.y)),
                           min(0.2, max(0.001, req.r)))

    found = await _run_blocking(_find)
    if found is None:
        raise HTTPException(status_code=404, detail="Could not read this frame")
    return {"sx": found[0], "sy": found[1]}


class MatchLookRequest(BaseModel):
    filepath: str
    adjustment: AdjustmentParams = AdjustmentParams()
    reference: str = ""          # a library photo, as it is edited
    reference_image: str = ""    # or a picture from elsewhere, base64 (a data: URL is fine)


@router.post("/export/develop/match-look")
async def develop_match_look(req: MatchLookRequest, request: Request):
    """The controls that give this frame the look of a reference photograph
    (adjust.match): three channel curves and a saturation, or black and white
    with a curve. Returned at full strength; the develop view scales them."""
    import base64
    import io as _io

    import numpy as np
    from PIL import Image, ImageOps

    from adjust.develop import CURVE_KEYS
    from adjust.match import describe, match
    from adjust.render import develop_frame
    from export.engine import camera_image, linear_base, resolve_export_source

    config = request.app.state.config
    catalog = _get_catalog(request)
    if not req.reference and not req.reference_image:
        raise HTTPException(status_code=400, detail="No reference picture")

    def developed(filepath, settings):
        source = resolve_export_source(_library_file(config, filepath), config.export_prefer_raw)
        frame = linear_base(source, 512)
        if frame is not None:
            return develop_frame(frame, settings)
        return camera_image(source, 512)

    def _match():
        # The frame as it is now, less what the match will set.
        edit = {k: v for k, v in _develop_payload(req.adjustment).items()
                if k not in CURVE_KEYS and k not in ("saturation", "mono")}
        frame = developed(req.filepath, edit)
        if req.reference:
            reference = developed(req.reference, catalog.photo_edit(req.reference))
        else:
            data = req.reference_image.split(",", 1)[-1]
            try:
                reference = ImageOps.exif_transpose(Image.open(_io.BytesIO(base64.b64decode(data))))
            except Exception:
                raise HTTPException(status_code=400, detail="That file is not a picture this can read")
            reference.thumbnail((512, 512))
        if frame is None or reference is None:
            return None
        found = match(np.asarray(frame.convert("RGB")), np.asarray(reference.convert("RGB")))
        return {"settings": found, "note": describe(found)}

    result = await _run_blocking(_match)
    if result is None:
        raise HTTPException(status_code=404, detail="Could not read the frame or the reference")
    return result


class AutoToneRequest(BaseModel):
    filepath: str
    size: int = 512

@router.post("/export/develop/auto")
async def develop_auto(req: AutoToneRequest, request: Request):
    """Suggest tone settings for one photo, measured from the frame itself.

    Returns numbers for the sliders rather than an image: the panel moves and
    the preview follows, so the person sees the same frame they were looking at
    with the suggestion applied.

    It measures the frame the preview is showing, from the same cache, and not
    the stored thumbnail statistics. Those were taken from the camera's own
    rendering, which already has the camera's tone curve in it; a RAW export is
    decoded without that curve, so auto tone built on the stored numbers would
    over-correct every RAW file in the library. For the same reason it measures
    the file the export will develop rather than the half that was basked.
    """
    from adjust.render import auto_develop
    from export.engine import linear_base, resolve_export_source

    config = request.app.state.config
    source = resolve_export_source(
        _library_file(config, req.filepath), config.export_prefer_raw
    )

    loop = asyncio.get_event_loop()

    def _measure():
        frame = linear_base(source, req.size)
        if frame is None:
            return None
        return auto_develop(frame)

    settings = await loop.run_in_executor(None, _measure)
    if settings is None:
        raise HTTPException(status_code=500, detail="Could not read this file")
    return {"settings": settings}


# ── Faces and people ─────────────────────────────────────────────────
# One scan at a time, one job slot: the models are a shared resource, the work
# is CPU-bound, and two passes over the same folder would only fight over the
# same thumbnail cache. A second request while one is running is told so rather
# than queued, because the answer to "scan this folder now" is "it is already
# scanning", which the panel can say.

from faces import jobs as face_jobs  # noqa: E402

_face_job = face_jobs.JOB


class FaceScanRequest(BaseModel):
    folder: Optional[str] = None   # one folder, or the whole library when absent
    threshold: Optional[float] = None


class PersonNameRequest(BaseModel):
    name: str


class PersonMergeRequest(BaseModel):
    into: int


def _face_defaults(request: Request) -> tuple:
    config = request.app.state.config
    return (
        config.models_dir,
        config.effective_face_threshold(),
        float(config.face_min_px),
    )


def _face_model(config) -> str:
    return face_jobs.recognizer(config)


@router.get("/faces/status")
async def faces_status(request: Request):
    """Whether faces can be found here, and how much of the library has been.

    Nothing is downloaded to answer this, so the panel can offer the feature and
    explain what is missing before anything is asked for.
    """
    from faces.detect import available, engine_present

    config = request.app.state.config
    catalog = _get_catalog(request)
    usable, reason = await _run_blocking(available, config.models_dir, False, _face_model(config))
    return {
        "available": usable,
        "engine": await _run_blocking(engine_present),
        "reason": reason,
        "models_dir": config.models_dir,
        "model": _face_model(config),
        "scan_size": getattr(config, "face_scan_size", "preview"),
        # Frames described by another model: the next scan looks at them again.
        "other_model": await _run_blocking(catalog.db.face_scans_by_other_model, _face_model(config)),
        # Photographs no scan has looked at yet, which the sweep will get to.
        "unscanned": sum((await _run_blocking(
            face_jobs.folders_needing_scan, catalog.db, _face_model(config))).values()),
        **catalog.db.face_scan_status(),
        "job": dict(_face_job),
    }


@router.post("/faces/scan")
async def faces_scan(req: FaceScanRequest, request: Request):
    """Start looking for faces, in one folder or everywhere.

    Refused while a scan is running: the second one would compete for the same
    models and the same cache, and the honest answer is that the work is already
    under way.
    """
    catalog = _get_catalog(request)
    if req.folder:
        folders = [req.folder]
        scope = req.folder
    else:
        folders = [f["folder"] for f in catalog.db.get_all_folders() if f.get("count")]
        scope = "the whole library"
    if not folders:
        raise HTTPException(status_code=400, detail="There is nothing to scan yet")

    # One at a time. Two passes would compete for the same models and the same
    # thumbnail cache, and they would double every counter the panel is showing.
    # The honest answer to "scan this folder now" is that it is already scanning.
    if not face_jobs.start(request.app, folders, scope, threshold=req.threshold):
        raise HTTPException(
            status_code=409,
            detail=f"Already scanning {_face_job.get('scope') or 'the library'}",
        )
    return {"ok": True, "scope": scope, "folders": len(folders)}


@router.post("/faces/scan/stop")
async def faces_scan_stop():
    """Ask the running scan to stop. It stops between frames, not mid-frame."""
    return {"ok": True, "stopped": face_jobs.stop()}


@router.get("/faces/scan/status")
async def faces_scan_status():
    return dict(_face_job)


@router.post("/faces/people/rebuild")
async def faces_rebuild(req: FaceScanRequest, request: Request):
    """Group the stored faces again, without looking at any photo."""
    from faces.service import rebuild_people

    _models_dir, threshold, min_px = _face_defaults(request)
    if req.threshold is not None:
        threshold = float(req.threshold)
    catalog = _get_catalog(request)
    summary = await _run_blocking(
        rebuild_people, catalog.db, threshold, min_group_px=min_px,
        model=_face_model(request.app.state.config),
    )
    return {"ok": True, **summary}


@router.get("/faces/people")
async def faces_people(request: Request):
    """The groups, in the order the panel shows them."""
    _models_dir, threshold, _min_px = _face_defaults(request)
    catalog = _get_catalog(request)
    people = await _run_blocking(catalog.db.list_clusters)
    return {
        "people": people,
        "threshold": threshold,
        **catalog.db.face_scan_status(),
    }


@router.get("/faces/people/{cluster_id}/photos")
async def faces_person_photos(cluster_id: int, request: Request):
    """Every photo one person appears in.

    A person's photographs are spread across folders, and the grid shows one
    folder at a time, so this is the only view in the app where they are
    together. Nothing new is drawn for it: these are the same grid thumbnails
    the library shows.
    """
    catalog = _get_catalog(request)
    cluster = await _run_blocking(catalog.db.get_cluster, cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="No such person")
    filepaths = await _run_blocking(catalog.db.cluster_filepaths, cluster_id)
    return {"id": cluster_id, "name": cluster.get("name") or "", "filepaths": filepaths}


@router.get("/faces/crop/{filepath:path}")
async def faces_crop(
    filepath: str,
    request: Request,
    index: int = 0,
    size: int = 112,
):
    """One face as a small JPEG, cropped from the rendering it was found in."""
    from faces.service import crop

    config = request.app.state.config
    catalog = _get_catalog(request)
    cache = _get_cache(request)
    if not catalog.db.get_photo(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    size = max(32, min(256, int(size)))
    jpeg = await _run_blocking(
        crop, catalog.db, cache, config.library_root, filepath, index, size,
        scan_size=getattr(config, "face_scan_size", "preview"),
    )
    if not jpeg:
        raise HTTPException(status_code=404, detail="No such face")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/faces/people/{cluster_id}/timeline")
async def faces_person_timeline(
    cluster_id: int,
    request: Request,
    period: str = Query("month", pattern="^(month|year)$"),
):
    """One person's best face for every month (or year) they were photographed,
    oldest first (:mod:`faces.timeline`)."""
    from faces.timeline import timeline

    catalog = _get_catalog(request)
    cluster = await _run_blocking(catalog.db.get_cluster, cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="No such person")
    faces = await _run_blocking(catalog.db.faces_with_dates, cluster_id)
    return {"id": cluster_id, "name": cluster.get("name") or "", "period": period,
            **timeline(faces, period)}


@router.get("/faces/portrait/{filepath:path}")
async def faces_portrait(
    filepath: str,
    request: Request,
    index: int = 0,
    width: int = 1200,
):
    """One face's photograph cut to a timeline portrait: the face in the same
    place and at the same size in every one."""
    from faces.service import portrait

    config = request.app.state.config
    catalog = _get_catalog(request)
    cache = _get_cache(request)
    if not catalog.db.get_photo(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    width = max(240, min(2400, int(width)))
    jpeg = await _run_blocking(
        portrait, catalog.db, cache, config.library_root, filepath, index, width,
        scan_size=getattr(config, "face_scan_size", "preview"),
    )
    if not jpeg:
        raise HTTPException(status_code=404, detail="No such face")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


def _edit_frame(request: Request, shown: str, half: str) -> Optional[tuple]:
    """The edit the viewer's picture of ``shown`` is drawn with, and the size of
    the picture ``half``'s faces were found in, when that edit crops or levels
    it: ``(settings, width, height)``, else None.

    Faces are found, and kept, on the whole unedited frame; the viewer shows the
    finished picture. Between the two, a box has to go through the crop and the
    turn (adjust.geometry)."""
    from io import BytesIO

    from PIL import Image

    from adjust.geometry import normalize_geometry
    from thumbnails import ThumbnailSize

    catalog = _get_catalog(request)
    # The viewer draws the photo's cover, so its boxes go through the cover's crop.
    settings = catalog.edit_of(shown) or catalog.edit_of(half)
    if not normalize_geometry(settings):
        return None
    source = Path(request.app.state.config.library_root) / half
    try:
        jpeg = get_thumbnail(source, _get_cache(request), ThumbnailSize.PREVIEW)
        width, height = Image.open(BytesIO(jpeg)).size
    except Exception:
        return None
    return settings, width, height


@router.get("/faces/photo/{filepath:path}")
async def faces_of_photo(filepath: str, request: Request):
    """The faces stored for one photograph, with who each one is.

    Either half of a RAW+JPG pair answers for the photograph: the faces are kept
    against the half the scan looked at, which is not always the one on screen.

    A face's box is a fraction of the whole frame; when the photograph's edit
    crops or levels it, ``shown`` is the box on the finished picture the viewer
    draws (None when the crop left the face out).
    """
    from adjust.geometry import shown_box
    from faces.service import scanned_half

    catalog = _get_catalog(request)
    half = await _run_blocking(scanned_half, catalog.db, filepath)
    rows = await _run_blocking(catalog.db.list_faces, half)
    scanned = await _run_blocking(catalog.db.faces_are_scanned, half)
    names = await _run_blocking(catalog.db.cluster_names, [r["cluster_id"] for r in rows])
    min_px = _face_defaults(request)[2]
    frame = await _run_blocking(_edit_frame, request, filepath, half) if rows else None

    def shown(r) -> dict:
        if frame is None:
            return {}
        box = shown_box(frame[0], (r["x"], r["y"], r["w"], r["h"]), frame[1], frame[2])
        return {"shown": dict(zip("xywh", box)) if box else None}

    return {
        "filepath": half,
        "scanned": scanned,
        "faces": [
            {
                "face_index": r["face_index"],
                "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
                **shown(r),
                "px": r["px"],
                "cluster_id": r["cluster_id"],
                "name": names.get(r["cluster_id"], ""),
                "grouped": r["px"] >= min_px and r["cluster_id"] > 0,
                "drawn": bool(r.get("drawn")),
            }
            for r in rows
        ],
    }


@router.get("/faces/named")
async def faces_named(request: Request):
    """The people with a name, for the viewer's "who is this?" — a short list,
    where the full one is thousands of unnamed groups."""
    catalog = _get_catalog(request)
    return {"people": await _run_blocking(catalog.db.named_clusters)}


class DrawnFaceRequest(BaseModel):
    filepath: str           # the photograph on screen
    x: float                # the box, as fractions of the frame
    y: float
    w: float
    h: float
    shown: bool = False     # drawn on the finished picture (cropped, levelled), not the frame
    into: int = 0           # a person, or
    name: str = ""          # a person by name (made when there is none)


@router.post("/faces/drawn")
async def faces_draw(req: DrawnFaceRequest, request: Request):
    """Draw a face where the detector found none — an artistic shot, a
    silhouette — and say who it is. It names the photo, never the grouping."""
    from faces.service import scanned_half
    from thumbnails import ThumbnailSize

    if not req.into and not req.name.strip():
        raise HTTPException(status_code=400, detail="Say who it is: a person or a name")
    box = tuple(min(1.0, max(0.0, float(v))) for v in (req.x, req.y, req.w, req.h))
    if box[2] <= 0.002 or box[3] <= 0.002:
        raise HTTPException(status_code=400, detail="Draw a box around the face")
    catalog = _get_catalog(request)
    half = await _run_blocking(scanned_half, catalog.db, req.filepath)
    if req.shown:
        from adjust.geometry import source_box

        frame = await _run_blocking(_edit_frame, request, req.filepath, half)
        if frame is not None:
            box = source_box(frame[0], box, frame[1], frame[2])
    result = await _run_blocking(
        # Its size in the preview the scan reads, as a found face's is given.
        catalog.draw_face, half, box, req.into, req.name, max(box[2], box[3]) * ThumbnailSize.PREVIEW.value)
    if result["face_index"] is None:
        raise HTTPException(status_code=404, detail="No such person")
    return {"ok": True, "filepath": half, **result}


class DrawnFaceRef(BaseModel):
    filepath: str
    face_index: int


@router.post("/faces/drawn/remove")
async def faces_remove_drawn(req: DrawnFaceRef, request: Request):
    """Take away a face drawn by hand (only those: a found face is taken out of
    its person instead, which is remembered)."""
    catalog = _get_catalog(request)
    if not await _run_blocking(catalog.remove_drawn_face, req.filepath, req.face_index):
        raise HTTPException(status_code=404, detail="No such drawn face")
    return {"ok": True}


class LookCloserRequest(BaseModel):
    filepath: Optional[str] = None   # one photograph
    folder: Optional[str] = None     # or every photograph of a folder


@router.post("/faces/closer")
async def faces_look_closer(req: LookCloserRequest, request: Request):
    """Look for faces again at full size, in one photograph or a folder.

    A few seconds a frame, so it runs as the face job: the People panel shows
    it, Stop stops it, and it waits for no one — a scan already running is
    reported, as for a scan.
    """
    catalog = _get_catalog(request)
    if req.filepath:
        filepaths, scope = [req.filepath], req.filepath.rsplit("/", 1)[-1]
    elif req.folder:
        photos = await _run_blocking(catalog.db.get_photos_in_folder, req.folder, pair_mode="paired")
        filepaths, scope = [p.filepath for p in photos], req.folder
    else:
        raise HTTPException(status_code=400, detail="Say which photograph or folder")
    if not filepaths:
        raise HTTPException(status_code=400, detail="There is nothing to look at")
    if not face_jobs.start_closer(request.app, filepaths, scope):
        raise HTTPException(
            status_code=409,
            detail=f"Already scanning {_face_job.get('scope') or 'the library'}",
        )
    return {"ok": True, "scope": scope, "photos": len(filepaths)}


@router.post("/faces/people/{cluster_id}/name")
async def faces_name(cluster_id: int, req: PersonNameRequest, request: Request):
    """Give a group of faces a name, and put that name on its photographs.

    The tag is the point of naming: from here the existing tag filter, the bulk
    actions, the export basket and the sidecars all work on a person without any
    of them knowing that people exist.
    """
    catalog = _get_catalog(request)
    if not catalog.db.get_cluster(cluster_id):
        raise HTTPException(status_code=404, detail="No such person")
    name = (req.name or "").strip()
    result = await _run_blocking(catalog.name_person, cluster_id, name)
    return {"ok": True, **result}


@router.post("/faces/people/{cluster_id}/merge")
async def faces_merge(cluster_id: int, req: PersonMergeRequest, request: Request):
    """Fold one group into another: two looks at the same person, told once."""
    catalog = _get_catalog(request)
    if not catalog.db.get_cluster(cluster_id) or not catalog.db.get_cluster(req.into):
        raise HTTPException(status_code=404, detail="No such person")
    if cluster_id == req.into:
        return {"ok": True, "moved": 0, "people": len(catalog.db.list_clusters())}
    moved = await _run_blocking(catalog.merge_people, cluster_id, req.into)
    return {"ok": True, **moved}


@router.post("/faces/people/{cluster_id}/not/{other_id}")
async def faces_not_this_person(cluster_id: int, other_id: int, request: Request):
    """The answer "no" to "is this Alice?": the group's faces remember that they
    are not that person, so no rebuild offers it or makes it again."""
    catalog = _get_catalog(request)
    if not catalog.db.get_cluster(cluster_id):
        raise HTTPException(status_code=404, detail="No such person")
    faces = await _run_blocking(catalog.db.refuse_person, cluster_id, other_id)
    return {"ok": True, "faces": faces}


@router.post("/faces/people/{cluster_id}/detach")
async def faces_detach(cluster_id: int, request: Request):
    """Take a group apart, for the case the grouping should not have been made."""
    catalog = _get_catalog(request)
    if not catalog.db.get_cluster(cluster_id):
        raise HTTPException(status_code=404, detail="No such person")
    result = await _run_blocking(catalog.detach_person, cluster_id)
    return {"ok": True, **result}


class FaceRef(BaseModel):
    filepath: str
    face_index: int = 0


class FacesRequest(BaseModel):
    faces: list[FaceRef]


class MoveFacesRequest(FacesRequest):
    into: int = 0           # a group, or
    name: str = ""          # a person by name (made when there is none)


class SortOutRequest(BaseModel):
    not_them: list[FaceRef] = []   # clear sets that are someone else: remembered
    unclear: list[FaceRef] = []    # too unclear to place: let go, not remembered


class TidyRequest(BaseModel):
    apply: bool = False


@router.get("/faces/people/{cluster_id}/sort")
async def faces_sort_person(cluster_id: int, request: Request):
    """One person's faces in look-alike sets, to pick which of them are really them."""
    from faces.service import sort_person

    catalog = _get_catalog(request)
    if not catalog.db.get_cluster(cluster_id):
        raise HTTPException(status_code=404, detail="No such person")
    _models_dir, threshold, _min_px = _face_defaults(request)
    sets = await _run_blocking(sort_person, catalog.db, cluster_id, threshold)
    return {"sets": sets}


@router.post("/faces/people/{cluster_id}/sort")
async def faces_sort_apply(cluster_id: int, req: SortOutRequest, request: Request):
    """Keep the chosen sets: the rest of the faces leave this person."""
    catalog = _get_catalog(request)
    if not catalog.db.get_cluster(cluster_id):
        raise HTTPException(status_code=404, detail="No such person")
    result = await _run_blocking(
        catalog.sort_out_person, cluster_id,
        [(f.filepath, f.face_index) for f in req.not_them],
        [(f.filepath, f.face_index) for f in req.unclear])
    return {"ok": True, **result}


@router.post("/faces/people/tidy")
async def faces_tidy(req: TidyRequest, request: Request):
    """Poor faces on named people that do not look like them: counted, or let go."""
    _models_dir, threshold, _min_px = _face_defaults(request)
    catalog = _get_catalog(request)
    result = await _run_blocking(catalog.tidy_named, threshold, req.apply)
    return {"ok": True, **result}


@router.get("/faces/people/{cluster_id}/faces")
async def faces_of_person(cluster_id: int, request: Request):
    """One person's faces, least alike first, for checking the group by eye."""
    from faces.service import person_faces

    catalog = _get_catalog(request)
    cluster = await _run_blocking(catalog.db.get_cluster, cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="No such person")
    threshold = _face_defaults(request)[1]
    faces = await _run_blocking(person_faces, catalog.db, cluster_id, threshold)
    return {"id": cluster_id, "name": cluster.get("name") or "",
            "threshold": threshold, "faces": faces}


@router.post("/faces/remove")
async def faces_remove(req: FacesRequest, request: Request):
    """Take several faces out of their groups. Each remembers it was not them."""
    catalog = _get_catalog(request)
    return {"ok": True, **await _run_blocking(
        catalog.remove_faces, [(f.filepath, f.face_index) for f in req.faces])}


@router.post("/faces/move")
async def faces_move(req: MoveFacesRequest, request: Request):
    """Put several faces with another person, or a new person by name."""
    if not req.into and not req.name.strip():
        raise HTTPException(status_code=400, detail="Say who they are: a person or a name")
    catalog = _get_catalog(request)
    result = await _run_blocking(
        catalog.move_faces, [(f.filepath, f.face_index) for f in req.faces], req.into, req.name)
    if not result["into"]:
        raise HTTPException(status_code=404, detail="No such person")
    return {"ok": True, **result}


@router.post("/faces/photo/{filepath:path}/detach")
async def faces_detach_one(
    filepath: str, request: Request, index: int = 0
):
    """Take one face out of its group.

    The controls that matter most are the ones that undo a wrong answer, and the
    wrong answer here is a face that joined the wrong person: one photograph of
    a child in a group named for their sibling.
    """
    catalog = _get_catalog(request)
    detached = await _run_blocking(catalog.detach_face, filepath, index)
    if not detached:
        raise HTTPException(status_code=404, detail="No such face")
    return {"ok": True, "detached": True}
