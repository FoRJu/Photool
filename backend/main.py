"""
Photool — FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000
"""

import no_console  # noqa: F401  (before anything starts a helper program)
import exiftool_path
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

import mcp_server
from config import load_config
from storage_health import summarize
from api.commands import router as commands_router
from api.routes import router
from commands import COMMANDS
from thumbnails import ThumbnailCache, ThumbnailQueueManager
from catalog import CatalogDB, CatalogService


def _report_storage(app: FastAPI, config) -> dict:
    """
    Check once, at startup, that the configured databases can actually be written.

    Both live under library_root, which is usually a secondary volume. When that
    volume goes read-only — a remount, an ACL change, a drive that has dropped
    to write-protect — reads keep working and every write fails quietly: ratings
    look saved but are not, and the thumbnail queue spins without producing
    anything. Saying it once, in plain words, is better than that.
    """
    storage = {
        "catalog": summarize("catalog database", config.db_path),
        "thumbnail_cache": summarize(
            "thumbnail cache", str(Path(config.thumbnail_cache) / "thumbnails.db")
        ),
    }
    app.state.storage = storage

    broken = [b for b in storage.values() if not b["writable"]]
    if broken:
        print("=" * 74)
        print("[startup] STORAGE WARNING — these paths cannot be written:")
        for block in broken:
            print(f"[startup]   {block['label']}: {block['path']}")
            print(f"[startup]     {block['error']}")
        print("[startup] Browsing still works, but ratings, imports and thumbnails")
        print("[startup] will NOT be saved. Fix the volume, or point db_path /")
        print("[startup] thumbnail_cache in config.yaml at a writable location.")
        print("=" * 74)
    return storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown tasks."""
    config = load_config()
    app.state.config = config

    storage = _report_storage(app, config)

    # exiftool, looked for where Homebrew puts it as well as on the PATH.
    if exiftool_path.ensure_on_path() is None:
        print("[startup] exiftool not found: capture settings, lens data and the camera's "
              "Picture Style will be missing until it is installed (see docs/setup.md)")

    # Ensure catalog directory exists. On a read-only volume this fails, which
    # is already reported above — the app still starts and serves reads.
    for directory in (Path(config.db_path).parent, Path(config.thumbnail_cache)):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[startup] Could not create {directory}: {exc}")

    # Initialize thumbnail cache
    max_bytes = max(0, int(config.thumbnail_cache_max_mb or 0)) * 1024 * 1024
    app.state.thumb_cache = ThumbnailCache(config.thumbnail_cache, max_bytes=max_bytes)

    # The camera looks learned from RAW+JPEG pairs, pooled per camera and Picture
    # Style, kept beside the thumbnails so they go on improving across sessions.
    from export.engine import set_profile_dir

    profiles_ok, profiles_error = set_profile_dir(Path(config.thumbnail_cache) / "camera_profiles")
    if not profiles_ok:
        print(f"[startup] Camera profiles are read only, nothing new will be learned: {profiles_error}")

    # Subject and sky masks find their selection with models kept beside the
    # face models, fetched on first use.
    from adjust import segment

    segment.configure(config.models_dir)
    # Object masks are chosen by pointing (adjust.objects); what was chosen is
    # kept beside the catalog, since an edit names it and needs it to export.
    from adjust import objects

    objects.configure(config.models_dir, str(Path(config.db_path).parent / "mattes"))

    # Noise reduction's model is kept there too; the denoised mosaics it makes are
    # cache, beside the thumbnails.
    from adjust import noise

    noise.configure(config.models_dir, str(Path(config.thumbnail_cache) / "denoised"))
    from thumbnails import video

    video.configure(str(Path(config.thumbnail_cache) / "videos"))
    print(f"[startup] Videos: {'ffmpeg found' if video.available() else 'no ffmpeg: videos have no thumbnails'}")
    print(f"[startup] Noise reduction: {'available' if noise.available() else 'onnxruntime not installed'}")

    # Named styles, learned on purpose and put on other cameras' files. Kept
    # beside the catalog rather than the thumbnail cache: they are work, not cache.
    from adjust import styles

    styles.set_dir(Path(config.db_path).parent / "styles")

    # Devices allowed in from the network (see access.py). Beside the catalog,
    # in a file of its own.
    from access import DeviceStore

    try:
        app.state.devices = DeviceStore(str(Path(config.db_path).parent / "devices.db"))
    except Exception as exc:
        app.state.devices = None
        print(f"[startup] Device pairing unavailable: {exc}")
    if config.lan_access:
        print("[startup] Network access ON: other devices must be paired (Devices, in the header)")

    # Initialize catalog
    db = CatalogDB(config.db_path)
    try:
        forgotten = db.forget_non_photos(())
        if forgotten:
            print(f"[startup] Catalog: forgot {forgotten} files that are not photographs")
    except Exception as exc:  # a read-only catalog still browses
        print(f"[startup] Catalog cleanup skipped: {exc}")
    app.state.catalog = CatalogService(
        db,
        config.library_root,
        burst_max_gap_seconds=config.burst_max_gap_seconds,
    )

    # Initialize thumbnail queue manager
    queue_manager = ThumbnailQueueManager(
        db=db,
        cache=app.state.thumb_cache,
        library_root=config.library_root,
        import_extensions=config.photo_extensions,
        enabled=config.thumbnail_queue_enabled,
        idle_interval_seconds=config.thumbnail_queue_interval_seconds,
        low_priority=config.thumbnail_queue_priority_low,
        auto_generate=config.thumbnail_queue_auto_generate,
        idle_rescan_seconds=config.thumbnail_queue_idle_rescan_seconds,
        auto_analyze=config.analysis_auto_enabled,
    )
    queue_manager.start()
    app.state.queue_manager = queue_manager

    print(f"[startup] Library root: {config.library_root}")
    print(f"[startup] Database: {config.db_path}")
    print(f"[startup] Thumbnail cache: {config.thumbnail_cache}")
    print(f"[startup] Thumbnail cache budget: "
          f"{'unlimited' if not max_bytes else f'{config.thumbnail_cache_max_mb} MB'}")
    print(f"[startup] Storage writable: "
          + ", ".join(f"{name}={'yes' if b['writable'] else 'NO'}" for name, b in storage.items()))
    print(f"[startup] Catalog: {db.count()} photos indexed")
    print(f"[startup] Photo measurements: {db.photo_stats_count()} measured, "
          f"{'automatic' if config.analysis_auto_enabled else 'manual only'}")
    print(f"[startup] Thumbnail queue: {'enabled' if config.thumbnail_queue_enabled else 'disabled'}")

    # Camera data a catalog row is missing (an older rebuild indexed without it),
    # read in the background; nothing to do costs one query.
    from catalog import exif_fill
    from export.jobs import EXPORT_DIR

    # Exports were indexed as photographs once (every exported frame counted
    # twice); they are not now, and what was indexed then is forgotten.
    if db.writable:
        forgotten = db.forget_folder_tree(EXPORT_DIR)
        if forgotten:
            print(f"[startup] Exports are not library photos: forgot {forgotten} indexed export(s)")

    exif_fill.wake(config.library_root, db)

    # The zone each camera's clock was on, read once per file, and every
    # photograph's time where it was taken (catalog/clock.py).
    from catalog import clock
    from places import gazetteer

    clock.configure(lambda: gazetteer.get(config.models_dir))
    clock.wake(config.library_root, db)

    # Faces drawn on a cropped photo before issue #37 were kept on the cropped
    # picture: moved onto the frame, once, off the request path.
    import threading

    from faces.service import repair_drawn_on_edits

    threading.Thread(target=repair_drawn_on_edits, name="drawn-faces",
                     args=(db, app.state.catalog, app.state.thumb_cache, config.library_root),
                     daemon=True).start()

    # Smart search: what each photograph shows, described in the background once
    # the models are here (they are fetched only when it is set up).
    from analysis import clip

    clip.configure(config.models_dir)
    if clip.available():
        clip.wake(config.library_root, db, app.state.thumb_cache)
    print(f"[startup] Smart search: {'set up' if clip.available() else 'not set up'}")

    # Photographs no face scan has looked at — a folder never opened, a scan a
    # restart cut short — are scanned once the server has settled.
    from faces import jobs as face_jobs

    face_jobs.wake(app, delay=face_jobs.STARTUP_DELAY_SECONDS)

    # The MCP endpoint's HTTP app is made per lifespan: its session manager can
    # be run only once (see mcp_server.Endpoint).
    app.state.mcp_http = mcp.streamable_http_app(streamable_http_path="/mcp")
    print(f"[startup] MCP: {len(COMMANDS)} tools at /mcp")

    async with mcp.session_manager.run():
        yield  # app runs

    queue_manager.stop()
    exif_fill.stop()
    clip.stop()
    face_jobs.stop()
    print("[shutdown] Cleaning up...")


app = FastAPI(
    title="Photool",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Readable by the page: whether a frame has a camera look to dial.
    expose_headers=["X-Camera-Look", "X-Camera-Look-State", "X-Camera-Look-Frames", "X-Camera-Look-Hint",
                    "X-Noise-Reduction"],
)

import background  # noqa: E402


@app.middleware("http")
async def note_use(request, call_next):
    """Background work steps aside while someone is using the app
    (background.py). A long request counts from its start to its answer."""
    using = background.counts_as_use(request.method, request.url.path)
    if using:
        background.touch()
    try:
        return await call_next(request)
    finally:
        if using:
            background.touch()


import access  # noqa: E402

access.install(app)
app.include_router(access.router, prefix="/api")
app.include_router(router, prefix="/api")
app.include_router(commands_router, prefix="/api")

from places.routes import router as places_router  # noqa: E402

app.include_router(places_router, prefix="/api")

# Agents (Claude Code, Claude Desktop, …) drive the library over MCP: one tool
# per command in the registry. See mcp_server.py.
mcp = mcp_server.build(app)
app.router.routes.append(Route("/mcp", endpoint=mcp_server.Endpoint()))

# The built page (`npm run build` in frontend/), so a browser on another device
# needs only this server. During development the Vite server serves the page on
# :5173 and proxies /api here, so this is skipped when nothing has been built.
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class PageFiles(StaticFiles):
    """The page is asked for afresh each time, so a rebuild shows on a plain
    refresh; the files in assets/ are named by their content and never change."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        # Starlette hands over an OS path: assets\x.js on Windows.
        if Path(path).parts[:1] == ("assets",):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response


if _DIST.is_dir():
    app.mount("/", PageFiles(directory=str(_DIST), html=True), name="page")
