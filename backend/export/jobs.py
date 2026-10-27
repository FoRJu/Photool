"""
Running an export of the basket, in the background.

The export panel and an agent driving the app over MCP start the same job, so
the job lives here rather than inside either of them. ``options`` is anything
with the export panel's fields (see ``api.routes.ExportRunRequest``); it is read
by attribute so the two callers can each describe the options in their own terms.
"""

import logging
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# job id -> progress. In memory on purpose: a job is only worth asking about
# while the process that runs it is alive.
_progress: dict[str, dict] = {}


def copy_suffix(version: int, name: str) -> str:
    """The filename suffix of a virtual copy: ``_vc2`` or ``_vc2-black-and-white``.

    The number is always there, so two copies named alike cannot overwrite each
    other; the name is only for the person reading the folder.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:24]
    return f"_vc{version}" + (f"-{slug}" if slug else "")


def progress(job_id: str, owner: Optional[str] = None):
    """Where a job has got to, or ``None`` if there is no such job — or, when
    ``owner`` is given, no such job of theirs."""
    state = _progress.get(job_id)
    if state is None or (owner is not None and state["owner"] != owner):
        return None
    return _public(state)


def recent(owner: Optional[str] = None, limit: int = 12) -> list[dict]:
    """The latest exports, newest first; everyone's when ``owner`` is None."""
    _prune_downloads()
    jobs = [s for s in _progress.values() if owner is None or s["owner"] == owner]
    jobs.sort(key=lambda s: s["started_at"], reverse=True)
    return [_public(s) for s in jobs[:limit]]


def _public(state: dict) -> dict:
    """A job as the API shows it: the written names are only counted, since a
    download serves them by name and nobody else needs the list."""
    return {**state, "files": len(state["files"])}


# ── Where an export may write ────────────────────────────────────────
#
# On this computer, only under <library>/export: a mistyped path cannot put
# files on C:\ or into a folder of originals, and the whole export folder can be
# moved somewhere else afterwards. A paired device writes nothing on this
# computer at all: its export is rendered into a private temporary folder,
# downloaded, and deleted an hour later.

# Characters Windows will not take in a name, and the names it reserves.
_BAD_NAME = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


# The folder under the library that exports are written to. It is not part of the
# library's photographs: an export is a copy of photographs already there, and
# indexing it counted every exported frame twice — in search, among a person's
# faces, in On this day (catalog.service.is_export_folder).
EXPORT_DIR = "export"


def export_root(config) -> Path:
    return Path(config.library_root) / EXPORT_DIR


def export_folder(config, name: str) -> Path:
    """The folder under ``export_root`` an export named ``name`` writes to.

    ``name`` is a folder name, or a few joined by ``/``; empty means "now".
    Anything that is a path of its own — a drive, a leading slash, ``..`` — is
    refused rather than corrected, so what is written is where the person said.
    """
    name = (name or "").strip().replace("\\", "/")
    if name.startswith("/"):
        raise ValueError("Exports go inside the export folder; give a folder name, not a path")
    name = name.strip("/")
    if not name:
        name = datetime.now().strftime("%Y-%m-%d %H%M")
    parts = [p.strip() for p in name.split("/")]
    for part in parts:
        if (not part or part in (".", "..") or _BAD_NAME.search(part)
                or part.split(".")[0].lower() in _RESERVED or part.endswith(".")):
            raise ValueError(
                f'"{name}" is not a folder name. Exports go inside the export folder; '
                'give a name such as "Hawaii 2026".'
            )
    root = export_root(config).resolve()
    folder = root.joinpath(*parts).resolve()
    try:
        folder.relative_to(root)
    except ValueError:
        raise ValueError("Exports go inside the export folder")
    return folder


DOWNLOAD_SECONDS = 3600
_downloads_root = Path(tempfile.gettempdir()) / "photool-exports"
_zip_lock = threading.Lock()


def _prune_downloads() -> None:
    """Delete downloads older than an hour, and any no job knows about (left by
    a run of the app that has since stopped)."""
    if not _downloads_root.is_dir():
        return
    now = time.time()
    for folder in _downloads_root.iterdir():
        state = _progress.get(folder.name)
        if state is not None and (state["status"] == "running"
                                  or now - state["finished_at"] < DOWNLOAD_SECONDS):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        if state is not None:
            state["expired"] = True


def _finished_download(job_id: str, owner: str) -> dict:
    state = _progress.get(job_id)
    if (state is None or state["owner"] != owner or state["kind"] != "download"
            or state["status"] != "completed" or state["expired"] or not state["files"]):
        raise LookupError("No such export, or it has expired")
    return state


def download_file(job_id: str, owner: str, name: str) -> Path:
    """One written JPEG of a finished download job of ``owner``'s."""
    state = _finished_download(job_id, owner)
    path = _downloads_root / job_id / name
    if name not in state["files"] or not path.is_file():
        raise LookupError("No such file")
    return path


def download_names(job_id: str, owner: str) -> list[str]:
    return list(_finished_download(job_id, owner)["files"])


def download_zip(job_id: str, owner: str) -> Path:
    """Every JPEG of a finished download job, as one ZIP, built once. Stored
    rather than deflated: a JPEG does not compress, and storing is instant."""
    state = _finished_download(job_id, owner)
    folder = _downloads_root / job_id
    path = folder / "_photos.zip"
    with _zip_lock:
        if not path.is_file():
            part = folder / "_photos.zip.part"
            with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
                for name in state["files"]:
                    z.write(folder / name, name)
            part.replace(path)
    return path


def start_export(config, catalog, options, owner: str = "local",
                 owner_name: str = "This computer", download: bool = False) -> dict:
    """Start writing ``owner``'s basket and return at once.

    ``download`` renders into a private temporary folder for the caller to
    fetch (a paired device); otherwise the JPEGs go to ``options.folder`` under
    the library's export folder. Raises ``ValueError`` when the basket is empty,
    the folder name is unusable or a download is too big; the caller says so in
    its own terms.
    """
    _prune_downloads()
    basket_items = catalog.db.list_basket(owner)
    if not basket_items:
        raise ValueError("Export basket is empty")
    if not download:
        output_dir = export_folder(config, getattr(options, "folder", ""))

    # One output per version to write: the photograph's own edit, then each of its
    # copies that is marked for export. Counted here so progress is honest.
    export_jobs = []
    for item in basket_items:
        export_jobs.append((item, 0, None, ""))
        for copy in catalog.photo_versions(item.filepath):
            if copy["export"]:
                export_jobs.append((item, copy["version"], copy["settings"],
                                    copy_suffix(copy["version"], copy["name"])))

    limit = getattr(config, "remote_export_limit", 200)
    if download and len(export_jobs) > limit:
        raise ValueError(
            f"{len(export_jobs)} pictures is more than one download takes ({limit}). "
            "Export part of the basket, or export on the library computer."
        )

    job_id = uuid.uuid4().hex[:12]
    if download:
        output_dir = _downloads_root / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _forget_old_jobs()
    state = _progress[job_id] = {
        "job_id": job_id,
        "status": "running",
        "kind": "download" if download else "folder",
        "owner": owner,
        "owner_name": owner_name,
        # Where the files went, for the person at this computer; a download's
        # temporary folder is nobody's business.
        "folder": "" if download else str(output_dir),
        "total": len(export_jobs),
        "completed": 0,
        "failed": 0,
        "current_file": "",
        "error": "",
        "started_at": time.time(),
        "finished_at": 0.0,
        "expired": False,
        # Names written, in order; a download serves these and nothing else.
        "files": [],
    }

    def _do_export():
        from export.engine import ExportTemplate, export_photo, resolve_export_source

        template = ExportTemplate(
            max_resolution=options.max_resolution,
            quality=options.quality,
            # What leaves this computer carries its dates and nothing else: no
            # GPS (where home is), no camera serial number.
            strip_exif=options.strip_exif or download,
            photographer=options.photographer,
        )

        for item, _version, copy_settings, suffix in export_jobs:
            state["current_file"] = item.filename + (suffix and f" ({suffix.lstrip('_')})")
            # A CR3+JPG pair is one frame with two renderings of it: an edit is
            # developed from the RAW when there is one (see resolve_export_source).
            source = resolve_export_source(
                Path(config.library_root) / item.filepath,
                config.export_prefer_raw,
            )
            try:
                # Each photograph as it stands, at the size asked for: a photo or a
                # copy that has been developed is written as it was developed, and
                # one that has not is the camera's own picture. A set is made to
                # look alike in the develop view (copy and paste), not here.
                #
                # The edit was already written to the sidecar when it was made,
                # so nothing is recorded here: an export reads the library, it
                # does not decide what the library says.
                sliders = (
                    copy_settings if _version else catalog.photo_edit(item.filepath)
                )
                written = export_photo(
                    source,
                    output_dir,
                    template,
                    develop=sliders or None,
                    suffix=suffix,
                    resize_only=True,
                )
                state["files"].append(Path(written).name)
                state["completed"] += 1
            except Exception as e:
                logging.getLogger(__name__).error(f"Export failed for {item.filename}: {e}")
                state["failed"] += 1
        state["status"] = "completed"
        state["current_file"] = ""

    def _run():
        try:
            _do_export()
        except Exception as e:
            logging.getLogger(__name__).exception("Export stopped")
            state["status"] = "failed"
            state["error"] = str(e)
            state["current_file"] = ""
        finally:
            state["finished_at"] = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "total": len(export_jobs), "kind": state["kind"],
            "folder": state["folder"]}


def written_folder(config, job_id: str) -> Path:
    """The folder a finished export on this computer wrote to, for showing it to
    the person at this computer. Raises ``LookupError`` for no such export, or one
    that was a download, or a folder that is not under the export folder (which
    no export writes to, but this is what lets a folder be opened on its say-so)."""
    state = _progress.get(job_id)
    if state is None or state["kind"] != "folder" or not state["folder"]:
        raise LookupError("No such export on this computer")
    folder = Path(state["folder"]).resolve()
    root = export_root(config).resolve()
    if folder != root and root not in folder.parents:
        raise LookupError("No such export on this computer")
    if not folder.is_dir():
        raise LookupError("That export's folder is gone")
    return folder


def _forget_old_jobs(keep: int = 30) -> None:
    finished = sorted((s["started_at"], jid) for jid, s in _progress.items()
                      if s["status"] != "running")
    for _, jid in finished[: max(0, len(_progress) - keep)]:
        state = _progress.pop(jid)
        if state["kind"] == "download":
            shutil.rmtree(_downloads_root / jid, ignore_errors=True)
