"""
Core thumbnail extraction, resizing, and disk caching.

Performance notes for Canon R6 II CR3 files (~25MB each):
  - exiftool JpgFromRaw (one-shot):  ~500ms on Windows (process startup)
  - exiftool with -stay_open:         ~20ms  (no startup overhead)
  - Pillow resize to 400px grid:      ~30ms
  - Pillow resize to 2000px preview:  ~60ms
  - rawpy full decode (fallback):     ~800ms
  - Total for cache miss (grid):      ~50ms with persistent exiftool
  - Cache hit:                         ~5ms
"""

import io
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_THUMBNAIL_CACHE_VERSION = "v3"


# ── Sizes (imported from store) ──────────────────────────────────────

from .store import ThumbnailSize, ThumbnailStore, _JPEG_QUALITY

# exiftool is handed its file list on the command line, so a folder with more
# than a couple of hundred RAWs has to be chunked or the process never starts.
from exif_batches import batches as _batches


# ── Cache (SQLite-backed) ────────────────────────────────────────────

class ThumbnailCache:
    """Thumbnail cache backed by a SQLite BLOB store.

    Replaces the old file-based cache to eliminate block-size waste.
    get() returns JPEG bytes (or None) instead of a file path.
    """

    def __init__(self, cache_dir: str, max_bytes: int = 0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(self.cache_dir / "thumbnails.db")
        self.store = ThumbnailStore(db_path, max_bytes=max_bytes)

    @property
    def writable(self) -> bool:
        """False when the cache volume will not accept writes."""
        return self.store.writable

    @property
    def write_error(self) -> str:
        return self.store.write_error

    def recheck_writable(self) -> bool:
        """Probe the cache volume again; True when it accepts writes now."""
        return self.store.recheck_writable()

    def get(self, source: Path, size: ThumbnailSize, variant: str = "") -> Optional[bytes]:
        """Return cached JPEG bytes or None."""
        return self.store.get(source, size, variant)

    def has(self, source: Path, size: ThumbnailSize, variant: str = "") -> bool:
        """Check if a thumbnail exists without reading the blob."""
        return self.store.has(source, size, variant)

    def put(
        self, source: Path, size: ThumbnailSize, jpeg_data: bytes, variant: str = ""
    ) -> None:
        """Store a thumbnail."""
        self.store.put(source, size, jpeg_data, variant)

    def clear(self):
        self.store.clear()

    def prune(self) -> dict:
        """Drop thumbnails whose source file is gone."""
        return self.store.prune()

    def vacuum(self) -> dict:
        return self.store.vacuum()

    def stats(self) -> dict:
        return self.store.stats()


# ── Exiftool: one-shot subprocess (thread-safe, no shared state) ─────

_exiftool_available: Optional[bool] = None


def _check_exiftool() -> bool:
    """Check if exiftool is installed. Cached after first call."""
    global _exiftool_available
    if _exiftool_available is not None:
        return _exiftool_available
    try:
        subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True, timeout=5,
        )
        _exiftool_available = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _exiftool_available = False
        logger.info("exiftool not found, will use rawpy fallback")
    return _exiftool_available


def extract_embedded_jpeg(source: Path) -> Optional[bytes]:
    """
    Extract embedded full-resolution JPEG from a CR3 file using exiftool.
    Canon R6 II embeds a full-res JPEG (~6000x4000) in every CR3.
    Without exiftool, LibRaw reads it instead (``_rawpy_embedded_jpeg``).
    """
    if not _check_exiftool():
        return _rawpy_embedded_jpeg(source)

    for tag in ["-JpgFromRaw", "-PreviewImage"]:
        try:
            result = subprocess.run(
                ["exiftool", "-b", tag, str(source)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and len(result.stdout) > 1000:
                if result.stdout[:2] == b"\xff\xd8":
                    logger.debug(
                        f"Extracted {tag} from {source.name}: "
                        f"{len(result.stdout) / 1024:.0f} KB"
                    )
                    return result.stdout
        except subprocess.TimeoutExpired:
            continue

    return None


def _rawpy_embedded_jpeg(source: Path) -> Optional[bytes]:
    """The camera's JPEG inside a RAW, read by LibRaw, for when exiftool is not
    installed. LibRaw hands back the largest preview the file holds: the full-size
    JPEG of a CR2 or CR3, a smaller one on some other makes. Without it the
    camera's look has nothing to learn from and every RAW is drawn flat."""
    try:
        import rawpy

        with rawpy.imread(str(source)) as raw:
            thumb = raw.extract_thumb()
        data = bytes(thumb.data)
        if thumb.format == rawpy.ThumbFormat.JPEG and len(data) > 1000 and data[:2] == b"\xff\xd8":
            return data
    except Exception as e:
        logger.debug(f"No embedded JPEG from LibRaw for {source.name}: {e}")
    return None


def _batch_extract_embedded_jpegs(sources: list[Path]) -> dict[str, bytes]:
    """
    Extract embedded JPEGs from multiple RAW files, one exiftool call per batch.
    Returns dict mapping filename to JPEG bytes.

    Chunked rather than one process for the whole list: this is much faster
    than calling exiftool per-file on Windows (it avoids repeated ~500ms
    process startup), but a folder of a few hundred RAWs is longer than the
    Windows command line allows, and then *no* file in the folder gets an
    embedded JPEG and every one falls back to a full rawpy decode.
    """
    if not _check_exiftool() or not sources:
        return {}

    results = {}

    # exiftool can process multiple files but outputs them concatenated.
    # Use a temp directory approach: extract each to a temp file.
    # -W flag writes output files with a pattern.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for chunk in _batches(sources):
            try:
                # Extract JpgFromRaw for the batch at once into temp dir
                # -W pattern: %d = source dir, %f = filename without ext
                cmd = [
                    "exiftool", "-b", "-JpgFromRaw",
                    "-W", str(tmpdir / "%f.jpg"),
                ] + [str(s) for s in chunk]

                subprocess.run(cmd, capture_output=True, timeout=60)

            except (subprocess.TimeoutExpired, Exception) as e:
                logger.warning(f"Batch exiftool extraction failed: {e}")
                continue

            # Collect results
            for source in chunk:
                out_file = tmpdir / f"{source.stem}.jpg"
                if not out_file.exists():
                    continue
                data = out_file.read_bytes()
                # Consume it: two folders can hold the same filename, and a
                # later chunk whose file has no embedded JPEG would otherwise
                # pick up this chunk's bytes as its own.
                out_file.unlink()
                if len(data) > 1000 and data[:2] == b"\xff\xd8":
                    results[str(source)] = data

    return results


def _batch_extract_exif(sources: list[Path], tags: list[str], fast: bool = False) -> dict[str, dict]:
    """
    Extract EXIF data from multiple files, one exiftool call per batch.
    Returns dict mapping str(source_path) to EXIF dict.

    An entry is matched to its file by the path exiftool reports, compared as a
    path (exiftool writes ``/`` where Windows wrote ``\\``), and by input order
    only when every file answered: a file exiftool skips would otherwise shift
    every later file's data onto its neighbour. A batch in which one file failed
    still answers for the rest (exiftool then exits 1 with the others' JSON).

    A batch that fails is skipped rather than aborting the rest: the caller
    fills the gap per-file, and a folder too long for one command line still
    gets metadata for every file.
    """
    if not _check_exiftool() or not sources:
        return {}

    tag_args = [f"-{t}" for t in tags]
    out: dict[str, dict] = {}

    for chunk in _batches(sources):
        try:
            # ``fast``: stop reading each file once its metadata is past, rather
            # than scanning to the end for trailers (half the time on a RAW).
            cmd = ["exiftool", "-j"] + (["-fast"] if fast else []) + tag_args + [str(s) for s in chunk]
            # A cold folder on a slow disk takes ~0.3 s a file; a batch that
            # times out loses every file in it.
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=60 + len(chunk))
            if not result.stdout.strip():
                continue
            data = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Batch EXIF extraction failed: {e}")
            continue

        by_path = {Path(e["SourceFile"]): e for e in data if e.get("SourceFile")}
        for i, source in enumerate(chunk):
            entry = by_path.get(Path(source))
            if entry is None and len(data) == len(chunk):
                entry = data[i]
            if entry is not None and "Error" not in entry:
                out[str(source)] = entry

    return out


def _batch_get_orientations(sources: list[Path]) -> dict[str, int]:
    """
    Get EXIF Orientation for multiple RAW files, one exiftool call per batch.
    Returns dict mapping str(source_path) to orientation int (1-8).

    Keys are matched by input order, NOT exiftool's SourceFile,
    to avoid Windows path separator mismatches.

    Orientation is the one field where a silent batch failure is invisible:
    the thumbnail is still produced, just rotated wrong, so callers must
    treat a missing key as "ask again per file" rather than "assume 1".
    """
    if not _check_exiftool() or not sources:
        return {}

    out: dict[str, int] = {}

    for chunk in _batches(sources):
        try:
            cmd = (
                ["exiftool", "-j", "-n", "-Orientation"]
                + [str(s) for s in chunk]
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                continue
            data = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Batch orientation extraction failed: {e}")
            continue

        # exiftool returns results in same order as input files
        for i, entry in enumerate(data):
            if i >= len(chunk):
                break
            orient = entry.get("Orientation", 1)
            try:
                out[str(chunk[i])] = int(orient)
            except (ValueError, TypeError):
                out[str(chunk[i])] = 1

    return out


# ── Extraction: rawpy fallback ───────────────────────────────────────

def _decode_raw_with_rawpy(source: Path) -> Optional[bytes]:
    """Full RAW decode via rawpy/libraw. Slow (~800ms) but always works."""
    try:
        import rawpy
    except ImportError:
        logger.warning("rawpy not installed, cannot decode RAW files")
        return None

    try:
        with rawpy.imread(str(source)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
            )
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"rawpy decode failed for {source.name}: {e}")
        return None


# ── Extraction: regular JPEG/TIFF ────────────────────────────────────

def _read_standard_image(source: Path) -> Optional[bytes]:
    """Read a standard image file, applying EXIF orientation."""
    try:
        img = Image.open(source)
        img.load()
        # Apply EXIF orientation (rotation + mirroring)
        img = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Failed to read {source.name}: {e}")
        return None


# ── Resize with auto-rotation ────────────────────────────────────────

def _normalize_image(
    jpeg_data: bytes,
    raw_orientation: int = 1,
) -> tuple[Image.Image, bool]:
    """Open an image and normalize orientation exactly once."""
    img = Image.open(io.BytesIO(jpeg_data))
    img.load()

    if raw_orientation > 1:
        # Guard against double-rotation: if the embedded JPEG is already
        # in portrait layout (h > w) and the container says portrait
        # (orientation 5-8), the JPEG was pre-rotated — skip transform.
        w, h = img.size
        is_portrait_pixel = h > w
        is_portrait_orient = raw_orientation in (5, 6, 7, 8)
        if is_portrait_pixel and is_portrait_orient:
            return img, False
        return _apply_orientation(img, raw_orientation), True

    orientation = 1
    try:
        orientation = int(img.getexif().get(274, 1) or 1)
    except Exception:
        orientation = 1

    if orientation > 1:
        return ImageOps.exif_transpose(img), True
    return img, False


def _resize_jpeg(
    jpeg_data: bytes, max_edge: int, quality: int = 85, raw_orientation: int = 1
) -> bytes:
    """
    Resize JPEG data so the longest edge is max_edge pixels.
    For RAW-extracted JPEGs, applies raw_orientation manually.
    For standard JPEGs, uses EXIF transpose.
    """
    img, transformed = _normalize_image(jpeg_data, raw_orientation)

    w, h = img.size
    if max(w, h) <= max_edge:
        if not transformed:
            return jpeg_data
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    if w >= h:
        new_w = max_edge
        new_h = int(h * (max_edge / w))
    else:
        new_h = max_edge
        new_w = int(w * (max_edge / h))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── Orientation from source file ──────────────────────────────────────

# EXIF Orientation values → Pillow transpose operations
_ORIENTATION_OPS = {
    2: [Image.FLIP_LEFT_RIGHT],
    3: [Image.ROTATE_180],
    4: [Image.FLIP_TOP_BOTTOM],
    5: [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
    6: [Image.ROTATE_270],
    7: [Image.FLIP_LEFT_RIGHT, Image.ROTATE_270],
    8: [Image.ROTATE_90],
}


def _get_raw_orientation(source: Path) -> int:
    """
    Get EXIF Orientation value from a RAW file's container metadata.
    The embedded JPEG often lacks this tag — it's in the CR3/ARW/NEF container.
    Returns orientation int (1-8), or 1 (normal) if not found.
    """
    if not _check_exiftool():
        return _rawpy_orientation(source)
    try:
        result = subprocess.run(
            ["exiftool", "-n", "-Orientation", "-s3", str(source)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return 1


# LibRaw's flip, as the EXIF Orientation it stands for.
_FLIP_TO_ORIENTATION = {0: 1, 3: 3, 5: 8, 6: 6}


def _rawpy_orientation(source: Path) -> int:
    """The orientation LibRaw reads from a RAW, for when exiftool is not installed."""
    try:
        import rawpy

        with rawpy.imread(str(source)) as raw:
            return _FLIP_TO_ORIENTATION.get(raw.sizes.flip, 1)
    except Exception:
        return 1


def _orientation_for(source: Path, batch: dict[str, int]) -> int:
    """
    Orientation for one RAW: the batch answer when it has one, else read it
    from this file alone.

    The batch covers a whole folder in a few processes. Treating a missing
    key as 1 instead would make a batch failure invisible — the thumbnail is
    still written, just rotated the wrong way, and a sideways thumbnail looks
    like a thumbnail rather than a bug. The per-file read only happens for
    files the batch did not answer for, and costs nothing when exiftool is
    unavailable (it returns 1 without spawning anything).
    """
    try:
        return batch[str(source)]
    except KeyError:
        return _get_raw_orientation(source)


def _apply_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """Apply EXIF orientation value to a Pillow Image."""
    ops = _ORIENTATION_OPS.get(orientation)
    if not ops:
        return img
    for op in ops:
        img = img.transpose(op)
    return img


# ── Main API ─────────────────────────────────────────────────────────

RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef"}


def _sibling_jpeg_bytes(source: Path) -> Optional[bytes]:
    """The JPEG with the same name beside a RAW file, if there is one."""
    try:
        wanted = source.stem.lower()
        for sibling in source.parent.iterdir():
            if sibling.stem.lower() == wanted and sibling.suffix.lower() in (".jpg", ".jpeg"):
                return _read_standard_image(sibling)
    except OSError:
        pass
    return None


def _extract_source_jpeg(source: Path) -> Optional[bytes]:
    """The file's own picture as JPEG bytes (see _source_jpeg); ``None`` for a
    file with none, which is remembered until the file changes, so the grid,
    the face scan and smart search do not each try it again and again."""
    if has_no_picture(source):
        return None
    data = _source_jpeg(source)
    if not data:
        _remember_no_picture(source)
    return data


def _source_jpeg(source: Path) -> Optional[bytes]:
    """Get a viewable JPEG from any supported source file."""
    suffix = source.suffix.lower()

    if suffix in (".jpg", ".jpeg"):
        return _read_standard_image(source)

    if suffix in RAW_EXTENSIONS:
        data = extract_embedded_jpeg(source)
        if data:
            return data
        # A camera that records no viewable JPEG in its RAW (a Ricoh GR's DNG, for
        # one) still wrote one beside it: the camera's own rendering, in a moment,
        # where decoding the RAW takes seconds.
        data = _sibling_jpeg_bytes(source)
        if data:
            return data
        logger.info(f"No embedded JPEG in {source.name}, falling back to rawpy")
        return _decode_raw_with_rawpy(source)

    if suffix in (".tif", ".tiff"):
        return _read_standard_image(source)

    from config import is_video

    if is_video(suffix):
        # A frame of the clip, already upright (ffmpeg turns it as it plays).
        from .video import poster_jpeg

        return poster_jpeg(source)

    logger.warning(f"Unsupported file type: {suffix}")
    return None


# Files with no picture to give (an empty recording, a damaged file), by path,
# with the size and time they had then: asked again, they are not read again
# until the file changes. Without this, every view of a folder and every
# background pass tried the same two empty clips again, hundreds of times.
_NO_PICTURE: dict[str, tuple[int, int]] = {}
_NO_PICTURE_LIMIT = 4096
_no_picture_lock = threading.Lock()


def _stamp(source: Path) -> Optional[tuple[int, int]]:
    try:
        st = source.stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def has_no_picture(source: Path) -> bool:
    """Whether ``source`` was found to have no picture, as it is now."""
    kept = _NO_PICTURE.get(str(source))
    return kept is not None and kept == _stamp(source)


def _remember_no_picture(source: Path) -> None:
    stamp = _stamp(source)
    if stamp is None:
        return
    with _no_picture_lock:
        if len(_NO_PICTURE) >= _NO_PICTURE_LIMIT:
            _NO_PICTURE.pop(next(iter(_NO_PICTURE)))
        _NO_PICTURE[str(source)] = stamp


# Renders in progress, by file and size. The grid, the face scan, smart search
# and the measuring pass all ask for the same pictures, often at the same moment:
# the log showed one preview made four times over, side by side, each taking
# seventeen seconds because the other three were competing with it. A second
# caller now waits for the first one's result instead.
_rendering: dict[tuple[str, str], threading.Event] = {}
_rendering_lock = threading.Lock()


def get_thumbnail(
    source: Path,
    cache: ThumbnailCache,
    size: ThumbnailSize = ThumbnailSize.GRID,
) -> Optional[bytes]:
    """Get a thumbnail as JPEG bytes, generating and caching if needed; ``None``
    for a file with no picture, which is remembered (until it changes)."""
    cached = cache.get(source, size)
    if cached:
        return cached
    if has_no_picture(source):
        return None

    key = (str(source), size.name)
    with _rendering_lock:
        running = _rendering.get(key)
        if running is None:
            _rendering[key] = threading.Event()
    if running is not None:
        # The other caller's result, which is None when the file has no picture
        # to give: trying again at once would fail the same way.
        if running.wait(120):
            return cache.get(source, size)
        return _render_thumbnail(source, cache, size)
    try:
        return _render_thumbnail(source, cache, size)
    finally:
        with _rendering_lock:
            _rendering.pop(key).set()


def _render_thumbnail(source: Path, cache: ThumbnailCache, size: ThumbnailSize) -> Optional[bytes]:
    t0 = time.monotonic()

    jpeg_data = _extract_source_jpeg(source)
    if not jpeg_data:
        return None

    # For RAW files, the embedded JPEG doesn't have orientation EXIF.
    # Read it from the source CR3/ARW/NEF container and apply manually.
    is_raw = source.suffix.lower() in RAW_EXTENSIONS
    orientation = _get_raw_orientation(source) if is_raw else 1

    if size != ThumbnailSize.FULL and size.value > 0:
        quality = _JPEG_QUALITY.get(size, 85)
        jpeg_data = _resize_jpeg(jpeg_data, size.value, quality, orientation)
    else:
        img, _ = _normalize_image(jpeg_data, orientation if is_raw else 1)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY.get(size, 95))
        jpeg_data = buf.getvalue()

    cache.put(source, size, jpeg_data)

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        f"Generated {size.name} thumbnail for {source.name} "
        f"in {elapsed:.0f}ms ({len(jpeg_data) / 1024:.0f} KB)"
    )
    return jpeg_data


def get_preview(source: Path, cache: ThumbnailCache) -> Optional[bytes]:
    return get_thumbnail(source, cache, ThumbnailSize.PREVIEW)


# ── Batch operations ─────────────────────────────────────────────────

def generate_thumbnails_for_folder(
    folder: Path,
    cache: ThumbnailCache,
    extensions: list[str],
    sizes: Optional[list[ThumbnailSize]] = None,
    progress_cb=None,
    max_workers: Optional[int] = None,
) -> dict:
    """
    Pre-generate thumbnails for all images in a folder.

    Optimized pipeline:
      1. Batch-extract embedded JPEGs with a single exiftool call
      2. Parallel resize/cache across CPU cores
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if sizes is None:
        sizes = [ThumbnailSize.GRID, ThumbnailSize.PREVIEW]
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    ext_set = {ext.lower() for ext in extensions}

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in ext_set
    )

    stats = {"total": len(files), "generated": 0, "cached": 0, "failed": 0}
    t0 = time.monotonic()

    # Find work to do (skip cached)
    work = []
    for f in files:
        for sz in sizes:
            if cache.get(f, sz):
                stats["cached"] += 1
            else:
                work.append((f, sz))

    if not work:
        stats["elapsed_seconds"] = round(time.monotonic() - t0, 2)
        stats["workers"] = max_workers
        return stats

    # Step 1: Batch extract embedded JPEGs for RAW files (single exiftool call)
    raw_files_needing_extract = list({
        f for f, sz in work if f.suffix.lower() in RAW_EXTENSIONS
    })
    extracted_cache = {}
    orientation_cache = {}

    if raw_files_needing_extract:
        logger.info(
            f"Batch extracting {len(raw_files_needing_extract)} RAW files..."
        )
        extracted_cache = _batch_extract_embedded_jpegs(raw_files_needing_extract)

        # Orientations come from the same chunked exiftool passes
        orientation_cache = _batch_get_orientations(raw_files_needing_extract)

        logger.info(
            f"Batch extraction got {len(extracted_cache)} / "
            f"{len(raw_files_needing_extract)} embedded JPEGs"
        )

    # Step 2: Parallel resize + cache
    def _process_one(file_path: Path, size: ThumbnailSize) -> bool:
        """Process a single file+size combo. Uses batch-extracted data if available."""
        if cache.get(file_path, size):
            return True

        key = str(file_path)
        jpeg_data = None

        if key in extracted_cache:
            jpeg_data = extracted_cache[key]
        else:
            jpeg_data = _extract_source_jpeg(file_path)

        if not jpeg_data:
            return False

        is_raw = file_path.suffix.lower() in RAW_EXTENSIONS
        orientation = _orientation_for(file_path, orientation_cache) if is_raw else 1

        if size != ThumbnailSize.FULL and size.value > 0:
            quality = _JPEG_QUALITY.get(size, 85)
            jpeg_data = _resize_jpeg(jpeg_data, size.value, quality, orientation)
        else:
            img, _ = _normalize_image(jpeg_data, orientation if is_raw else 1)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY.get(size, 95))
            jpeg_data = buf.getvalue()

        cache.put(file_path, size, jpeg_data)
        return True

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, f, sz): (f, sz)
            for f, sz in work
        }
        for future in as_completed(futures):
            done += 1
            f, sz = futures[future]
            try:
                if future.result():
                    stats["generated"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Thumbnail generation failed for {f.name}: {e}")
                stats["failed"] += 1

            if progress_cb and done % max_workers == 0:
                progress_cb(done, len(work), f.name)

    # Free batch extraction memory
    extracted_cache.clear()

    stats["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    stats["workers"] = max_workers
    return stats


def generate_thumbnails_batch(
    files: list[Path],
    cache: ThumbnailCache,
    size: ThumbnailSize = ThumbnailSize.GRID,
    max_workers: Optional[int] = None,
) -> dict:
    """Generate thumbnails for a specific list of files (prefetch)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    uncached = [f for f in files if not cache.get(f, size)]

    stats = {
        "total": len(files),
        "generated": 0,
        "cached": len(files) - len(uncached),
        "failed": 0,
    }
    t0 = time.monotonic()

    if uncached:
        # Batch extract RAWs first
        raws = [f for f in uncached if f.suffix.lower() in RAW_EXTENSIONS]
        extracted = _batch_extract_embedded_jpegs(raws) if raws else {}
        orientations = _batch_get_orientations(raws) if raws else {}

        def _do_one(f: Path) -> bool:
            if cache.get(f, size):
                return True
            key = str(f)
            jpeg_data = extracted.get(key) or _extract_source_jpeg(f)
            if not jpeg_data:
                return False

            is_raw = f.suffix.lower() in RAW_EXTENSIONS
            orientation = _orientation_for(f, orientations) if is_raw else 1

            if size != ThumbnailSize.FULL and size.value > 0:
                jpeg_data = _resize_jpeg(
                    jpeg_data, size.value, _JPEG_QUALITY.get(size, 85), orientation
                )
            else:
                img, _ = _normalize_image(jpeg_data, orientation if is_raw else 1)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                jpeg_data = buf.getvalue()
            cache.put(f, size, jpeg_data)
            return True

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do_one, f): f for f in uncached}
            for future in as_completed(futures):
                try:
                    if future.result():
                        stats["generated"] += 1
                    else:
                        stats["failed"] += 1
                except Exception:
                    stats["failed"] += 1

    stats["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    return stats


# ── EXIF extraction ──────────────────────────────────────────────────

def get_exif_summary(source: Path) -> dict:
    """Extract key EXIF fields for the UI."""
    from config import is_video

    if is_video(source.suffix):
        # Its date and length are worked out as indexing works them out.
        found = get_exif_batch([source]).get(str(source))
        if found:
            return found
    info = {
        "camera": "", "lens": "", "focal_length": "", "aperture": "",
        "shutter_speed": "", "iso": "", "date": "", "dimensions": "",
        "file_size_mb": round(source.stat().st_size / (1024 * 1024), 2),
    }

    if not _check_exiftool():
        return info

    try:
        result = subprocess.run(
            [
                "exiftool", "-j",
                "-Model", "-LensModel", "-FocalLength", "-FNumber",
                "-ExposureTime", "-ISO", "-DateTimeOriginal", "-ImageSize",
                str(source),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data:
                d = data[0]
                info["camera"] = d.get("Model", "")
                info["lens"] = d.get("LensModel", "")
                info["focal_length"] = d.get("FocalLength", "")
                info["aperture"] = f"f/{d['FNumber']}" if "FNumber" in d else ""
                info["shutter_speed"] = d.get("ExposureTime", "")
                info["iso"] = str(d.get("ISO", ""))
                info["date"] = d.get("DateTimeOriginal", "")
                info["dimensions"] = d.get("ImageSize", "")

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.debug(f"EXIF extraction failed for {source.name}: {e}")

    return info


def _video_date(source: Path, d: dict) -> str:
    """When a video was taken, as EXIF writes a date. A Canon that writes
    DateTimeOriginal says it in local time; a DJI Pocket writes none, but its
    file name carries the local time to the second; failing both, CreateDate,
    which some cameras write in local time and some in UTC."""
    if d.get("DateTimeOriginal"):
        return d["DateTimeOriginal"]
    from importer.copy import _get_date_from_filename

    named = _get_date_from_filename(source)
    if named is not None:
        return named.strftime("%Y:%m:%d %H:%M:%S")
    created = d.get("CreateDate", "")
    return "" if str(created).startswith("0000") else created


def get_exif_batch(sources: list[Path], fast: bool = False) -> dict[str, dict]:
    """
    Extract EXIF for multiple files in one exiftool call.
    Returns dict mapping filepath string to EXIF info dict; a video's carries
    its ``duration`` in seconds.
    """
    from config import is_video

    tags = [
        "Model", "LensModel", "FocalLength", "FNumber",
        "ExposureTime", "ISO", "DateTimeOriginal", "ImageSize",
    ]
    if any(is_video(s.suffix) for s in sources):
        tags += ["CreateDate", "Duration#"]
    raw = _batch_extract_exif(sources, tags, fast=fast)
    results = {}
    for source in sources:
        d = raw.get(str(source), {})
        try:
            size_mb = round(source.stat().st_size / (1024 * 1024), 2)
        except OSError:                     # gone since it was listed
            size_mb = 0.0
        results[str(source)] = {
            "camera": d.get("Model", ""),
            "lens": d.get("LensModel", ""),
            "focal_length": d.get("FocalLength", ""),
            "aperture": f"f/{d['FNumber']}" if "FNumber" in d else "",
            "shutter_speed": d.get("ExposureTime", ""),
            "iso": str(d.get("ISO", "")),
            "date": d.get("DateTimeOriginal", ""),
            "dimensions": d.get("ImageSize", ""),
            "file_size_mb": size_mb,
        }
        if is_video(source.suffix):
            results[str(source)]["date"] = _video_date(source, d)
            try:
                results[str(source)]["duration"] = float(d.get("Duration") or 0.0)
            except (TypeError, ValueError):
                results[str(source)]["duration"] = 0.0
    return results
