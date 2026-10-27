"""
File importer: copies photos from SD card to library with date-based organization.

Features:
  - Reads EXIF DateTimeOriginal to determine folder placement
  - Falls back to file modification time if EXIF unavailable
  - SHA-256 checksum verification after copy
  - Duplicate detection (skip/rename/overwrite)
  - Progress callback for UI integration

Why this module is structured the way it is
------------------------------------------
A full SD card import used to spawn one ``exiftool`` process *per file*, before
deciding whether the file even needed copying. Measured on a 9,000-file card
that is ~770 ms per file — roughly two hours, most of it spent on photos that
are already in the library. Because files are processed in card order, the
photos that actually needed importing sit at the end of the list, so a mostly
backed-up card looks like it never starts.

The import now runs in three phases:

  1. classify — one directory walk of the library builds a
     name → [(size, path)] index. A card file whose name *and* byte size match
     a library file is already backed up, so it is skipped without reading any
     EXIF at all. Size is part of the test because Canon reuses filenames
     across sessions: in this library 7,507 names appear more than once, and
     name alone would silently drop new photos. A spot check of 40 groups that
     shared both name and size found them byte-identical every time, so this
     is a safe test — and it costs no file reads.
  2. dates — EXIF is read only for the files that survived phase 1, in batches
     of EXIFTOOL_BATCH_SIZE per exiftool process, with ``-fast`` so only the
     file header is read (~12 ms/file instead of ~770 ms).
  3. copy — copies run on a small thread pool, and the source SHA-256 is
     computed while streaming the copy instead of re-reading the source
     afterwards (3 full passes over the data become 2).

Measured end to end on a 9,202-file card with 8,984 photos already backed up:
the pre-copy pass goes from ~118 minutes to ~6 seconds.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from config import VIDEO_EXTENSIONS, Config
# The command-line chunking for exiftool is shared with the thumbnail
# pipeline, which reads the same limit the hard way (see exif_batches).
# Re-exported under its original private name so existing callers and tests
# keep working.
from exif_batches import (
    EXIFTOOL_ARG_CHARS,
    EXIFTOOL_BATCH_SIZE,
    batches as _batches,
)

# Optional extra check when deciding whether a name+size match is really the
# same photo: compare this many bytes at the head and tail of both files.
# Off by default — measured at ~50 ms per file on a card (random 64 KB reads
# are seek-bound), which costs more than just reading the EXIF instead. Set it
# on import_files(duplicate_probe_bytes=...) if you want belt-and-braces.
DUPLICATE_PROBE_BYTES = 64 * 1024

# Concurrent copy streams. The bottleneck is SD-card read + library write, which
# overlap well across a few streams; more than this just thrashes the card.
DEFAULT_PARALLEL_WORKERS = 4


@dataclass
class ImportResult:
    """Result of importing a single file."""
    source: Path
    destination: Path
    status: str  # "copied" | "skipped_duplicate" | "renamed" | "failed"
    checksum: str = ""
    error: str = ""


@dataclass
class ImportSummary:
    """Summary of an import batch."""
    total_files: int = 0
    copied: int = 0
    skipped: int = 0
    failed: int = 0
    total_bytes: int = 0
    results: list[ImportResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0


# ── Path keys ────────────────────────────────────────────────────────

def _norm_key(path) -> str:
    """
    Normalise a path for use as a dict key. exiftool echoes the paths we pass
    it but rewrites separators to "/", so both sides must be normalised.
    """
    s = str(path).replace("\\", "/")
    return s.casefold() if os.name == "nt" else s


# ── Date extraction ──────────────────────────────────────────────────

def _get_date_from_exif(filepath: Path) -> Optional[datetime]:
    """
    Extract DateTimeOriginal from EXIF using exiftool (single file).
    This is the most reliable method for CR3 files.

    Prefer batch_capture_dates() for anything above a handful of files — this
    pays full process-startup cost and is only used as a per-file fallback.
    """
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-fast",
                "-DateTimeOriginal",
                "-d", "%Y-%m-%dT%H:%M:%S",
                "-j",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data and "DateTimeOriginal" in data[0]:
                return datetime.fromisoformat(data[0]["DateTimeOriginal"])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, FileNotFoundError):
        pass
    return None


# A date, and perhaps the time, somewhere in a file name: DJI's
# DJI_20251018143005_0001_D.MP4, a phone's 20251018_143005.jpg, a renamed
# 2025-10-18_001.CR3.
_NAME_DATE = re.compile(
    r"(?<!\d)((?:19|20)\d\d)-?(\d\d)-?(\d\d)(?:[_-]?(\d\d)(\d\d)(\d\d))?(?!\d)"
)


def _get_date_from_filename(filepath: Path) -> Optional[datetime]:
    """
    The date a file's name carries, if it carries one.

    A camera's own names mostly don't (6U4A1234.CR3, GX010042.MP4), but DJI's
    do, to the second, in the camera's local time — which is what a video needs,
    having no DateTimeOriginal.
    """
    for match in _NAME_DATE.finditer(filepath.stem):
        parts = [int(g) for g in match.groups() if g is not None]
        try:
            return datetime(*parts)
        except ValueError:
            continue
    return None


def get_capture_date(filepath: Path) -> datetime:
    """
    Get the capture date for a photo file. Priority:
      1. EXIF DateTimeOriginal (most accurate)
      2. Filename parsing (if date-encoded)
      3. File modification time (last resort)
    """
    date = _get_date_from_exif(filepath)
    if date:
        return date

    date = _get_date_from_filename(filepath)
    if date:
        return date

    # Fall back to mtime
    return datetime.fromtimestamp(filepath.stat().st_mtime)


def parse_exiftool_json(stdout: str) -> dict[str, datetime]:
    """Parse an exiftool -j payload into {normalised_path: datetime}."""
    out: dict[str, datetime] = {}
    if not stdout or not stdout.strip():
        return out
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return out
    if not isinstance(data, list):
        return out
    for entry in data:
        if not isinstance(entry, dict):
            continue
        value = entry.get("DateTimeOriginal")
        source = entry.get("SourceFile")
        if not value or not source:
            continue
        try:
            out[_norm_key(source)] = datetime.fromisoformat(value)
        except ValueError:
            continue
    return out


def batch_capture_dates(paths: list[Path]) -> tuple[dict[str, datetime], bool]:
    """
    Read DateTimeOriginal for many files using one exiftool process per batch.

    Returns (dates, exiftool_worked). ``exiftool_worked`` is False when no
    batch produced anything — typically exiftool is not installed — so callers
    can skip a per-file retry that would fail the same way 9,000 times.
    """
    dates: dict[str, datetime] = {}
    if not paths:
        return dates, False

    worked = False
    for batch in _batches(paths):
        try:
            result = subprocess.run(
                [
                    "exiftool",
                    "-fast",
                    "-DateTimeOriginal",
                    "-d", "%Y-%m-%dT%H:%M:%S",
                    "-j",
                ] + [str(p) for p in batch],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(60, len(batch)),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        if not result.stdout.strip():
            continue
        found = parse_exiftool_json(result.stdout)
        if found:
            worked = True
            dates.update(found)

    return dates, worked


# ── Library index (the "already backed up?" fast path) ───────────────

def build_library_index(
    library_root: str | Path,
    extensions: Optional[Iterable[str]] = None,
) -> dict[str, list[tuple[int, Path]]]:
    """
    Walk the library once and map lowercase filename → [(size, path), ...].

    os.scandir is used deliberately: DirEntry.stat() reuses the metadata the
    directory listing already returned, so a 68k-file library costs a few
    seconds instead of 68k separate stat syscalls.
    """
    root = Path(library_root)
    index: dict[str, list[tuple[int, Path]]] = {}
    ext_set = {ext.lower() for ext in extensions} if extensions else None

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = os.scandir(current)
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.startswith("."):
                            continue
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.name.startswith("."):
                        continue
                    if ext_set is not None and Path(entry.name).suffix.lower() not in ext_set:
                        continue
                    index.setdefault(entry.name.lower(), []).append(
                        (entry.stat().st_size, Path(entry.path))
                    )
                except OSError:
                    continue
    return index


def _files_identical(source: Path, other: Path, probe_bytes: int) -> bool:
    """
    Cheap identity check: same size, and the first and last ``probe_bytes``
    blocks match. Two different photos reusing a Canon filename essentially
    never pass this, while reading 128 KB is orders of magnitude cheaper than
    hashing a 16 MB CR3.
    """
    try:
        src = os.stat(source)
        dst = os.stat(other)
    except OSError:
        return False
    if src.st_size != dst.st_size:
        return False
    if src.st_size == 0:
        return True

    n = min(probe_bytes, src.st_size)
    try:
        with open(source, "rb") as fa, open(other, "rb") as fb:
            if fa.read(n) != fb.read(n):
                return False
            if n < src.st_size:
                fa.seek(src.st_size - n)
                fb.seek(dst.st_size - n)
                if fa.read(n) != fb.read(n):
                    return False
    except OSError:
        return False
    return True


def _find_in_library(
    source: Path,
    index: dict[str, list[tuple[int, Path]]],
    probe_bytes: int = 0,
) -> Optional[Path]:
    """
    Return the library path this card file is already backed up as, if any.

    Name + byte size is the default test and costs no file reads. Pass a
    non-zero probe_bytes to additionally require that the head and tail of both
    files match — slower, and only worth it if you distrust size.
    """
    candidates = index.get(source.name.lower())
    if not candidates:
        return None
    try:
        size = source.stat().st_size
    except OSError:
        return None
    for lib_size, lib_path in candidates:
        if lib_size != size:
            continue
        if not probe_bytes or _files_identical(source, lib_path, probe_bytes):
            return lib_path
    return None


# ── Checksum ─────────────────────────────────────────────────────────

def sha256_file(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file, reading in chunks to handle large CR3s."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_with_hash(source: Path, dest: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Copy source → dest while hashing the bytes already streamed through.
    Equivalent to shutil.copy2 (metadata preserved) but saves a full extra
    read of the source during verification.
    """
    h = hashlib.sha256()
    with open(source, "rb") as fs, open(dest, "wb") as fd:
        while True:
            chunk = fs.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            fd.write(chunk)
    shutil.copystat(source, dest)
    return h.hexdigest()


# ── Destination path building ────────────────────────────────────────

def build_dest_path(
    source: Path,
    capture_date: datetime,
    config: Config,
) -> Path:
    """
    Build the destination path based on config folder pattern.

    Example: library_root/2024/06/15/6U4A1234.CR3
    """
    folder = config.folder_pattern.format(
        year=capture_date.strftime("%Y"),
        month=capture_date.strftime("%m"),
        day=capture_date.strftime("%d"),
    )
    return Path(config.library_root) / folder / source.name


def _handle_duplicate(dest: Path, strategy: str) -> tuple[Path, str]:
    """
    Handle case where destination file already exists.
    Returns (final_path, status).
    """
    if strategy == "skip":
        return dest, "skipped_duplicate"
    elif strategy == "overwrite":
        return dest, "copied"
    elif strategy == "rename":
        return _reserve_unique_name(dest)
    else:
        return dest, "skipped_duplicate"


def _reserve_unique_name(dest: Path) -> tuple[Path, str]:
    """
    Atomically claim a free destination name with O_EXCL, so two concurrent
    workers can never pick the same "_001" fallback.
    """
    parent = dest.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    stem, suffix = dest.stem, dest.suffix
    candidate = dest
    counter = 0
    while True:
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            counter += 1
            candidate = parent / f"{stem}_{counter:03d}{suffix}"
            continue
        os.close(fd)
        return candidate, ("renamed" if candidate != dest else "copied")


def _claim_destination(dest: Path, strategy: str, lock: threading.Lock) -> tuple[Path, str]:
    """Decide where a file lands, serialised so concurrent workers stay safe."""
    with lock:
        if dest.exists():
            if strategy == "rename":
                return _reserve_unique_name(dest)
            if strategy == "overwrite":
                return dest, "copied"
            return dest, "skipped_duplicate"
        return dest, "copied"


# ── Main import function ─────────────────────────────────────────────

ProgressCallback = Callable[[int, int, str], None]  # (current, total, filename)
StageCallback = Callable[[str], None]               # ("indexing_library" | ...)


def import_files(
    files: list[Path],
    config: Config,
    progress_cb: Optional[ProgressCallback] = None,
    verify_checksums: bool = True,
    reuse_existing: bool = True,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    stage_cb: Optional[StageCallback] = None,
    duplicate_probe_bytes: int = 0,
) -> ImportSummary:
    """
    Import a list of files into the library.

    Args:
        files: List of source file paths to import
        config: Application config
        progress_cb: Optional callback for progress updates (current, total, filename)
        verify_checksums: Whether to verify SHA-256 after copy (slower but safer)
        reuse_existing: Skip files already present in the library without reading
            their EXIF. Disable to force a full EXIF pass over every file.
        parallel_workers: Concurrent copy streams (1 disables threading)
        stage_cb: Optional callback announcing which phase the import is in
        duplicate_probe_bytes: Extra bytes to compare at the head/tail of a
            name+size match before calling it a duplicate. 0 (default) trusts
            name+size, which is both safe for camera files and ~50x cheaper.

    Returns:
        ImportSummary with details of every file processed
    """
    total = len(files)
    summary = ImportSummary(total_files=total, started_at=datetime.now())
    results: list[Optional[ImportResult]] = [None] * total

    done = 0
    cb_lock = threading.Lock()

    def _stage(name: str):
        if stage_cb:
            try:
                stage_cb(name)
            except Exception:
                pass

    def _report(source: Path):
        nonlocal done
        if not progress_cb:
            return
        with cb_lock:
            done += 1
            current = done
        try:
            progress_cb(current, total, source.name)
        except Exception:
            pass

    # ── Phase 1: classify against the library, no EXIF ───────────────
    index: dict[str, list[tuple[int, Path]]] = {}
    if reuse_existing and total:
        _stage("indexing_library")
        index = build_library_index(config.library_root, config.import_extensions)

    _stage("checking")
    pending: list[int] = []
    for i, source in enumerate(files):
        existing = _find_in_library(source, index, duplicate_probe_bytes) if index else None
        if existing is not None:
            results[i] = ImportResult(
                source=source,
                destination=existing,
                status="skipped_duplicate",
            )
            _report(source)
        else:
            pending.append(i)

    # ── Phase 2: EXIF for the survivors only ─────────────────────────
    capture_dates: dict[int, datetime] = {}
    if pending:
        _stage("reading_exif")
        dates, exif_ok = batch_capture_dates([files[i] for i in pending])
        for i in pending:
            source = files[i]
            date = dates.get(_norm_key(source))
            if date is None and source.suffix.lower() not in VIDEO_EXTENSIONS:
                # Not in the batch result: no EXIF in the file, or the batch
                # failed. Retry individually only if exiftool works at all. A
                # video never has DateTimeOriginal, so asking again only costs.
                date = _get_date_from_exif(source) if exif_ok else None
            if date is None:
                date = _get_date_from_filename(source)
            if date is None:
                try:
                    date = datetime.fromtimestamp(source.stat().st_mtime)
                except OSError:
                    date = datetime.now()
            capture_dates[i] = date

    # ── Phase 3: copy + verify ───────────────────────────────────────
    _stage("copying")
    dest_lock = threading.Lock()

    def _work(i: int) -> tuple[int, ImportResult]:
        result = _import_single_file(
            files[i], capture_dates[i], config, verify_checksums, dest_lock
        )
        _report(files[i])
        return i, result

    if pending:
        workers = max(1, min(int(parallel_workers or 1), len(pending)))
        if workers == 1:
            for i in pending:
                j, result = _work(i)
                results[j] = result
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="import") as pool:
                for j, result in pool.map(_work, pending):
                    results[j] = result

    for result in results:
        if result is None:
            continue
        if result.status in ("copied", "renamed"):
            summary.copied += 1
            try:
                summary.total_bytes += result.source.stat().st_size
            except OSError:
                pass
        elif result.status == "skipped_duplicate":
            summary.skipped += 1
        elif result.status == "failed":
            summary.failed += 1
        summary.results.append(result)

    summary.finished_at = datetime.now()
    _stage("done")
    return summary


def _import_single_file(
    source: Path,
    capture_date: datetime,
    config: Config,
    verify_checksums: bool,
    dest_lock: Optional[threading.Lock] = None,
) -> ImportResult:
    """Import a single file to the library. capture_date comes from phase 2."""
    try:
        # 1. Build destination path and claim it
        dest = build_dest_path(source, capture_date, config)
        if dest_lock is not None:
            dest, status = _claim_destination(dest, config.duplicate_strategy, dest_lock)
        else:
            if dest.exists():
                dest, status = _handle_duplicate(dest, config.duplicate_strategy)
            else:
                status = "copied"
        if status == "skipped_duplicate":
            return ImportResult(
                source=source,
                destination=dest,
                status="skipped_duplicate",
            )

        # 2. Create directory and copy
        dest.parent.mkdir(parents=True, exist_ok=True)
        checksum = ""
        if verify_checksums:
            # Hash the source while streaming it into the destination.
            src_hash = _copy_with_hash(source, dest)
            checksum = src_hash
            dst_hash = sha256_file(dest)
            if src_hash != dst_hash:
                dest.unlink(missing_ok=True)  # remove corrupted copy
                return ImportResult(
                    source=source,
                    destination=dest,
                    status="failed",
                    error=f"Checksum mismatch: {src_hash[:12]}… ≠ {dst_hash[:12]}…",
                )
        else:
            shutil.copy2(source, dest)  # copy2 preserves metadata

        return ImportResult(
            source=source,
            destination=dest,
            status=status,
            checksum=checksum,
        )

    except Exception as e:
        return ImportResult(
            source=source,
            destination=Path(""),
            status="failed",
            error=str(e),
        )


# ── CLI entry point for standalone testing ───────────────────────────

_STAGE_LABELS = {
    "indexing_library": "Indexing library…",
    "checking": "Checking which photos are already backed up…",
    "reading_exif": "Reading capture dates…",
    "copying": "Copying…",
    "done": "Done.",
}


def _print_stage(stage: str):
    print(f"\n  {_STAGE_LABELS.get(stage, stage)}")


def _print_progress(current: int, total: int, filename: str):
    pct = current / total * 100
    print(f"\r  [{current}/{total}] {pct:5.1f}%  {filename:<30s}", end="", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import load_config
    from importer.sd_detect import detect_sd_cards, find_importable_files

    config = load_config()

    print("Scanning for SD cards...")
    cards = detect_sd_cards(config)

    if not cards:
        print("No SD cards found. Check config.yaml sd_mount_paths.")
        sys.exit(1)

    for card in cards:
        print(f"\nFound: {card.mount_point}")
        print(f"  {card.total_files} files, {card.total_size_gb:.1f} GB")

        files = find_importable_files(card, config.import_extensions)

        print(f"  Importing to: {config.library_root}")
        summary = import_files(
            files, config,
            progress_cb=_print_progress,
            stage_cb=_print_stage,
        )
        print()  # newline after progress

        print(f"  ✓ Copied: {summary.copied}")
        print(f"  → Skipped: {summary.skipped}")
        print(f"  ✗ Failed: {summary.failed}")
        print(f"  Time: {summary.elapsed_seconds:.1f}s")

        if summary.failed > 0:
            print("\n  Failed files:")
            for r in summary.results:
                if r.status == "failed":
                    print(f"    {r.source.name}: {r.error}")
