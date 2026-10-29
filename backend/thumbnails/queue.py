"""Persistent low-priority thumbnail queue."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import background as gate
from catalog.database import CatalogDB, ThumbnailJobRecord

from .extract import ThumbnailCache, ThumbnailSize, get_thumbnail
from .store import CacheNotWritable

logger = logging.getLogger(__name__)

# When the cache volume refuses writes there is no point walking the library
# again every idle_rescan_seconds — every job fails and is re-enqueued, which
# looks like activity and produces nothing. Pause the queue and retry slowly.
_STORAGE_BACKOFF_SECONDS = 300.0

# How long the measurement sweep may be starved by thumbnail work before it gets
# a turn anyway. Thumbnails win because the UI needs them, but a library that is
# still backfilling previews would otherwise never be idle long enough to
# measure anything, and every feature that reads measurements would wait forever.
_MEASURE_STARVATION_SECONDS = 120.0


def discover_missing_previews(
    folder: Path,
    cache: ThumbnailCache,
    extensions: list[str],
) -> dict:
    """Return per-folder thumbnail availability stats."""
    ext_set = {ext.lower() for ext in extensions}
    files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in ext_set
    ]

    missing_grid = 0
    missing_preview = 0
    missing_any = 0
    for file_path in files:
        no_grid = not cache.get(file_path, ThumbnailSize.GRID)
        no_preview = not cache.get(file_path, ThumbnailSize.PREVIEW)
        if no_grid:
            missing_grid += 1
        if no_preview:
            missing_preview += 1
        if no_grid or no_preview:
            missing_any += 1

    total_files = len(files)
    return {
        "total_files": total_files,
        "missing_grid": missing_grid,
        "missing_preview": missing_preview,
        "missing_any": missing_any,
        "ready_grid": total_files - missing_grid,
        "ready_preview": total_files - missing_preview,
    }


def _turbo_worker_count() -> int:
    """Return number of worker threads for turbo mode (~80% of CPU cores)."""
    return max(1, int((os.cpu_count() or 4) * 0.8))


class ThumbnailQueueManager:
    """Processes thumbnail jobs in background thread(s).

    Normal mode: single thread, 0.25s sleep between items.
    Turbo mode: ThreadPoolExecutor with ~80% of CPU cores, no sleep.

    The same thread also measures folders for culling and export matching —
    right after a folder's thumbnails are generated, and in the gaps when there
    is nothing left to generate. Thumbnail work always comes first.
    """

    def __init__(
        self,
        db: CatalogDB,
        cache: ThumbnailCache,
        library_root: str,
        import_extensions: list[str],
        *,
        enabled: bool = True,
        idle_interval_seconds: float = 0.25,
        low_priority: int = 100,
        auto_generate: bool = True,
        idle_rescan_seconds: float = 60.0,
        auto_analyze: bool = True,
        turbo: bool = False,
    ):
        self.db = db
        self.cache = cache
        self.library_root = Path(library_root)
        self.import_extensions = import_extensions
        self.enabled = enabled
        self.idle_interval_seconds = idle_interval_seconds
        self.low_priority = low_priority
        self.auto_generate = auto_generate
        self.idle_rescan_seconds = idle_rescan_seconds
        self.auto_analyze = auto_analyze
        self._turbo = turbo
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Set when the cache refuses writes, so the API can report why the queue
        # is idle instead of showing an empty queue with no explanation.
        self.storage_error = ""
        self._last_storage_error = ""
        # How far the idle measurement sweep has got, so it walks the library
        # once instead of returning to the same folder forever.
        self._measure_cursor = ""
        # When the sweep last ran, so thumbnail work cannot postpone it forever.
        self._measure_starvation_seconds = _MEASURE_STARVATION_SECONDS
        self._last_measure_at = time.monotonic()
        # What this thread is doing at the moment someone asks. An empty job
        # list looks the same on screen whether the worker is measuring a folder,
        # backing off because the cache refused writes, or genuinely finished, and
        # "it looks finished" is how a stalled queue hides for hours.
        self._activity = ""
        self._activity_folder = ""
        self._activity_started = 0.0
        self._iterations = 0
        self._completed_jobs = 0
        self._measure_attempts = 0
        # Whether the job in hand was queued by the app rather than by someone:
        # those wait their turn behind the person using it (background.py).
        self._gated = False

    @property
    def turbo(self) -> bool:
        return self._turbo

    def set_turbo(self, enabled: bool) -> None:
        self._turbo = enabled
        logger.info(f"Thumbnail queue turbo mode {'ON' if enabled else 'OFF'}"
                     f" ({_turbo_worker_count()} workers)" if enabled else "")

    def start(self):
        if not self.enabled or self._thread:
            return
        self.db.reset_running_thumbnail_jobs()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="thumbnail-queue",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def enqueue_folder(self, folder: str) -> ThumbnailJobRecord:
        return self.db.create_thumbnail_job(folder, priority=self.low_priority)

    def list_jobs(self) -> list[dict]:
        return [self._serialize_job(job) for job in self.db.list_thumbnail_jobs()]

    def status(self) -> dict:
        """
        What the worker is doing, as far as it can be known from outside.

        The job table says what is queued; it cannot say whether anything is
        working on it, what that work is, or why an empty queue is empty. Those
        are the three questions a stalled backfill actually asks, and until now
        all three were answered by watching the library and guessing.
        """
        thread = self._thread
        activity = self._activity
        return {
            "enabled": self.enabled,
            "alive": bool(thread and thread.is_alive()),
            "auto_generate": self.auto_generate,
            "auto_analyze": self.auto_analyze,
            "turbo": self._turbo,
            # "" when idle, otherwise "thumbnails" / "measuring" / "paused".
            "activity": activity,
            "activity_folder": self._activity_folder,
            "activity_seconds": (
                round(time.monotonic() - self._activity_started, 1) if activity else 0.0
            ),
            "storage_error": self.storage_error,
            "cache_writable": self.cache.writable,
            "write_error": self.cache.write_error,
            # How far the idle measurement sweep has got. Not a progress figure:
            # it is a cursor, and seeing it move is what distinguishes a slow
            # sweep from one that is stuck on a folder it cannot read.
            "measure_cursor": self._measure_cursor,
            "iterations": self._iterations,
            "jobs_completed": self._completed_jobs,
            "measure_attempts": self._measure_attempts,
        }

    def _set_activity(self, activity: str, folder: str = "") -> None:
        self._activity = activity
        self._activity_folder = folder
        self._activity_started = time.monotonic()

    def _clear_activity(self) -> None:
        self._activity = ""
        self._activity_folder = ""

    def pause_job(self, job_id: int) -> bool:
        job = self.db.get_thumbnail_job(job_id)
        if not job or job.status not in {"queued", "running"}:
            return False
        return self.db.set_thumbnail_job_status(job_id, "paused")

    def resume_job(self, job_id: int) -> bool:
        job = self.db.get_thumbnail_job(job_id)
        if not job or job.status != "paused":
            return False
        return self.db.set_thumbnail_job_status(job_id, "queued")

    def cancel_job(self, job_id: int) -> bool:
        job = self.db.get_thumbnail_job(job_id)
        if not job or job.status in {"completed", "cancelled"}:
            return False
        return self.db.set_thumbnail_job_status(job_id, "cancelled")

    def reorder_job(self, job_id: int, direction: str) -> bool:
        return self.db.reorder_thumbnail_job(job_id, direction)

    def folder_status(self, folder: str) -> dict | None:
        job = self.db.get_active_thumbnail_job_for_folder(folder)
        return self._serialize_job(job) if job else None

    def discover_folder(self, folder: str) -> dict:
        folder_path = (self.library_root / folder).resolve()
        status = discover_missing_previews(folder_path, self.cache, self.import_extensions)
        status["job"] = self.folder_status(folder)
        return status

    def _report_storage_error(self) -> None:
        """Log the cache write failure once, not once per retry."""
        error = self.cache.write_error or "thumbnail cache is not writable"
        self.storage_error = error
        if error != self._last_storage_error:
            logger.error(
                "Thumbnail queue paused: thumbnails cannot be written. %s "
                "(retrying in %ss)",
                error, int(max(_STORAGE_BACKOFF_SECONDS, self.idle_rescan_seconds)),
            )
            self._last_storage_error = error

    def _run_loop(self):
        while not self._stop_event.is_set():
            # Everything the loop touches can raise: the volume goes read-only,
            # a folder is mid-copy, the catalog is being rebuilt. It used to be
            # that only a job raised into a handler, so a failure anywhere else
            # took the thread down with it — the queue silently stops working,
            # the app keeps serving from cache, and nothing says why.
            try:
                if not self._run_iteration():
                    self._stop_event.wait(self.idle_rescan_seconds)
            except Exception:
                logger.exception("Thumbnail queue iteration failed")
                self._stop_event.wait(self.idle_interval_seconds)

    def _run_iteration(self) -> bool:
        """One turn of the queue. True means there is more work to do right now."""
        self._iterations += 1
        if not self.cache.writable:
            self._set_activity("paused")
            self._report_storage_error()
            self._stop_event.wait(max(_STORAGE_BACKOFF_SECONDS, self.idle_rescan_seconds))
            # The backoff is also the retry. A drive that was offline, read-only,
            # or simply busy should come back on its own; a latch that only a
            # restart clears is the difference between a hiccup and a queue that
            # looks alive forever while it does nothing.
            if self.cache.recheck_writable():
                self.storage_error = ""
                self._last_storage_error = ""
                self._clear_activity()
                logger.info("Thumbnail queue resumed: the cache accepts writes again")
            return False

        job = self.db.claim_next_thumbnail_job()
        if job:
            self._process_job_guarded(job)
            return True

        if not self.auto_generate:
            self._stop_event.wait(self.idle_interval_seconds)
            return False

        enqueued = self._auto_enqueue_all()
        if enqueued > 0:
            logger.info(f"Auto-enqueued {enqueued} folders for thumbnail generation")
            # Thumbnail work still comes first, but not without limit: a library
            # that is backfilling previews is never idle, so the sweep below
            # would never run and the catalog would stay unmeasured forever.
            if time.monotonic() - self._last_measure_at >= self._measure_starvation_seconds:
                self._measure_idle_folder()
            return True

        # Nothing to generate — use the idle time to measure one folder, so a
        # library that predates measurements still gets one.
        if self._measure_idle_folder():
            return True

        # Nothing left to do at all — long idle sleep.
        return False

    def _my_turn(self, name: str = "Thumbnails") -> bool:
        if not self._gated:
            return not self._stop_event.is_set()
        return gate.wait_turn(name, self._stop_event.is_set)

    def _process_job_guarded(self, job: ThumbnailJobRecord) -> None:
        self._gated = job.priority > self.low_priority
        self._set_activity("thumbnails", job.folder)
        try:
            self._process_job(job.id)
            self._completed_jobs += 1
        except CacheNotWritable as exc:
            if exc.transient:
                # The cache was locked, not unwritable. Failing the job would
                # report a problem that has already gone away; the next turn gets
                # the lock. The short wait is what stops a persistent lock from
                # turning into a claim/fail loop that burns the CPU.
                logger.warning("Thumbnail job deferred, cache locked: %s", exc)
                self.db.set_thumbnail_job_status(job.id, "queued")
                self._stop_event.wait(1.0)
                return
            # The volume stopped accepting writes partway through. Put the
            # reason on the job — "attempt to write a readonly database" on
            # its own does not tell anyone which drive to go and fix.
            self.storage_error = str(exc)
            logger.error("Thumbnail generation stopped: %s", exc)
            self.db.set_thumbnail_job_status(job.id, "failed", str(exc))
        except Exception as exc:
            logger.exception("Thumbnail queue job failed")
            self.db.set_thumbnail_job_status(job.id, "failed", str(exc))
        finally:
            self._clear_activity()

    def _auto_enqueue_all(self) -> int:
        """Auto-enqueue all indexed folders that have missing thumbnails.

        Uses lower priority (200) than user-triggered jobs (100) so manual
        requests always preempt auto-generated ones.
        """
        auto_priority = self.low_priority + 100  # 200 by default
        try:
            all_folders = self.db.get_all_folders()
        except Exception:
            return 0

        enqueued = 0
        for folder_info in all_folders:
            folder = folder_info["folder"]
            # Skip if there's already an active job for this folder
            existing = self.db.get_active_thumbnail_job_for_folder(folder)
            if existing:
                continue

            folder_path = self.library_root / folder
            if not folder_path.exists() or not folder_path.is_dir():
                continue

            # Quick check: are there missing thumbnails?
            status = discover_missing_previews(folder_path, self.cache, self.import_extensions)
            if status["missing_any"] > 0:
                self.db.create_thumbnail_job(folder, priority=auto_priority)
                enqueued += 1

        return enqueued

    def _process_job(self, job_id: int):
        job = self.db.get_thumbnail_job(job_id)
        if not job:
            return

        folder_path = self.library_root / job.folder
        if not folder_path.exists() or not folder_path.is_dir():
            self.db.set_thumbnail_job_status(job_id, "failed", "Folder not found")
            return

        work_items = self._build_work_items(folder_path)
        total_items = job.completed_items + job.failed_items + len(work_items)
        self.db.update_thumbnail_job_progress(job_id, total_items=total_items, last_file="")

        if not work_items:
            self.db.set_thumbnail_job_status(job_id, "completed")
            self._analyze_folder(job.folder)
            return

        if self._turbo:
            self._process_items_turbo(job_id, work_items)
        else:
            self._process_items_normal(job_id, work_items)

        final = self.db.get_thumbnail_job(job_id)
        if final and final.status == "running":
            self.db.set_thumbnail_job_status(job_id, "completed")
            self._analyze_folder(job.folder)

    def _analyze_folder(self, folder: str) -> dict | None:
        """Measure a folder once its thumbnails are done.

        Runs on the queue thread, right after the thumbnails it reads have just
        been written, so the pass is a few hundred small JPEG decodes and never
        competes with the UI. Statistics are a second-order concern: if anything
        here fails the thumbnail job has still succeeded, so the failure is
        logged and dropped rather than retried or reported as a failed job.
        """
        if not self.auto_analyze or not self.db.writable:
            return None
        if not self._my_turn("Measuring"):
            return None
        try:
            # Deferred: analysis.service imports the thumbnails package, which
            # imports this module, so a top-level import would be circular.
            from analysis.service import analyze_folder

            result = analyze_folder(
                self.db,
                self.cache,
                str(self.library_root),
                folder,
                should_stop=self._stop_event.is_set,
            )
            if result.get("analyzed") or result.get("failed"):
                logger.info(
                    f"Measured {folder}: {result.get('analyzed', 0)} analyzed, "
                    f"{result.get('skipped', 0)} unchanged, "
                    f"{result.get('failed', 0)} failed"
                )
            elif result.get("error"):
                # Without this the sweep walks an unreachable or renamed library
                # folder by folder, measuring nothing, and the log stays clean.
                logger.warning(f"Cannot measure {folder}: {result['error']}")
            return result
        except Exception as e:
            logger.warning(f"Measurement pass failed for {folder}: {e}")
            return None

    def _measure_idle_folder(self) -> bool:
        """Measure the next folder that needs it, while the queue is idle.

        A folder whose thumbnails are already warm never produces a job, so
        without this sweep a library that predates measurements would stay
        unmeasured until someone asked for it folder by folder.
        """
        if not self.auto_analyze or not self.db.writable:
            return False
        try:
            folder = self.db.next_unmeasured_folder(self._measure_cursor)
        except Exception as e:
            logger.warning(f"Could not look for unmeasured folders: {e}")
            return False
        if not folder:
            self._measure_cursor = ""
            return False
        # Someone is using the app: not now, but without blocking this thread,
        # so a folder they queue themselves is still picked up at once.
        if gate.must_wait():
            self._stop_event.wait(1.0)
            return True
        self._gated = True
        # The cursor moves before the pass, not after: a folder that cannot be
        # measured at all must not stop the ones behind it.
        self._measure_cursor = folder
        self._last_measure_at = time.monotonic()
        self._set_activity("measuring", folder)
        try:
            self._analyze_folder(folder)
        finally:
            self._clear_activity()
            self._measure_attempts += 1
        # True whenever a folder was attempted, not only when the pass measured
        # something. A folder that could not be measured is still progress: the
        # next one is already queued up, and sleeping a minute per failure would
        # turn a backfill of a few hundred folders into a few hours of waiting.
        return True

    def _process_items_normal(self, job_id: int, work_items: list):
        """Single-threaded processing with idle sleep between items."""
        for source, size in work_items:
            if not self._my_turn():
                return
            current = self.db.get_thumbnail_job(job_id)
            if not current or current.status in {"paused", "cancelled", "failed"}:
                return

            result = get_thumbnail(source, self.cache, size)
            if result:
                self.db.update_thumbnail_job_progress(
                    job_id, completed_delta=1,
                    last_file=f"{source.name}:{size.name.lower()}",
                )
            else:
                self.db.update_thumbnail_job_progress(
                    job_id, failed_delta=1,
                    last_file=f"{source.name}:{size.name.lower()}",
                )

            if self._stop_event.wait(self.idle_interval_seconds):
                return

    def _process_items_turbo(self, job_id: int, work_items: list):
        """Multi-threaded processing using ~80% of CPU cores, no sleep."""
        workers = _turbo_worker_count()
        batch_size = workers * 4  # process in batches to allow cancellation checks

        for i in range(0, len(work_items), batch_size):
            if not self._my_turn():
                return
            current = self.db.get_thumbnail_job(job_id)
            if not current or current.status in {"paused", "cancelled", "failed"}:
                return

            batch = work_items[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(get_thumbnail, source, self.cache, size): (source, size)
                    for source, size in batch
                }
                for future in as_completed(futures):
                    source, size = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.db.update_thumbnail_job_progress(
                                job_id, completed_delta=1,
                                last_file=f"{source.name}:{size.name.lower()}",
                            )
                        else:
                            self.db.update_thumbnail_job_progress(
                                job_id, failed_delta=1,
                                last_file=f"{source.name}:{size.name.lower()}",
                            )
                    except Exception:
                        self.db.update_thumbnail_job_progress(
                            job_id, failed_delta=1,
                            last_file=f"{source.name}:{size.name.lower()}",
                        )

    def _build_work_items(self, folder: Path) -> list[tuple[Path, ThumbnailSize]]:
        ext_set = {ext.lower() for ext in self.import_extensions}
        work_items: list[tuple[Path, ThumbnailSize]] = []
        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in ext_set:
                continue

            for size in (ThumbnailSize.GRID, ThumbnailSize.PREVIEW):
                if not self.cache.has(file_path, size):
                    work_items.append((file_path, size))
        return work_items

    def _serialize_job(self, job: ThumbnailJobRecord | None) -> dict | None:
        if not job:
            return None
        payload = asdict(job)
        payload["pending_items"] = max(
            payload["total_items"] - payload["completed_items"] - payload["failed_items"],
            0,
        )
        return payload