"""
Tests for the thumbnail extraction module.

These tests use synthetic JPEG data to test the resize and cache logic
without needing real CR3 files or exiftool installed.
"""

import io
import json
import os
import sqlite3
import stat
import tempfile
import time
import types
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from PIL import Image
from catalog.database import CatalogDB
from exif_batches import EXIFTOOL_ARG_CHARS
from thumbnails import extract as extract_module
from thumbnails.extract import (
    ThumbnailCache,
    ThumbnailSize,
    _resize_jpeg,
    get_thumbnail,
)
from thumbnails.queue import ThumbnailQueueManager
from thumbnails import store as store_module
from thumbnails.store import CacheNotWritable, ThumbnailStore
from storage_health import open_readonly, probe_writable


def _make_test_jpeg(width: int = 6000, height: int = 4000) -> bytes:
    """Create a synthetic JPEG image at the given dimensions."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_test_jpeg_file(folder: Path, name: str, width: int = 6000, height: int = 4000) -> Path:
    """Write a test JPEG to disk."""
    path = folder / name
    path.write_bytes(_make_test_jpeg(width, height))
    return path


class TestResize:
    def test_downsizes_to_max_edge(self):
        jpeg = _make_test_jpeg(6000, 4000)
        resized = _resize_jpeg(jpeg, 400, quality=80)

        img = Image.open(io.BytesIO(resized))
        assert max(img.size) == 400
        assert img.size == (400, 266) or img.size == (400, 267)

    def test_preserves_aspect_ratio_portrait(self):
        jpeg = _make_test_jpeg(4000, 6000)
        resized = _resize_jpeg(jpeg, 400, quality=80)

        img = Image.open(io.BytesIO(resized))
        assert max(img.size) == 400
        # Should be portrait
        assert img.size[1] > img.size[0]

    def test_no_upscale_if_smaller(self):
        jpeg = _make_test_jpeg(200, 150)
        resized = _resize_jpeg(jpeg, 400, quality=80)

        # Should return original data unchanged
        assert resized == jpeg

    def test_output_is_smaller(self):
        jpeg = _make_test_jpeg(6000, 4000)
        resized = _resize_jpeg(jpeg, 400, quality=80)
        assert len(resized) < len(jpeg)


class TestCache:
    def test_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))

            # Create a source file
            src = _make_test_jpeg_file(tmp, "test.jpg", 800, 600)

            # Cache miss
            assert cache.get(src, ThumbnailSize.GRID) is None

            # Put something in cache
            resized = _resize_jpeg(src.read_bytes(), 400, 80)
            cache.put(src, ThumbnailSize.GRID, resized)

            # Cache hit
            hit = cache.get(src, ThumbnailSize.GRID)
            assert hit is not None
            assert isinstance(hit, bytes)

    def test_different_sizes_cached_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))
            src = _make_test_jpeg_file(tmp, "test.jpg")

            grid_data = _resize_jpeg(src.read_bytes(), 400, 80)
            preview_data = _resize_jpeg(src.read_bytes(), 2000, 90)

            cache.put(src, ThumbnailSize.GRID, grid_data)
            cache.put(src, ThumbnailSize.PREVIEW, preview_data)

            grid_hit = cache.get(src, ThumbnailSize.GRID)
            preview_hit = cache.get(src, ThumbnailSize.PREVIEW)
            assert grid_hit is not None
            assert preview_hit is not None
            assert grid_hit != preview_hit  # different data

    def test_cache_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))
            src = _make_test_jpeg_file(tmp, "test.jpg", 800, 600)

            cache.put(src, ThumbnailSize.GRID, b"fake thumb data")
            stats = cache.stats()
            assert stats["files"] == 1

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))
            src = _make_test_jpeg_file(tmp, "test.jpg", 800, 600)

            cache.put(src, ThumbnailSize.GRID, b"data")
            cache.clear()
            assert cache.get(src, ThumbnailSize.GRID) is None
            assert cache.stats()["files"] == 0


class TestGetThumbnail:
    def test_generates_thumbnail_for_jpeg(self):
        """get_thumbnail should work end-to-end for JPEG source files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))

            # Create a source JPEG
            src = _make_test_jpeg_file(tmp, "IMG_0001.JPG", 6000, 4000)

            # Generate grid thumbnail
            result = get_thumbnail(src, cache, ThumbnailSize.GRID)
            assert result is not None
            assert isinstance(result, bytes)

            # Verify it's a valid JPEG at the right size
            img = Image.open(io.BytesIO(result))
            assert img.format == "JPEG"
            assert max(img.size) == 400

    def test_caches_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))
            src = _make_test_jpeg_file(tmp, "IMG_0001.JPG", 4000, 3000)

            # First call generates
            data1 = get_thumbnail(src, cache, ThumbnailSize.GRID)
            # Second call hits cache
            data2 = get_thumbnail(src, cache, ThumbnailSize.GRID)

            assert data1 == data2

    def test_returns_none_for_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = ThumbnailCache(str(tmp / "cache"))

            # Create a non-image file
            src = tmp / "readme.txt"
            src.write_text("not an image")

            result = get_thumbnail(src, cache, ThumbnailSize.GRID)
            assert result is None


# ── exiftool command-line batching ───────────────────────────────────
#
# exiftool receives its file list as command-line arguments, so a folder of a
# few hundred RAWs (one 5D Mark II card's DCIM directory) is longer than
# Windows allows and the process never starts. The failure is quiet — the
# caller's catch turns it into "no metadata found" — so these tests pin both
# that the calls are chunked and that a chunk that still dies is not read as
# "orientation 1".


class TestExiftoolBatching:
    @staticmethod
    def _file_args(cmd: list[str]) -> list[str]:
        """The file arguments, ignoring the executable and the switches."""
        return [a for a in cmd[1:] if not a.startswith("-")]

    def test_orientations_are_chunked_and_merged(self, monkeypatch):
        monkeypatch.setattr(extract_module, "_check_exiftool", lambda: True)
        sources = [Path(f"E:/DCIM/100EOS5D/IMG_{i:04d}.CR2") for i in range(250)]
        wanted = {str(p): (i % 8) + 1 for i, p in enumerate(sources)}
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            files = self._file_args(cmd)
            payload = [{"Orientation": wanted[f]} for f in files]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(payload))

        monkeypatch.setattr(extract_module.subprocess, "run", fake_run)

        out = extract_module._batch_get_orientations(sources)

        assert len(calls) >= 2, "250 files must not go into a single command line"
        assert len(out) == len(sources)
        assert all(out[str(p)] == wanted[str(p)] for p in sources)

    def test_no_command_line_exceeds_the_argument_cap(self, monkeypatch):
        monkeypatch.setattr(extract_module, "_check_exiftool", lambda: True)
        # Deep DCIM paths with long leaf names: the shape that actually broke.
        sources = [
            Path("E:/DCIM/100EOS5D/" + "x" * 120 + f"_{i:04d}.CR2")
            for i in range(120)
        ]
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            files = self._file_args(cmd)
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Orientation": 1} for _ in files]),
            )

        monkeypatch.setattr(extract_module.subprocess, "run", fake_run)

        out = extract_module._batch_get_orientations(sources)

        assert calls
        assert len(out) == len(sources)
        for cmd in calls:
            arg_chars = sum(len(f) + 1 for f in self._file_args(cmd))
            assert arg_chars <= EXIFTOOL_ARG_CHARS

    def test_exif_batches_are_chunked_and_merged(self, monkeypatch):
        monkeypatch.setattr(extract_module, "_check_exiftool", lambda: True)
        sources = [Path(f"E:/DCIM/100EOS5D/IMG_{i:04d}.CR2") for i in range(250)]
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            files = self._file_args(cmd)
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Model": "Canon EOS 5D Mark II"} for _ in files]),
            )

        monkeypatch.setattr(extract_module.subprocess, "run", fake_run)

        out = extract_module._batch_extract_exif(sources, ["Model"])

        assert len(calls) >= 2
        assert len(out) == len(sources)
        assert out[str(sources[-1])]["Model"] == "Canon EOS 5D Mark II"

    def test_failed_batch_is_absent_rather_than_defaulted(self, monkeypatch):
        """
        A chunk that dies must leave its files *missing* from the batch result.
        Defaulting them to 1 would write a sideways thumbnail that still looks
        like a successful thumbnail.
        """
        monkeypatch.setattr(extract_module, "_check_exiftool", lambda: True)
        sources = [Path(f"E:/DCIM/100EOS5D/IMG_{i:04d}.CR2") for i in range(250)]
        calls = []

        def fake_run(cmd, **kwargs):
            files = self._file_args(cmd)
            calls.append(files)
            if len(calls) == 1:
                raise OSError(206, "The filename or extension is too long")
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Orientation": 6} for _ in files]),
            )

        monkeypatch.setattr(extract_module.subprocess, "run", fake_run)

        batch = extract_module._batch_get_orientations(sources)

        failed, survived = calls[0], calls[1:]
        assert len(batch) == sum(len(c) for c in survived)
        assert not any(f in batch for f in failed)

    def test_missing_batch_entry_falls_back_to_a_per_file_read(self, monkeypatch):
        monkeypatch.setattr(extract_module, "_check_exiftool", lambda: True)
        sources = [Path(f"E:/DCIM/100EOS5D/IMG_{i:04d}.CR2") for i in range(250)]
        calls = []

        def fake_run(cmd, **kwargs):
            files = self._file_args(cmd)
            calls.append(files)
            if len(calls) == 1:
                raise OSError(206, "The filename or extension is too long")
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Orientation": 6} for _ in files]),
            )

        monkeypatch.setattr(extract_module.subprocess, "run", fake_run)

        batch = extract_module._batch_get_orientations(sources)

        per_file = []

        def fake_single(path):
            per_file.append(path)
            return 6

        monkeypatch.setattr(extract_module, "_get_raw_orientation", fake_single)

        # Answered by the batch: no second process.
        assert extract_module._orientation_for(sources[-1], batch) == 6
        assert per_file == []

        # Missed by the failed chunk: read this one file instead of guessing 1.
        assert extract_module._orientation_for(sources[0], batch) == 6
        assert per_file == [sources[0]]

    def test_generate_batch_reads_orientation_when_the_batch_missed_it(
        self, monkeypatch, tmp_path
    ):
        """
        Wiring check on the folder path: a file the batch did not answer for
        still gets its orientation read. Keeping the old
        ``orientations.get(key, 1)`` in the loop would pass every other test
        here and only show up as sideways thumbnails in the grid.
        """
        cache = ThumbnailCache(str(tmp_path / "cache"))
        raw = tmp_path / "IMG_0001.CR2"
        raw.write_bytes(b"not a real raw")

        monkeypatch.setattr(
            extract_module, "_batch_extract_embedded_jpegs", lambda sources: {}
        )
        monkeypatch.setattr(
            extract_module, "_batch_get_orientations", lambda sources: {}
        )
        # A landscape embedded JPEG plus orientation 6 must come out portrait.
        monkeypatch.setattr(
            extract_module, "_extract_source_jpeg", lambda source: _make_test_jpeg(6000, 4000)
        )
        asked = []
        monkeypatch.setattr(
            extract_module,
            "_get_raw_orientation",
            lambda source: asked.append(source) or 6,
        )

        stats = extract_module.generate_thumbnails_batch(
            [raw], cache, ThumbnailSize.GRID
        )

        assert stats["generated"] == 1
        assert asked == [raw]
        img = Image.open(io.BytesIO(cache.get(raw, ThumbnailSize.GRID)))
        assert (img.width, img.height) == (266, 400)

    def test_embedded_jpeg_batches_do_not_leak_across_chunks(self, monkeypatch):
        """
        -W writes to <stem>.jpg, so the same filename in two folders shares one
        output path. A later chunk whose file has no embedded JPEG must not
        pick up the earlier chunk's bytes as its own result.
        """
        monkeypatch.setattr(extract_module, "_check_exiftool", lambda: True)
        first = Path("E:/DCIM/100EOS5D/IMG_0001.CR2")
        second = Path("E:/DCIM/101EOS5D/IMG_0001.CR2")

        def fake_run(cmd, **kwargs):
            pattern = Path(cmd[4])
            files = [Path(a) for a in cmd[5:]]
            for f in files:
                if f == first:
                    (pattern.parent / f"{f.stem}.jpg").write_bytes(
                        b"\xff\xd8" + b"x" * 2000
                    )
                # `second` has no embedded JPEG: exiftool writes nothing.
            return types.SimpleNamespace(returncode=0, stdout=b"")

        monkeypatch.setattr(extract_module.subprocess, "run", fake_run)
        # Force one file per chunk; the chunking policy itself is covered above.
        monkeypatch.setattr(
            extract_module, "_batches", lambda paths, **kw: iter([[first], [second]])
        )

        out = extract_module._batch_extract_embedded_jpegs([first, second])

        assert str(first) in out
        assert str(second) not in out


# ── Thumbnail job DB tests ───────────────────────────────────────────

from catalog.database import CatalogDB, ThumbnailJobRecord


def _make_db(tmp_path: Path) -> CatalogDB:
    return CatalogDB(str(tmp_path / "test.db"))


class TestThumbnailJobCRUD:
    def test_create_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            job = db.create_thumbnail_job("2025/03-07", priority=50)
            assert job.id is not None
            assert job.folder == "2025/03-07"
            assert job.status == "queued"
            assert job.priority == 50

            fetched = db.get_thumbnail_job(job.id)
            assert fetched is not None
            assert fetched.id == job.id
            assert fetched.folder == job.folder

    def test_list_jobs_ordered_by_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            db.create_thumbnail_job("low", priority=100)
            db.create_thumbnail_job("high", priority=10)
            db.create_thumbnail_job("mid", priority=50)

            jobs = db.list_thumbnail_jobs()
            assert [j.folder for j in jobs] == ["high", "mid", "low"]

    def test_get_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            assert db.get_thumbnail_job(999) is None


class TestThumbnailJobStatus:
    def test_set_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            job = db.create_thumbnail_job("folder")
            assert db.set_thumbnail_job_status(job.id, "paused")
            assert db.get_thumbnail_job(job.id).status == "paused"

    def test_set_status_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            job = db.create_thumbnail_job("folder")
            db.set_thumbnail_job_status(job.id, "failed", "disk full")
            fetched = db.get_thumbnail_job(job.id)
            assert fetched.status == "failed"
            assert fetched.error_message == "disk full"

    def test_set_status_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            assert not db.set_thumbnail_job_status(999, "paused")


class TestThumbnailJobClaim:
    def test_claim_respects_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            db.create_thumbnail_job("low", priority=100)
            db.create_thumbnail_job("high", priority=10)

            claimed = db.claim_next_thumbnail_job()
            assert claimed.folder == "high"
            assert claimed.status == "running"
            assert claimed.started_at != ""

    def test_claim_skips_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            db.create_thumbnail_job("first", priority=10)
            db.create_thumbnail_job("second", priority=20)

            db.claim_next_thumbnail_job()  # claims "first"
            second = db.claim_next_thumbnail_job()
            assert second.folder == "second"

    def test_claim_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            assert db.claim_next_thumbnail_job() is None


class TestThumbnailJobReset:
    def test_reset_running_to_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            db.create_thumbnail_job("a")
            db.create_thumbnail_job("b")
            db.claim_next_thumbnail_job()  # "a" → running
            db.claim_next_thumbnail_job()  # "b" → running

            count = db.reset_running_thumbnail_jobs()
            assert count == 2

            jobs = db.list_thumbnail_jobs()
            assert all(j.status == "queued" for j in jobs)


class TestThumbnailJobProgress:
    def test_update_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            job = db.create_thumbnail_job("folder")
            db.update_thumbnail_job_progress(job.id, total_items=10, last_file="img.cr3:grid")
            db.update_thumbnail_job_progress(job.id, completed_delta=3, last_file="img3.cr3:grid")
            db.update_thumbnail_job_progress(job.id, failed_delta=1, last_file="bad.cr3:grid")

            updated = db.get_thumbnail_job(job.id)
            assert updated.total_items == 10
            assert updated.completed_items == 3
            assert updated.failed_items == 1
            assert updated.last_file == "bad.cr3:grid"


class TestThumbnailJobFolder:
    def test_get_active_job_for_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            db.create_thumbnail_job("2025/03-07")
            active = db.get_active_thumbnail_job_for_folder("2025/03-07")
            assert active is not None
            assert active.folder == "2025/03-07"

    def test_no_active_job_after_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            job = db.create_thumbnail_job("folder")
            db.set_thumbnail_job_status(job.id, "completed")
            assert db.get_active_thumbnail_job_for_folder("folder") is None

    def test_reorder_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(Path(tmp))
            j1 = db.create_thumbnail_job("a", priority=10)
            j2 = db.create_thumbnail_job("b", priority=20)

            # Move j2 up (swap with j1)
            assert db.reorder_thumbnail_job(j2.id, "up")
            jobs = db.list_thumbnail_jobs()
            assert jobs[0].folder == "b"
            assert jobs[1].folder == "a"


def _rows(db_path: str, sql: str) -> list:
    """Read from the cache DB directly. Connections must close: Windows locks them."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _row_count(db_path: str) -> int:
    return _rows(db_path, "SELECT COUNT(*) FROM thumbnails")[0][0]


class TestWriteProbe:
    """A read-only volume must be reported, not worked around silently."""

    def test_probe_fails_when_the_parent_cannot_be_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("x")
            writable, error = probe_writable(blocker / "nested" / "thumbnails.db")
            assert writable is False
            assert "not-a-directory" in error

    def test_probe_accepts_a_new_file_in_a_writable_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            writable, error = probe_writable(Path(tmp) / "catalog.db")
            assert writable is True
            assert error == ""

    def test_probe_creates_a_missing_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / ".photomanager" / "thumbnails" / "thumbnails.db"
            writable, _ = probe_writable(str(target))
            assert writable is True
            assert target.parent.is_dir()

    def test_open_readonly_reads_a_wal_database(self):
        """Both databases are WAL; a read-only open must still read them."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog.db"
            writer = sqlite3.connect(str(db_path))
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY)")
            writer.execute("INSERT INTO photos DEFAULT VALUES")
            writer.commit()
            writer.close()

            conn = open_readonly(str(db_path))
            try:
                assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 1
            finally:
                conn.close()

            with pytest.raises(sqlite3.OperationalError):
                conn = open_readonly(str(db_path))
                try:
                    conn.execute("INSERT INTO photos DEFAULT VALUES")
                finally:
                    conn.close()


class TestCacheResilience:
    """Unwritable cache, size budget, and orphan cleanup."""

    def _store(self, tmp: Path, max_bytes: int = 0) -> ThumbnailStore:
        return ThumbnailStore(str(tmp / "cache" / "thumbnails.db"), max_bytes=max_bytes)

    def test_put_records_source_path_and_last_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp)
            src = _make_test_jpeg_file(tmp, "IMG_1.JPG", 800, 600)

            store.put(src, ThumbnailSize.GRID, b"x" * 100)

            source_path, last_access = _rows(
                store.db_path, "SELECT source_path, last_access FROM thumbnails"
            )[0]
            assert source_path == os.path.normcase(os.path.abspath(str(src)))
            assert last_access > 0

    def test_old_schema_is_migrated_and_existing_rows_still_read(self):
        """A cache written before source_path existed keeps working in place."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "cache" / "thumbnails.db"
            db_path.parent.mkdir(parents=True)

            legacy = sqlite3.connect(str(db_path))
            legacy.execute(
                """CREATE TABLE thumbnails (
                       cache_key TEXT NOT NULL,
                       size TEXT NOT NULL,
                       jpeg_data BLOB NOT NULL,
                       byte_length INTEGER NOT NULL,
                       created_at TEXT DEFAULT '',
                       PRIMARY KEY (cache_key, size)
                   )"""
            )
            src = _make_test_jpeg_file(tmp, "IMG_1.JPG", 800, 600)
            legacy.execute(
                "INSERT INTO thumbnails (cache_key, size, jpeg_data, byte_length)"
                " VALUES (?, ?, ?, ?)",
                (ThumbnailStore.cache_key(src), "grid", b"legacy thumb", 12),
            )
            legacy.commit()
            legacy.close()

            store = ThumbnailStore(str(db_path))
            assert store.writable is True
            assert store.get(src, ThumbnailSize.GRID) == b"legacy thumb"
            # Rows with no source path cannot be checked by prune, but they are
            # visible and they sort first under LRU, so a budget retires them.
            assert store.stats()["unverifiable_rows"] == 1

    def test_budget_evicts_previews_before_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp, max_bytes=1000)
            a = _make_test_jpeg_file(tmp, "A.JPG", 800, 600)
            b = _make_test_jpeg_file(tmp, "B.JPG", 800, 600)

            store.put(a, ThumbnailSize.GRID, b"g" * 300)
            store.put(a, ThumbnailSize.PREVIEW, b"p" * 600)   # 900 total, under cap
            store.put(b, ThumbnailSize.PREVIEW, b"p" * 600)   # 1500 -> evict A's preview

            assert store.get(b, ThumbnailSize.PREVIEW) == b"p" * 600
            assert store.get(a, ThumbnailSize.PREVIEW) is None
            assert store.get(a, ThumbnailSize.GRID) == b"g" * 300
            assert store.stats()["files"] == 2

    def test_budget_falls_back_to_grids_when_previews_are_not_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp, max_bytes=100)
            a = _make_test_jpeg_file(tmp, "A.JPG", 800, 600)

            store.put(a, ThumbnailSize.GRID, b"g" * 300)

            assert store.get(a, ThumbnailSize.GRID) is None
            assert store.stats()["files"] == 0

    def test_prune_removes_thumbnails_whose_source_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp)
            keep = _make_test_jpeg_file(tmp, "keep.JPG", 800, 600)
            gone = _make_test_jpeg_file(tmp, "gone.JPG", 800, 600)

            store.put(keep, ThumbnailSize.GRID, b"k" * 10)
            store.put(gone, ThumbnailSize.GRID, b"g" * 10)
            store.put(gone, ThumbnailSize.PREVIEW, b"G" * 20)
            gone.unlink()

            result = store.prune()

            assert result["deleted"] == 2
            assert result["unverifiable"] == 0
            assert _row_count(store.db_path) == 1
            assert store.get(keep, ThumbnailSize.GRID) == b"k" * 10

    def test_prune_leaves_rows_without_a_source_path_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp)

            conn = sqlite3.connect(store.db_path)
            try:
                conn.execute(
                    "INSERT INTO thumbnails (cache_key, size, jpeg_data, byte_length,"
                    " source_path, last_access) VALUES (?, ?, ?, ?, '', 0)",
                    ("legacykey", "grid", b"legacy", 6),
                )
                conn.commit()
            finally:
                conn.close()

            result = store.prune()

            assert result["deleted"] == 0
            assert result["unverifiable"] == 1
            assert _row_count(store.db_path) == 1

    def test_read_only_cache_still_serves_reads_and_refuses_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp)
            src = _make_test_jpeg_file(tmp, "IMG_1.JPG", 800, 600)
            store.put(src, ThumbnailSize.GRID, b"data")

            store.writable = False
            store.write_error = "simulated read-only volume"

            assert store.get(src, ThumbnailSize.GRID) == b"data"
            with pytest.raises(CacheNotWritable):
                store.put(src, ThumbnailSize.PREVIEW, b"nope")
            with pytest.raises(CacheNotWritable):
                store.prune()

            stats = store.stats()
            assert stats["writable"] is False
            assert stats["write_error"] == "simulated read-only volume"
            assert stats["files"] == 1

    def test_store_on_an_unwritable_volume_starts_anyway(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            blocker = tmp / "blocked"
            blocker.write_text("x")
            src = _make_test_jpeg_file(tmp, "IMG_1.JPG", 800, 600)

            store = ThumbnailStore(str(blocker / "nested" / "thumbnails.db"))

            assert store.writable is False
            assert store.write_error
            assert store.get(src, ThumbnailSize.GRID) is None
            with pytest.raises(CacheNotWritable):
                store.put(src, ThumbnailSize.GRID, b"data")
            assert store.stats()["writable"] is False

    def test_legacy_cache_stays_readable_while_the_volume_is_read_only(self):
        """
        A cache from before source_path / cache_meta existed, on a volume that
        will not take writes, must still serve its thumbnails and report its size.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "cache" / "thumbnails.db"
            db_path.parent.mkdir(parents=True)
            src = _make_test_jpeg_file(tmp, "IMG_1.JPG", 800, 600)

            legacy = sqlite3.connect(str(db_path))
            legacy.execute("PRAGMA journal_mode=WAL")
            legacy.execute(
                "CREATE TABLE thumbnails (cache_key TEXT NOT NULL, size TEXT NOT NULL,"
                " jpeg_data BLOB NOT NULL, byte_length INTEGER NOT NULL,"
                " created_at TEXT DEFAULT '', PRIMARY KEY (cache_key, size))"
            )
            legacy.execute(
                "INSERT INTO thumbnails (cache_key, size, jpeg_data, byte_length)"
                " VALUES (?, ?, ?, ?)",
                (ThumbnailStore.cache_key(src), "grid", b"legacy thumb", 12),
            )
            legacy.commit()
            legacy.close()

            os.chmod(db_path, stat.S_IREAD)
            try:
                store = ThumbnailStore(str(db_path))
                assert store.writable is False

                assert store.get(src, ThumbnailSize.GRID) == b"legacy thumb"
                stats = store.stats()
                assert stats["files"] == 1
                assert stats["by_size"]["grid"]["count"] == 1
                # Nothing in the old schema says which files still exist.
                assert stats["unverifiable_rows"] == 1
            finally:
                os.chmod(db_path, stat.S_IWRITE | stat.S_IREAD)

    def test_stats_reports_the_configured_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = self._store(tmp, max_bytes=5 * 1024 * 1024)
            stats = store.stats()
            assert stats["max_bytes"] == 5 * 1024 * 1024
            assert stats["max_mb"] == 5.0

    def test_cache_forwards_the_size_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ThumbnailCache(str(Path(tmp) / "cache"), max_bytes=1234)
            assert cache.store.max_bytes == 1234
            assert cache.writable is True
            assert cache.write_error == ""


class TestQueueScheduling:
    """The queue thread has two jobs: generate thumbnails, and measure folders.

    Thumbnail work comes first, but it must not postpone measuring forever, and a
    raise anywhere in the loop must not take the thread down. Those are
    scheduling decisions, so they are tested as scheduling decisions with the
    actual work stubbed out.
    """

    def _queue(self, tmp_path: Path, **overrides) -> ThumbnailQueueManager:
        db = CatalogDB(str(tmp_path / "catalog.db"))
        cache = ThumbnailCache(str(tmp_path / "cache"))
        return ThumbnailQueueManager(
            db, cache, str(tmp_path), [".jpg", ".cr3"],
            idle_interval_seconds=0.0, idle_rescan_seconds=0.0, **overrides,
        )

    def test_measures_a_folder_when_there_is_nothing_to_generate(self, tmp_path):
        q = self._queue(tmp_path)
        q._auto_enqueue_all = lambda: 0
        measured = []
        q._measure_idle_folder = lambda: measured.append(1) or False

        assert q._run_iteration() is False
        assert measured == [1]

    def test_thumbnail_work_goes_first_inside_the_starvation_window(self, tmp_path):
        q = self._queue(tmp_path)
        q._auto_enqueue_all = lambda: 5
        measured = []
        q._measure_idle_folder = lambda: measured.append(1) or True

        # _last_measure_at is set when the queue is built, so this is the first
        # turn: previews are more urgent, and the loop returns True to keep going.
        assert q._run_iteration() is True
        assert measured == []

    def test_measurement_is_not_starved_by_endless_thumbnail_work(self, tmp_path):
        """A library backfilling previews is never idle. Without a deadline the
        sweep would never run and the catalog would stay unmeasured forever."""
        q = self._queue(tmp_path)
        q._auto_enqueue_all = lambda: 5
        q._last_measure_at = time.monotonic() - (q._measure_starvation_seconds + 1)
        measured = []
        q._measure_idle_folder = lambda: measured.append(1) or True

        assert q._run_iteration() is True
        assert measured == [1]

    def test_an_unwritable_cache_backs_off_without_touching_the_catalog(self, tmp_path):
        q = self._queue(tmp_path)
        waited = []
        q._stop_event = types.SimpleNamespace(is_set=lambda: False, wait=waited.append)
        q.cache = types.SimpleNamespace(
            writable=False, write_error="read-only volume", recheck_writable=lambda: False
        )
        claimed = []
        q.db.claim_next_thumbnail_job = lambda: claimed.append(1)

        assert q._run_iteration() is False
        assert claimed == []
        assert waited and waited[0] >= 300
        assert q.storage_error == "read-only volume"

    def test_a_cache_that_accepts_writes_again_resumes_on_its_own(self, tmp_path):
        """The backoff is also the retry. A latch only a restart clears turns a
        drive that slept for a minute into a queue that does nothing all day."""
        q = self._queue(tmp_path)
        q._stop_event = types.SimpleNamespace(is_set=lambda: False, wait=lambda s: None)
        back = types.SimpleNamespace(
            writable=False, write_error="read-only volume", recheck_writable=lambda: True
        )
        q.cache = back

        assert q._run_iteration() is False
        assert q.storage_error == ""
        assert q._activity == ""

        # And the next turn is back to normal work.
        back.writable = True
        q._auto_enqueue_all = lambda: 0
        q._measure_idle_folder = lambda: False
        assert q._run_iteration() is False
        assert q.status()["storage_error"] == ""

    def test_a_locked_cache_requeues_the_job_instead_of_failing_it(self, tmp_path):
        """A lock says the next attempt will work. Failing the job reports a
        problem that has already gone away and leaves a red row for nothing."""
        q = self._queue(tmp_path)
        job = q.db.create_thumbnail_job("2024/2024-06-15")
        q.db.claim_next_thumbnail_job()
        q._stop_event = types.SimpleNamespace(is_set=lambda: False, wait=lambda s: None)

        def refuse(job_id):
            raise CacheNotWritable("thumbnail cache is busy: database is locked", transient=True)

        q._process_job = refuse
        q._process_job_guarded(job)

        assert q.db.get_thumbnail_job(job.id).status == "queued"
        assert q.storage_error == ""

    def test_status_says_what_the_worker_is_doing(self, tmp_path):
        """An empty job list looks the same whether the worker is measuring a
        folder, paused on a write failure, or finished. The panel needs to know."""
        q = self._queue(tmp_path)
        status = q.status()
        assert status["alive"] is False
        assert status["activity"] == ""
        assert status["measure_cursor"] == ""

        q.db.next_unmeasured_folder = lambda after="": "2024/2024-06-15"
        seen = []
        q._analyze_folder = lambda folder: seen.append(q.status()["activity"]) or {}

        q._measure_idle_folder()

        # The activity is visible while the pass runs, and cleared after it.
        assert seen == ["measuring"]
        assert q.status()["activity"] == ""
        assert q.status()["measure_cursor"] == "2024/2024-06-15"
        assert q.status()["measure_attempts"] == 1

    def test_an_iteration_that_raises_does_not_end_the_thread(self, tmp_path):
        """The old loop only caught errors inside a job, so anything else raised
        stopped the queue silently while the app kept serving from cache."""
        q = self._queue(tmp_path)
        turns = []

        def iteration():
            turns.append(1)
            raise RuntimeError("the volume went away mid-scan")

        q._run_iteration = iteration
        q.start()
        try:
            deadline = time.monotonic() + 5
            while len(turns) < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            q.stop()

        assert len(turns) >= 3
        assert q._thread is None

    def test_a_folder_that_cannot_be_measured_still_moves_the_sweep_on(self, tmp_path):
        q = self._queue(tmp_path)
        q.db.next_unmeasured_folder = lambda after="": "2024/2024-06-15"
        attempted = []
        q._analyze_folder = lambda folder: attempted.append(folder) or {"error": "Folder not found"}

        assert q._measure_idle_folder() is True
        assert attempted == ["2024/2024-06-15"]
        assert q._measure_cursor == "2024/2024-06-15"

    def test_the_sweep_starts_over_once_the_library_is_measured(self, tmp_path):
        q = self._queue(tmp_path)
        q._measure_cursor = "2024/2024-06-15"
        q.db.next_unmeasured_folder = lambda after="": ""

        assert q._measure_idle_folder() is False
        assert q._measure_cursor == ""

    def test_the_sweep_stays_off_when_measurements_are_disabled(self, tmp_path):
        q = self._queue(tmp_path, auto_analyze=False)
        asked = []
        q.db.next_unmeasured_folder = lambda after="": asked.append(1) or "2024/2024-06-15"

        assert q._measure_idle_folder() is False
        assert asked == []


def _connection_that_raises(message: str):
    """Stand in for a volume that refuses the write, without a real volume."""

    def connect(*args, **kwargs):
        raise sqlite3.OperationalError(message)

    return connect


class TestRefusedWrites:
    """A locked cache and a read-only volume are different problems.

    Treating them the same is what makes a queue stop forever over something the
    next attempt would have got, and leaves the screen saying the cache is
    unwritable while a fresh probe of the same path reports that it is fine.
    """

    def _store(self, tmp_path: Path) -> ThumbnailStore:
        return ThumbnailStore(str(tmp_path / "thumbnails.db"))

    def _source(self, tmp_path: Path) -> Path:
        """The cache key is built from the file's mtime and size, so it has to exist."""
        source = tmp_path / "IMG_0001.JPG"
        source.write_bytes(b"not really a jpeg")
        return source

    def test_a_locked_write_leaves_the_cache_marked_writable(self, tmp_path):
        store = self._store(tmp_path)
        store._connect = _connection_that_raises("database is locked")

        with pytest.raises(CacheNotWritable) as caught:
            store.put(self._source(tmp_path), ThumbnailSize.GRID, b"data")

        assert caught.value.transient is True
        assert store.writable is True
        assert store.write_error == ""

    def test_a_read_only_database_still_latches_off(self, tmp_path):
        store = self._store(tmp_path)
        store._connect = _connection_that_raises("attempt to write a readonly database")

        with pytest.raises(CacheNotWritable) as caught:
            store.put(self._source(tmp_path), ThumbnailSize.GRID, b"data")

        assert caught.value.transient is False
        assert store.writable is False
        assert "readonly" in store.write_error

    def test_rechecking_clears_the_latch_when_the_volume_comes_back(self, tmp_path):
        store = self._store(tmp_path)
        source = self._source(tmp_path)
        store.writable = False
        store.write_error = "read-only volume"

        assert store.recheck_writable() is True
        assert store.writable is True
        assert store.write_error == ""

        # The point of clearing the latch is that writes work again.
        store.put(source, ThumbnailSize.GRID, b"data")
        assert store.get(source, ThumbnailSize.GRID) == b"data"

    def test_rechecking_keeps_the_latch_when_the_volume_is_still_refused(
        self, tmp_path, monkeypatch
    ):
        store = self._store(tmp_path)
        store.writable = False
        monkeypatch.setattr(
            store_module, "probe_writable", lambda path: (False, "still read-only")
        )

        assert store.recheck_writable() is False
        assert store.writable is False
        assert store.write_error == "still read-only"

    def test_the_cache_facade_passes_the_recheck_through(self, tmp_path):
        """The queue holds the facade, not the store, so this is the path that
        actually runs; a missing method would be an exception inside the thread."""
        cache = ThumbnailCache(str(tmp_path / "cache"))
        cache.store.writable = False
        assert cache.recheck_writable() is True
        assert cache.writable is True
