"""
Tests for photo measurements: the statistics themselves, the table that caches
them, and the folder pass that fills it.

The measurements are the foundation for culling hints and for matching a set of
frames at export, so the tests care about two things: that the numbers order
photos the way they should (a bright frame measures brighter), and that a stored
row is only trusted while it still matches the file it came from.
"""

import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
from PIL import Image, ImageFilter

from analysis.image_stats import (
    STATS_VERSION,
    compute_image_stats,
    hamming_distance,
    stats_are_current,
    stats_from_jpeg,
)
from analysis.service import analyze_folder
from catalog.database import CatalogDB, PhotoRecord
from thumbnails import ThumbnailCache
from thumbnails.queue import ThumbnailQueueManager


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, color)


def _noisy(color: tuple[int, int, int], size=(64, 64)) -> Image.Image:
    """A solid colour plus per-pixel jitter, so sharpness has something to see."""
    base = np.asarray(_solid(color, size), dtype=np.int16)
    rng = np.random.default_rng(7)
    jitter = rng.integers(-40, 40, size=(size[1], size[0], 3), dtype=np.int16)
    return Image.fromarray(np.clip(base + jitter, 0, 255).astype(np.uint8))


def _blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=3))


# ── The measurements ─────────────────────────────────────────────────

class TestImageStats:
    def test_keys_match_the_stats_table(self):
        """The measurement code and the table cannot drift apart: every key the
        measurements produce has a column to land in."""
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            with db._connect() as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(photo_stats)")
                }
            stats = compute_image_stats(_solid((128, 128, 128)))
            assert set(stats) - columns == set()

    def test_bright_frame_measures_brighter(self):
        dark = compute_image_stats(_solid((20, 20, 20)))
        bright = compute_image_stats(_solid((230, 230, 230)))
        assert bright["mean_lum"] > dark["mean_lum"]
        assert bright["lum_p50"] > dark["lum_p50"]
        assert bright["lum_p95"] > dark["lum_p95"]

    def test_clipping_is_a_fraction(self):
        blown = compute_image_stats(_solid((255, 255, 255)))
        assert blown["clip_high"] == 1.0
        assert blown["clip_low"] == 0.0
        normal = compute_image_stats(_solid((128, 128, 128)))
        assert normal["clip_high"] == 0.0

    def test_grey_world_reports_the_colour_cast(self):
        # A scene the camera is rendering warm has a red channel that dominates
        # green, so wb_r rises above 1 and wb_b falls below it.
        warm = compute_image_stats(_solid((200, 150, 100)))
        assert warm["wb_r"] > 1.0
        assert warm["wb_b"] < 1.0
        assert warm["wb_g"] == 1.0
        neutral = compute_image_stats(_solid((150, 150, 150)))
        assert abs(neutral["wb_r"] - 1.0) < 1e-6
        assert abs(neutral["wb_b"] - 1.0) < 1e-6

    def test_identical_frames_share_a_hash(self):
        a = compute_image_stats(_noisy((120, 130, 140)))
        b = compute_image_stats(_noisy((120, 130, 140)))
        assert a["dhash"] == b["dhash"]
        assert hamming_distance(a["dhash"], b["dhash"]) == 0

    def test_exposure_change_barely_moves_the_hash(self):
        """The point of a *difference* hash: two exposures of one frame group
        together, because only the relative brightness of neighbours matters."""
        frame = _noisy((120, 130, 140))
        brighter = Image.fromarray(
            np.clip(np.asarray(frame, dtype=np.int16) + 45, 0, 255).astype(np.uint8)
        )
        a = compute_image_stats(frame)
        b = compute_image_stats(brighter)
        assert hamming_distance(a["dhash"], b["dhash"]) <= 4

    def test_different_frames_do_not_share_a_hash(self):
        left = Image.new("RGB", (64, 64))
        left.paste(Image.new("RGB", (32, 64), (255, 255, 255)), (0, 0))
        right = Image.new("RGB", (64, 64))
        right.paste(Image.new("RGB", (32, 64), (255, 255, 255)), (32, 0))
        a = compute_image_stats(left)
        b = compute_image_stats(right)
        assert hamming_distance(a["dhash"], b["dhash"]) > 10

    def test_blurred_frame_scores_lower_sharpness(self):
        sharp = compute_image_stats(_noisy((40, 128, 220)))
        soft = compute_image_stats(_blur(_noisy((40, 128, 220))))
        assert sharp["sharpness"] > soft["sharpness"]

    def test_hash_is_64_bits_of_hex(self):
        stats = compute_image_stats(_noisy((10, 20, 30)))
        assert len(stats["dhash"]) == 16
        int(stats["dhash"], 16)

    def test_hamming_distance_treats_unknown_as_no_match(self):
        assert hamming_distance("", "0000000000000000") == 64
        assert hamming_distance("zz", "0000000000000000") == 64

    def test_stats_from_jpeg_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            _noisy((90, 100, 110)).save(path, format="JPEG", quality=90)
            stats = stats_from_jpeg(path.read_bytes())
            assert stats["width"] == 64
            assert stats["mean_lum"] > 0

    def test_too_small_to_measure(self):
        try:
            compute_image_stats(_solid((0, 0, 0), size=(1, 1)))
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestStatsFreshness:
    def test_matching_file_is_current(self):
        row = {"stats_version": STATS_VERSION, "source_mtime": 100, "source_size": 20}
        assert stats_are_current(row, 100, 20)

    def test_changed_file_is_stale(self):
        row = {"stats_version": STATS_VERSION, "source_mtime": 100, "source_size": 20}
        assert not stats_are_current(row, 101, 20)
        assert not stats_are_current(row, 100, 21)

    def test_missing_row_is_stale(self):
        assert not stats_are_current(None, 100, 20)

    def test_older_measurement_code_is_stale(self):
        row = {"stats_version": STATS_VERSION - 1, "source_mtime": 100, "source_size": 20}
        assert not stats_are_current(row, 100, 20)


# ── The table ────────────────────────────────────────────────────────

class TestPhotoStatsTable:
    def _db_with_photo(self, tmp: Path, filepath: str) -> CatalogDB:
        db = CatalogDB(str(tmp / "test.db"))
        folder, filename = filepath.rsplit("/", 1)
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        db.upsert_photo(PhotoRecord(
            filepath=filepath,
            filename=filename,
            folder=folder,
            extension=ext,
            file_size=1234,
        ))
        return db

    def test_upsert_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_with_photo(Path(tmp), "2024/06/15/IMG_0001.CR3")
            stats = compute_image_stats(_noisy((100, 110, 120)))
            db.upsert_photo_stats(
                "2024/06/15/IMG_0001.CR3", stats,
                source_mtime=111, source_size=222, source_kind="grid_thumbnail",
            )
            row = db.get_photo_stats("2024/06/15/IMG_0001.CR3")
            assert row["source_mtime"] == 111
            assert row["source_size"] == 222
            assert row["source_kind"] == "grid_thumbnail"
            assert row["stats_version"] == STATS_VERSION
            assert row["dhash"] == stats["dhash"]
            assert abs(row["mean_lum"] - stats["mean_lum"]) < 1e-3

    def test_upsert_replaces_the_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_with_photo(Path(tmp), "2024/06/15/IMG_0001.CR3")
            db.upsert_photo_stats("2024/06/15/IMG_0001.CR3",
                                 compute_image_stats(_solid((10, 10, 10))))
            db.upsert_photo_stats("2024/06/15/IMG_0001.CR3",
                                 compute_image_stats(_solid((200, 200, 200))))
            row = db.get_photo_stats("2024/06/15/IMG_0001.CR3")
            assert row["mean_lum"] > 150
            assert db.photo_stats_count() == 1

    def test_missing_measurements_get_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_with_photo(Path(tmp), "2024/06/15/IMG_0001.CR3")
            db.upsert_photo_stats("2024/06/15/IMG_0001.CR3", {"mean_lum": 42.0})
            row = db.get_photo_stats("2024/06/15/IMG_0001.CR3")
            assert row["mean_lum"] == 42.0
            assert row["wb_r"] == 1.0
            assert row["dhash"] == ""

    def test_folder_listing_follows_the_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_with_photo(Path(tmp), "2024/06/15/IMG_0001.CR3")
            self._db_with_photo(Path(tmp), "2024/06/16/IMG_0002.CR3")
            stats = compute_image_stats(_noisy((100, 110, 120)))
            db.upsert_photo_stats("2024/06/15/IMG_0001.CR3", stats)
            db.upsert_photo_stats("2024/06/16/IMG_0002.CR3", stats)
            # A row whose photo is gone from the catalog is not served.
            db.upsert_photo_stats("gone/IMG_9999.CR3", stats)

            listed = db.list_photo_stats_for_folder("2024/06/15")
            assert list(listed) == ["2024/06/15/IMG_0001.CR3"]

    def test_next_unmeasured_folder_walks_newest_event_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_with_photo(Path(tmp), "2023/2023-12-31/IMG_0001.CR3")
            self._db_with_photo(Path(tmp), "2024/2024-01-01/IMG_0002.CR3")
            # Folder keys are YYYY/YYYY-MM-DD, so descending name order is
            # descending event date. A backfill that starts in 2016 measures
            # nothing the user is about to cull until it has already finished.
            assert db.next_unmeasured_folder() == "2024/2024-01-01"
            assert db.next_unmeasured_folder("2024/2024-01-01") == "2023/2023-12-31"
            assert db.next_unmeasured_folder("2023/2023-12-31") is None

            db.upsert_photo_stats("2024/2024-01-01/IMG_0002.CR3",
                                 compute_image_stats(_solid((10, 10, 10))))
            assert db.next_unmeasured_folder() == "2023/2023-12-31"

    def test_status_counts_measured_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_with_photo(Path(tmp), "2024/06/15/IMG_0001.CR3")
            self._db_with_photo(Path(tmp), "2024/06/15/IMG_0002.CR3")
            assert db.photo_stats_status("2024/06/15") == {
                "folder": "2024/06/15", "total": 2, "measured": 0, "missing": 2,
            }
            db.upsert_photo_stats("2024/06/15/IMG_0001.CR3",
                                 compute_image_stats(_solid((10, 10, 10))))
            status = db.photo_stats_status("2024/06/15")
            assert status["measured"] == 1
            assert status["missing"] == 1

    def test_existing_database_gains_the_table(self):
        """The table arrives through the migration list, so a catalog written by
        an older version upgrades in place instead of losing its photos."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            db = CatalogDB(str(db_path))
            db.upsert_photo(PhotoRecord(
                filepath="2024/06/15/IMG_0001.CR3",
                filename="IMG_0001.CR3",
                folder="2024/06/15",
                extension=".cr3",
            ))
            with db._connect() as conn:
                conn.execute("DROP TABLE photo_stats")
            del db

            upgraded = CatalogDB(str(db_path))
            assert upgraded.count() == 1
            assert upgraded.photo_stats_count() == 0
            upgraded.upsert_photo_stats("2024/06/15/IMG_0001.CR3",
                                       compute_image_stats(_solid((10, 10, 10))))
            assert upgraded.photo_stats_count() == 1


# ── The folder pass ──────────────────────────────────────────────────

def _make_library(tmp: Path) -> Path:
    """A library with two indexed JPEGs of clearly different brightness."""
    root = tmp / "library"
    folder = root / "2024/06/15"
    folder.mkdir(parents=True)
    _noisy((110, 120, 130)).save(folder / "IMG_0001.JPG", format="JPEG", quality=90)
    _noisy((60, 60, 60)).save(folder / "IMG_0002.JPG", format="JPEG", quality=90)
    return root


def _index_library(root: Path) -> CatalogDB:
    db = CatalogDB(str(root / "catalog.db"))
    folder = root / "2024/06/15"
    for name in sorted(p.name for p in folder.iterdir()):
        stat = (folder / name).stat()
        db.upsert_photo(PhotoRecord(
            filepath=f"2024/06/15/{name}",
            filename=name,
            folder="2024/06/15",
            extension=".jpg",
            file_size=stat.st_size,
        ))
    return db


class TestAnalyzeFolder:
    def _library(self, tmp: Path) -> Path:
        return _make_library(tmp)

    def _index(self, root: Path) -> CatalogDB:
        return _index_library(root)

    def test_measures_every_indexed_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._library(Path(tmp))
            db = self._index(root)
            cache = ThumbnailCache(str(root / "thumbs"))

            result = analyze_folder(db, cache, str(root), "2024/06/15")
            assert result["analyzed"] == 2
            assert result["failed"] == 0
            assert db.photo_stats_status("2024/06/15")["measured"] == 2

            rows = db.list_photo_stats_for_folder("2024/06/15")
            bright = rows["2024/06/15/IMG_0001.JPG"]["mean_lum"]
            dark = rows["2024/06/15/IMG_0002.JPG"]["mean_lum"]
            assert bright > dark

    def test_second_pass_skips_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._library(Path(tmp))
            db = self._index(root)
            cache = ThumbnailCache(str(root / "thumbs"))

            analyze_folder(db, cache, str(root), "2024/06/15")
            again = analyze_folder(db, cache, str(root), "2024/06/15")
            assert again["analyzed"] == 0
            assert again["skipped"] == 2

    def test_rewritten_file_is_measured_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._library(Path(tmp))
            db = self._index(root)
            cache = ThumbnailCache(str(root / "thumbs"))
            analyze_folder(db, cache, str(root), "2024/06/15")

            target = root / "2024/06/15" / "IMG_0001.JPG"
            stat = target.stat()
            os.utime(target, (stat.st_atime, stat.st_mtime + 120))

            result = analyze_folder(db, cache, str(root), "2024/06/15")
            assert result["analyzed"] == 1
            assert result["skipped"] == 1

    def test_unreadable_file_does_not_stop_the_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._library(Path(tmp))
            db = self._index(root)
            db.upsert_photo(PhotoRecord(
                filepath="2024/06/15/IMG_0003.JPG",
                filename="IMG_0003.JPG",
                folder="2024/06/15",
                extension=".jpg",
            ))
            cache = ThumbnailCache(str(root / "thumbs"))

            result = analyze_folder(db, cache, str(root), "2024/06/15")
            assert result["analyzed"] == 2
            assert result["failed"] == 1

    def test_missing_folder_reports_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._library(Path(tmp))
            db = CatalogDB(str(root / "catalog.db"))
            cache = ThumbnailCache(str(root / "thumbs"))
            result = analyze_folder(db, cache, str(root), "2024/06/99")
            assert result["error"] == "Folder not found"

    def test_stop_between_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._library(Path(tmp))
            db = self._index(root)
            cache = ThumbnailCache(str(root / "thumbs"))
            result = analyze_folder(db, cache, str(root), "2024/06/15",
                                   should_stop=lambda: True)
            assert result["analyzed"] == 0


# ── The background sweep ─────────────────────────────────────────────

class TestIdleMeasurement:
    """Folders whose thumbnails are already warm never produce a job, so the
    queue measures them in the gaps between jobs instead of never."""

    def _manager(self, root: Path, **kwargs) -> ThumbnailQueueManager:
        db = _index_library(root)
        cache = ThumbnailCache(str(root / "thumbs"))
        return ThumbnailQueueManager(db, cache, str(root), [".jpg"], **kwargs)

    def test_measures_a_folder_with_no_thumbnail_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_library(Path(tmp))
            manager = self._manager(root)

            assert manager._measure_idle_folder() is True
            assert manager.db.photo_stats_status("2024/06/15")["measured"] == 2
            # Nothing left to measure: the sweep stops instead of spinning.
            assert manager._measure_idle_folder() is False

    def test_sweep_walks_forward_through_the_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_library(Path(tmp))
            (root / "2024/06/16").mkdir()
            _noisy((80, 80, 80)).save(root / "2024/06/16" / "IMG_0003.JPG",
                                      format="JPEG", quality=90)
            db = _index_library(root)
            db.upsert_photo(PhotoRecord(
                filepath="2024/06/16/IMG_0003.JPG",
                filename="IMG_0003.JPG",
                folder="2024/06/16",
                extension=".jpg",
            ))
            cache = ThumbnailCache(str(root / "thumbs"))
            manager = ThumbnailQueueManager(db, cache, str(root), [".jpg"])

            assert manager._measure_idle_folder() is True
            assert manager._measure_idle_folder() is True
            assert db.photo_stats_status("2024/06/15")["measured"] == 2
            assert db.photo_stats_status("2024/06/16")["measured"] == 1
            assert manager._measure_idle_folder() is False

    def test_an_unmeasurable_folder_does_not_block_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_library(Path(tmp))
            db = _index_library(root)
            # Indexed, but the file is not on disk, and it is the newest folder,
            # so the newest-first sweep reaches it before any real one.
            db.upsert_photo(PhotoRecord(
                filepath="2025/2025-06-01/IMG_0000.JPG",
                filename="IMG_0000.JPG",
                folder="2025/2025-06-01",
                extension=".jpg",
            ))
            cache = ThumbnailCache(str(root / "thumbs"))
            manager = ThumbnailQueueManager(db, cache, str(root), [".jpg"])

            # Attempting the bad folder counts as progress, not as a stall: the
            # cursor moved past it, so the next turn measures the real folder
            # instead of sleeping out the idle interval on every failure.
            assert manager._measure_idle_folder() is True    # 2025-06-01, nothing to read
            assert manager._measure_cursor == "2025/2025-06-01"
            assert manager._measure_idle_folder() is True    # 2024-06-15, carried on
            assert db.photo_stats_status("2024/06/15")["measured"] == 2

    def test_can_be_turned_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_library(Path(tmp))
            manager = self._manager(root, auto_analyze=False)
            assert manager._measure_idle_folder() is False
            assert manager.db.photo_stats_count() == 0
