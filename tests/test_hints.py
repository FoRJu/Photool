"""
Tests for quality hints: the pure classification in ``analysis.hints`` and the
folder-level read in ``CatalogService.folder_hints``.

The two hints are different in kind and the tests keep them that way. Soft frame
is relative to the frame's peers (the same camera, moments apart, at a similar
focal length), so it needs them and goes quiet without; only a frame far below the
whole folder is soft on its own. Blown highlights is an absolute reading of the file, so it
applies anywhere and never depends on the neighbours.
"""

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analysis.hints import (  # noqa: E402
    ACCIDENT_RATIO,
    BLOWN_CLIP,
    MIN_FOLDER_SAMPLE,
    SOFT_RATIO,
    compute_hints,
    folder_baseline,
    median,
    peer_sharpness,
    peers,
    photo_hints,
)
from analysis.image_stats import STATS_VERSION  # noqa: E402
from catalog.database import CatalogDB  # noqa: E402
from catalog.service import CatalogService  # noqa: E402


def _row(sharpness=10.0, clip_high=0.0, clip_low=0.0, **overrides):
    row = {
        "stats_version": STATS_VERSION,
        "sharpness": sharpness,
        "clip_high": clip_high,
        "clip_low": clip_low,
    }
    row.update(overrides)
    return row


def _rows(count, sharpness=10.0, **overrides):
    return {f"f/IMG_{i:04d}.CR3": _row(sharpness=sharpness, **overrides) for i in range(count)}


# ── The median ───────────────────────────────────────────────────────

class TestMedian:
    def test_empty_has_no_median(self):
        assert median([]) is None

    def test_odd_count_takes_the_middle(self):
        assert median([5, 1, 3]) == 3

    def test_even_count_averages_the_middle(self):
        assert median([1, 2, 8, 9]) == 5

    def test_a_single_outlier_does_not_move_it(self):
        # One razor-sharp frame must not make the other twelve look soft.
        values = [10.0] * 12 + [400.0]
        assert median(values) == 10.0


# ── The folder baseline ──────────────────────────────────────────────

class TestFolderBaseline:
    def test_a_small_folder_has_no_usable_median(self):
        baseline = folder_baseline([_row() for _ in range(MIN_FOLDER_SAMPLE - 1)])
        assert baseline["measured"] == MIN_FOLDER_SAMPLE - 1
        assert baseline["median_sharpness"] is None

    def test_a_folder_at_the_minimum_gets_a_median(self):
        baseline = folder_baseline([_row(sharpness=float(i)) for i in range(1, MIN_FOLDER_SAMPLE + 1)])
        assert baseline["measured"] == MIN_FOLDER_SAMPLE
        assert baseline["median_sharpness"] == (MIN_FOLDER_SAMPLE + 1) / 2

    def test_a_frame_without_measurements_is_not_counted(self):
        rows = [_row(), _row(sharpness=None), _row(clip_high=None), {}]
        assert folder_baseline(rows)["measured"] == 1

    def test_a_zero_sharpness_does_not_drag_the_median_down(self):
        # A blank thumbnail measures zero sharpness; it is a read failure, not a
        # soft frame, and it must not redefine what this folder looks like.
        rows = [_row(sharpness=0.0) for _ in range(6)] + [
            _row(sharpness=float(i)) for i in range(1, MIN_FOLDER_SAMPLE + 1)
        ]
        assert folder_baseline(rows)["measured"] == MIN_FOLDER_SAMPLE
        assert folder_baseline(rows)["median_sharpness"] == (MIN_FOLDER_SAMPLE + 1) / 2


# ── One frame ────────────────────────────────────────────────────────

class TestPhotoHints:
    def test_a_frame_well_below_its_peers_is_soft(self):
        ratio = SOFT_RATIO - 0.05
        hints = photo_hints(_row(sharpness=10.0 * ratio), 10.0, peer_sharpness=10.0)
        assert hints["soft_frame"] is True

    def test_a_frame_at_the_ratio_is_not_soft(self):
        hints = photo_hints(_row(sharpness=10.0 * SOFT_RATIO), 10.0, peer_sharpness=10.0)
        assert hints["soft_frame"] is False

    def test_below_the_folder_but_not_its_peers_is_not_soft(self):
        # A telephoto frame against a folder of busy wide shots.
        hints = photo_hints(_row(sharpness=3.0), 10.0, peer_sharpness=3.5)
        assert hints["soft_frame"] is False

    def test_far_below_the_whole_folder_is_soft_without_peers(self):
        # Camera shake, or a frame taken by accident.
        assert photo_hints(_row(sharpness=10.0 * ACCIDENT_RATIO * 0.9), 10.0)["soft_frame"] is True
        assert photo_hints(_row(sharpness=10.0 * SOFT_RATIO * 0.9), 10.0)["soft_frame"] is False

    def test_no_baseline_means_no_relative_hint(self):
        hints = photo_hints(_row(sharpness=0.01), None)
        assert hints["soft_frame"] is False

    def test_blown_highlights_does_not_need_a_baseline(self):
        hints = photo_hints(_row(clip_high=BLOWN_CLIP + 0.01), None)
        assert hints["blown_highlights"] is True

    def test_blown_highlights_at_the_threshold_is_not_flagged(self):
        hints = photo_hints(_row(clip_high=BLOWN_CLIP), 10.0)
        assert hints["blown_highlights"] is False

    def test_an_unmeasured_frame_gets_no_hints(self):
        for row in ({}, {"sharpness": None, "clip_high": None}, None):
            hints = photo_hints(row, 10.0)
            assert hints == {"soft_frame": False, "blown_highlights": False}

    def test_both_hints_can_be_true_at_once(self):
        hints = photo_hints(_row(sharpness=0.5, clip_high=0.5), 10.0)
        assert hints == {"soft_frame": True, "blown_highlights": True}

    def test_a_zero_sharpness_frame_is_soft_when_the_folder_is_sharp(self):
        assert photo_hints(_row(sharpness=0.0), 10.0)["soft_frame"] is True


class TestPeers:
    @staticmethod
    def _record(path, second, focal="105 mm", camera="Canon EOS R6m2"):
        from types import SimpleNamespace

        return SimpleNamespace(filepath=path, capture_date=f"2026:10:03 10:29:{second:02d}",
                               focal_length=focal, camera=camera)

    def test_peers_are_the_same_camera_moments_apart_at_a_like_focal_length(self):
        found = peers([
            self._record("a", 0), self._record("b", 5), self._record("c", 10, focal="30 mm"),
            self._record("d", 12, camera="RICOH GR III"), self._record("e", 40),
        ])
        assert sorted(found["a"]) == ["b"]          # e is 40 s on; c is wide; d another camera
        assert found["c"] == [] and found["d"] == []
        assert sorted(found["e"]) == []

    def test_one_second_with_and_without_a_focal_length_does_not_fail(self):
        # Sorting the frames used to compare 105.0 with None and raise, so the
        # folder's hints answered 500.
        found = peers([self._record("a", 0, focal=""), self._record("b", 0), self._record("c", 1)])
        assert found["a"] == []
        assert sorted(found["b"]) == ["c"] and sorted(found["c"]) == ["b"]

    def test_a_frame_with_too_few_peers_has_no_peer_median(self):
        rows = {p: _row(sharpness=v) for p, v in (("a", 10.0), ("b", 9.0), ("c", 2.0))}
        assert peer_sharpness(rows, {"a": ["b", "c"], "b": ["a"], "c": ["a", "b"]}) == {
            "a": 9.0, "c": 9.0}


# ── A whole folder ───────────────────────────────────────────────────

class TestComputeHints:
    def test_only_the_soft_one_is_flagged(self):
        rows = _rows(MIN_FOLDER_SAMPLE)
        rows["f/IMG_0005.CR3"] = _row(sharpness=0.5)
        result = compute_hints(rows)
        assert result["median_sharpness"] == 10.0
        assert result["hints"]["f/IMG_0005.CR3"]["soft_frame"] is True
        assert sum(1 for h in result["hints"].values() if h["soft_frame"]) == 1

    def test_an_unmeasured_frame_is_absent_not_clean(self):
        rows = _rows(MIN_FOLDER_SAMPLE)
        rows["f/IMG_9999.CR3"] = {"stats_version": STATS_VERSION, "sharpness": None, "clip_high": None}
        result = compute_hints(rows)
        assert "f/IMG_9999.CR3" not in result["hints"]
        assert len(result["hints"]) == MIN_FOLDER_SAMPLE

    def test_a_small_folder_still_gets_absolute_hints(self):
        rows = _rows(3, clip_high=0.5)
        result = compute_hints(rows)
        assert result["median_sharpness"] is None
        assert all(h["blown_highlights"] for h in result["hints"].values())
        assert not any(h["soft_frame"] for h in result["hints"].values())

    def test_an_empty_folder_reports_nothing(self):
        assert compute_hints({}) == {"measured": 0, "median_sharpness": None, "hints": {}}


# ── Catalog service ──────────────────────────────────────────────────

_NAMES = tuple(f"IMG_{i:04d}" for i in range(1, MIN_FOLDER_SAMPLE + 3))


class TestServiceHints:
    def _setup(self, tmp: Path):
        lib = tmp / "library"
        folder = lib / "2024-06-15"
        folder.mkdir(parents=True)
        for name in _NAMES:
            (folder / f"{name}.CR3").write_bytes(b"fake raw " * 100)
            (folder / f"{name}.JPG").write_bytes(b"fake jpg " * 50)
        db = CatalogDB(str(tmp / "catalog.db"))
        return CatalogService(db, str(lib)), lib

    def _measure(self, service, name, sharpness=10.0, clip_high=0.0):
        service.db.upsert_photo_stats(
            f"2024-06-15/{name}.CR3",
            {
                "stats_version": STATS_VERSION,
                "dhash": "0000000000000000",
                "sharpness": sharpness,
                "clip_high": clip_high,
            },
            source_mtime=1000,
            source_size=900,
        )

    def test_a_folder_reports_its_soft_and_blown_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name in _NAMES:
                self._measure(service, name)
            self._measure(service, "IMG_0002", sharpness=0.5)
            self._measure(service, "IMG_0003", clip_high=0.5)

            result = service.folder_hints("2024-06-15", pair_mode="raw_only")
            assert result["total"] == len(_NAMES)
            assert result["measured"] == len(_NAMES)
            assert result["soft"] == 1
            assert result["blown"] == 1
            assert result["hints"]["2024-06-15/IMG_0002.CR3"]["soft_frame"] is True
            assert result["hints"]["2024-06-15/IMG_0003.CR3"]["blown_highlights"] is True

    def test_an_unmeasured_folder_says_so_instead_of_claiming_all_is_well(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])

            result = service.folder_hints("2024-06-15", pair_mode="raw_only")
            assert result["total"] == len(_NAMES)
            assert result["measured"] == 0
            assert result["median_sharpness"] is None
            assert result["hints"] == {}

    def test_measurements_from_older_code_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name in _NAMES:
                service.db.upsert_photo_stats(
                    f"2024-06-15/{name}.CR3",
                    {"stats_version": STATS_VERSION - 1, "sharpness": 10.0, "clip_high": 0.0},
                    source_mtime=1000,
                    source_size=900,
                )

            result = service.folder_hints("2024-06-15", pair_mode="raw_only")
            assert result["measured"] == 0
            assert result["hints"] == {}

    def test_a_pair_is_one_frame_not_two_opinions(self):
        # The grid collapses CR3+JPG into one tile, so the hints answer for the
        # frame on screen: one entry per tile, not one per file on disk. Which
        # half of the pair the grid shows is the grid's business, so the
        # measurements are the same for both.
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3", ".jpg"])
            for name in _NAMES:
                sharpness = 0.1 if name == "IMG_0002" else 10.0
                for ext in (".CR3", ".JPG"):
                    service.db.upsert_photo_stats(
                        f"2024-06-15/{name}{ext}",
                        {"stats_version": STATS_VERSION, "sharpness": sharpness, "clip_high": 0.0},
                        source_mtime=1000,
                        source_size=900,
                    )

            shown = [
                record.filepath
                for record in service.get_photos_in_folder("2024-06-15", pair_mode="paired")
            ]
            paired = service.folder_hints("2024-06-15", pair_mode="paired")
            assert paired["total"] == len(_NAMES)
            assert sorted(paired["hints"]) == sorted(shown)
            assert paired["soft"] == 1

    def test_the_grid_filters_narrow_what_is_judged(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name in _NAMES:
                self._measure(service, name)
            self._measure(service, "IMG_0002", sharpness=0.5)
            service.set_flag("2024-06-15/IMG_0002.CR3", -1)

            # A rejected frame is not on the grid, so it gets no answer here.
            kept = service.folder_hints("2024-06-15", pair_mode="raw_only", flag_filter=0)
            assert "2024-06-15/IMG_0002.CR3" not in kept["hints"]
            assert kept["total"] == len(_NAMES) - 1

    def test_a_filter_does_not_redefine_what_sharp_means(self):
        # Two tagged frames are far too few to describe the folder, so the
        # reference has to come from the whole folder for the hint to survive
        # the filter being turned on.
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name in _NAMES:
                self._measure(service, name)
            self._measure(service, "IMG_0002", sharpness=0.5)
            for name in ("IMG_0001", "IMG_0002"):
                service.set_photo_tags(f"2024-06-15/{name}.CR3", ["graduation"])

            tagged = service.folder_hints("2024-06-15", pair_mode="raw_only", tag_filter="graduation")
            assert tagged["total"] == 2
            assert tagged["measured"] == 2
            assert tagged["median_sharpness"] == 10.0
            assert tagged["soft"] == 1
