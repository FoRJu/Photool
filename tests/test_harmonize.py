"""
Tests for matching a set — the small corrections that make a batch of photos
look like one shoot instead of a hundred separate exposures.

Two things are being pinned down: that the target is the middle of the set
rather than whatever the brightest frame happened to look like, and that a
correction stays a nudge no matter how far apart the frames started.
"""

import math
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analysis.harmonize import (  # noqa: E402
    MAX_EV_SHIFT,
    MAX_WB_DELTA,
    apply_correction,
    correction,
    set_target,
)
from analysis.image_stats import compute_image_stats  # noqa: E402
from export.engine import ExportTemplate, MatchSettings, export_photo  # noqa: E402


def _row(lum, wb_r=1.0, wb_b=1.0):
    """A stored measurement, shaped like a catalog row."""
    return {"lum_p50": lum, "wb_r": wb_r, "wb_b": wb_b}


def _target(lum, wb_r=1.0, wb_b=1.0):
    """A set target, shaped like what set_target returns."""
    return {"lum": lum, "wb_r": wb_r, "wb_b": wb_b}


def _spread(rows):
    """How far apart a set is, in EV — the thing this feature exists to reduce."""
    values = [row["lum_p50"] for row in rows if row and row["lum_p50"] > 0]
    return max(math.log2(v) for v in values) - min(math.log2(v) for v in values)


class TestSetTarget:
    def test_the_middle_of_the_set_wins_not_the_brightest_frame(self):
        target = set_target([_row(100), _row(120), _row(900)])
        assert target["lum"] == pytest.approx(120)
        assert target["count"] == 3

    def test_an_even_set_sits_between_its_two_middles(self):
        target = set_target([_row(100), _row(200)])
        assert target["lum"] == pytest.approx(150)

    def test_one_photo_has_nothing_to_match_against(self):
        assert set_target([_row(100)]) is None
        assert set_target([]) is None
        assert set_target([None, None]) is None

    def test_unmeasurable_members_are_left_out(self):
        target = set_target([
            _row(100),
            _row(120),
            _row(0),                      # measured black: not a brightness
            _row(140, wb_r=0, wb_b=1),    # no usable white balance
            {"lum_p50": None},
            {"lum_p50": "not a number"},
            None,
        ])
        assert target is not None
        assert target["count"] == 2
        assert target["lum"] == pytest.approx(110)


class TestCorrection:
    def test_a_darker_frame_is_lifted(self):
        change = correction(_row(100), _target(130))
        assert change["ev"] == pytest.approx(math.log2(1.3))

    def test_a_brighter_frame_is_brought_down(self):
        change = correction(_row(130), _target(100))
        assert change["ev"] == pytest.approx(-math.log2(1.3))

    def test_the_reach_is_limited_however_far_apart_they_started(self):
        assert correction(_row(10), _target(2000))["ev"] == pytest.approx(MAX_EV_SHIFT)
        assert correction(_row(2000), _target(10))["ev"] == pytest.approx(-MAX_EV_SHIFT)

    def test_a_frame_that_already_matches_is_left_alone(self):
        assert correction(_row(100), _target(101)) is None
        assert correction(_row(100, 1.0, 1.0), _target(100, 1.01, 0.99)) is None

    def test_strength_scales_the_nudge(self):
        full = correction(_row(100), _target(130))
        half = correction(_row(100), _target(130), strength=0.5)
        assert full["ev"] == pytest.approx(math.log2(1.3))
        assert half["ev"] == pytest.approx(full["ev"] * 0.5)
        assert correction(_row(100), _target(130), strength=0) is None

    def test_white_balance_is_a_nudge_too(self):
        change = correction(_row(100), _target(100, wb_r=1.5, wb_b=0.5))
        assert change["wb_r"] == pytest.approx(1.0 + MAX_WB_DELTA)
        assert change["wb_b"] == pytest.approx(1.0 - MAX_WB_DELTA)
        assert change["ev"] == 0.0

    def test_a_measurement_that_is_not_there_is_not_corrected(self):
        assert correction(_row(0), _target(100)) is None
        assert correction({"lum_p50": 100}, _target(100)) is None


class TestApplyCorrection:
    def test_identity_changes_nothing(self):
        image = Image.new("RGB", (8, 8), (120, 130, 140))
        same = apply_correction(image, {"ev": 0.0, "wb_r": 1.0, "wb_b": 1.0})
        assert same.tobytes() == image.tobytes()

    def test_a_stop_of_brightness_multiplies_the_pixels(self):
        image = Image.new("RGB", (8, 8), (100, 100, 100))
        lifted = apply_correction(image, {"ev": 1.0, "wb_r": 1.0, "wb_b": 1.0})
        assert lifted.getpixel((0, 0)) == (200, 200, 200)

    def test_a_lift_that_would_blow_out_stops_at_white(self):
        image = Image.new("RGB", (8, 8), (200, 200, 200))
        lifted = apply_correction(image, {"ev": 2.0, "wb_r": 1.0, "wb_b": 1.0})
        assert lifted.getpixel((0, 0)) == (255, 255, 255)

    def test_white_balance_moves_the_channels_apart(self):
        image = Image.new("RGB", (8, 8), (128, 128, 128))
        warm = apply_correction(image, {"ev": 0.0, "wb_r": 1.1, "wb_b": 0.9})
        red, green, blue = warm.getpixel((0, 0))
        assert red > green > blue


class TestExportMatchesTheSet:
    """The whole point: three frames of one moment, exported, end up together."""

    def test_a_matched_set_ends_up_closer_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()

            sources, rows = [], []
            for index, level in enumerate((60, 120, 240)):
                path = root / f"IMG_{index:04d}.JPG"
                Image.new("RGB", (240, 160), (level,) * 3).save(path, format="JPEG", quality=95)
                sources.append(path)
                rows.append(compute_image_stats(Image.open(path)))

            target = set_target(rows)
            assert target["lum"] == pytest.approx(120, abs=6)
            before = _spread(rows)

            match = MatchSettings(target=target, strength=1.0)
            for path in sources:
                export_photo(
                    path, out_dir,
                    ExportTemplate(max_resolution=512, quality=95),
                    match=match,
                )

            exported = [compute_image_stats(Image.open(out_dir / p.name)) for p in sources]
            assert _spread(exported) < before * 0.6
            # The middle frame was already the target, so it is written untouched.
            assert match.applied == 2

    def test_a_set_that_has_not_been_measured_exports_as_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            path = root / "IMG_0000.JPG"
            Image.new("RGB", (120, 80), (90, 90, 90)).save(path, format="JPEG", quality=95)

            export_photo(
                path, out_dir,
                ExportTemplate(max_resolution=512, quality=95),
                match=MatchSettings(target=None),
            )
            written = compute_image_stats(Image.open(out_dir / path.name))
            assert written["lum_p50"] == pytest.approx(90, abs=3)
