"""
Tests for the edit model: one person's own develop settings for a single photo.

What these tests hold still is the shared shape. A matching correction and the
point half of a hand-made edit are both "stops plus a gain per channel", so they
compose into one curve instead of two passes over the pixels — and the numbers a
slider shows are the numbers the export applies. The tonal ranges are the other
half: gains over part of the luminance range, which cannot live in a per-channel
curve and so get their own pass, and their own tests.
"""

import json
import math
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import (
    MAX_TONE_GAIN,
    adjustment,
    apply_develop,
    apply_tone,
    auto_tone,
    combine,
    describe,
    describe_settings,
    describe_tone,
    is_identity,
    normalize,
    point_of,
    to_crs,
    tone_table,
)
from catalog.database import CatalogDB
from export import engine
from export.engine import MatchSettings, export_photo
from analysis.harmonize import apply_correction


# ── The sliders ──────────────────────────────────────────────────────

class TestSliders:
    def test_everything_at_zero_is_no_adjustment(self):
        assert adjustment() is None
        assert adjustment(0.0, 0, 0) is None

    def test_exposure_is_stops(self):
        change = adjustment(1.0)
        assert change["ev"] == pytest.approx(1.0)
        assert change["wb_r"] == 1.0
        assert change["wb_g"] == 1.0
        assert change["wb_b"] == 1.0

    def test_warming_adds_red_and_takes_blue(self):
        warm = adjustment(temperature=100)
        assert warm["wb_r"] > 1.0
        assert warm["wb_b"] < 1.0
        assert warm["wb_g"] == 1.0

        cool = adjustment(temperature=-100)
        assert cool["wb_r"] < 1.0
        assert cool["wb_b"] > 1.0

    def test_tint_moves_green_the_other_way(self):
        magenta = adjustment(tint=100)
        green = adjustment(tint=-100)
        assert magenta["wb_g"] < 1.0 < green["wb_g"]
        # Tint is a green-channel move, not a red/blue one.
        assert magenta["wb_r"] == 1.0
        assert magenta["wb_b"] == 1.0

    def test_a_slider_dragged_too_far_is_pulled_back_not_rejected(self):
        change = adjustment(exposure=9.0, temperature=500, tint=-900)
        assert change["ev"] == pytest.approx(5.0)
        assert change["wb_r"] == pytest.approx(1.25)
        assert change["wb_b"] == pytest.approx(0.75)
        assert change["wb_g"] == pytest.approx(1.15)

    def test_nonsense_is_no_adjustment(self):
        assert adjustment(exposure="not a number") is None
        assert adjustment(temperature="warm") is None

    def test_a_personal_nudge_is_not_clamped_to_a_matching_nudge(self):
        """The matching limits exist because that feature promises consistency.
        Someone asking for two stops means two stops."""
        from analysis.harmonize import MAX_EV_SHIFT, MAX_WB_DELTA

        change = adjustment(exposure=2.0, temperature=100)
        assert change["ev"] > MAX_EV_SHIFT
        assert change["wb_r"] - 1.0 > MAX_WB_DELTA


# ── Composing ────────────────────────────────────────────────────────

class TestCombine:
    def test_nothing_composes_into_nothing(self):
        assert combine() is None
        assert combine(None, None) is None
        assert combine(adjustment(0.0, 0, 0)) is None

    def test_stops_add_and_gains_multiply(self):
        matched = {"ev": 0.5, "wb_r": 1.1, "wb_b": 0.9}
        personal = adjustment(1.0, 100, 0)
        combined = combine(matched, personal)
        assert combined["ev"] == pytest.approx(1.5)
        assert combined["wb_r"] == pytest.approx(1.1 * 1.25)
        assert combined["wb_b"] == pytest.approx(0.9 * 0.75)

    def test_the_order_does_not_matter(self):
        matched = {"ev": 0.5, "wb_r": 1.1, "wb_b": 0.9}
        personal = adjustment(1.0, 100, 0)
        assert combine(matched, personal) == combine(personal, matched)

    def test_a_nudge_cancelled_out_is_no_curve(self):
        """Two stops up and two stops down is a photo left alone, and the caller
        should be told there is nothing to apply."""
        assert combine(adjustment(1.0), adjustment(-1.0)) is None

    def test_identity_is_reported_as_such(self):
        assert is_identity(None)
        assert is_identity({"ev": 0.0, "wb_r": 1.0, "wb_g": 1.0, "wb_b": 1.0})
        assert not is_identity({"ev": 0.5, "wb_r": 1.0, "wb_g": 1.0, "wb_b": 1.0})
        assert not is_identity({"ev": 0.0, "wb_r": 1.2, "wb_g": 1.0, "wb_b": 1.0})

    def test_describe_says_what_is_about_to_happen(self):
        assert describe(None) == "no adjustment"
        text = describe(adjustment(0.5, 20, 0))
        assert "+0.50 EV" in text


# ── The pixels ───────────────────────────────────────────────────────

def _solid(value: int, size: int = 8) -> Image.Image:
    return Image.new("RGB", (size, size), (value, value, value))


def _channel(image: Image.Image, index: int) -> int:
    return int(np.array(image)[:, :, index].mean())


class TestPixels:
    def test_one_stop_doubles_the_recorded_values(self):
        out = apply_correction(_solid(100), adjustment(1.0))
        assert _channel(out, 0) == 200

    def test_one_stop_down_halves_them(self):
        out = apply_correction(_solid(100), adjustment(-1.0))
        assert _channel(out, 0) == 50

    def test_a_white_balance_gain_reaches_one_channel(self):
        out = apply_correction(_solid(100), adjustment(temperature=100))
        assert _channel(out, 0) == 125
        assert _channel(out, 1) == 100
        assert _channel(out, 2) == 75

    def test_tint_moves_the_green_channel(self):
        out = apply_correction(_solid(200), adjustment(tint=100))
        assert _channel(out, 0) == 200
        assert _channel(out, 1) == 170

    def test_a_nudge_and_a_match_land_as_one_pass(self):
        """Applying the composed curve gives the same frame as applying both in
        turn, which is the whole reason they share a shape."""
        matched = {"ev": 0.5, "wb_r": 1.1, "wb_b": 0.9}
        personal = adjustment(0.5, 40, 0)

        together = apply_correction(_solid(100), combine(matched, personal))
        in_turn = apply_correction(apply_correction(_solid(100), matched), personal)

        for index in range(3):
            assert abs(_channel(together, index) - _channel(in_turn, index)) <= 1

    def test_highlights_are_clipped_not_wrapped(self):
        out = apply_correction(_solid(200), adjustment(2.0))
        assert _channel(out, 0) == 255


# ── Decoding once ────────────────────────────────────────────────────

class TestPreviewBase:
    def _source(self, tmp: Path) -> Path:
        source = tmp / "IMG_0001.JPG"
        _solid(120, 64).save(source, format="JPEG")
        return source

    def test_the_frame_is_decoded_once_and_reused(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        source = self._source(tmp_path)
        calls = []
        real_load = engine._load_image

        def counting_load(path, max_resolution=0):
            calls.append(path)
            return real_load(path, max_resolution)

        monkeypatch.setattr(engine, "_load_image", counting_load)

        first = engine.preview_base(source, 32)
        second = engine.preview_base(source, 32)

        assert first is second
        assert len(calls) == 1

    def test_a_different_size_is_a_different_frame(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        source = self._source(tmp_path)
        calls = []
        real_load = engine._load_image
        monkeypatch.setattr(
            engine, "_load_image",
            lambda path, max_resolution=0: calls.append(path) or real_load(path, max_resolution),
        )
        small = engine.preview_base(source, 16)
        large = engine.preview_base(source, 64)
        assert small.size != large.size
        assert len(calls) == 2

    def test_a_file_that_changed_on_disk_is_not_served_from_memory(self, tmp_path):
        engine.clear_preview_cache()
        source = self._source(tmp_path)
        before = engine.preview_base(source, 64)
        _solid(20, 64).save(source, format="JPEG")
        after = engine.preview_base(source, 64)
        assert before is not after
        assert _channel(after, 0) < _channel(before, 0)

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert engine.preview_base(tmp_path / "nope.JPG", 32) is None


# ── The export applies it ────────────────────────────────────────────

class TestExportAppliesIt:
    def _template(self):
        # No EXIF step: this test is about pixels, and exiftool is a subprocess.
        return engine.ExportTemplate(max_resolution=0, quality=95, strip_exif=False)

    def _grey_jpeg(self, tmp: Path, value: int = 100) -> Path:
        source = tmp / "IMG_0001.JPG"
        _solid(value, 64).save(source, format="JPEG", quality=95)
        return source

    def test_an_unadjusted_export_is_the_photo_as_it_is(self, tmp_path):
        source = self._grey_jpeg(tmp_path)
        out = export_photo(
            source, tmp_path, self._template()
        )
        assert abs(_channel(Image.open(out), 0) - 100) <= 3

    @staticmethod
    def _after_stops(value: int, stops: float) -> float:
        """What an encoded grey becomes under an exposure move in linear light."""
        from adjust.render import decode, encode

        linear = decode(np.array([value / 255.0], dtype=np.float32))[0]
        return float(encode(np.array([linear * 2.0 ** stops], dtype=np.float32))[0] * 255.0)

    def test_the_nudge_reaches_the_written_file(self, tmp_path):
        """One stop is one stop of light, not a doubling of the encoded value."""
        source = self._grey_jpeg(tmp_path)
        out = export_photo(
            source, tmp_path, self._template(),
            develop={"exposure": 1.0},
        )
        assert abs(_channel(Image.open(out), 0) - self._after_stops(100, 1.0)) <= 4
        assert _channel(Image.open(out), 0) < 180  # not the old doubling to 200

    def test_the_nudge_and_the_match_travel_in_one_pass(self, tmp_path):
        """A frame a stop under the set, nudged one stop by hand, lands where
        both together say it should — not where either one alone would put it.
        The match is asked for on encoded values (a 0.5 EV match doubles them
        by 2**0.5), so it is translated to the same gain in linear light."""
        source = self._grey_jpeg(tmp_path, value=40)
        match = MatchSettings(
            target={"count": 2, "lum": 80.0, "wb_r": 1.0, "wb_b": 1.0},
            strength=1.0,
        )
        out = export_photo(
            source, tmp_path, self._template(),
            
            match=match,
            develop={"exposure": 1.0},
        )
        # The match is capped at 0.5 EV on encoded values, which is the same
        # light as 0.5 / 0.45 stops in linear; the person's stop is added to it.
        matched = min(0.5, math.log2(80.0 / 40.0)) / 0.45
        expected = self._after_stops(40, matched + 1.0)
        assert abs(_channel(Image.open(out), 0) - expected) <= 6
        assert match.applied == 1

    def test_a_nudge_alone_still_counts_as_no_match(self, tmp_path):
        source = self._grey_jpeg(tmp_path)
        out = export_photo(
            source, tmp_path, self._template(),
            develop={"exposure": -1.0},
        )
        assert abs(_channel(Image.open(out), 0) - self._after_stops(100, -1.0)) <= 3

    def test_a_tone_control_reaches_the_written_file_on_its_own(self, tmp_path):
        """No exposure, no white balance, no match: only the tone pass runs."""
        source = self._grey_jpeg(tmp_path, value=20)
        out = export_photo(
            source, tmp_path, self._template(),
            develop={"shadows": 100},
        )
        # A flat grey is all base layer, so the shadow lift is the whole story,
        # and it is a real lift: several stops for a frame this dark.
        assert _channel(Image.open(out), 0) > 20 + 40

    def test_the_point_half_lands_before_the_tone_half(self, tmp_path):
        """The ranges are defined relative to the exposure that was chosen, so
        a frame lifted a stop is then toned as the lifted frame, not the dark
        one. A brighter frame has less shadow left to open, so it gains less
        from the same slider than the dark one would have."""
        source = self._grey_jpeg(tmp_path, value=20)
        plain = export_photo(
            source, tmp_path, self._template(),
            develop={"shadows": 100},
        )
        plain_value = _channel(Image.open(plain), 0)
        lifted = export_photo(
            source, tmp_path, self._template(),
            
            develop={"exposure": 1.0, "shadows": 100},
        )
        lifted_value = _channel(Image.open(lifted), 0)
        assert lifted_value > plain_value
        # The stop it was lifted by is not simply added on top of the fill.
        assert lifted_value - plain_value < self._after_stops(20, 1.0) - 20 + 40


class TestNormalize:
    """The settings dictionary is what the panel edits, what the basket stores
    and what a preset holds. Every caller reads an empty dict as "no edit", so
    a control sitting at neutral has to be absent rather than present at zero."""

    def test_nothing_set_is_no_edit(self):
        assert normalize({}) == {}
        assert normalize(None) == {}
        assert normalize({"exposure": 0, "shadows": 0}) == {}

    def test_a_control_at_neutral_is_not_kept(self):
        assert normalize({"exposure": 0.0, "temperature": 0, "highlights": 5}) == {
            "highlights": 5
        }

    def test_a_control_that_cannot_be_read_falls_back_to_neutral(self):
        """A hand-edited or older blob degrades instead of failing an export."""
        assert normalize({"exposure": "bright", "shadows": 20}) == {"shadows": 20}

    def test_a_control_out_of_range_is_pulled_back(self):
        assert normalize({"exposure": 40.0, "shadows": 900, "blacks": -900}) == {
            "exposure": 5.0, "shadows": 100, "blacks": -100,
        }

    def test_a_control_this_build_does_not_know_is_dropped(self):
        assert normalize({"sparkle": 40, "shadows": 10}) == {"shadows": 10}

    def test_both_halves_survive_together(self):
        clean = normalize({"exposure": 0.7, "temperature": 20, "shadows": 30})
        assert clean == {"exposure": 0.7, "temperature": 20, "shadows": 30}
        assert point_of(clean) is not None
        assert tone_table(clean) is not None


class TestToneTable:
    """The tone controls are gains over part of the luminance range. The table
    is the whole behaviour of a slider, so it can be read directly instead of
    inferred from pixels."""

    def test_no_tone_control_is_no_pass(self):
        assert tone_table({}) is None
        assert tone_table({"exposure": 1.0, "temperature": 50}) is None

    def test_the_table_covers_every_luma(self):
        table = tone_table({"shadows": 50})
        assert len(table) == 256
        assert all(0.0 <= gain <= MAX_TONE_GAIN for gain in table)

    def test_shadows_lift_the_bottom_and_leave_the_top_alone(self):
        table = tone_table({"shadows": 100})
        assert table[0] == pytest.approx(1.9, abs=0.01)
        assert table[255] == pytest.approx(1.0)
        assert table[10] > table[100] > table[255]

    def test_highlights_pull_the_top_back_and_leave_the_bottom_alone(self):
        table = tone_table({"highlights": -100})
        assert table[255] == pytest.approx(0.15, abs=0.01)
        assert table[0] == pytest.approx(1.0)
        assert table[250] < table[100]

    def test_blacks_only_move_the_bottom_end(self):
        table = tone_table({"blacks": -100})
        assert table[0] == pytest.approx(0.5, abs=0.01)
        assert table[128] == pytest.approx(1.0)
        assert table[255] == pytest.approx(1.0)

    def test_whites_only_move_the_top_end(self):
        table = tone_table({"whites": 100})
        assert table[255] == pytest.approx(1.5, abs=0.01)
        assert table[0] == pytest.approx(1.0)
        assert table[128] == pytest.approx(1.0)

    def test_contrast_moves_away_from_the_midtones_and_not_through_them(self):
        """A multiplicative contrast is nearly a no-op in the shadows, which is
        why this one is a power law anchored on the pivot."""
        table = tone_table({"contrast": 100})
        assert table[64] < 1.0 < table[192]
        assert abs(table[128] - 1.0) < 0.02

    def test_negative_contrast_is_the_mirror_image(self):
        table = tone_table({"contrast": -100})
        assert table[64] > 1.0 > table[192]

    def test_stacked_ranges_are_clamped_not_run_away_with(self):
        table = tone_table({"shadows": 100, "blacks": 100, "contrast": -100})
        assert table[0] == pytest.approx(MAX_TONE_GAIN)

    def test_the_ranges_compose_into_one_table(self):
        """Highlights down and shadows up at the same time is the point of the
        two controls; neither may cancel the other out in the middle."""
        table = tone_table({"highlights": -100, "shadows": 100})
        assert table[0] > 1.0 > table[255]


class TestTonePixels:
    def test_a_shadow_lift_reaches_a_dark_frame(self):
        out = apply_tone(_solid(20), tone_table({"shadows": 100}))
        assert abs(_channel(out, 0) - 37) <= 1

    def test_a_tone_control_does_not_invent_detail(self):
        """A pixel that is already pure black stays black. This is the honest
        limit of a multiplicative control, and the reason auto tone is not
        allowed to claim it recovered anything from a clipped shadow."""
        out = apply_tone(_solid(0), tone_table({"shadows": 100}))
        assert _channel(out, 0) == 0

    def test_highlight_recovery_pulls_a_bright_frame_down(self):
        out = apply_tone(_solid(250), tone_table({"highlights": -100}))
        assert abs(_channel(out, 0) - 38) <= 2

    def test_a_gain_on_luma_leaves_colour_alone(self):
        """Every channel is scaled by the same number, so a tone control cannot
        shift hue — that is what keeps it separate from white balance."""
        source = Image.new("RGB", (8, 8), (200, 100, 50))
        out = np.array(apply_tone(source, tone_table({"shadows": 60})))
        ratios = out[:, :, 0] / np.maximum(out[:, :, 1], 1)
        assert np.allclose(ratios, 2.0, atol=0.05)

    def test_apply_develop_is_the_same_two_pass_result(self):
        settings = {"exposure": 1.0, "shadows": 100}
        direct = apply_develop(_solid(20), settings)
        assert abs(_channel(direct, 0) - 69) <= 2

    def test_an_empty_edit_set_leaves_the_image_alone(self):
        source = _solid(120)
        assert apply_develop(source, {}) is source
        assert apply_develop(source, None) is source


def _frame(*bands, size: int = 200) -> Image.Image:
    """A frame with a chosen tone distribution: ``(value, share)`` pairs.

    A test photo is a histogram, not a gradient. Auto tone reads percentiles and
    then asks whether acting on them would show, so the fixture has to say how
    much of the frame sits where — a uniform ramp answers that question with a
    distribution no camera produces.
    """
    values: list[int] = []
    for value, share in bands:
        values += [value] * int(round(share * size * size))
    values = (values + [values[-1]] * (size * size))[: size * size]
    row = np.array(values, dtype=np.uint8).reshape(size, size, 1)
    return Image.fromarray(np.repeat(row, 3, axis=2), "RGB")


class TestAutoTone:
    """Auto tone sets this frame's endpoints from its own histogram. What makes
    it worth having is what it declines to do: a frame that is fine, or dark on
    purpose, gets nothing."""

    def test_a_frame_that_spans_the_range_is_left_alone(self):
        """The endpoints already sit where they should, so there is nothing for
        either end to act on and no curve worth adding."""
        assert auto_tone(_frame((12, 0.3), (120, 0.4), (236, 0.3))) == {}

    def test_a_frame_sitting_on_white_loses_highlights(self):
        assert auto_tone(_frame((200, 0.6), (255, 0.4))) == {"highlights": -16}

    def test_a_milky_frame_gets_its_black_point_lowered_and_white_raised(self):
        """Dark ground, mid-tones, bright sky: the two ends move, the curve
        does not, because the frame already has contrast."""
        assert auto_tone(_frame((45, 0.3), (120, 0.4), (195, 0.3))) == {
            "whites": 42, "blacks": -43,
        }

    def test_a_flat_midtone_band_gets_contrast_and_nothing_else(self):
        """Nothing at either end to move, so only the curve is worth offering.
        The band spans 16% of the range, and the amount is scaled by how far
        that is short of a frame that already has contrast."""
        assert auto_tone(_frame((105, 0.5), (145, 0.5))) == {"contrast": 40}

    def test_a_dark_frame_stays_dark(self):
        """Auto tone does not touch exposure, so a frame that is meant to be
        dark is not brightened into something it is not."""
        assert auto_tone(_solid(3)) == {}

    def test_a_control_that_would_not_change_the_frame_is_not_offered(self):
        """A night frame's darkest pixels are already flat black. The percentile
        rule alone would see a frame five levels off its black point and offer a
        shadow lift; no gain a slider can reach moves a pixel at 2. The frame is
        asked, and the answer is that the suggestion would do nothing."""
        night = _frame((0, 0.6), (1, 0.25), (2, 0.1), (180, 0.05))
        assert auto_tone(night) == {}

    def test_a_high_key_frame_is_not_told_its_blacks_are_wrong(self):
        """A control with nothing to act on is noise, not a suggestion."""
        assert "blacks" not in auto_tone(_frame((200, 0.6), (255, 0.4)))

    def test_it_never_touches_exposure_or_white_balance(self):
        frames = [
            _frame((12, 0.3), (120, 0.4), (236, 0.3)),
            _frame((45, 0.3), (120, 0.4), (195, 0.3)),
            _solid(3), _solid(250),
        ]
        for image in frames:
            settings = auto_tone(image)
            assert "exposure" not in settings
            assert "temperature" not in settings
            assert "tint" not in settings

    def test_it_never_asks_for_more_than_a_slider_can_reach(self):
        from adjust.develop import (
            AUTO_MAX_BLACKS, AUTO_MAX_CONTRAST, AUTO_MAX_HIGHLIGHTS,
            AUTO_MAX_SHADOWS, AUTO_MAX_WHITES,
        )

        ceilings = {
            "highlights": AUTO_MAX_HIGHLIGHTS, "whites": AUTO_MAX_WHITES,
            "shadows": AUTO_MAX_SHADOWS, "blacks": AUTO_MAX_BLACKS,
            "contrast": AUTO_MAX_CONTRAST,
        }
        frames = [
            _frame((12, 0.3), (120, 0.4), (236, 0.3)),
            _frame((45, 0.3), (120, 0.4), (195, 0.3)),
            _frame((105, 0.5), (145, 0.5)),
            _frame((200, 0.6), (255, 0.4)),
            _solid(3),
        ]
        for image in frames:
            for key, value in auto_tone(image).items():
                assert abs(value) <= ceilings[key]

    def test_what_it_returns_is_a_valid_edit_set(self):
        settings = auto_tone(_frame((45, 0.3), (120, 0.4), (195, 0.3)))
        assert normalize(settings) == settings
        assert tone_table(settings) is not None


class TestDescribingAnEdit:
    def test_no_edit_says_so(self):
        assert describe_settings({}) == "no edit"
        assert describe_settings({"exposure": 0, "shadows": 0}) == "no edit"

    def test_it_names_every_control_that_is_set(self):
        line = describe_settings(
            {"exposure": 0.7, "highlights": -40, "temperature": 20}
        )
        assert line == "+0.70 EV, Highlights -40, WB r1.050 g1.000 b0.950"

    def test_the_tone_half_can_be_described_on_its_own(self):
        """The export log wants the tone line separately, because the point
        half is already logged as part of the correction."""
        assert describe_tone({"shadows": 25, "contrast": 10}) == "Shadows +25, Contrast +10"
        assert describe_tone({"exposure": 1.0}) == "no tone"


class TestSidecarNames:
    """The same edit has to mean the same thing in Lightroom. Where a value
    carries over exactly it is written as-is; where it would be a lie, it is
    named for what it actually is."""

    def test_no_edit_writes_nothing(self):
        assert to_crs({}) == {}
        assert to_crs(None) == {}

    def test_exposure_and_the_tone_controls_carry_over_exactly(self):
        crs = to_crs({"exposure": 0.7, "contrast": 15, "shadows": 30, "blacks": -10})
        assert crs["crs:Exposure2012"] == "+0.70"
        assert crs["crs:Contrast2012"] == "+15"
        assert crs["crs:Shadows2012"] == "+30"
        assert crs["crs:Blacks2012"] == "-10"

    def test_tint_is_rescaled_to_lightroom_s_range(self):
        assert to_crs({"tint": 40})["crs:Tint2012"] == "+60"

    def test_warmth_is_not_claimed_to_be_a_colour_temperature(self):
        """Lightroom's crs:Temperature is kelvin. This app's warmth slider is a
        relative channel shift, so it goes under a name that says so."""
        crs = to_crs({"temperature": -30})
        assert crs["crs:TemperatureShift"] == "-30"
        assert "crs:Temperature" not in crs


# ── The photograph remembers it ──────────────────────────────────────

class TestPhotoEdit:
    """
    The edit belongs to the photograph, not to a row in the export basket. That
    is what makes it non-destructive: it is there when the basket is cleared,
    it is there when a folder is browsed months later, and it does not depend on
    having decided to export anything.
    """

    def _db(self, tmp: Path) -> CatalogDB:
        return CatalogDB(str(tmp / "catalog.db"))

    def test_an_edit_is_stored_against_one_photo(self, tmp_path):
        """The row holds the sliders, not the curve they turn into, so the
        develop panel can put them back exactly where they were left."""
        db = self._db(tmp_path)
        sliders = {"exposure": 0.5, "temperature": 20, "tint": -10}
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", sliders)
        assert db.get_photo_edit("2024-06-15/IMG_0001.CR3") == sliders

    def test_walking_the_sliders_back_clears_it(self, tmp_path):
        db = self._db(tmp_path)
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", {"exposure": 0.5})
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", None)
        assert db.get_photo_edit("2024-06-15/IMG_0001.CR3") == {}
        assert db.photo_edit_count() == 0

    def test_a_photo_that_was_never_edited_has_no_edit(self, tmp_path):
        db = self._db(tmp_path)
        assert db.get_photo_edit("2024-06-15/IMG_9999.CR3") == {}

    def test_one_photo_s_edit_does_not_move_another(self, tmp_path):
        db = self._db(tmp_path)
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", {"exposure": 1.0})
        assert db.get_photo_edit("2024-06-15/IMG_0002.CR3") == {}

    def test_an_edit_survives_the_basket_being_cleared(self, tmp_path):
        """The whole point of moving it off the basket row: choosing to export a
        frame and deciding how it should look are separate, and undoing the first
        must not throw away the second."""
        db = self._db(tmp_path)
        db.add_to_basket("2024-06-15/IMG_0001.CR3", "2024-06-15", "IMG_0001.CR3")
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", {"shadows": 30})
        db.clear_basket()
        assert db.get_photo_edit("2024-06-15/IMG_0001.CR3") == {"shadows": 30}

    def test_a_folder_comes_back_as_one_map(self, tmp_path):
        """One query for a whole grid: asking per frame would be thousands of
        round trips to decide which thumbnails get a marker."""
        db = self._db(tmp_path)
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", {"exposure": 1.0})
        db.set_photo_edit("2024-06-15/IMG_0002.CR3", {"shadows": 20})
        db.set_photo_edit("2024-06-16/IMG_0003.CR3", {"contrast": 10})

        assert set(db.list_photo_edits("2024-06-15")) == {
            "2024-06-15/IMG_0001.CR3", "2024-06-15/IMG_0002.CR3",
        }
        assert len(db.list_photo_edits()) == 3

    def test_an_older_database_gains_the_table_and_moves_its_basket_edits(self, tmp_path):
        """A catalog written before this feature existed still opens, and the
        nudges it was holding on basket rows become edits on the photographs —
        once, not on every start."""
        db_path = tmp_path / "old.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """CREATE TABLE export_basket (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT UNIQUE NOT NULL,
                    folder TEXT NOT NULL DEFAULT '',
                    filename TEXT NOT NULL DEFAULT '',
                    added_at TEXT DEFAULT '',
                    adjustment TEXT NOT NULL DEFAULT ''
                )"""
            )
            conn.execute(
                "INSERT INTO export_basket (filepath, folder, filename, adjustment) "
                "VALUES ('2024-06-15/IMG_0001.CR3', '2024-06-15', 'IMG_0001.CR3', "
                "'{\"exposure\": 1.0}')"
            )
            conn.commit()

        db = CatalogDB(str(db_path))
        # Moved onto the photograph...
        assert db.get_photo_edit("2024-06-15/IMG_0001.CR3") == {"exposure": 1.0}
        # ...and taken off the basket row, so a second start finds nothing to
        # move and cannot resurrect an edit the user has since cleared.
        assert db.list_basket()[0].adjustment == ""
        db.set_photo_edit("2024-06-15/IMG_0001.CR3", None)
        assert CatalogDB(str(db_path)).get_photo_edit("2024-06-15/IMG_0001.CR3") == {}
