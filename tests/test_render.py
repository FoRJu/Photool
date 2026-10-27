"""
Tests for the develop pipeline: an edit performed in linear light.

These hold still the properties that make the controls useful rather than
decorative — exposure is a stop of light, highlights can bring back what an
8-bit rendering would have thrown away, shadows open without crushing the
mid-tones, a squeeze of the scene's range keeps its local detail, and nothing
leaves a halo along an edge.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import render
from adjust.develop import from_crs, normalize, to_crs
from adjust.render import (
    LOG_MID,
    LinearFrame,
    auto_develop,
    decode,
    develop_frame,
    encode,
    linear_from_image,
    tone_curve,
)


def _frame(rgb) -> LinearFrame:
    return LinearFrame(np.asarray(rgb, dtype=np.float32))


def _flat(value, size=64, tint=(1.0, 1.0, 1.0)) -> LinearFrame:
    rgb = np.empty((size, size, 3), dtype=np.float32)
    for i, k in enumerate(tint):
        rgb[..., i] = value * k
    return _frame(rgb)


def _px(image: Image.Image, x=None, y=None, channel=1) -> float:
    arr = np.asarray(image, dtype=np.float32)
    if x is None:
        return float(arr[..., channel].mean())
    return float(arr[y, x, channel])


def _textured(base_value: float, amplitude: float, size=128, seed=1) -> np.ndarray:
    """A surface: a base brightness with fine texture riding on it."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size)).astype(np.float32)
    plane = base_value * np.exp2(amplitude * noise)
    return np.stack([plane] * 3, axis=-1)


# ── The display curve ────────────────────────────────────────────────

class TestDisplayCurve:
    def test_encode_inverts_decode(self):
        values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        assert np.allclose(decode(encode(values)), values, atol=2e-4)

    def test_an_8_bit_picture_survives_the_round_trip_unchanged(self):
        """No edit means the same bytes: the pipeline must not drift a frame
        that was merely passed through it."""
        rng = np.random.default_rng(0)
        pic = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), "RGB")
        back = _frame(linear_from_image(pic)).encoded()
        assert np.abs(np.asarray(back, int) - np.asarray(pic, int)).max() <= 1

    def test_encoding_clips_instead_of_wrapping(self):
        assert float(encode(np.array([3.0]))[0]) == pytest.approx(1.0, abs=1e-3)
        assert float(encode(np.array([-1.0]))[0]) == 0.0


# ── Light ────────────────────────────────────────────────────────────

class TestDisplayShoulder:
    def test_a_frame_with_no_headroom_is_untouched(self):
        """A JPEG's white must stay 255 under an edit that does not touch the
        highlights."""
        frame = linear_from_frame(255)
        assert np.asarray(develop_frame(frame, {"temperature": 1}))[0, 0, 1] == 255

    def test_the_rolloff_tightens_smoothly_as_headroom_grows(self):
        """No step between a frame that just exceeds white and one that does not."""
        def mid_gain(peak):
            rgb = np.full((8, 8, 3), 0.9, dtype=np.float32)
            rgb[0, 0] = peak
            return float(np.asarray(_frame(rgb).encoded())[4, 4, 1])
        values = [mid_gain(p) for p in (1.0, 1.01, 1.05, 1.5, 3.0)]
        assert values[0] == pytest.approx(float(encode(np.array([0.9]))[0] * 255), abs=1)
        assert all(a >= b - 1 for a, b in zip(values, values[1:]))   # never rises
        assert abs(values[0] - values[1]) <= 1.5                      # and no step

    def test_the_brightest_value_lands_on_white(self):
        rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
        rgb[0, 0] = 2.5
        assert np.asarray(_frame(rgb).encoded())[0, 0, 1] >= 254

    def test_no_edit_and_an_edit_that_changes_nothing_are_the_same_picture(self):
        rng = np.random.default_rng(9)
        frame = _frame(rng.random((16, 16, 3)).astype(np.float32) * 2.0)
        a = np.asarray(frame.encoded(), dtype=int)
        b = np.asarray(develop_frame(frame, {"exposure": 0.01}), dtype=int)
        assert np.abs(a - b).max() <= 2


class TestExposure:
    def test_one_stop_is_one_stop_of_light(self):
        frame = _flat(0.1)
        out = develop_frame(frame, {"exposure": 1.0})
        linear = decode(np.array([_px(out) / 255.0], dtype=np.float32))[0]
        assert linear == pytest.approx(0.2, rel=0.03)

    def test_the_edit_leaves_the_decoded_frame_alone(self):
        frame = _flat(0.1)
        before = frame.rgb.copy()
        develop_frame(frame, {"exposure": 2.0, "shadows": 50, "compress": 60})
        assert np.array_equal(frame.rgb, before)

    def test_an_empty_edit_is_the_frame_as_it_stands(self):
        frame = _flat(0.2)
        assert np.array_equal(
            np.asarray(develop_frame(frame, {})), np.asarray(frame.encoded())
        )

    def test_a_stop_up_does_not_blow_a_midtone_the_way_scaling_encoded_values_did(self):
        """The old exposure doubled the encoded value, which is more than two
        stops of light: a frame at 100/255 was sent to 200."""
        out = develop_frame(linear_from_frame(100), {"exposure": 1.0})
        assert _px(out) < 170

    def test_a_large_move_rescues_a_frame_shot_far_under(self):
        out = develop_frame(_flat(0.004), {"exposure": 4.0})
        expected = float(encode(np.array([0.004 * 16.0], dtype=np.float32))[0] * 255.0)
        assert _px(out) == pytest.approx(expected, abs=3)
        # From 5 levels of black to a picture you can see.
        assert _px(out) > 10 * _px(develop_frame(_flat(0.004), {"exposure": 0.01}))


def linear_from_frame(value: int) -> LinearFrame:
    pic = Image.new("RGB", (16, 16), (value, value, value))
    return _frame(linear_from_image(pic))


# ── Highlights: recovering what 8 bits would have thrown away ────────

class TestHighlights:
    def _sky(self) -> LinearFrame:
        """A ramp of cloud from white up to three times white, in linear light
        — the headroom a RAW decode keeps and an 8-bit rendering clips."""
        ramp = np.linspace(1.0, 3.0, 256, dtype=np.float32)
        rgb = np.tile(ramp[None, :, None], (64, 1, 3))
        return _frame(rgb)

    def test_without_an_edit_the_headroom_is_squeezed_into_a_sliver_of_white(self):
        """Everything above white lands in the last few levels: it is on the
        record, but it does not show as detail until the highlights are pulled."""
        row = np.asarray(develop_frame(self._sky(), {"exposure": 0.01}))[0, :, 1].astype(float)
        assert row.max() - row.min() < 12
        assert row.max() >= 250

    def test_pulling_highlights_back_shows_the_detail_that_was_above_white(self):
        out = develop_frame(self._sky(), {"highlights": -100})
        row = np.asarray(out)[0, :, 1].astype(float)
        assert row.max() - row.min() > 40
        blocks = row.reshape(16, 16).mean(axis=1)   # dither is noise; the ramp is not
        assert np.all(np.diff(blocks) > 0)

    def test_it_is_a_real_pull_not_a_nudge(self):
        bright = _flat(0.9)
        plain = _px(develop_frame(bright, {"exposure": 0.01}))
        pulled = _px(develop_frame(bright, {"highlights": -100}))
        assert plain - pulled > 40

    def test_an_exposure_boost_no_longer_costs_the_highlights(self):
        """A bright area and a dark one; brighten two stops and take the bright
        one back. The dark area is opened while the bright one stays below
        white and keeps its texture."""
        rng = np.random.default_rng(2)
        rgb = np.full((64, 128, 3), 0.01, dtype=np.float32)
        rgb[:, 64:] = 0.7 * np.exp2(0.25 * rng.standard_normal((64, 64, 1))).astype(np.float32)
        out = np.asarray(develop_frame(_frame(rgb), {"exposure": 2.0, "highlights": -100}))
        assert out[:, :64, 1].mean() > 33            # the dark half was opened (from ~5)
        right = out[:, 64:, 1]
        assert right.mean() < 250                      # the bright half is not white
        assert right.std() > 3                         # and it still has texture


# ── Shadows ──────────────────────────────────────────────────────────

class TestShadows:
    def _scene(self) -> np.ndarray:
        rgb = np.empty((64, 128, 3), dtype=np.float32)
        rgb[:, :64] = 0.004      # a deep shadow
        rgb[:, 64:] = 0.18       # middle grey
        return rgb

    def test_lifting_shadows_opens_them_a_long_way(self):
        out = develop_frame(_frame(self._scene()), {"shadows": 100})
        before = develop_frame(_frame(self._scene()), {"exposure": 0.01})
        assert _px(out, 10, 10) - _px(before, 10, 10) > 25      # the deepest black

    def test_an_ordinary_shadow_is_opened_decisively(self):
        """Three stops under grey is where a backlit face lives."""
        shadow = _flat(0.18 * 2.0 ** -3)
        before = _px(develop_frame(shadow, {"exposure": 0.01}))
        assert _px(develop_frame(shadow, {"shadows": 100})) > before + 45

    def test_it_leaves_the_mid_tones_alone(self):
        out = develop_frame(_frame(self._scene()), {"shadows": 100})
        before = develop_frame(_frame(self._scene()), {"exposure": 0.01})
        assert abs(_px(out, 100, 10) - _px(before, 100, 10)) < 12

    def test_deepening_shadows_darkens_them(self):
        shadow = _flat(0.02)
        assert _px(develop_frame(shadow, {"shadows": -100})) < _px(
            develop_frame(shadow, {"exposure": 0.01})
        ) - 15

    def test_the_texture_of_a_lifted_shadow_survives(self):
        """Opening a shadow moves its level. It must not wash out the surface
        that is in it — that is what the detail layer is for."""
        rgb = _textured(0.01, 0.5)
        plain = np.asarray(develop_frame(_frame(rgb), {"exposure": 0.01}))[..., 1]
        lifted = np.asarray(develop_frame(_frame(rgb), {"shadows": 100}))[..., 1]
        assert lifted.mean() > plain.mean() + 30
        assert lifted.std() >= plain.std() * 0.9


# ── Squeezing a wide scene into a normal picture ─────────────────────

class TestCompress:
    def _backlit(self) -> np.ndarray:
        """A dark room with a bright window: about ten stops across."""
        rng = np.random.default_rng(4)
        rgb = np.empty((128, 128, 3), dtype=np.float32)
        rgb[:, :64] = (0.003 * np.exp2(0.3 * rng.standard_normal((128, 64, 1)))).astype(np.float32)
        rgb[:, 64:] = (2.5 * np.exp2(0.3 * rng.standard_normal((128, 64, 1)))).astype(np.float32)
        return rgb

    def test_it_pulls_the_ends_of_the_range_towards_each_other(self):
        frame = _frame(self._backlit())
        plain = np.asarray(develop_frame(frame, {"exposure": 0.01}))[..., 1].astype(float)
        squeezed = np.asarray(develop_frame(frame, {"compress": 100}))[..., 1].astype(float)
        gap_before = plain[:, 64:].mean() - plain[:, :64].mean()
        gap_after = squeezed[:, 64:].mean() - squeezed[:, :64].mean()
        assert gap_after < gap_before * 0.75

    def test_it_keeps_both_halves_inside_the_picture(self):
        out = np.asarray(develop_frame(_frame(self._backlit()), {"compress": 100}))[..., 1]
        assert out[:, :64].mean() > 25                  # the room is no longer black
        assert out[:, 64:].mean() < 245                 # the window is no longer white

    def test_local_detail_is_kept_through_the_squeeze(self):
        frame = _frame(self._backlit())
        plain = np.asarray(develop_frame(frame, {"exposure": 0.01}))[..., 1].astype(float)
        squeezed = np.asarray(develop_frame(frame, {"compress": 100}))[..., 1].astype(float)
        # Texture is judged where it is visible: in the window, which the plain
        # rendering clips to a flat white and the squeeze brings back into range.
        assert squeezed[:, 64:].std() > plain[:, 64:].std()
        # And the room's texture is not flattened by lifting it.
        assert squeezed[:, :64].std() > 0.7 * plain[:, :64].std()

    def test_no_halo_runs_along_a_hard_edge(self):
        """A bright sky over a dark ground, ten stops apart. A base layer that is
        blurred across the edge puts a bright or dark line along it once the
        squeeze is applied; the line is the artefact this guards against."""
        rgb = np.full((128, 256, 3), 0.004, dtype=np.float32)
        rgb[:64] = 3.0
        out = np.asarray(develop_frame(_frame(rgb), {"compress": 100, "highlights": -50}))[..., 1]
        sky = out[:64].astype(float)
        interior = sky[:40].mean()
        assert sky[56:64].max() <= interior + 6          # no bright rim above the edge
        ground = out[64:].astype(float)
        assert ground[:8].min() >= ground[24:].mean() - 6  # no dark rim below it


# ── The curve ────────────────────────────────────────────────────────

class TestToneCurve:
    EXTREMES = [
        {"shadows": 100, "highlights": -100, "whites": -100, "blacks": 100,
         "contrast": 100, "compress": 100},
        {"shadows": -100, "highlights": 100, "whites": 100, "blacks": -100,
         "contrast": -100},
        {"contrast": 100},
        {"compress": 100, "highlights": -100},
    ]

    @pytest.mark.parametrize("settings", EXTREMES)
    def test_the_curve_never_reverses_the_picture(self, settings):
        curve = tone_curve(settings)
        assert np.all(np.diff(curve) > 0)

    def test_contrast_alone_holds_middle_grey_still(self):
        curve = tone_curve({"contrast": 100})
        assert float(np.interp(0.0, render._CURVE_X, curve)) == pytest.approx(0.0, abs=0.02)

    def test_more_contrast_spreads_the_stops_about_grey(self):
        curve = tone_curve({"contrast": 100})
        up = float(np.interp(2.0, render._CURVE_X, curve))
        down = float(np.interp(-2.0, render._CURVE_X, curve))
        assert up > 2.0 and down < -2.0

    def test_each_control_owns_its_own_end_of_the_range(self):
        x = render._CURVE_X
        shadows = tone_curve({"shadows": 100}) - x
        highlights = tone_curve({"highlights": 100}) - x
        deep, bright = np.searchsorted(x, -4.0), np.searchsorted(x, 2.0)
        assert shadows[deep] > 1.0 and abs(shadows[bright]) < 0.05
        assert highlights[bright] > 1.0 and abs(highlights[deep]) < 0.05

    def test_the_whites_slider_is_narrower_than_highlights(self):
        x = render._CURVE_X
        mid = np.searchsorted(x, -0.5)
        assert abs((tone_curve({"whites": 100}) - x)[mid]) < abs(
            (tone_curve({"highlights": 100}) - x)[mid]
        )


# ── The look ─────────────────────────────────────────────────────────

class TestLook:
    def test_saturation_down_all_the_way_is_grey(self):
        frame = _flat(0.2, tint=(1.4, 1.0, 0.6))
        out = np.asarray(develop_frame(frame, {"saturation": -100})).astype(int)
        assert np.abs(out[..., 0] - out[..., 2]).max() <= 2

    def test_vibrance_moves_a_flat_colour_more_than_a_saturated_one(self):
        flat = _flat(0.2, tint=(1.1, 1.0, 0.95))
        loud = _flat(0.2, tint=(2.2, 0.6, 0.3))

        def spread(frame):
            """How much more colourful the frame got, relative to how colourful
            it was: vibrance is proportional, not a flat addition."""
            a = np.asarray(frame.encoded()).astype(float)
            b = np.asarray(develop_frame(frame, {"vibrance": 100})).astype(float)
            before = (a.max(axis=2) - a.min(axis=2)).mean()
            return (b.max(axis=2) - b.min(axis=2)).mean() / before

        assert spread(flat) > spread(loud)

    def test_fade_lifts_the_blacks(self):
        out = develop_frame(_flat(0.0005), {"fade": 100})
        assert _px(out) > 18

    def test_split_toning_tints_each_end_with_its_own_colour(self):
        rgb = np.empty((32, 64, 3), dtype=np.float32)
        rgb[:, :32] = 0.01
        rgb[:, 32:] = 0.8
        out = np.asarray(develop_frame(_frame(rgb), {
            "shadow_hue": 220, "shadow_sat": 100,   # blue in the shadows
            "highlight_hue": 40, "highlight_sat": 100,  # amber in the highlights
        })).astype(int)
        shadow, light = out[10, 5], out[10, 50]
        assert shadow[2] > shadow[0]
        assert light[0] > light[2]

    def test_a_look_alone_does_not_move_the_tones(self):
        frame = _flat(0.18)
        plain = _px(develop_frame(frame, {"exposure": 0.01}))
        toned = _px(develop_frame(frame, {"vibrance": 60, "saturation": 20}))
        assert abs(plain - toned) < 4


# ── White balance and set matching ───────────────────────────────────

class TestColourAndMatching:
    def test_warmth_adds_red_and_takes_blue(self):
        out = np.asarray(develop_frame(_flat(0.2), {"temperature": 100}))
        assert out[..., 0].mean() > out[..., 2].mean() + 20

    def test_a_match_is_translated_not_reinterpreted(self):
        """Matching measures on encoded values. A half-stop of match is the same
        change of light as a stop and a bit in linear light, whichever way the
        frame is brightened."""
        frame = _flat(0.1)
        matched = develop_frame(frame, {"exposure": 0.0, "clarity": 1}, matched={"ev": 0.5})
        direct = develop_frame(frame, {"exposure": 0.5 / 0.45})
        assert abs(_px(matched) - _px(direct)) <= 2


# ── Auto ─────────────────────────────────────────────────────────────

class TestAuto:
    def test_a_frame_shot_under_is_brought_up(self):
        out = auto_develop(_frame(_textured(0.01, 0.6)))
        assert out.get("exposure", 0) > 1.0

    def test_a_frame_with_blown_highlights_gets_them_pulled_back(self):
        rng = np.random.default_rng(5)
        rgb = (0.15 * np.exp2(1.5 * rng.standard_normal((96, 96, 1)))).astype(np.float32)
        rgb = np.repeat(rgb, 3, axis=2)
        rgb[:30] = 2.5
        out = auto_develop(_frame(rgb))
        assert out.get("highlights", 0) < -20

    def test_a_frame_that_is_already_right_is_left_alone(self):
        assert auto_develop(_frame(_textured(0.14, 0.35))) == {}

    def test_a_wide_scene_is_offered_the_squeeze(self):
        rng = np.random.default_rng(6)
        rgb = np.empty((96, 96, 3), dtype=np.float32)
        rgb[:, :48] = (0.002 * np.exp2(0.4 * rng.standard_normal((96, 48, 1)))).astype(np.float32)
        rgb[:, 48:] = (1.5 * np.exp2(0.4 * rng.standard_normal((96, 48, 1)))).astype(np.float32)
        out = auto_develop(_frame(rgb))
        assert out.get("compress", 0) > 0 and out.get("shadows", 0) > 0

    def test_its_suggestion_is_a_valid_edit(self):
        out = auto_develop(_frame(_textured(0.01, 0.6)))
        assert normalize(out) == out


# ── The settings model ───────────────────────────────────────────────

class TestSettingsModel:
    NEW = {
        "compress": 40, "clarity": -20, "vibrance": 30, "saturation": -10, "fade": 25,
        "shadow_hue": 215, "shadow_sat": 30, "highlight_hue": 40, "highlight_sat": 20,
    }

    def test_the_new_controls_are_kept(self):
        assert normalize(self.NEW) == self.NEW

    def test_one_way_ranges_do_not_go_negative(self):
        assert normalize({"compress": -50, "fade": -3}) == {}

    def test_a_hue_without_a_strength_is_not_an_edit(self):
        assert normalize({"shadow_hue": 200}) == {}
        assert normalize({"shadow_hue": 200, "shadow_sat": 10}) == {
            "shadow_hue": 200, "shadow_sat": 10,
        }

    def test_the_whole_edit_survives_the_sidecar(self):
        edit = {"exposure": 1.35, "highlights": -70, "tint": 7, **self.NEW}
        assert from_crs(to_crs(edit)) == normalize(edit)

    def test_the_names_lightroom_reads_are_used_where_they_exist(self):
        out = to_crs({"clarity": 20, "vibrance": 10, "saturation": -5})
        assert out["crs:Clarity2012"] == "+20"
        assert out["crs:Vibrance"] == "+10"
        assert out["crs:Saturation"] == "-5"

    def test_an_edit_saved_before_the_new_controls_still_reads(self):
        assert normalize({"exposure": 0.7, "shadows": 30}) == {"exposure": 0.7, "shadows": 30}


def test_a_frame_shrunk_to_a_size_is_that_size_on_its_long_side():
    from adjust.render import resize_linear

    # 2048/6000 * 6000 truncates to 2047 in floating point.
    assert resize_linear(np.zeros((4000, 6000, 3), dtype=np.float32), 2048).shape[:2] == (1365, 2048)
