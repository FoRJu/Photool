"""
White balance in Kelvin and tint (adjust.whitebalance): Lightroom's numbers, the
re-balance of a RAW as a fresh decode at that white would make it, the grey
picker, and the edit model around them (normalize, the sidecar, paste).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import whitebalance as wb  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402
from export.engine import _srgb_from_camera  # noqa: E402
from export.paste_match import adjuster, relative_white  # noqa: E402

# A Canon EOS 5D Mark III's ColorMatrix (XYZ to camera), as LibRaw carries it.
CAM_XYZ = np.array([
    [6722, -635, -963],
    [-4287, 12460, 2028],
    [-908, 2162, 5668],
]) / 10000.0
SRGB_FROM_CAM = _srgb_from_camera(SimpleNamespace(num_colors=3, rgb_xyz_matrix=np.vstack([CAM_XYZ, [0, 0, 0]])))


def _neutral(kelvin, tint=0.0):
    n = CAM_XYZ @ wb.xyz_of(wb.xy_of(kelvin, tint))
    return n / n[1]


def _raw_wb(kelvin=5500, tint=0.0):
    """A RAW shot with its camera's white balance set to ``kelvin``, ``tint``."""
    return wb.of_camera(CAM_XYZ, 1.0 / _neutral(kelvin, tint), SRGB_FROM_CAM)


def _grey_lit_by(kelvin, tint, shot: wb.WhiteBalance, level=0.3):
    """A grey card lit by ``kelvin``, decoded at ``shot``'s white balance: the
    linear sRGB the frame holds."""
    camera = CAM_XYZ @ wb.xyz_of(wb.xy_of(kelvin, tint)) * level
    return SRGB_FROM_CAM @ (camera / shot.neutral)


def _spread(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return float(rgb.max() / rgb.min() - 1.0)


def _frame(rgb, white=None, size=24):
    frame = LinearFrame(np.tile(np.asarray(rgb, dtype=np.float32), (size, size, 1)))
    frame.wb = (lambda: white) if white is not None else None
    return frame


class TestKelvinAndTint:
    def test_round_trip_through_xy(self):
        for kelvin in np.geomspace(2000, 25000, 40):
            for tint in range(-100, 101, 20):
                back, t = wb.kelvin_tint_of(*wb.xy_of(kelvin, tint))
                assert abs(1e6 / back - 1e6 / kelvin) < 1.0
                assert abs(t - tint) < 0.5

    def test_illuminant_a(self):
        kelvin, tint = wb.kelvin_tint_of(0.4476, 0.4074)
        assert abs(kelvin - 2856) <= 15
        assert abs(tint) <= 3

    def test_d65_reads_as_lightroom_reads_daylight(self):
        kelvin, tint = wb.kelvin_tint_of(*wb.D65_XY)
        assert abs(kelvin - 6504) < 20 and 5 < tint < 15


class TestTheRebalance:
    def test_a_grey_lit_by_one_light_and_set_to_it_comes_out_grey(self):
        shot = _raw_wb(5500)
        assert abs(shot.as_shot[0] - 5500) < 2 and abs(shot.as_shot[1]) < 0.5
        for light, tint in ((2850, 0), (3800, 21), (7500, 10), (12000, -20)):
            grey = _grey_lit_by(light, tint, shot)
            assert _spread(grey) > 0.05          # cast, as shot
            out = wb.matrix_for(shot, light, tint) @ grey
            assert _spread(out) < 0.005

    def test_direction(self):
        grey = np.full(3, 0.18)
        for white in (_raw_wb(5500), None):
            k0, t0 = wb.as_shot(white)
            warmer = wb.matrix_for(white, k0 * 1.4, t0) @ grey
            cooler = wb.matrix_for(white, k0 / 1.4, t0) @ grey
            assert warmer[0] / warmer[2] > 1.05 and cooler[0] / cooler[2] < 0.95
            magenta = wb.matrix_for(white, k0, t0 + 30) @ grey
            assert magenta[1] < magenta[0] and magenta[1] < magenta[2]

    def test_as_shot_is_no_pass(self):
        shot = _raw_wb(5200, 8)
        k, t = shot.as_shot
        assert wb.matrix_for(shot, k, t) is None
        assert wb.matrix_for(shot, 1e6 / (1e6 / k + 0.9), t + 0.9) is None
        assert wb.matrix_for(shot, 1e6 / (1e6 / k + 1.5), t) is not None


class TestThePicker:
    @pytest.mark.parametrize("white", [_raw_wb(4800, 5), None])
    def test_the_answer_makes_the_patch_neutral(self, white):
        if white is None:
            patch = np.array([0.32, 0.27, 0.19])
        else:
            patch = _grey_lit_by(3300, 12, white)
        kelvin, tint = wb.white_point(white, patch)
        assert isinstance(kelvin, int) and isinstance(tint, int)
        assert _spread(wb.matrix_for(white, kelvin, tint) @ patch) < 0.005

    def test_the_answer_applied_in_a_render_is_neutral(self):
        white = _raw_wb(5500)
        patch = _grey_lit_by(3000, 0, white, level=0.25)
        kelvin, tint = wb.white_point(white, patch)
        out = np.asarray(develop_frame(_frame(patch, white), {"wb_kelvin": kelvin, "wb_tint": tint}),
                         dtype=np.float64)
        assert np.ptp(out.reshape(-1, 3).mean(axis=0)) <= 1.0

    def test_refusals(self):
        with pytest.raises(ValueError, match="clipped"):
            wb.white_point(None, [0.99, 0.6, 0.5])
        with pytest.raises(ValueError, match="dark"):
            wb.white_point(None, [0.001, 0.001, 0.001])
        with pytest.raises(ValueError, match="grey"):
            wb.white_point(None, [0.6, 0.05, 0.02])


class TestRendering:
    def test_custom_equal_to_as_shot_is_bit_identical_to_none(self):
        white = _raw_wb(5300, 6)
        rng = np.random.default_rng(1)
        rgb = rng.uniform(0.02, 0.8, (32, 32, 3)).astype(np.float32)
        k, t = white.as_shot
        frame = LinearFrame(rgb)
        frame.wb = lambda: white
        plain = np.asarray(develop_frame(frame, {}))
        same = np.asarray(develop_frame(frame, {"wb_kelvin": round(k), "wb_tint": round(t)}))
        assert np.array_equal(plain, same)
        edited = {"contrast": 20, "temperature": 10}
        assert np.array_equal(np.asarray(develop_frame(frame, edited)),
                              np.asarray(develop_frame(frame, {**edited, "wb_kelvin": round(k),
                                                               "wb_tint": round(t)})))

    def test_old_warmth_and_tint_render_as_they_did(self):
        rgb = np.random.default_rng(2).uniform(0.02, 0.8, (32, 32, 3)).astype(np.float32)
        old = {"temperature": 12, "tint": -3}
        without = np.asarray(develop_frame(LinearFrame(rgb.copy()), old))
        with_wb = LinearFrame(rgb.copy())
        with_wb.wb = lambda: _raw_wb(4000)
        assert np.array_equal(without, np.asarray(develop_frame(with_wb, old)))
        # The gains they always were: red up, blue down, by the old slider's reach.
        flat = np.asarray(develop_frame(_frame([0.18, 0.18, 0.18]), {"temperature": 100}), dtype=float)
        assert flat[..., 0].mean() > flat[..., 2].mean() + 20

    def test_a_patch_and_a_draft_read_the_whole_frame_s_white(self):
        from adjust.render import draft_of, healed

        white = _raw_wb(5000)
        frame = _frame([0.2, 0.2, 0.2], white, size=40)
        assert wb.of_frame(draft_of(frame)) is white
        assert wb.of_frame(healed(frame, [{"x": 0.5, "y": 0.5, "r": 0.05, "sx": 0.2, "sy": 0.2}])) is white
        child = LinearFrame(frame.rgb[:4, :4], anchor_of=frame)
        assert wb.of_frame(child) is white


class TestTheEdit:
    def test_normalize(self):
        assert normalize({"wb_tint": 20}) == {}
        assert normalize({"wb_kelvin": 900, "wb_tint": 400}) == {"wb_kelvin": 2000, "wb_tint": 150}
        assert normalize({"wb_kelvin": 99999.4, "wb_tint": -12.6}) == {"wb_kelvin": 50000, "wb_tint": -13}
        assert normalize({"wb_kelvin": 5500, "wb_tint": 0}) == {"wb_kelvin": 5500}
        assert normalize({"wb_kelvin": None, "wb_tint": None}) == {}
        # A zero is no colour temperature, not one to clamp up to 2000 K.
        assert normalize({"wb_kelvin": 0, "wb_tint": 12}) == {}
        assert normalize({"wb_kelvin": -5, "exposure": 1}) == {"exposure": 1.0}

    def test_sidecar_round_trip(self):
        for edit in ({"wb_kelvin": 5500, "wb_tint": 10}, {"wb_kelvin": 2850},
                     {"wb_kelvin": 7500, "wb_tint": -150, "temperature": 12, "tint": -3}):
            crs = to_crs(edit)
            assert "crs:Temperature" not in crs
            assert crs["crs:WBKelvin"] == str(edit["wb_kelvin"])
            assert from_crs(crs) == normalize(edit)


class TestPaste:
    def test_keep_light_keeps_the_frame_s_own(self):
        keep = adjuster(Path("."), None, {}, "keep_light")
        pasted = {"wb_kelvin": 3000, "wb_tint": 5, "contrast": 20}
        out, _ = keep("b.CR3", pasted, {"wb_kelvin": 6100, "wb_tint": -4})
        assert out == {"wb_kelvin": 6100, "wb_tint": -4, "contrast": 20}
        out, _ = keep("b.CR3", pasted, {})
        assert out == {"contrast": 20}

    def test_matching_moves_the_target_s_own_as_shot_by_the_copied_difference(self):
        # Copied: shot at 5000 K, set 250 mireds... set to 4000 K (+50 mireds) and +6 tint.
        got = relative_white({"wb_kelvin": 4000, "wb_tint": 8}, (5000, 2), (6250, -1))
        assert abs(1e6 / got["wb_kelvin"] - (1e6 / 6250 + 50)) < 0.5
        assert got["wb_tint"] == 5
        assert relative_white({"exposure": 1}, (5000, 2), (6250, -1)) == {}
