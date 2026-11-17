"""
Colour grading, the vignette's shape, dehaze and texture (brief 3).

The first thing held still is the old picture: an edit that carries none of the
new keys renders bit for bit as it did before they were added. The hashes below
were taken on the code before the change, from a fixed synthetic frame.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.render import LinearFrame, develop_frame  # noqa: E402


def scene(w=240, h=180, seed=7) -> np.ndarray:
    """A fixed linear frame: a sky-to-ground ramp, saturated patches, and fine noise."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = 0.02 + 0.9 * (1 - y) * (0.6 + 0.4 * x)
    rgb[..., 1] = 0.03 + 0.7 * (1 - y) * (0.8 - 0.3 * x)
    rgb[..., 2] = 0.05 + 0.5 * (1 - y) + 0.1 * x
    rgb[60:110, 30:90] = (0.6, 0.08, 0.04)
    rgb[60:110, 140:200] = (0.05, 0.3, 0.6)
    rgb[130:170, 80:160] = (1.6, 1.4, 1.1)
    rgb *= 1.0 + 0.08 * rng.standard_normal((h, w, 1)).astype(np.float32)
    return np.clip(rgb, 0.0, None).astype(np.float32)


def digest(edit: dict) -> str:
    out = develop_frame(LinearFrame(scene()), edit)
    return hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16]


BAND = {"kind": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.6}

# The edits as they were before this change, and the picture each made.
OLD_EDITS = {
    "split toning": {"shadow_hue": 210, "shadow_sat": 40, "highlight_hue": 40, "highlight_sat": 30},
    "vignette": {"vignette": -45},
    "vignette light": {"vignette": 30, "grain": 20},
    "clarity": {"clarity": 50, "contrast": 20, "shadows": 30},
    "masks": {"exposure": 0.3, "masks": [dict(BAND, exposure=-0.8, clarity=40, saturation=-30)]},
    "everything": {"exposure": -0.2, "clarity": -30, "fade": 20, "shadow_hue": 30, "shadow_sat": 25,
                   "vignette": -20, "sharpen": 40,
                   "masks": [dict(BAND, contrast=30, temperature=20)]},
}
OLD_HASHES = {
    "clarity": "57dd14c7811964f2",
    "everything": "4706c966e264c2d3",
    "masks": "2e6184f87d92ca51",
    "split toning": "36cad66dd62a70db",
    "vignette": "6ed57ff798baed86",
    "vignette light": "1b62f136456f5868",
}


@pytest.mark.parametrize("name", sorted(OLD_EDITS))
def test_an_old_edit_renders_as_before(name):
    assert digest(OLD_EDITS[name]) == OLD_HASHES[name]


# ── Colour grading ───────────────────────────────────────────────────

from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import _look_of, _vignette  # noqa: E402

GREYS = np.linspace(0.0, 1.0, 101, dtype=np.float32)[None, :, None].repeat(3, axis=2)


def graded(edit):
    return _look_of(GREYS.copy(), edit)[0]


class TestGrading:
    SPLIT = {"shadow_hue": 220, "shadow_sat": 50, "highlight_hue": 40, "highlight_sat": 50}

    def test_balance_zero_and_no_midtone_is_the_old_picture(self):
        assert np.array_equal(graded(dict(self.SPLIT, tone_balance=0, midtone_hue=90)), graded(self.SPLIT))
        assert digest(dict(OLD_EDITS["split toning"], tone_balance=0, midtone_hue=120)) \
            == OLD_HASHES["split toning"]

    def test_a_midtone_tint_moves_mid_grey_and_leaves_black_and_white(self):
        out = graded({"midtone_hue": 0, "midtone_sat": 60})
        mid = out[50]
        assert mid[0] - mid[2] > 0.08                 # mid grey turns red
        assert np.array_equal(out[0], GREYS[0, 0]) and np.array_equal(out[100], GREYS[0, 100])
        # The tint fades toward the ends, and is a colour, not a brightness.
        spread = out.max(axis=1) - out.min(axis=1)
        assert spread[5] < spread[25] < spread[50] and spread[95] < spread[75] < spread[50]
        assert abs(float(mid.mean()) - 0.5) < 1e-3

    @staticmethod
    def crossover(edit):
        """The grey at which the shadows' blue hands over to the highlights' orange."""
        out = graded(edit)
        warm = out[:, 0] - out[:, 2]
        return int(np.argmax(warm > 0))

    def test_balance_moves_the_crossover(self):
        plain = self.crossover(self.SPLIT)
        left = self.crossover(dict(self.SPLIT, tone_balance=-100))
        right = self.crossover(dict(self.SPLIT, tone_balance=100))
        assert 45 <= plain <= 55
        assert left > plain + 15                      # the shadows' tint has more of the range
        assert right < plain - 15

    def test_the_balance_needs_a_tint_and_the_midtone_hue_its_strength(self):
        assert normalize({"tone_balance": 40}) == {}
        assert normalize({"tone_balance": 40, "highlight_sat": 10})["tone_balance"] == 40
        assert normalize({"midtone_hue": 40}) == {}


# ── The vignette's shape ─────────────────────────────────────────────

FLAT = np.full((181, 241, 3), 0.5, np.float32)
WHOLE = (0.0, 0.0, 1.0, 1.0)


class TestVignette:
    def test_size_and_feather_at_zero_are_the_old_picture(self):
        assert np.array_equal(_vignette(FLAT, -60, WHOLE, 0, 0), _vignette(FLAT, -60, WHOLE))
        for name in ("vignette", "vignette light"):
            edit = dict(OLD_EDITS[name], vignette_size=0, vignette_feather=0)
            assert digest(edit) == OLD_HASHES[name]

    @pytest.mark.parametrize("size,feather", [(-100, 0), (100, 0), (0, -100), (0, 100), (-100, 100), (-100, -100)])
    def test_the_centre_is_never_changed(self, size, feather):
        for amount in (-100, 100):
            out = _vignette(FLAT, amount, WHOLE, size, feather)
            assert np.array_equal(out[89:92, 119:122], FLAT[89:92, 119:122])

    def test_size_reaches_further_in_or_keeps_to_the_corners(self):
        def darkening(size):
            return 0.5 - _vignette(FLAT, -80, WHOLE, size, 0)[90, 30, 0]    # part way out, on the middle row
        assert darkening(-100) > darkening(0) + 0.05 > darkening(100) + 0.05
        corner = [_vignette(FLAT, -80, WHOLE, s, 0)[0, 0, 0] for s in (-100, 0, 100)]
        assert max(corner) - min(corner) < 0.02          # the corners darken as much

    def test_feather_makes_the_edge_harder_or_softer(self):
        def steepest(feather):
            out = _vignette(FLAT, -100, WHOLE, 0, feather)
            diagonal = out[np.arange(181), np.round(np.arange(181) * 240 / 180).astype(int), 0]
            return float(np.abs(np.diff(diagonal)).max())
        assert steepest(-100) > 2 * steepest(0) > 2 * steepest(100)


def test_the_sidecar_keeps_the_new_keys():
    edit = {"shadow_hue": 200, "shadow_sat": 20, "midtone_hue": 35, "midtone_sat": 30, "tone_balance": -40,
            "vignette": -30, "vignette_size": -50, "vignette_feather": 60}
    crs = to_crs(edit)
    assert crs["crs:ColorGradeMidtoneHue"] == "35" and crs["crs:SplitToningBalance"] == "-40"
    back = from_crs(crs)
    assert {k: back.get(k) for k in edit} == edit


def test_controls_say_what_each_new_key_means():
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from commands.develop import controls

    listed = {c["key"]: c for c in controls()}
    for key in NEW_KEYS:
        assert key in listed and listed[key]["meaning"], key
        assert "\n" not in listed[key]["meaning"]
    assert (listed["midtone_hue"]["min"], listed["midtone_hue"]["max"]) == (0.0, 360.0)
    assert (listed["tone_balance"]["min"], listed["tone_balance"]["max"]) == (-100.0, 100.0)


NEW_KEYS = ("midtone_hue", "midtone_sat", "tone_balance", "vignette_size", "vignette_feather")


def test_a_lut_holds_the_grading():
    from adjust.lut import split

    point, left_out = split({"midtone_hue": 40, "midtone_sat": 30, "shadow_sat": 10, "tone_balance": 20,
                             "vignette": -20, "vignette_size": 30})
    assert set(point) == {"midtone_hue", "midtone_sat", "shadow_sat", "tone_balance"}
    assert left_out == ["Vignette"]
