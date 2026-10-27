"""
Spot removal (adjust.spots): a spot is painted over from a clean place nearby,
in the light and colour of its own surroundings.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import describe_settings, from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LUMA, LinearFrame, develop_frame, healed, region_frame  # noqa: E402
from adjust.spots import apply_spots, find_source, heal_patch, normalize_spots  # noqa: E402


def _scene(h=120, w=180, seed=0):
    """A sky getting brighter to the right, with a little grain, and a dark
    speck of dust at (60, 60)."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.08, 0.32, w, dtype=np.float32)[None, :, None]
    rgb = np.broadcast_to(ramp * np.array([0.8, 0.9, 1.2], np.float32), (h, w, 3)).copy()
    rgb *= 1.0 + 0.03 * rng.standard_normal((h, w, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    dust = np.hypot(xx - 60, yy - 60) <= 3
    rgb[dust] *= 0.2
    return rgb, dust


DUST = {"x": 60.5 / 180, "y": 60.5 / 120, "r": 5 / 180, "sx": 110.5 / 180, "sy": 60.5 / 120}


class TestHeal:
    def test_the_speck_is_gone_and_the_light_is_the_spots_own(self):
        rgb, dust = _scene()
        out = apply_spots(rgb, normalize_spots([DUST]))
        around = rgb[50:71, 50:71][~dust[50:71, 50:71]].mean(axis=0)
        # The speck's pixels now look like the sky round it, not like the
        # brighter sky 50 px to the right that they were taken from.
        assert np.allclose(out[dust].mean(axis=0), around, rtol=0.08)
        assert out[dust].mean() < rgb[60, 110].mean() * 0.85
        # Nothing outside the circle and its soft edge is touched.
        assert np.array_equal(out[:, 80:100], rgb[:, 80:100])

    def test_a_clone_is_the_source_as_it_is(self):
        rgb, _ = _scene()
        out = apply_spots(rgb, normalize_spots([{**DUST, "mode": "clone", "feather": 0}]))
        assert np.allclose(out[58:63, 58:63], rgb[58:63, 108:113])

    def test_opacity_paints_part_way(self):
        dst = np.full((21, 21, 3), 0.1, np.float32)
        src = np.full((21, 21, 3), 0.5, np.float32)
        half = heal_patch(dst, src, 4, "clone", feather=0, opacity=50)
        assert np.allclose(half[10, 10], 0.3)

    def test_a_loupe_patch_agrees_with_the_whole_frame(self):
        rgb, _ = _scene()
        spots = normalize_spots([DUST])
        whole = apply_spots(rgb, spots)
        top, left, side = 40, 40, 40          # the source (x 110) is outside the patch
        h, w = rgb.shape[:2]

        def fetch(x0, y0, size):
            xs = np.clip(np.arange(x0, x0 + size), 0, w - 1)
            ys = np.clip(np.arange(y0, y0 + size), 0, h - 1)
            return rgb[np.ix_(ys, xs)]

        window = (left / w, top / h, (left + side) / w, (top + side) / h)
        patch = apply_spots(rgb[top:top + side, left:left + side], spots, window, fetch)
        assert np.allclose(patch, whole[top:top + side, left:left + side], atol=1e-5)


class TestFindSource:
    def test_a_source_is_clean_nearby_and_inside_the_frame(self):
        rgb, _ = _scene()
        # A second speck where a lazy choice would take from.
        yy, xx = np.mgrid[0:120, 0:180]
        rgb[np.hypot(xx - 76, yy - 60) <= 3] *= 0.2
        sx, sy = find_source(rgb, DUST["x"], DUST["y"], DUST["r"])
        px, py = sx * 180, sy * 120
        assert 7 <= px <= 172 and 7 <= py <= 112
        assert np.hypot(px - 76, py - 60) > 8           # not on the other speck
        assert np.hypot(px - 60, py - 60) < 40           # but near the spot


class TestEdit:
    def test_cleaned_given_a_source_and_round_tripped(self):
        clean = normalize({"spots": [{"x": 0.99, "y": 0.5, "r": 2, "mode": "smudge", "opacity": 140},
                                     {"y": 0.2}, "junk"]})
        (spot,) = clean["spots"]
        assert spot["r"] == 0.2 and spot["mode"] == "heal" and "opacity" not in spot
        assert spot["sx"] < spot["x"]                     # no room to the right
        assert from_crs(to_crs(clean)) == clean
        assert "1 spot removed" in describe_settings(clean)

    def test_develop_removes_it_before_the_rest_of_the_edit(self):
        rgb, dust = _scene()
        frame = LinearFrame(rgb)
        out = np.asarray(develop_frame(frame, {"spots": [DUST], "exposure": 0.5}), np.float32) @ LUMA
        plain = np.asarray(develop_frame(frame, {"exposure": 0.5}), np.float32) @ LUMA
        assert out[dust].mean() > plain[dust].mean() + 20
        # Kept: a second render with another slider does not heal again.
        first = healed(frame, normalize_spots([DUST]))
        assert healed(frame, normalize_spots([DUST])) is first

    def test_a_spot_only_edit_is_still_an_edit(self):
        rgb, dust = _scene()
        frame = LinearFrame(rgb)
        out = np.asarray(develop_frame(frame, {"spots": [DUST]}), np.float32) @ LUMA
        plain = np.asarray(frame.encoded(), np.float32) @ LUMA
        assert out[dust].mean() > plain[dust].mean() + 20
