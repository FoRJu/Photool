"""A mask's blur (adjust.defocus): out of focus where the mask reaches, as a lens would."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from adjust import defocus  # noqa: E402
from adjust.masks import active, normalize_mask  # noqa: E402
from adjust.render import LinearFrame, develop_frame, mask_reach  # noqa: E402


def stripes(w=300, h=200):
    rgb = np.full((h, w, 3), 0.05, np.float32)
    rgb[:, ::8] = 0.8
    return rgb


def detail(rgb):
    return float(np.abs(np.diff(rgb[..., 1], axis=1)).mean())


# Whole on the left half, nothing on the right.
LEFT = {"kind": "linear", "x1": 0.4, "y1": 0.5, "x2": 0.41, "y2": 0.5}


class TestSettings:
    def test_kept_in_range(self):
        assert normalize_mask({**LEFT, "blur": 40})["blur"] == 40
        assert normalize_mask({**LEFT, "blur": 400})["blur"] == 100
        assert "blur" not in normalize_mask({**LEFT, "blur": -30})

    def test_a_mask_that_only_blurs_is_an_edit(self):
        assert active([normalize_mask({**LEFT, "blur": 30})])


class TestBlur:
    def test_blurs_where_it_reaches_and_nowhere_else(self):
        rgb = stripes()
        frame = LinearFrame(rgb)
        out = defocus.defocused(frame, [normalize_mask({**LEFT, "blur": 60})], mask_reach).rgb
        assert detail(out[:, 10:100]) < detail(rgb[:, 10:100]) * 0.2
        assert np.array_equal(out[:, 140:], rgb[:, 140:])

    def test_what_it_does_not_reach_is_not_spread_into_it(self):
        """A bright thing left out of a background mask does not glow into the background."""
        rgb = np.full((100, 200, 3), 0.05, np.float32)
        rgb[:, 125:] = 1.0                      # the subject, right, left out of the mask
        frame = LinearFrame(rgb)
        out = defocus.defocused(frame, [normalize_mask({**LEFT, "blur": 80})], mask_reach).rgb
        assert out[:, :75].max() < 0.06

    def test_a_bright_point_opens_into_a_disc(self):
        rgb = np.full((101, 101, 3), 0.01, np.float32)
        rgb[50, 50] = 50.0
        whole = {"kind": "linear", "x1": 0.5, "y1": 2.0, "x2": 0.5, "y2": 2.5}   # reaches everywhere
        out = defocus.defocused(LinearFrame(rgb), [normalize_mask({**whole, "blur": 100})], mask_reach).rgb
        lit = out[..., 0] > 0.02
        ys, xs = np.nonzero(lit)
        # About a disc: as wide as high, its middle not much brighter than its edge.
        assert abs((ys.max() - ys.min()) - (xs.max() - xs.min())) <= 2
        assert out[50, 50, 0] < out[50, xs.max() - 1, 0] * 2.5

    def test_a_part_cut_wider_is_blurred_as_the_whole(self):
        rgb = stripes(400, 300)
        masks = [normalize_mask({**LEFT, "blur": 50})]
        whole = defocus.pixels(rgb, (0, 0, 1, 1), masks, lambda m: mask_reach(LinearFrame(rgb), m))
        margin = defocus.reach(masks, 400)
        x0, x1, y0, y1 = 100 - margin, 200 + margin, 100 - margin, 200 + margin
        window = (x0 / 400, y0 / 300, x1 / 400, y1 / 300)
        part = rgb[y0:y1, x0:x1]
        probe = LinearFrame(part, window=window)
        got = defocus.pixels(part, window, masks, lambda m: mask_reach(probe, m))
        inner = got[margin:-margin, margin:-margin]
        assert np.abs(inner - whole[100:200, 100:200]).max() < 0.02

    def test_drawn_through_develop(self):
        rgb = stripes()
        out = np.asarray(develop_frame(LinearFrame(rgb), {"masks": [{**LEFT, "blur": 60}]}), np.float32)
        plain = np.asarray(develop_frame(LinearFrame(rgb), {"masks": [{**LEFT, "exposure": 0.001}]}), np.float32)
        assert detail(out[:, 10:100]) < detail(plain[:, 10:100]) * 0.3
        assert np.abs(out[:, 150:] - plain[:, 150:]).max() <= 1.0
