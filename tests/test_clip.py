"""
Clipped highlights and the pink they turn into.

A sensor clips one channel at a time; white balance then boosts red and blue past
the clipped green, and a flash-blown face comes out magenta. These tests build
that situation (a skin-coloured highlight whose brightness peaks well past what
the sensor holds) and check the colour of the clipped area comes from the skin
around it rather than from the clipping.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.clip import clip_weight, reconstruct_highlights, saturation_levels
from adjust.render import LUMA, LinearFrame, develop_frame
from export import engine
from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng, tifffile  # noqa: F401

GAINS = np.array([2.0, 1.0, 1.67])
NOMINAL = GAINS / GAINS.min()
SKIN = np.array([1.25, 1.0, 0.8])          # a warm surface, relative to its green


def _face(sat=0.88, size=(120, 180)):
    """A skin-coloured highlight, in white-balanced linear light as the decoder
    returns it: brightness peaks near 3.3 and each channel saturates at ``sat`` of
    its own level."""
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot((x - w / 2) / (w * 0.32), (y - h / 2) / (h * 0.40))
    v = 0.12 + 3.2 * np.exp(-(r ** 2) * 1.6)
    true = v[..., None] * SKIN
    observed = np.minimum(true, sat * NOMINAL)
    return true.astype(np.float32), observed.astype(np.float32)


def _ratios(px):
    """Colour as ratios to green: what a hue is, once brightness is taken out."""
    px = np.asarray(px, dtype=float)
    return px[..., 0] / px[..., 1], px[..., 2] / px[..., 1]


class TestFindingTheSaturationLevel:
    def test_a_plateau_below_the_stated_level_is_found(self):
        """Real sensors often saturate a little under the file's white level.
        Trusting the file there is how the clipping gets missed."""
        _, observed = _face(sat=0.88)
        levels = saturation_levels(observed, NOMINAL)
        assert levels[1] == pytest.approx(0.88, abs=0.005)
        assert levels[0] == pytest.approx(0.88 * NOMINAL[0], abs=0.01)

    def test_a_smooth_bright_surface_is_not_mistaken_for_a_plateau(self):
        """A white wall with a little noise has no pile of pixels at its top."""
        rng = np.random.default_rng(1)
        wall = 0.8 + 0.02 * rng.standard_normal((120, 180, 1))
        rgb = np.repeat(wall, 3, axis=2).astype(np.float32)
        assert np.allclose(saturation_levels(rgb, NOMINAL), NOMINAL)

    def test_a_frame_with_nothing_clipped_comes_back_untouched(self):
        true, _ = _face()
        rgb = (true * 0.1).astype(np.float32)
        assert reconstruct_highlights(rgb, NOMINAL) is rgb


class TestTheColourOfAClippedArea:
    def test_without_the_fix_the_clipped_area_is_pink(self):
        """The problem, stated: green stops at its ceiling and red and blue go on."""
        _, observed = _face()
        centre = observed[60, 90]
        r, b = _ratios(centre)
        skin_r, skin_b = _ratios(SKIN)
        # Red and blue both far above where skin has them, against a green that
        # stopped: that is a magenta cast.
        assert r > skin_r + 0.4 and b > skin_b + 0.4

    def test_the_clipped_area_takes_the_colour_of_the_skin_around_it(self):
        true, observed = _face()
        fixed = reconstruct_highlights(observed, NOMINAL)
        r, b = _ratios(fixed[60, 90])
        skin_r, skin_b = _ratios(SKIN)
        assert r == pytest.approx(skin_r, abs=0.06)
        assert b == pytest.approx(skin_b, abs=0.06)

    def test_it_holds_after_the_exposure_is_pulled_down(self):
        """The reported symptom: pull down a stop or two and the highlight is pink."""
        true, observed = _face()
        before = develop_frame(LinearFrame(observed), {"exposure": -1.5})
        after = develop_frame(LinearFrame(reconstruct_highlights(observed, NOMINAL)),
                              {"exposure": -1.5})
        core, ring = (60, 90), (60, 90 + 70)
        skin_r, skin_b = _ratios(np.asarray(after, dtype=float)[ring])
        pink = _ratios(np.asarray(before, dtype=float)[core])
        fixed = _ratios(np.asarray(after, dtype=float)[core])
        assert abs(pink[1] - skin_b) > 0.25               # it was pink
        assert fixed[0] == pytest.approx(skin_r, abs=0.12)
        assert fixed[1] == pytest.approx(skin_b, abs=0.12)

    def test_pixels_that_were_not_clipped_are_not_touched(self):
        _, observed = _face()
        fixed = reconstruct_highlights(observed, NOMINAL)
        safe = clip_weight(observed, saturation_levels(observed, NOMINAL)) == 0
        assert safe.sum() > 1000
        assert np.array_equal(fixed[safe], observed[safe])

    def test_the_rebuilt_area_is_no_brighter_than_the_colour_could_first_have_clipped(self):
        """It claims what the file supports and no more."""
        true, observed = _face()
        fixed = reconstruct_highlights(observed, NOMINAL)
        assert float((fixed @ LUMA).max()) < float((true @ LUMA).max()) * 0.6

    def test_a_frame_that_is_clipped_all_over_has_no_colour_to_borrow_so_it_goes_white(self):
        flat = np.full((60, 90, 3), 0.88, dtype=np.float32) * NOMINAL.astype(np.float32)
        fixed = reconstruct_highlights(flat, NOMINAL)
        r, b = _ratios(fixed[30, 45])
        assert r == pytest.approx(1.0, abs=0.05) and b == pytest.approx(1.0, abs=0.05)

    def test_a_clipped_edge_does_not_borrow_across_a_dark_one(self):
        """Dark pixels carry no colour worth borrowing; the highlight takes its
        colour from skin, not from the black behind it."""
        true, observed = _face()
        observed[:, :20] = 0.001                      # a black border
        fixed = reconstruct_highlights(observed, NOMINAL)
        r, b = _ratios(fixed[60, 90])
        skin_r, skin_b = _ratios(SKIN)
        assert r == pytest.approx(skin_r, abs=0.08) and b == pytest.approx(skin_b, abs=0.08)


@pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
class TestOnARealRawDecode:
    """Through the real LibRaw, with a sensor that saturates below its stated
    white level — the case where a threshold taken from the file finds nothing."""

    def _dng(self, tmp_path, sat=0.88):
        true, _ = _face(size=(240, 360))
        raw = np.minimum(true / GAINS, sat)           # what the sensor records
        h, w = raw.shape[:2]
        bayer = np.zeros((h, w))
        bayer[0::2, 0::2] = raw[0::2, 0::2, 0]
        bayer[0::2, 1::2] = raw[0::2, 1::2, 1]
        bayer[1::2, 0::2] = raw[1::2, 0::2, 1]
        bayer[1::2, 1::2] = raw[1::2, 1::2, 2]
        from test_edited_thumbnails import _XYZ_TO_SRGB, _rational, _WHITE

        extra = [
            (50706, "B", 4, (1, 4, 0, 0), True), (50708, "s", 0, "Synthetic", True),
            (33421, "H", 2, (2, 2), True), (33422, "B", 4, (0, 1, 1, 2), True),
            (50714, "H", 1, (0,), True), (50717, "H", 1, (_WHITE,), True),
            (50721, "i", 18, _rational(_XYZ_TO_SRGB), True),
            (50728, "2I", 3, _rational(1.0 / GAINS), True), (50778, "H", 1, (21,), True),
        ]
        path = tmp_path / "flash.dng"
        tifffile.imwrite(path, (bayer * _WHITE).astype(np.uint16), photometric=32803,
                         compression=None, extratags=extra)
        return path

    def test_a_flash_blown_face_is_not_pink_after_a_pull_down(self, tmp_path):
        decoded = engine._load_raw_linear(self._dng(tmp_path), 0)
        out = np.asarray(develop_frame(LinearFrame(decoded), {"exposure": -1.5}), dtype=float)
        true, _ = _face(size=(240, 360))
        skin = (true[..., 1] > 0.3) & (true[..., 1] < 0.6)          # unclipped skin nearby
        want = _ratios(out[skin].mean(axis=0))
        core = _ratios(out[120, 180])
        assert core[0] == pytest.approx(want[0], abs=0.08)
        assert core[1] == pytest.approx(want[1], abs=0.08)


class TestChannelByChannel:
    """In the camera's own primaries one channel clips at a time, and only the
    clipped ones are rebuilt: the others still hold real data."""

    LEVELS = np.array([1.26, 1.0, 2.14], dtype=np.float32)      # an R6 II's, white-balanced

    def _sheet(self, surround, size=(120, 180)):
        """A sheet of white paper with printed lines, lit far past where green and
        red clip but not blue, on a ``surround`` coloured background."""
        h, w = size
        rgb = np.empty((h, w, 3), dtype=np.float32)
        rgb[:] = surround
        y, x = np.mgrid[0:h, 0:w]
        paper = (abs(y - h / 2) < h * 0.3) & (abs(x - w / 2) < w * 0.3)
        lines = (y % 6) < 2
        value = np.where(lines, 1.6, 1.9)                       # ink: a little darker
        for c in range(3):
            rgb[..., c][paper] = np.minimum(value[paper], self.LEVELS[c])
        return rgb, paper, lines

    def test_a_channel_that_did_not_clip_keeps_its_data(self):
        rgb, paper, lines = self._sheet(surround=(0.3, 0.3, 0.3))
        fixed = reconstruct_highlights(rgb, self.LEVELS)
        assert np.array_equal(fixed[paper][:, 2], rgb[paper][:, 2])

    def test_the_printing_on_a_blown_sheet_comes_back(self):
        """Green and red clipped flat; blue still records the lines, and the rebuilt
        sheet shows them in every channel."""
        rgb, paper, lines = self._sheet(surround=(0.3, 0.3, 0.3))
        fixed = reconstruct_highlights(rgb, self.LEVELS)
        assert np.ptp(rgb[paper][:, 1]) == 0
        ink, blank = fixed[paper & lines][:, 1].mean(), fixed[paper & ~lines][:, 1].mean()
        assert blank - ink > 0.2

    def test_white_paper_beside_a_warm_surface_stays_white(self):
        """Colour is borrowed from bright neighbours, and only when it agrees with
        the channel that kept its data; a warm stage does not tint the paper."""
        rgb, paper, _ = self._sheet(surround=(0.55, 0.35, 0.15))
        fixed = reconstruct_highlights(rgb, self.LEVELS)
        r, b = _ratios(fixed[paper].mean(axis=0))
        assert r == pytest.approx(1.0, abs=0.08) and b == pytest.approx(1.0, abs=0.08)

    def test_the_middle_of_a_fully_clipped_patch_takes_its_edges_colour(self):
        """Clipped in every channel there is no data of its own; the colour is
        carried in from the edge, which kept a channel."""
        rgb, paper, _ = self._sheet(surround=(0.55, 0.35, 0.15))
        h, w = paper.shape
        core = np.zeros_like(paper)
        core[h // 2 - 8:h // 2 + 8, w // 2 - 12:w // 2 + 12] = True
        rgb[core] = self.LEVELS                                     # every channel at its level
        fixed = reconstruct_highlights(rgb, self.LEVELS)
        r, b = _ratios(fixed[core].mean(axis=0))
        assert r == pytest.approx(1.0, abs=0.1) and b == pytest.approx(1.0, abs=0.1)
        # and no darker than the partly clipped paper around it
        assert fixed[core][:, 1].mean() >= 0.95 * fixed[paper & ~core][:, 1].mean()
