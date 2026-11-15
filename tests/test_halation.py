"""
Halation and bloom (adjust.halation): bright light spreading as it does on film.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import halation  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame, linear_from_image  # noqa: E402


def frame_of(rgb8) -> LinearFrame:
    return LinearFrame(linear_from_image(Image.fromarray(np.asarray(rgb8, np.uint8), "RGB")))


def picture(w=400, h=300, side=30, level=255, ground=4):
    """Black, with a square of ``level`` in the middle."""
    a = np.full((h, w, 3), ground, np.uint8)
    top, left = (h - side) // 2, (w - side) // 2
    a[top:top + side, left:left + side] = level
    return a


def out(frame, edit):
    return np.asarray(develop_frame(frame, edit), dtype=np.float32)


# The standard picture's square spans x 185..214, y 135..164.
RIGHT_EDGE, MID_Y = 214, 150


def added(edit, rgb8=None, base=None):
    rgb8 = picture() if rgb8 is None else rgb8
    frame = frame_of(rgb8)
    return out(frame, edit) - out(frame, base or {})


class TestOff:
    def test_no_key_is_the_picture(self):
        frame = frame_of(picture())
        assert np.array_equal(out(frame, {}), out(frame_of(picture()), {"glow_size": 50, "glow_threshold": -30}))

    def test_normalize_drops_what_has_nothing_to_act_on(self):
        clean = normalize({"halation_colour": 40, "glow_size": 30, "glow_threshold": -20})
        assert not set(clean) & set(halation.KEYS)
        clean = normalize({"bloom": 30, "glow_size": 30, "halation_colour": 40})
        assert clean.get("glow_size") == 30 and "halation_colour" not in clean


class TestHalation:
    def test_red_rises_most(self):
        d = added({"halation": 60})
        r, g, b = d[MID_Y, RIGHT_EDGE + 3]
        assert r > g >= b and r > 5
        assert np.abs(d[2, 2]).max() <= 1
        # Nothing outside the lamp is darker. The lamp's own white takes a little of the
        # red light above white, and the display's roll-off keeps its hue: a warm white
        # core, a few levels under 255 in green and blue, never grey.
        core = np.zeros(d.shape[:2], bool)
        core[135:165, 185:215] = True
        assert d[~core].min() >= 0
        lamp = out(frame_of(picture()), {"halation": 60})[core]
        assert lamp[:, 0].min() >= 253 and lamp.min() >= 240

    def test_more_is_more(self):
        at = [added({"halation": v})[MID_Y, RIGHT_EDGE + 5].sum() for v in (20, 60, 100)]
        assert at[0] < at[1] < at[2]

    def test_colour(self):
        def ratio(c):
            r, g, _ = added({"halation": 60, "halation_colour": c})[MID_Y, RIGHT_EDGE + 3]
            return g / r
        assert ratio(-100) < ratio(0) < ratio(100)

    def test_mono_is_grey(self):
        o = out(frame_of(picture()), {"halation": 60, "mono": 1})
        assert np.array_equal(o[..., 0], o[..., 1]) and np.array_equal(o[..., 1], o[..., 2])


class TestBloom:
    def test_white_light_blooms_white(self):
        r, g, b = added({"bloom": 60})[MID_Y, RIGHT_EDGE + 3]
        assert min(r, g, b) > 0 and max(r, g, b) <= min(r, g, b) * 1.15

    def test_bloom_reaches_further_than_halation(self):
        # In linear light: 40 px out (a tenth of the picture) the bloom is under a level
        # on 8-bit black, and rounds away there.
        frame = frame_of(picture())
        far = RIGHT_EDGE + 40
        bloom = halation.glow(frame.rgb, frame.window, {"bloom": 60})[MID_Y, far].sum()
        halo = halation.glow(frame.rgb, frame.window, {"halation": 60})[MID_Y, far].sum()
        assert bloom > halo * 10 > 0 or (bloom > 0 and halo == 0)


class TestSizeAndThreshold:
    def test_size(self):
        far = RIGHT_EDGE + 40
        wide = added({"bloom": 60, "halation": 60, "glow_size": 100})[MID_Y, far].sum()
        tight = added({"bloom": 60, "halation": 60, "glow_size": -100})[MID_Y, far].sum()
        assert wide > tight

    def test_threshold(self):
        grey = picture(level=150)
        low = added({"halation": 60, "bloom": 60, "glow_threshold": -100}, grey)[MID_Y, RIGHT_EDGE + 3]
        high = added({"halation": 60, "bloom": 60, "glow_threshold": 100}, grey)[MID_Y, RIGHT_EDGE + 3]
        assert low.max() > 3 and high.max() < 1

    def test_exposure_decides_what_is_bright(self):
        grey = picture(level=150)
        exposed = added({"halation": 60, "exposure": 2}, grey, {"exposure": 2})[MID_Y, RIGHT_EDGE + 3].sum()
        plain = added({"halation": 60}, grey)[MID_Y, RIGHT_EDGE + 3].sum()
        assert exposed > 3 and plain < exposed / 3


class TestScale:
    def test_the_same_at_any_size(self):
        edit = {"halation": 50, "bloom": 50}
        large = develop_frame(frame_of(picture(1600, 1200, 120)), edit)
        small = np.asarray(develop_frame(frame_of(picture(800, 600, 60)), edit), dtype=np.float32)
        shrunk = np.asarray(large.resize((800, 600), Image.BOX), dtype=np.float32)
        assert np.abs(shrunk - small).mean() < 2

    def test_the_loupe_takes_its_glow_from_the_preview(self):
        edit = {"halation": 50, "bloom": 50}
        full = frame_of(picture(1600, 1200, 120))
        preview = frame_of(np.asarray(Image.fromarray(picture(1600, 1200, 120)).resize((800, 600), Image.BOX)))
        # A quarter of the picture, right of the square (which spans x 740..859).
        window = (0.55, 0.35, 0.80, 0.60)
        x0, y0, x1, y1 = 880, 420, 1280, 720
        patch = full.rgb[y0:y1, x0:x1]
        mine = halation.pixels(preview, patch, window, edit)
        whole = halation.lit(full, edit).rgb[y0:y1, x0:x1]
        a = np.asarray(LinearFrame(np.ascontiguousarray(mine)).encoded(), dtype=np.float32)
        b = np.asarray(LinearFrame(np.ascontiguousarray(whole)).encoded(), dtype=np.float32)
        assert np.abs(a - b).mean() < 1.5
        assert np.abs(a - np.asarray(LinearFrame(np.ascontiguousarray(patch)).encoded(), np.float32)).mean() > 1


def test_the_sidecar_keeps_them():
    edit = {"halation": 40, "halation_colour": -30, "bloom": 20, "glow_size": 25, "glow_threshold": -10}
    back = from_crs(to_crs(edit))
    assert {k: back.get(k) for k in edit} == edit
