"""
Lights and Darks, the middle of the tone range; and skin retouched on the people
a mask reaches (adjust.skin).

The people models are not needed: the face boxes and the skin the parsers would
find are put on the frame as they would leave them, so what is tested is the
retouch itself, on a patch of synthetic skin with blemishes and pores in it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import skin  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame, mask_reach, tone_curve, _CURVE_X  # noqa: E402

H, W = 240, 320
FACE = (0.40, 0.20, 0.20, 0.25)      # x, y, w, h as fractions: 64 px across


def _stops_moved(settings, at):
    curve = tone_curve(settings)
    return float(np.interp(at, _CURVE_X, curve) - at)


class TestLightsAndDarks:
    def test_each_moves_its_own_middle_of_the_range(self):
        assert _stops_moved({"lights": 100}, 0.7) > 1.0          # a lit face
        assert abs(_stops_moved({"lights": 100}, 4.0)) < 0.05    # the sky: highlights' own
        assert abs(_stops_moved({"lights": 100}, -4.0)) < 0.05
        assert _stops_moved({"darks": -100}, -1.8) < -1.0        # dark cloth
        assert abs(_stops_moved({"darks": -100}, -6.0)) < 0.05   # deep shadow: the shadows' own
        assert abs(_stops_moved({"darks": -100}, 1.5)) < 0.05

    def test_they_are_kept_where_lightroom_reads_them(self):
        clean = normalize({"lights": 30, "darks": -20})
        crs = to_crs(clean)
        assert crs["crs:ParametricLights"] == "+30" and crs["crs:ParametricDarks"] == "-20"
        assert from_crs(crs) == clean

    def test_a_mask_takes_them_too(self):
        clean = normalize({"masks": [{"kind": "radial", "lights": 40, "darks": -10, "smooth": 150, "even": -5,
                                      "texture": 30}]})
        m = clean["masks"][0]
        assert (m["lights"], m["darks"], m["smooth"], m["texture"]) == (40, -10, 100, 30)
        assert "even" not in m                                   # 0..100: -5 is none


def _skin_frame(seed=1):
    """A face-sized patch of warm skin: smooth shading, blotches at the middle
    scale, fine pores; the rest of the frame a grey wall."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.full((H, W, 3), 0.18, np.float32)
    lum = 0.30 * (1.0 + 0.15 * np.sin(xx / 40.0))                  # the face's own shape
    import cv2
    blotch = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3.0)
    blotch *= 0.35 / blotch.std()
    pores = rng.standard_normal((H, W)).astype(np.float32) * 0.04
    red = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 4.0)
    red *= 0.25 / red.std()
    lum = lum * np.exp2(blotch + pores)
    face = np.stack([lum * 1.6 * (1 + red), lum * 0.9, lum * 0.6], axis=-1)
    region = (slice(30, 210), slice(80, 240))
    out = base.copy()
    out[region] = face[region]
    frame = LinearFrame(out.astype(np.float32))
    skin_matte = np.zeros((H, W), np.float32)
    skin_matte[region] = 1.0
    frame._mattes["_faces"] = [FACE]
    frame._mattes[("_parts", tuple(sorted(skin.SKIN_AREAS)), 0)] = skin_matte
    frame._mattes["people"] = skin_matte
    frame._mattes["_parts_read"] = []           # read by the parsers: nothing more to it
    return frame, region


def _bands(rgb, region):
    import cv2
    lum = np.log2(np.maximum(rgb @ skin._LUMA, 1e-5))[region]
    fine = lum - cv2.GaussianBlur(lum, (0, 0), 0.7)
    middle = cv2.GaussianBlur(lum, (0, 0), 0.7) - cv2.GaussianBlur(lum, (0, 0), 6.0)
    return float(middle.std()), float(fine.std())


class TestSkin:
    def test_smooth_calms_the_blemishes_and_keeps_the_pores(self):
        frame, region = _skin_frame()
        before = _bands(frame.rgb, region)
        after = _bands(skin.retouch(frame, [{"kind": "people", "smooth": 100}], mask_reach), region)
        assert after[0] < 0.75 * before[0]                        # the middle band: calmer
        assert after[1] > 0.8 * before[1]                         # the pores: still there

    def test_smooth_takes_the_dark_spots_and_keeps_the_glow(self):
        # A blemish is darker than the skin round it; its sheen is brighter.
        import cv2
        frame, region = _skin_frame()
        out = skin.retouch(frame, [{"kind": "people", "smooth": 100}], mask_reach)

        def middle(rgb):
            lum = np.log2(np.maximum(rgb @ skin._LUMA, 1e-5))[region]
            return cv2.GaussianBlur(lum, (0, 0), 0.7) - cv2.GaussianBlur(lum, (0, 0), 6.0)
        was, now = middle(frame.rgb), middle(out)
        dark, bright = was < -0.1, was > 0.1
        assert np.abs(now[dark]).mean() < 0.55 * np.abs(was[dark]).mean()
        assert np.abs(now[bright]).mean() > 0.6 * np.abs(was[bright]).mean()

    def test_texture_softens_or_crisps_the_pores(self):
        frame, region = _skin_frame()
        fine = _bands(frame.rgb, region)[1]
        soft = _bands(skin.retouch(frame, [{"kind": "people", "texture": -100}], mask_reach), region)[1]
        crisp = _bands(skin.retouch(frame, [{"kind": "people", "texture": 100}], mask_reach), region)[1]
        assert soft < 0.6 * fine < crisp * 0.6

    def test_even_tone_evens_the_redness_and_keeps_the_skin_its_colour(self):
        frame, region = _skin_frame()
        out = skin.retouch(frame, [{"kind": "people", "even": 100}], mask_reach)
        ratio = lambda rgb: (rgb[..., 0] / np.maximum(rgb[..., 1], 1e-5))[region]  # noqa: E731
        assert ratio(out).std() < 0.6 * ratio(frame.rgb).std()
        assert abs(ratio(out).mean() / ratio(frame.rgb).mean() - 1.0) < 0.05

    def test_the_skin_keeps_its_brightness(self):
        frame, region = _skin_frame()
        out = skin.retouch(frame, [{"kind": "people", "smooth": 100, "even": 100}], mask_reach)
        before, after = (frame.rgb @ skin._LUMA)[region].mean(), (out @ skin._LUMA)[region].mean()
        assert abs(after / before - 1.0) < 0.03

    def test_nothing_but_skin_where_the_mask_reaches_is_touched(self):
        frame, region = _skin_frame()
        out = skin.retouch(frame, [{"kind": "people", "smooth": 100, "even": 100}], mask_reach)
        outside = np.ones((H, W), bool)
        outside[region] = False
        assert np.array_equal(out[outside], frame.rgb[outside])
        # A radial over the left half of the face: the right half stays as it was.
        half = skin.retouch(frame, [{"kind": "radial", "cx": 0.3, "cy": 0.5, "rx": 0.1, "ry": 0.5, "feather": 0,
                                     "smooth": 100}], mask_reach)
        assert np.array_equal(half[:, 200:], frame.rgb[:, 200:])

    def test_the_develop_view_retouches_once_for_other_sliders(self):
        frame, region = _skin_frame()
        edit = {"masks": [{"kind": "people", "smooth": 60, "exposure": 0.5}]}
        develop_frame(frame, edit)
        kept = frame._retouched
        develop_frame(frame, {"masks": [{"kind": "people", "smooth": 60, "exposure": 1.0}]})
        assert frame._retouched is kept                          # an exposure move: not retouched again
        develop_frame(frame, {"masks": [{"kind": "people", "smooth": 70, "exposure": 1.0}]})
        assert frame._retouched is not kept
