"""
Lens corrections: the model, its inversion, correcting a frame and a patch of
it alike, the controls in the edit, and the camera's correction learned from a
real frame's JPEG.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import lens  # noqa: E402
from adjust.develop import describe_settings, from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame, lens_corrected  # noqa: E402
from samples import sample  # noqa: E402

# About the camera's own correction of a 24 mm zoom (measured on an RF24-105).
CAMERA = {"ox": -0.001, "oy": 0.0002, "s": 0.9885, "a1": 0.075, "a2": -0.04}


def _grid(h=120, w=180):
    rng = np.random.default_rng(5)
    return rng.random((h, w, 3)).astype(np.float32)


class TestTheModel:
    def test_inverting_it_finds_the_point_it_maps(self):
        qx, qy = np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.5, 0.5, 7))
        px, py = lens.source_of(CAMERA, qx, qy)
        fx, fy = lens._forward(CAMERA, px, py)
        assert np.abs(fx - qx).max() < 1e-7 and np.abs(fy - qy).max() < 1e-7

    def test_nothing_asked_is_no_model(self):
        assert lens.model_for({}, CAMERA, 1.5) is None
        assert lens.model_for({"lens_profile": 1}, None, 1.5) is None

    def test_the_cameras_correction_only_when_asked(self):
        assert lens.model_for({"lens_profile": 1}, CAMERA, 1.5) == CAMERA
        assert lens.model_for({"distortion": 0}, CAMERA, 1.5) is None

    def test_by_hand_keeps_the_corners_full(self):
        for amount in (-100, -40, 40, 100):
            model = lens.model_for({"distortion": amount}, None, 1.5)
            h = 1.0 / np.hypot(1.5, 1.0)
            w = 1.5 * h
            # Every point of the edge comes from inside the frame.
            t = np.linspace(-1, 1, 21)
            ex = np.concatenate([t * w, t * w, -w + 0 * t, w + 0 * t])
            ey = np.concatenate([-h + 0 * t, h + 0 * t, t * h, t * h])
            sx, sy = lens.source_of(model, ex, ey)
            assert (np.abs(sx) <= w * 1.001).all() and (np.abs(sy) <= h * 1.001).all(), amount

    def test_a_registration_is_kept_free_of_its_size(self):
        small = lens.from_registration(np.array([384.0, 256.0]), 461.5, [-0.5, 0.1, 0.99, 0.06, -0.03])
        large = lens.from_registration(np.array([768.0, 512.0]), 923.0, [-1.0, 0.2, 0.99, 0.06, -0.03])
        for key in small:
            assert small[key] == pytest.approx(large[key], abs=1e-6)


class TestCorrectingAFrame:
    def test_each_pixel_comes_from_where_the_model_says(self):
        rgb = _grid()
        out = lens.warp(rgb, CAMERA)
        h, w = rgb.shape[:2]
        map_x, map_y = lens.maps(CAMERA, (w, h), (0, 0, 1, 1), (h, w))
        for y, x in ((10, 12), (60, 90), (110, 170)):
            sx, sy = int(round(float(map_x[y, x]))), int(round(float(map_y[y, x])))
            if 0 <= sx < w and 0 <= sy < h and abs(map_x[y, x] - sx) < 0.05 and abs(map_y[y, x] - sy) < 0.05:
                assert np.allclose(out[y, x], rgb[sy, sx], atol=1e-4)

    def test_straight_lines_bowed_by_a_lens_come_out_straight(self):
        # A grid of lines drawn as a barrel lens would draw them (the inverse of
        # the camera's correction), then corrected.
        h, w = 400, 600
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        r = np.hypot(w / 2, h / 2)
        qx, qy = (xs + 0.5 - w / 2) / r, (ys + 0.5 - h / 2) / r
        fx, fy = lens._forward(CAMERA, qx, qy)          # where each RAW point lands, corrected
        cx, cy = fx * r + w / 2, fy * r + h / 2
        lines = ((np.abs((cx % 40) - 20) < 1.5) | (np.abs((cy % 40) - 20) < 1.5)).astype(np.float32)
        raw = np.repeat(lines[..., None], 3, axis=2)
        out = lens.warp(raw, CAMERA)[..., 0]
        # A horizontal line near the top: its row is the same at the middle and the side.
        top = out[:40]
        rows_mid = np.argmax(top[:, w // 2])
        rows_side = np.argmax(top[:, 20])
        assert abs(int(rows_mid) - int(rows_side)) <= 1
        bowed_mid, bowed_side = np.argmax(lines[:40, w // 2]), np.argmax(lines[:40, 20])
        assert abs(int(bowed_mid) - int(bowed_side)) >= 3      # it was bowed

    def test_a_patch_is_the_same_as_the_frame_corrected_whole(self):
        rgb = _grid(160, 240)
        whole = lens.warp(rgb, CAMERA)

        def fetch(x0, y0, x1, y1):
            xs = np.clip(np.arange(x0, x1), 0, 239)
            ys = np.clip(np.arange(y0, y1), 0, 159)
            return rgb[np.ix_(ys, xs)]

        patch = lens.warp_region(fetch, CAMERA, (240, 160), (20 / 240, 30 / 160, 80 / 240, 90 / 160), (60, 60))
        assert np.allclose(patch, whole[30:90, 20:80], atol=1e-4)


class TestInTheDevelop:
    def _frame(self):
        frame = LinearFrame(_grid())
        frame.lens = lambda: CAMERA
        return frame

    def test_the_profile_is_the_cameras_correction(self):
        frame = self._frame()
        corrected = lens_corrected(frame, 1, 0)
        assert corrected is not frame
        assert np.allclose(corrected.rgb, lens.warp(frame.rgb, CAMERA))
        assert lens_corrected(frame, 1, 0) is corrected            # kept

    def test_a_frame_that_cannot_know_is_left_alone_unless_by_hand(self):
        frame = LinearFrame(_grid())
        assert lens_corrected(frame, 1, 0) is frame
        assert lens_corrected(frame, 1, 50) is not frame

    def test_a_patch_is_never_corrected_on_its_own(self):
        frame = self._frame()
        frame.window = (0.2, 0.2, 0.6, 0.6)
        assert lens_corrected(frame, 1, 60) is frame

    def test_develop_frame_corrects(self):
        frame = self._frame()
        plain = np.asarray(develop_frame(frame, {"exposure": 0.01}))
        fixed = np.asarray(develop_frame(frame, {"exposure": 0.01, "lens_profile": 1}))
        assert np.abs(plain.astype(int) - fixed.astype(int)).mean() > 5
        assert develop_frame(frame, {"lens_profile": 1}).tobytes() != frame.encoded().tobytes()


class TestTheControls:
    def test_kept_and_written_under_this_apps_names(self):
        edit = {"lens_profile": 1, "distortion": -30}
        assert normalize({"lens_profile": 7, "distortion": -300}) == {"lens_profile": 1, "distortion": -100}
        crs = to_crs(edit)
        assert crs["crs:PhotoolLensProfile"] == "1" and crs["crs:PhotoolDistortion"] == "-30"
        assert from_crs(crs) == edit
        assert from_crs({"crs:LensProfileEnable": "1", "crs:LensManualDistortionAmount": "20"}) == {}
        assert "lens profile, distortion -30" in describe_settings(edit)


def test_a_jpeg_has_no_profile_to_learn(tmp_path):
    from export.engine import lens_profile

    path = tmp_path / "a.jpg"
    Image.new("RGB", (60, 40)).save(path)
    assert lens_profile(path, np.zeros((40, 60, 3), np.float32)) is None


def test_a_real_frame_lines_up_with_its_camera_jpeg_when_corrected():
    """On the R6 II and RF24-105 at 24 mm: the corners of the developed RAW sit
    25 px from the camera's JPEG at 1600 px, and under a pixel once corrected."""
    photo = sample("lens-24mm.CR3")
    from adjust.look import _lens_model, _register
    from adjust.render import LUMA
    from export import engine

    frame = engine.linear_base(photo, 1600)
    camera = engine._sibling_jpeg_rgb(photo)

    def corner_error(img):
        a = np.asarray(img, dtype=np.float32) / 255.0
        h, w = a.shape[:2]
        c = np.asarray(Image.fromarray(camera).resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
        found = _register(a @ LUMA, c @ LUMA)
        if found is None:
            return 0.0
        centre, radius, u = found
        return float(np.hypot(*(_lens_model(np.array([[0.0, 0.0]]), centre, radius, u)[0])))

    assert corner_error(develop_frame(frame, {"exposure": 0.01})) > 10
    assert corner_error(develop_frame(frame, {"lens_profile": 1})) < 1.0
