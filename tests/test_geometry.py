"""Crop and level: the framing half of an edit."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import geometry
from adjust.develop import CRS_KEYS, from_crs, normalize, to_crs
from adjust.render import LinearFrame, develop_frame, region_frame


def test_a_whole_frame_crop_and_a_zero_angle_are_no_edit():
    assert normalize({"crop_x": 0, "crop_y": 0, "crop_w": 1, "crop_h": 1, "angle": 0.01}) == {}


def test_a_crop_needs_all_four_numbers():
    assert normalize({"crop_x": 0.1, "crop_y": 0.1, "crop_w": 0.5}) == {}


def test_a_crop_is_pulled_inside_the_frame():
    out = normalize({"crop_x": 0.9, "crop_y": -1, "crop_w": 0.5, "crop_h": 0.5})
    assert out["crop_x"] == 0.5 and out["crop_y"] == 0.0


def test_crop_sets_the_size_of_the_picture():
    img = Image.new("RGB", (400, 200))
    out = geometry.apply_geometry(img, {"crop_x": 0.25, "crop_y": 0, "crop_w": 0.5, "crop_h": 1})
    assert out.size == (200, 200)


def test_levelling_keeps_no_empty_corners():
    img = Image.new("RGB", (600, 400), (200, 200, 200))
    out = geometry.apply_geometry(img, {"angle": 10})
    assert out.size[0] < 600 and out.size[1] < 400
    arr = np.asarray(out)
    assert arr.min() > 150, "a corner of the turned frame leaked into the crop"


def test_a_crop_beyond_the_turned_picture_is_pulled_back():
    box = geometry.fit_box((0, 0, 1, 1), 20, 600, 400)
    assert geometry._rotated_fits(box, 20, 600, 400)
    assert box[2] < 1 and box[3] < 1


def test_a_clockwise_angle_levels_a_counter_clockwise_tilt():
    # A bar tilted so its right end is higher; turning clockwise lowers it.
    img = Image.new("L", (400, 400), 0)
    px = img.load()
    for x in range(400):
        y = 200 - int((x - 200) * 0.1763)  # about 10 degrees, right side higher
        for dy in (-1, 0, 1):
            px[x, y + dy] = 255
    out = np.asarray(geometry.apply_geometry(img, {"angle": 10}))
    rows = [np.argmax(out[:, c]) for c in (10, out.shape[1] // 2, out.shape[1] - 10)]
    assert max(rows) - min(rows) <= 3


def test_source_point_inverts_the_turn_and_the_crop():
    img = Image.new("RGB", (800, 600), (0, 0, 0))
    px = img.load()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            px[500 + dx, 250 + dy] = (255, 255, 255)
    settings = {"angle": 12}
    out = geometry.apply_geometry(img, settings)
    arr = np.asarray(out).sum(axis=2)
    ys, xs = np.nonzero(arr > 600)
    u, v = xs.mean() / out.size[0], ys.mean() / out.size[1]
    su, sv = geometry.source_point(settings, u, v, 800, 600)
    assert abs(su * 800 - 500) < 3 and abs(sv * 600 - 250) < 3


def test_crop_survives_the_sidecar():
    edit = normalize({"exposure": 0.5, "angle": -2.5, "crop_x": 0.1, "crop_y": 0.2,
                      "crop_w": 0.6, "crop_h": 0.4})
    written = to_crs(edit)
    assert "crs:CropLeft" not in written, "a turned crop does not mean the same in Lightroom"
    assert from_crs(written) == edit


def test_an_unturned_crop_is_also_written_for_lightroom():
    written = to_crs(normalize({"crop_x": 0.1, "crop_y": 0.2, "crop_w": 0.6, "crop_h": 0.4}))
    assert written["crs:HasCrop"] == "True"
    assert float(written["crs:CropRight"]) == pytest.approx(0.7)
    assert float(written["crs:CropBottom"]) == pytest.approx(0.6)
    assert set(written) <= set(CRS_KEYS) | {"crs:HasCrop", "crs:CropLeft", "crs:CropTop",
                                            "crs:CropRight", "crs:CropBottom", "crs:CropAngle"}


def test_develop_frame_crops_a_framing_only_edit():
    frame = LinearFrame(np.full((100, 200, 3), 0.2, dtype=np.float32))
    out = develop_frame(frame, {"crop_x": 0, "crop_y": 0, "crop_w": 0.5, "crop_h": 0.5})
    assert out.size == (100, 50)


def test_develop_frame_tones_then_crops():
    frame = LinearFrame(np.full((100, 200, 3), 0.2, dtype=np.float32))
    plain = develop_frame(frame, {"exposure": 1})
    cropped = develop_frame(frame, {"exposure": 1, "crop_x": 0, "crop_y": 0, "crop_w": 0.5, "crop_h": 1})
    assert cropped.size == (100, 100)
    assert np.asarray(cropped)[0, 0].tolist() == np.asarray(plain)[0, 0].tolist()


def test_a_loupe_patch_is_toned_like_the_frame_it_was_cut_from():
    rng = np.random.default_rng(1)
    full = np.clip(rng.normal(0.15, 0.02, (400, 600, 3)), 0.01, 1).astype(np.float32)
    full[:, 300:] *= 4  # a bright half
    small = LinearFrame(full[::4, ::4].copy())
    settings = {"shadows": 60, "highlights": -40}
    preview = np.asarray(develop_frame(small, settings)).astype(float)

    patch = region_frame(small, full[100:200, 50:150], (50 / 600, 100 / 400, 150 / 600, 200 / 400))
    got = np.asarray(develop_frame(patch, settings)).astype(float).mean()
    want = preview[25:50, 12:37].mean()
    assert abs(got - want) < 6, (got, want)
