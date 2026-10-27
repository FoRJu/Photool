"""
Transform and upright: the tilt and aspect as a homography, drawn with no empty
corners, in the edit like any control, and Upright finding the tilt and the level
that straighten a frame's own lines, or saying it cannot.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import lens  # noqa: E402
from adjust.develop import OWN_KEYS, describe_settings, from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402
from adjust.upright import upright  # noqa: E402
from samples import sample  # noqa: E402
from test_develop_routes import PHOTO, client  # noqa: E402,F401  (a fixture)

H, W = 400, 600


def _posts(made=0, turned=0.0):
    """Straight dark posts and a level line, as a camera tilted by ``made`` (the
    slider's units, with no enlarging) and turned ``turned`` degrees would see them."""
    img = np.full((H, W, 3), 0.8, np.float32)
    for x in range(40, W, 60):
        img[:, x:x + 4] = 0.1
    img[300:304, :] = 0.1
    if made:
        img = lens.warp(img, {**lens.IDENTITY, "H": lens.homography({"tf_vertical": made}), "zoom": 1.0})
    if turned:
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).rotate(
            turned, resample=Image.BICUBIC, fillcolor=(204, 204, 204)), np.float32) / 255
    return img


class TestTheHomography:
    def test_nothing_asked_is_none(self):
        assert lens.homography({}) is None
        assert lens.model_for({}, None, 1.5) is None

    def test_the_middle_stays_where_it_was(self):
        for settings in ({"tf_vertical": 60}, {"tf_horizontal": -40}, {"tf_vertical": 30, "tf_aspect": 50}):
            x, y = lens._project(lens.homography(settings), np.array([0.0]), np.array([0.0]))
            assert abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9, settings

    def test_the_corners_stay_inside_the_frame(self):
        for v in (-100, -30, 45, 100):
            model = lens.model_for({"tf_vertical": v, "tf_aspect": 40}, None, 1.5)
            assert model["zoom"] >= 1.0
            h = 1.0 / np.hypot(1.5, 1.0)
            w = 1.5 * h
            corners = np.array([[-w, -h], [w, -h], [-w, h], [w, h]])
            fx, fy = lens._to_frame(model, corners[:, 0], corners[:, 1])
            px, py = lens.source_of(model, fx, fy)
            assert (np.abs(px) <= w * 1.001).all() and (np.abs(py) <= h * 1.001).all(), v

    def test_it_is_drawn_through_the_frame(self):
        frame = LinearFrame(np.random.default_rng(1).random((60, 90, 3)).astype(np.float32))
        plain = np.asarray(develop_frame(frame, {"exposure": 0.01}))
        tilted = np.asarray(develop_frame(frame, {"exposure": 0.01, "tf_vertical": 40}))
        assert plain.shape == tilted.shape and np.abs(plain.astype(int) - tilted.astype(int)).mean() > 5


class TestTheControls:
    def test_kept_and_written_under_this_apps_names(self):
        edit = {"tf_vertical": -16, "tf_horizontal": 5, "tf_aspect": 10}
        crs = to_crs(edit)
        assert crs["crs:PhotoolTransformVertical"] == "-16"
        assert from_crs(crs) == edit
        assert from_crs({"crs:PerspectiveVertical": "-16"}) == {}
        assert "transform vertical -16, horizontal +5, aspect +10" in describe_settings(edit)
        assert normalize({"tf_vertical": 400}) == {"tf_vertical": 100}

    def test_a_frames_own_like_its_level(self):
        assert {"tf_vertical", "tf_horizontal", "tf_aspect"} <= set(OWN_KEYS)


class TestUpright:
    @pytest.mark.parametrize("made", [25, -40, 60])
    def test_a_tilted_camera_is_found_and_undone(self, made):
        found = upright(_posts(made))
        assert found["found"] and abs(found["tf_vertical"] + made) <= max(4, 0.12 * abs(made)), found
        assert abs(found["angle"]) < 0.5

    def test_a_turned_camera_is_found_too(self):
        found = upright(_posts(-40, turned=3.0))
        assert found["found"] and abs(found["tf_vertical"] - 40) <= 4
        assert abs(found["angle"] - 3.0) < 0.5      # turned back clockwise

    def test_level_from_the_horizon(self):
        found = upright(_posts(turned=-2.0), "level")
        assert found["found"] and abs(found["angle"] + 2.0) < 0.5

    def test_a_picture_with_no_lines_is_left_alone(self):
        rng = np.random.default_rng(2)
        blobs = rng.random((H, W, 3)).astype(np.float32)
        found = upright(blobs)
        assert not found["found"] and found["reason"]

    def test_lines_that_do_not_agree_are_not_verticals(self):
        img = np.full((H, W, 3), 0.8, np.float32)
        rng = np.random.default_rng(3)
        from PIL import ImageDraw

        pic = Image.fromarray((img * 255).astype(np.uint8))
        d = ImageDraw.Draw(pic)
        for _ in range(30):           # arms and trunks: near vertical, every which way
            x, lean = rng.uniform(20, W - 20), rng.uniform(-20, 20)
            d.line((x, 40, x + np.tan(np.radians(lean)) * 300, 340), fill=(25, 25, 25), width=4)
        found = upright(np.asarray(pic, np.float32) / 255)
        assert not found["found"]


class TestThroughTheApi:
    def test_the_route_answers(self, client):
        res = client.post("/api/export/develop/upright", json={"filepath": PHOTO, "mode": "vertical", "size": 256})
        assert res.status_code == 200 and "found" in res.json()
        assert client.post("/api/export/develop/upright",
                           json={"filepath": PHOTO, "mode": "sideways"}).status_code == 400


def test_on_real_photographs():
    """A building shot from below is straightened; a portrait is left alone."""
    building = sample("building-from-below.CR3")
    portrait = sample("group-of-three.CR3")
    from export.engine import upright_for

    found = upright_for(building, {"lens_profile": 1}, "vertical")
    assert found["found"] and found["tf_vertical"] < -5, found
    assert not upright_for(portrait, {"lens_profile": 1}, "vertical")["found"]
