"""A quarter turn and a mirror: the framing applied before the level and crop."""

import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import geometry
from adjust.develop import from_crs, normalize, to_crs
from catalog.database import CatalogDB
from catalog.service import CatalogService

W, H = 400, 300
TURNS = [(t, f) for t in (0, 90, 180, 270) for f in (0, 1)]


def test_turns_are_kept_as_quarters_and_a_full_turn_is_none():
    assert normalize({"turn": 90}) == {"turn": 90}
    assert normalize({"turn": -90}) == {"turn": 270}
    assert normalize({"turn": 360}) == {}
    assert normalize({"turn": 95}) == {"turn": 90}
    assert normalize({"flip": 1}) == {"flip": 1}
    assert normalize({"flip": 0}) == {}


@pytest.mark.parametrize("turn,flip", TURNS)
def test_a_pixel_lands_where_shown_point_says(turn, flip):
    img = Image.new("RGB", (W, H))
    img.putpixel((10, 20), (255, 0, 0))
    settings = {"turn": turn, "flip": flip}
    out = geometry.apply_geometry(img, settings)
    assert out.size == ((H, W) if turn in (90, 270) else (W, H))
    u, v = geometry.shown_point(settings, 10.5 / W, 20.5 / H, W, H)
    assert out.getpixel((int(u * out.size[0]), int(v * out.size[1]))) == (255, 0, 0)


@pytest.mark.parametrize("turn,flip", TURNS)
def test_source_point_is_the_way_back_through_turn_level_and_crop(turn, flip):
    settings = {"turn": turn, "flip": flip, "angle": 7,
                "crop_x": 0.1, "crop_y": 0.2, "crop_w": 0.5, "crop_h": 0.6}
    for point in ((0.45, 0.5), (0.5, 0.55)):
        u, v = geometry.shown_point(settings, *point, W, H)
        assert 0 < u < 1 and 0 < v < 1
        assert geometry.source_point(settings, u, v, W, H) == pytest.approx(point, abs=1e-6)


def test_a_turned_crop_has_the_turned_frames_shape():
    img = Image.new("RGB", (W, H))
    out = geometry.apply_geometry(img, {"turn": 90, "crop_x": 0, "crop_y": 0, "crop_w": 1, "crop_h": 0.5})
    assert out.size == (H, W // 2)


def test_turning_takes_the_crop_with_the_picture():
    img = Image.new("RGB", (W, H))
    img.putpixel((150, 120), (255, 0, 0))
    before = {"crop_x": 0.3, "crop_y": 0.3, "crop_w": 0.2, "crop_h": 0.2}
    after = geometry.turned_by(before, 90)
    assert after["turn"] == 90
    out = geometry.apply_geometry(img, after)
    assert out.size == (60, 80)                 # the same 80 x 60 piece, stood up
    assert (255, 0, 0) in [out.getpixel((x, y)) for x in range(60) for y in range(80)]


def test_four_quarters_and_two_mirrors_are_where_it_started():
    start = {"angle": 3, "crop_x": 0.1, "crop_y": 0.2, "crop_w": 0.3, "crop_h": 0.4}
    edit = start
    for _ in range(4):
        edit = geometry.turned_by(edit, 90)
    assert edit == pytest.approx(normalize(start))
    assert geometry.turned_by(geometry.turned_by(start, 0, True), 0, True) == pytest.approx(normalize(start))
    assert geometry.turned_by(geometry.turned_by(start, 90), -90) == pytest.approx(normalize(start))


def test_a_mirror_keeps_the_horizon_level():
    assert geometry.turned_by({"angle": 3}, 0, True) == {"flip": 1, "angle": -3.0}


def test_turning_a_mirrored_picture_right_turns_it_right():
    img = Image.new("RGB", (W, H))
    img.putpixel((5, 5), (255, 0, 0))           # top left of the frame
    mirrored = geometry.turned_by({}, 0, True)  # now top right
    turned = geometry.turned_by(mirrored, 90)   # a quarter right: bottom right
    out = geometry.apply_geometry(img, turned)
    assert out.getpixel((out.size[0] - 6, out.size[1] - 6)) == (255, 0, 0)


def test_turn_and_flip_survive_the_sidecar_and_keep_lightrooms_crop_out():
    edit = normalize({"turn": 270, "flip": 1, "crop_x": 0.1, "crop_y": 0.2, "crop_w": 0.6, "crop_h": 0.4})
    written = to_crs(edit)
    assert "crs:CropLeft" not in written
    assert from_crs(written) == edit


def test_the_grid_turn_keeps_the_tone_and_can_be_undone():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lib = tmp / "library"
        folder = lib / "2024" / "06" / "15"
        folder.mkdir(parents=True)
        (folder / "IMG_0001.CR3").write_bytes(b"fake raw " * 100)
        catalog = CatalogService(CatalogDB(str(tmp / "catalog.db")), str(lib))
        catalog.index_folder("2024/06/15", [".cr3"])
        path = "2024/06/15/IMG_0001.CR3"
        catalog.set_photo_edit(path, {"exposure": 0.5})
        result = catalog.turn_photos([path], -90)
        assert result["edits"][path] == {"exposure": 0.5, "turn": 270}
        assert catalog.photo_edit(path) == {"exposure": 0.5, "turn": 270}
        catalog.restore_edits(result["previous"])
        assert catalog.photo_edit(path) == {"exposure": 0.5}
