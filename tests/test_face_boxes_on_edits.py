"""
Face boxes on a cropped or levelled photograph (issue #37).

A face is found, and kept, as a box on the whole unedited frame. The viewer
shows the finished picture, so a box goes through the crop and the turn on the
way there, and a box drawn on that picture comes back through them on the way
in. The face crops (people, the year) are cut from the frame and need neither.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import geometry  # noqa: E402

W, H = 800, 600


def test_a_crop_moves_and_scales_the_box():
    settings = {"crop_x": 0.2, "crop_y": 0.1, "crop_w": 0.5, "crop_h": 0.5}
    box = geometry.shown_box(settings, (0.3, 0.2, 0.1, 0.1), W, H)
    assert box == pytest.approx((0.2, 0.2, 0.2, 0.2))


def test_a_face_the_crop_left_out_has_no_box():
    settings = {"crop_x": 0.5, "crop_y": 0.0, "crop_w": 0.5, "crop_h": 1.0}
    assert geometry.shown_box(settings, (0.1, 0.2, 0.1, 0.1), W, H) is None
    # Half in: the part that is in.
    assert geometry.shown_box(settings, (0.45, 0.2, 0.1, 0.1), W, H) == pytest.approx((0.0, 0.2, 0.1, 0.1))


def test_shown_point_is_the_way_back_of_source_point():
    settings = {"angle": -8.5, "crop_x": 0.15, "crop_y": 0.2, "crop_w": 0.6, "crop_h": 0.55}
    for u, v in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.3), (0.25, 0.8)]:
        sx, sy = geometry.source_point(settings, u, v, W, H)
        assert geometry.shown_point(settings, sx, sy, W, H) == pytest.approx((u, v), abs=1e-6)


def test_the_box_follows_the_face_through_a_turn_and_a_crop():
    """A white square on the frame, put through the edit as the viewer's picture
    is, lands inside the box the server gives for it."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    face = (0.55, 0.35, 0.08, 0.1)
    left, top = int(face[0] * W), int(face[1] * H)
    img.paste((255, 255, 255), (left, top, left + int(face[2] * W), top + int(face[3] * H)))
    settings = {"angle": 10, "crop_x": 0.2, "crop_y": 0.15, "crop_w": 0.6, "crop_h": 0.6}
    out = geometry.apply_geometry(img, settings)
    ys, xs = np.nonzero(np.asarray(out).sum(axis=2) > 600)
    ow, oh = out.size
    x, y, w, h = geometry.shown_box(settings, face, W, H)
    assert x * ow - 2 <= xs.min() and xs.max() <= (x + w) * ow + 2
    assert y * oh - 2 <= ys.min() and ys.max() <= (y + h) * oh + 2
    # The box hugs the turned square: no bigger than its turned corners need.
    assert w * ow < (xs.max() - xs.min()) * 1.1 + 4 and h * oh < (ys.max() - ys.min()) * 1.1 + 4


def test_a_box_drawn_on_the_cropped_picture_lands_on_the_frame():
    settings = {"crop_x": 0.2, "crop_y": 0.1, "crop_w": 0.5, "crop_h": 0.5}
    assert geometry.source_box(settings, (0.2, 0.2, 0.2, 0.2), W, H) == pytest.approx((0.3, 0.2, 0.1, 0.1))


# ── Through the API ─────────────────────────────────────────────────

PHOTO = "2026/2026-10-17/IMG_0001.JPG"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    library = tmp_path / "lib"
    (library / "2026" / "2026-10-17").mkdir(parents=True)
    Image.new("RGB", (W, H), (90, 90, 90)).save(library / PHOTO)
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main
    from catalog.database import PhotoRecord

    with TestClient(main.app) as c:
        catalog = main.app.state.catalog
        catalog.db.upsert_photo(PhotoRecord(filepath=PHOTO, filename="IMG_0001.JPG", folder="2026/2026-10-17",
                                            base_name="IMG_0001", extension=".jpg"))
        face = SimpleNamespace(x=0.3, y=0.2, w=0.1, h=0.1, score=0.9, px=120.0, quality=0.8,
                               embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        catalog.db.replace_faces(PHOTO, [face], 1, 1, model="arcface")
        yield c, catalog


def test_the_viewer_gets_the_box_on_the_finished_picture(client):
    c, catalog = client
    plain = c.get(f"/api/faces/photo/{PHOTO}").json()["faces"][0]
    assert "shown" not in plain and plain["x"] == pytest.approx(0.3)

    catalog.set_photo_edit(PHOTO, {"crop_x": 0.2, "crop_y": 0.1, "crop_w": 0.5, "crop_h": 0.5})
    face = c.get(f"/api/faces/photo/{PHOTO}").json()["faces"][0]
    assert face["x"] == pytest.approx(0.3)                   # kept on the frame
    assert [face["shown"][k] for k in "xywh"] == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_a_box_drawn_on_the_cropped_picture_is_kept_on_the_frame(client):
    c, catalog = client
    catalog.set_photo_edit(PHOTO, {"crop_x": 0.2, "crop_y": 0.1, "crop_w": 0.5, "crop_h": 0.5})
    res = c.post("/api/faces/drawn", json={"filepath": PHOTO, "x": 0.6, "y": 0.6, "w": 0.2, "h": 0.2,
                                           "shown": True, "name": "Grandpa"})
    assert res.status_code == 200
    drawn = next(f for f in catalog.db.list_faces(PHOTO) if f["drawn"])
    assert (drawn["x"], drawn["y"], drawn["w"], drawn["h"]) == pytest.approx((0.5, 0.4, 0.1, 0.1))


def test_faces_drawn_on_a_crop_before_the_fix_are_put_on_the_frame_once(client):
    from faces.service import REPAIRED_KEY, repair_drawn_on_edits

    c, catalog = client
    db = catalog.db
    # A library from before the fix: the start-up pass has not been.
    with db._connect() as conn:
        conn.execute("DELETE FROM library_settings WHERE key = ?", (REPAIRED_KEY,))
    catalog.set_photo_edit(PHOTO, {"crop_x": 0.2, "crop_y": 0.1, "crop_w": 0.5, "crop_h": 0.5})
    # Drawn the old way, on the cropped picture, after the crop was saved.
    index = db.add_drawn_face(PHOTO, 0.6, 0.6, 0.2, 0.2, px=200)
    with db._connect() as conn:
        conn.execute("UPDATE photo_faces SET created_at = '2999-01-01' WHERE face_index = ?", (index,))
    cache = c.app.state.thumb_cache
    root = c.app.state.config.library_root
    assert repair_drawn_on_edits(db, catalog, cache, root) == 1
    drawn = next(f for f in db.list_faces(PHOTO) if f["drawn"])
    assert (drawn["x"], drawn["y"], drawn["w"], drawn["h"]) == pytest.approx((0.5, 0.4, 0.1, 0.1))
    assert repair_drawn_on_edits(db, catalog, cache, root) == 0      # once


def test_a_face_drawn_before_the_crop_stays(client):
    from faces.service import REPAIRED_KEY, repair_drawn_on_edits

    c, catalog = client
    db = catalog.db
    index = db.add_drawn_face(PHOTO, 0.6, 0.6, 0.2, 0.2, px=200)
    with db._connect() as conn:
        conn.execute("DELETE FROM library_settings WHERE key = ?", (REPAIRED_KEY,))
        conn.execute("UPDATE photo_faces SET created_at = '2000-01-01' WHERE face_index = ?", (index,))
    catalog.set_photo_edit(PHOTO, {"crop_x": 0.2, "crop_y": 0.1, "crop_w": 0.5, "crop_h": 0.5})
    assert repair_drawn_on_edits(db, catalog, c.app.state.thumb_cache, c.app.state.config.library_root) == 0
    drawn = next(f for f in db.list_faces(PHOTO) if f["drawn"])
    assert (drawn["x"], drawn["y"]) == pytest.approx((0.6, 0.6))
