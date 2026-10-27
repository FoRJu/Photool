"""
A person's timeline: the best face of each month (or year), oldest first, and
the portraits that hold the face in one place from frame to frame.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from faces import service as face_service  # noqa: E402
from faces.service import PORTRAIT_ASPECT, PORTRAIT_EYE_LINE, PORTRAIT_FACE, portrait  # noqa: E402
from faces.timeline import face_score, timeline  # noqa: E402


def _face(name, date="", folder="misc", px=200.0, quality=1.0, rating=0, flag=0, index=0):
    return {"filepath": f"{folder}/{name}", "face_index": index, "x": 0.4, "y": 0.3, "w": 0.1, "h": 0.15,
            "px": px, "quality": quality, "capture_date": date, "folder": folder,
            "rating": rating, "flag": flag}


def test_one_face_per_month_oldest_first_the_best_kept():
    faces = [
        _face("a.jpg", "2020:05:02 10:00:00", px=80),
        _face("b.jpg", "2020:05:20 10:00:00", px=220),
        _face("c.jpg", "2019:12:24 10:00:00"),
        _face("d.jpg", "2020:05:21 10:00:00", px=220, quality=0.3),
    ]
    out = timeline(faces)
    assert [p["period"] for p in out["periods"]] == ["2019-12", "2020-05"]
    may = out["periods"][1]
    assert may["filepath"] == "misc/b.jpg" and may["photos"] == 3 and may["date"] == "2020-05-20"


def test_a_picked_or_rated_frame_beats_a_slightly_bigger_face():
    plain = _face("plain.jpg", "2021:01:01 00:00:00", px=240)
    picked = _face("picked.jpg", "2021:01:02 00:00:00", px=200, flag=1)
    assert face_score(picked) > face_score(plain)
    assert timeline([plain, picked])["periods"][0]["filepath"] == "misc/picked.jpg"


def test_rejects_are_left_out_and_undated_photos_counted():
    faces = [
        _face("r.jpg", "2022:03:01 00:00:00", px=240, flag=-1),
        _face("k.jpg", "2022:03:02 00:00:00", px=60),
        _face("u1.jpg"), _face("u1.jpg", index=1), _face("u2.jpg"),
    ]
    out = timeline(faces)
    assert [p["filepath"] for p in out["periods"]] == ["misc/k.jpg"]
    assert out["undated"] == 2


def test_a_folder_date_stands_in_for_missing_exif():
    out = timeline([_face("x.jpg", folder="2018/2018-07-04")])
    assert out["periods"][0]["period"] == "2018-07"


def test_by_year():
    faces = [_face("a.jpg", "2020:01:01 00:00:00"), _face("b.jpg", "2020:11:01 00:00:00"),
             _face("c.jpg", "2023:06:01 00:00:00")]
    out = timeline(faces, "year")
    assert [(p["period"], p["photos"]) for p in out["periods"]] == [("2020", 2), ("2023", 1)]


def test_ties_keep_the_earlier_frame():
    a = _face("b_later_name.jpg", "2020:05:01 00:00:00")
    b = _face("a_earlier_name.jpg", "2020:05:09 00:00:00")
    assert timeline([b, a])["periods"][0]["filepath"] == "misc/b_later_name.jpg"
    assert timeline([a, b])["periods"][0]["filepath"] == "misc/b_later_name.jpg"


@pytest.fixture()
def person(tmp_path):
    db = CatalogDB(str(tmp_path / "c.db"))
    folder = "2024/2024-06-01"
    alice = db.create_cluster("Alice")
    for i, (date, rating) in enumerate([("2024:06:01 09:00:00", 0), ("", 3)]):
        name = f"IMG_{i}.JPG"
        db.upsert_photo(PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder,
                                    extension=".jpg", base_name=f"IMG_{i}", capture_date=date, rating=rating))
    with db._connect() as conn:
        for i in range(2):
            conn.execute(
                "INSERT INTO photo_faces (filepath, face_index, x, y, w, h, px, embedding, cluster_id, quality) "
                "VALUES (?, 0, 0.5, 0.25, 0.1, 0.2, 120, ?, ?, 1)",
                (f"{folder}/IMG_{i}.JPG", np.ones(8, np.float32).tobytes(), alice))
    return db, alice, folder


def test_the_catalog_gives_faces_with_their_photos_dates(person):
    db, alice, folder = person
    rows = db.faces_with_dates(alice)
    assert {r["filepath"] for r in rows} == {f"{folder}/IMG_0.JPG", f"{folder}/IMG_1.JPG"}
    assert {r["rating"] for r in rows} == {0, 3}
    out = timeline(rows)
    # Both are June 2024 — one by EXIF, one by its folder — and the rated one wins.
    assert len(out["periods"]) == 1 and out["periods"][0]["filepath"].endswith("IMG_1.JPG")


def test_a_portrait_holds_the_face_in_the_same_place(person, monkeypatch, tmp_path):
    """A bright box drawn where the face is lands centred across, on the eye
    line, at the set share of the height — and past the photo's edge is black."""
    db, _alice, folder = person
    w, h = 1000, 800
    img = np.full((h, w, 3), 90, np.uint8)
    # The face, near the top: x 0.5-0.6, y 0.02-0.22 of the frame, 100 x 160 px.
    img[16:176, 500:600] = 255
    with db._connect() as conn:
        conn.execute("UPDATE photo_faces SET y = 0.02 WHERE filepath = ?", (f"{folder}/IMG_0.JPG",))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    monkeypatch.setattr("thumbnails.get_thumbnail", lambda *a, **k: buf.getvalue())

    jpeg = portrait(db, None, str(tmp_path), f"{folder}/IMG_0.JPG", 0, width=600)
    out = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L")).astype(float)
    oh, ow = out.shape
    assert (ow, oh) == (600, round(600 / PORTRAIT_ASPECT))
    ys, xs = np.nonzero(out > 200)
    assert abs((xs.min() + xs.max()) / 2 - ow / 2) < 4
    assert abs((ys.min() + ys.max()) / 2 - oh * PORTRAIT_EYE_LINE) < 4
    assert abs((ys.max() - ys.min()) / oh - PORTRAIT_FACE) < 0.02
    # The frame runs past the photo's top: black there, grey where the photo is.
    assert out[2, ow // 2] < 10 and out[oh - 3, 5] > 60


def test_no_portrait_for_a_face_that_is_not_there(person, tmp_path):
    db, _alice, folder = person
    assert portrait(db, None, str(tmp_path), f"{folder}/IMG_0.JPG", 5) is None
