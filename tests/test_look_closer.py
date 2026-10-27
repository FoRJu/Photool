"""
Faces in the viewer: which half of a pair holds a photograph's faces, looking
closer at full size without losing the names already given, and the names the
viewer shows.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB
from catalog.service import CatalogService
from faces import service as face_service
from faces.clustering import unit


def _face(x, y, w=0.1, h=0.1, px=120.0, vector=(1.0, 0.0)):
    return SimpleNamespace(x=x, y=y, w=w, h=h, score=0.9, px=px, quality=1.0,
                           embedding=unit(np.asarray(vector, dtype=np.float32)))


def _library(tmp: Path, names):
    folder = tmp / "library" / "2026" / "2026-10-17"
    folder.mkdir(parents=True)
    for name in names:
        if name.endswith(".JPG"):
            Image.new("RGB", (60, 40)).save(folder / name, format="JPEG")
        else:
            (folder / name).write_bytes(b"raw")
    db = CatalogDB(str(tmp / "catalog.db"))
    service = CatalogService(db, str(tmp / "library"))
    service.index_folder("2026/2026-10-17", [".cr3", ".jpg"])
    return tmp / "library", db, service


def test_either_half_of_a_pair_answers_for_its_faces(tmp_path):
    _root, db, _service = _library(tmp_path, ["A.CR3", "A.JPG"])
    db.replace_faces("2026/2026-10-17/A.JPG", [_face(0.1, 0.1)], 1, 1, model="arcface")
    for shown in ("2026/2026-10-17/A.CR3", "2026/2026-10-17/A.JPG"):
        assert face_service.scanned_half(db, shown) == "2026/2026-10-17/A.JPG"


def test_an_unscanned_pair_answers_with_the_half_a_scan_would_read(tmp_path):
    _root, db, _service = _library(tmp_path, ["A.CR3", "A.JPG"])
    shown = {p.filepath for p in db.get_photos_in_folder("2026/2026-10-17", pair_mode="paired")}
    assert face_service.scanned_half(db, "2026/2026-10-17/A.CR3") in shown


def test_looking_closer_finds_more_and_keeps_the_names(tmp_path, monkeypatch):
    root, db, service = _library(tmp_path, ["A.JPG"])
    path = "2026/2026-10-17/A.JPG"
    db.replace_faces(path, [_face(0.1, 0.1)], 1, 1, model="arcface")
    service.move_faces([(path, 0)], name="Ada")

    # At full size the same face is found again, a little tighter, and a small
    # one at the back that the preview missed.
    monkeypatch.setattr("faces.detect.configure", lambda *a, **k: None)
    monkeypatch.setattr("faces.detect.find_faces", lambda image, min_px=40: [
        _face(0.3, 0.6, 0.02, 0.02, px=45.0, vector=(0.0, 1.0)), _face(0.102, 0.101)])
    monkeypatch.setattr(face_service, "full_image", lambda source: Image.new("RGB", (600, 400)))

    result = face_service.look_closer(db, str(root), [path], "models", min_group_px=40.0)
    assert result == {"scanned": 1, "skipped": 0, "failed": 0, "faces": 2, "found": 1}
    rows = db.list_faces(path)
    names = db.cluster_names(r["cluster_id"] for r in rows)
    assert sorted(names.get(r["cluster_id"], "") for r in rows) == ["", "Ada"]
    assert db.faces_are_scanned(path)


def test_a_photo_that_cannot_be_read_is_counted_not_fatal(tmp_path, monkeypatch):
    root, db, _service = _library(tmp_path, ["A.JPG", "B.JPG"])
    monkeypatch.setattr("faces.detect.configure", lambda *a, **k: None)
    monkeypatch.setattr("faces.detect.find_faces", lambda image, min_px=40: [])
    monkeypatch.setattr(face_service, "full_image",
                        lambda source: None if source.name == "A.JPG" else Image.new("RGB", (60, 40)))
    result = face_service.look_closer(
        db, str(root), ["2026/2026-10-17/A.JPG", "2026/2026-10-17/B.JPG"], "models")
    assert result["failed"] == 1 and result["scanned"] == 1


def test_the_viewer_gets_the_named_people_in_order(tmp_path):
    _root, db, service = _library(tmp_path, ["A.JPG", "B.JPG"])
    db.replace_faces("2026/2026-10-17/A.JPG", [_face(0.1, 0.1)], 1, 1, model="arcface")
    db.replace_faces("2026/2026-10-17/B.JPG", [_face(0.1, 0.1)], 1, 1, model="arcface")
    service.move_faces([("2026/2026-10-17/A.JPG", 0)], name="zoe")
    service.move_faces([("2026/2026-10-17/B.JPG", 0)], name="Ada")
    assert [p["name"] for p in db.named_clusters()] == ["Ada", "zoe"]
    assert db.cluster_names([0]) == {}
