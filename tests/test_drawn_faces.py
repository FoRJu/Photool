"""
A face drawn by hand where the detector found none: it names the photograph,
outlives every rescan and regroup, and never teaches the grouping.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord
from catalog.service import CatalogService
from faces import service as face_service
from faces.clustering import unit

PHOTO = "2025/2025-03-06/MK2A8100.JPG"


def _face(x, vector=(1.0, 0.0, 0.0)):
    return SimpleNamespace(x=x, y=0.2, w=0.1, h=0.1, score=0.9, px=120.0, quality=0.8,
                           embedding=unit(np.asarray(vector, dtype=np.float32)))


def _catalog(tmp: Path):
    db = CatalogDB(str(tmp / "catalog.db"))
    for name in ("MK2A8100.JPG", "MK2A8101.JPG"):
        db.upsert_photo(PhotoRecord(filepath=f"2025/2025-03-06/{name}", filename=name,
                                    folder="2025/2025-03-06", base_name=Path(name).stem, extension=".jpg"))
    service = CatalogService(db, str(tmp / "library"))
    # Ada, found and named in the other frame.
    db.replace_faces("2025/2025-03-06/MK2A8101.JPG", [_face(0.5)], 1, 1, model="arcface")
    service.move_faces([("2025/2025-03-06/MK2A8101.JPG", 0)], name="Ada")
    db.replace_faces(PHOTO, [], 1, 1, model="arcface")      # scanned: nobody found
    return db, service


def _drawn(db):
    return [f for f in db.list_faces(PHOTO) if f["drawn"]]


def test_a_drawn_face_names_the_photo(tmp_path):
    db, service = _catalog(tmp_path)
    ada = db.clusters_named("Ada")[0]
    result = service.draw_face(PHOTO, (0.4, 0.3, 0.1, 0.15), into=ada, px=300)
    assert result["face_index"] == CatalogDB.DRAWN_FACE_INDEX and result["into"] == ada
    assert "Ada" in db.get_photo_tags(PHOTO)
    assert db.cluster_names([_drawn(db)[0]["cluster_id"]]) == {ada: "Ada"}


def test_a_new_name_makes_a_person(tmp_path):
    db, service = _catalog(tmp_path)
    service.draw_face(PHOTO, (0.4, 0.3, 0.1, 0.15), name="Grandpa")
    assert db.clusters_named("Grandpa") and "Grandpa" in db.get_photo_tags(PHOTO)


def test_it_outlives_a_rescan_a_prune_and_a_regroup(tmp_path):
    db, service = _catalog(tmp_path)
    service.draw_face(PHOTO, (0.4, 0.3, 0.1, 0.15), name="Grandpa", px=10)

    db.replace_faces(PHOTO, [_face(0.1, (0.0, 1.0, 0.0))], 2, 2, model="arcface")   # a look closer
    db.prune_small_faces(40.0)                    # drawn small, still kept
    face_service.rebuild_people(db, 0.42, min_group_px=40.0, model="arcface")

    drawn = _drawn(db)
    assert len(drawn) == 1 and drawn[0]["face_index"] == CatalogDB.DRAWN_FACE_INDEX
    assert db.cluster_names([drawn[0]["cluster_id"]]) == {drawn[0]["cluster_id"]: "Grandpa"}
    assert {f["face_index"] for f in db.list_faces(PHOTO)} == {0, CatalogDB.DRAWN_FACE_INDEX}


def test_it_never_shapes_the_grouping(tmp_path):
    db, service = _catalog(tmp_path)
    service.draw_face(PHOTO, (0.4, 0.3, 0.1, 0.15), name="Ada")
    assert all(not f["drawn"] for f in db.list_all_faces(model="arcface"))


def test_taking_it_away_takes_the_name_off(tmp_path):
    db, service = _catalog(tmp_path)
    result = service.draw_face(PHOTO, (0.4, 0.3, 0.1, 0.15), name="Grandpa")
    assert service.remove_drawn_face(PHOTO, result["face_index"])
    assert _drawn(db) == [] and "Grandpa" not in db.get_photo_tags(PHOTO)
    # A face the detector found is not a drawn one, and is not taken away this way.
    assert not service.remove_drawn_face("2025/2025-03-06/MK2A8101.JPG", 0)
