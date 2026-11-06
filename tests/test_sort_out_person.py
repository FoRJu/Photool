"""
A named person who is really several people, and the blurred faces that
gathered on one: "Sort out" splits a person into look-alike sets to pick from,
and "tidy" lets go of poor faces that do not look like the person's clear ones.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord, parse_refused
from catalog.service import CatalogService
from faces import service as face_service
from faces.clustering import unit

DIM = 16


def _face(vector, quality=0.6, px=100.0):
    return SimpleNamespace(x=0.1, y=0.1, w=0.2, h=0.2, score=0.9, px=px, quality=quality,
                           embedding=unit(np.asarray(vector, dtype=np.float32)))


def _near(rng, axis, quality=0.6, spread=0.3):
    return _face(np.eye(DIM, dtype=np.float32)[axis] * 3 + rng.normal(0, spread, DIM), quality=quality)


def _library(tmp: Path, faces_by_file: dict) -> tuple[CatalogDB, CatalogService]:
    db = CatalogDB(str(tmp / "catalog.db"))
    for filepath, faces in faces_by_file.items():
        folder, _, filename = filepath.rpartition("/")
        db.upsert_photo(PhotoRecord(filepath=filepath, filename=filename, folder=folder,
                                    base_name=Path(filename).stem, extension=".jpg"))
        db.replace_faces(filepath, faces, 1, 1, model="arcface")
    return db, CatalogService(db, str(tmp))


def _name_all(service, paths, name):
    return service.move_faces([(p, 0) for p in paths], name=name)["into"]


def test_sort_out_splits_a_mixed_person(tmp_path):
    rng = np.random.default_rng(5)
    files = {f"f/robin_{i}.jpg": [_near(rng, 2)] for i in range(8)}
    files.update({f"f/sam_{i}.jpg": [_near(rng, 7)] for i in range(5)})
    files.update({f"f/soft_{i}.jpg": [_face(rng.normal(0, 1, DIM), quality=0.05)] for i in range(3)})
    db, service = _library(tmp_path, files)
    robin = _name_all(service, list(files), "Robin")

    sets = face_service.sort_person(db, robin, 0.42)
    clear_sets = [s for s in sets if s["kind"] == "set"]
    assert [s["count"] for s in clear_sets[:2]] == [8, 5]
    assert {f["filepath"].split("_")[0] for f in clear_sets[0]["faces"]} == {"f/robin"}
    assert sets[-1]["kind"] == "unclear" and sets[-1]["count"] == 3

    # Keep the first set: Sam's faces leave remembered, the unclear ones just leave.
    sam = [(f["filepath"], f["face_index"]) for f in clear_sets[1]["faces"]]
    soft = [(f["filepath"], f["face_index"]) for f in sets[-1]["faces"]]
    result = service.sort_out_person(robin, sam, soft)
    assert (result["removed"], result["released"]) == (5, 3)

    rows = {r["filepath"]: r for r in db.list_all_faces()}
    assert all(rows[f"f/robin_{i}.jpg"]["cluster_id"] == robin for i in range(8))
    assert all(rows[f"f/sam_{i}.jpg"]["cluster_id"] == 0 for i in range(5))
    assert all(robin in parse_refused(rows[f"f/sam_{i}.jpg"]["not_people"]) for i in range(5))
    assert all(robin not in parse_refused(rows[f"f/soft_{i}.jpg"]["not_people"]) for i in range(3))
    # The name came off the photos Robin is no longer in.
    assert "Robin" in db.get_photo_tags("f/robin_0.jpg")
    assert "Robin" not in db.get_photo_tags("f/sam_0.jpg")

    # A rebuild does not put Sam back with Robin.
    face_service.rebuild_people(db, 0.42, min_group_px=0, model="arcface")
    rows = {r["filepath"]: r for r in db.list_all_faces()}
    assert all(rows[f"f/sam_{i}.jpg"]["cluster_id"] != robin for i in range(5))


def test_tidy_lets_go_of_poor_faces_unlike_the_person(tmp_path):
    rng = np.random.default_rng(6)
    files = {f"f/alex_{i}.jpg": [_near(rng, 3)] for i in range(6)}
    files.update({f"f/alex_soft_{i}.jpg": [_near(rng, 3, quality=0.1, spread=0.4)] for i in range(4)})
    files.update({f"f/other_soft_{i}.jpg": [_near(rng, 11, quality=0.1)] for i in range(4)})
    db, service = _library(tmp_path, files)
    alex = _name_all(service, list(files), "Alex")

    looked = service.tidy_named(0.42, apply=False)
    assert looked["released"] == 4 and not looked["applied"]
    assert all(r["cluster_id"] == alex for r in db.list_all_faces())     # nothing moved yet

    done = service.tidy_named(0.42, apply=True)
    assert done["released"] == 4 and done["applied"] and done["untagged"] == 4
    rows = {r["filepath"]: r for r in db.list_all_faces()}
    assert all(rows[f"f/other_soft_{i}.jpg"]["cluster_id"] == 0 for i in range(4))
    assert all(rows[f"f/alex_soft_{i}.jpg"]["cluster_id"] == alex for i in range(4))
    assert all(rows[f"f/alex_{i}.jpg"]["cluster_id"] == alex for i in range(6))


def test_tidy_leaves_a_person_with_too_few_clear_faces(tmp_path):
    rng = np.random.default_rng(7)
    files = {"f/kit_0.jpg": [_near(rng, 4)]}
    files.update({f"f/kit_soft_{i}.jpg": [_near(rng, 9, quality=0.1)] for i in range(5)})
    db, service = _library(tmp_path, files)
    _name_all(service, list(files), "Kit")
    assert service.tidy_named(0.42, apply=True)["released"] == 0
