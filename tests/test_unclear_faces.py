"""
Faces that cannot say who they are: the recogniser puts blurred, tiny and
turned-away faces near one point whoever they are, and grouped like any other
face they became one "person" of thousands. They are found, but join no one.
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

DIM = 16
BLUR = np.eye(DIM, dtype=np.float32)[0]     # where unclear faces all point


def _face(vector, quality=0.6, px=100.0):
    return SimpleNamespace(x=0.1, y=0.1, w=0.2, h=0.2, score=0.9, px=px, quality=quality,
                           embedding=unit(np.asarray(vector, dtype=np.float32)))


def _blurred(rng, n):
    """Poor faces of many different people, all pulled toward BLUR."""
    return [_face(BLUR * 3 + rng.normal(0, 0.6, DIM), quality=0.02) for _ in range(n)]


def _person(rng, axis, n):
    base = np.eye(DIM, dtype=np.float32)[axis]
    return [_face(base * 3 + rng.normal(0, 0.3, DIM)) for _ in range(n)]


def _db(tmp: Path, faces_by_file: dict[str, list]) -> CatalogDB:
    db = CatalogDB(str(tmp / "catalog.db"))
    for filepath, faces in faces_by_file.items():
        folder, _, filename = filepath.rpartition("/")
        db.upsert_photo(PhotoRecord(filepath=filepath, filename=filename, folder=folder,
                                    base_name=Path(filename).stem, extension=".jpg"))
        db.replace_faces(filepath, faces, 1, 1, model="arcface")
    return db


def test_the_centre_is_where_the_poorest_faces_point():
    rng = np.random.default_rng(0)
    centre = face_service.unclear_centre(_blurred(rng, 150) + _person(rng, 3, 20))
    assert centre is not None and float(centre @ BLUR) > 0.9


def test_too_few_poor_faces_say_nothing():
    rng = np.random.default_rng(0)
    assert face_service.unclear_centre(_blurred(rng, 10) + _person(rng, 3, 50)) is None


def test_blurred_faces_join_no_one_and_people_still_group(tmp_path):
    rng = np.random.default_rng(1)
    blurred, ada, ben = _blurred(rng, 150), _person(rng, 3, 6), _person(rng, 5, 6)
    files = {f"f/blur_{i}.jpg": [face] for i, face in enumerate(blurred)}
    files.update({f"f/ada_{i}.jpg": [face] for i, face in enumerate(ada)})
    files.update({f"f/ben_{i}.jpg": [face] for i, face in enumerate(ben)})
    db = _db(tmp_path, files)

    summary = face_service.rebuild_people(db, 0.42, min_group_px=0, model="arcface")
    assert summary["unclear"] >= 140

    cluster = {f"{r['filepath']}": r["cluster_id"] for r in db.list_all_faces()}
    assert sum(1 for k, v in cluster.items() if k.startswith("f/blur") and v) <= 10
    assert len({cluster[f"f/ada_{i}.jpg"] for i in range(6)}) == 1
    assert len({cluster[f"f/ben_{i}.jpg"] for i in range(6)}) == 1
    assert cluster["f/ada_0.jpg"] != cluster["f/ben_0.jpg"]


def test_a_blurred_face_named_by_hand_keeps_its_person(tmp_path):
    rng = np.random.default_rng(2)
    blurred = _blurred(rng, 150)
    db = _db(tmp_path, {f"f/blur_{i}.jpg": [face] for i, face in enumerate(blurred)})
    CatalogService(db, str(tmp_path)).move_faces([("f/blur_0.jpg", 0)], name="Ada")

    face_service.rebuild_people(db, 0.42, min_group_px=0, model="arcface")
    named = {r["filepath"]: r["cluster_id"] for r in db.list_all_faces() if r["cluster_id"]}
    assert "f/blur_0.jpg" in named
    assert db.cluster_names([named["f/blur_0.jpg"]]) == {named["f/blur_0.jpg"]: "Ada"}


def test_groups_are_listed_by_their_clear_faces(tmp_path):
    rng = np.random.default_rng(3)
    db = _db(tmp_path, {
        **{f"f/sharp_{i}.jpg": [face] for i, face in enumerate(_person(rng, 3, 3))},
        # One clear face starts the group; eight soft ones join it.
        **{f"f/soft_{i}.jpg": [_face(np.eye(DIM)[5] * 3 + rng.normal(0, 0.1, DIM),
                                     quality=0.6 if i == 0 else 0.1)]
           for i in range(9)},
    })
    face_service.rebuild_people(db, 0.42, min_group_px=0, model="arcface")
    groups = db.list_clusters()
    assert [(g["faces"], g["clear"]) for g in groups[:2]] == [(3, 3), (9, 1)]
