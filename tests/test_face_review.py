"""
Checking a person's group by eye, and changing many photos at once: faces least
alike first, taking faces out or moving them to someone else (with the names on
the photos kept true), and one batch edit for stars, flags, colours and tags.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord, parse_refused  # noqa: E402
from catalog.service import CatalogService  # noqa: E402
from faces.service import person_faces  # noqa: E402

FOLDER = "2026/2026-07-04"


def _vec(*values):
    v = np.zeros(8, dtype=np.float32)
    v[: len(values)] = values
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture()
def lib(tmp_path):
    root = tmp_path / "lib"
    (root / FOLDER).mkdir(parents=True)
    db = CatalogDB(str(tmp_path / "c.db"))
    for i in range(5):
        name = f"IMG_{i}.JPG"
        (root / FOLDER / name).write_bytes(b"x")
        db.upsert_photo(PhotoRecord(filepath=f"{FOLDER}/{name}", filename=name, folder=FOLDER,
                                    extension=".jpg", base_name=f"IMG_{i}"))
    service = CatalogService(db, str(root))
    alice = db.create_cluster()
    bob = db.create_cluster()
    # Alice's faces point one way; IMG_3's face is really Bob's, and sits apart.
    faces = [
        (f"{FOLDER}/IMG_0.JPG", alice, _vec(1, 0.05)),
        (f"{FOLDER}/IMG_1.JPG", alice, _vec(1, 0.1)),
        (f"{FOLDER}/IMG_2.JPG", alice, _vec(1, 0.0, 0.05)),
        (f"{FOLDER}/IMG_3.JPG", alice, _vec(0.2, 1)),
        (f"{FOLDER}/IMG_4.JPG", bob, _vec(0.1, 1)),
    ]
    with db._connect() as conn:
        for fp, cluster, vec in faces:
            conn.execute(
                "INSERT INTO photo_faces (filepath, face_index, w, h, px, embedding, cluster_id, quality) "
                "VALUES (?, 0, 0.1, 0.1, 120, ?, ?, 1)", (fp, vec.tobytes(), cluster))
    service.name_person(alice, "Alice")
    service.name_person(bob, "Bob")
    return db, service, alice, bob


def test_the_least_alike_face_comes_first(lib):
    db, _service, alice, _bob = lib
    faces = person_faces(db, alice, 0.4)
    assert [f["filepath"].rsplit("/", 1)[1] for f in faces][0] == "IMG_3.JPG"
    assert faces[0]["similarity"] < faces[-1]["similarity"]
    assert len(faces) == 4


def test_a_face_is_not_made_alike_by_being_its_own_exemplar(lib):
    """IMG_3 is unlike the rest, so it is one of the group's exemplars; it must
    not score as alike through itself."""
    db, _service, alice, _bob = lib
    faces = person_faces(db, alice, 0.4)
    assert faces[0]["filepath"].endswith("IMG_3.JPG") and faces[0]["similarity"] < 0.6


def test_a_profile_is_alike_through_the_groups_other_profiles(tmp_path):
    """Two profiles of a mostly frontal person are scored the way they joined:
    through each other, not against the frontal average alone."""
    db = CatalogDB(str(tmp_path / "c.db"))
    alice = db.create_cluster("Alice")
    frontal = [_vec(1, 0.01 * k, 0.0, 0.01 * (k % 3)) for k in range(8)]
    profiles = [_vec(0.2, 0, 0.98), _vec(0.21, 0.02, 0.97)]
    vectors = frontal + profiles
    for i in range(len(vectors)):
        db.upsert_photo(PhotoRecord(filepath=f"{FOLDER}/IMG_{i}.JPG", filename=f"IMG_{i}.JPG",
                                    folder=FOLDER, extension=".jpg", base_name=f"IMG_{i}"))
    with db._connect() as conn:
        for i, vec in enumerate(vectors):
            fp = f"{FOLDER}/IMG_{i}.JPG"
            conn.execute(
                "INSERT INTO photo_faces (filepath, face_index, w, h, px, embedding, cluster_id, quality) "
                "VALUES (?, 0, 0.1, 0.1, 120, ?, ?, 1)", (fp, vec.tobytes(), alice))
    by_name = {f["filepath"].rsplit("/", 1)[1]: f["similarity"] for f in person_faces(db, alice, 0.4)}
    average = np.sum(vectors, axis=0)
    average /= np.linalg.norm(average)
    for i, vec in enumerate(profiles, start=len(frontal)):
        assert float(vec @ average) < 0.4 + 0.08    # the average alone would flag it
        assert by_name[f"IMG_{i}.JPG"] > 0.4 + 0.08  # not marked "worth a look"


def test_removing_a_face_takes_the_name_off_its_photo_and_is_remembered(lib):
    db, service, alice, _bob = lib
    wrong = f"{FOLDER}/IMG_3.JPG"
    assert "Alice" in db.get_photo_tags(wrong)
    result = service.remove_faces([(wrong, 0)])
    assert result == {"removed": 1, "untagged": 1}
    assert "Alice" not in db.get_photo_tags(wrong)
    row = db.list_faces(wrong)[0]
    assert row["cluster_id"] == 0
    assert alice in parse_refused(row["not_people"])
    # The other photos keep the name.
    assert "Alice" in db.get_photo_tags(f"{FOLDER}/IMG_0.JPG")


def test_moving_faces_to_another_person_moves_the_name_too(lib):
    db, service, alice, bob = lib
    wrong = f"{FOLDER}/IMG_3.JPG"
    result = service.move_faces([(wrong, 0)], into=bob)
    assert result == {"moved": 1, "into": bob}
    assert db.get_photo_tags(wrong) == ["Bob"]
    row = db.list_faces(wrong)[0]
    assert row["cluster_id"] == bob and alice in parse_refused(row["not_people"])


def test_moving_to_a_new_name_makes_the_person(lib):
    db, service, _alice, _bob = lib
    wrong = f"{FOLDER}/IMG_3.JPG"
    result = service.move_faces([(wrong, 0)], name="Carol")
    carol = result["into"]
    assert db.get_cluster(carol)["name"] == "Carol"
    assert db.get_photo_tags(wrong) == ["Carol"]
    # A second move by the same name finds Carol rather than making another.
    again = service.move_faces([(f"{FOLDER}/IMG_2.JPG", 0)], name="carol")
    assert again["into"] == carol


def test_batch_edit_rates_flags_and_tags_many_photos(lib):
    db, service, _alice, _bob = lib
    paths = [f"{FOLDER}/IMG_0.JPG", f"{FOLDER}/IMG_1.JPG", "not/a/photo.jpg"]
    result = service.edit_photos(paths, rating=4, flag=1, color_label=2, add_tags=["Fireworks"])
    assert result["photos"] == 2 and result["tagged"] == 2
    for fp in paths[:2]:
        photo = db.get_photo(fp)
        assert (photo.rating, photo.flag, photo.color_label) == (4, 1, 2)
        assert "Fireworks" in db.get_photo_tags(fp)
    service.edit_photos(paths[:1], rating=9, flag=-5)
    assert (db.get_photo(paths[0]).rating, db.get_photo(paths[0]).flag) == (5, -1)


def test_removing_a_persons_tag_takes_their_face_out_too(lib):
    db, service, alice, _bob = lib
    target = f"{FOLDER}/IMG_1.JPG"
    result = service.edit_photos([target], remove_tags=["alice"])
    assert result["untagged"] == 1 and result["faces_removed"] == 1
    assert db.list_faces(target)[0]["cluster_id"] == 0
    assert "Alice" not in db.get_photo_tags(target)
    # A rename puts the name back only on the faces still in the group.
    service.name_person(alice, "Alice B")
    assert "Alice B" not in db.get_photo_tags(target)


def test_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image

    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (40, 30), (40 * i, 80, 90)).save(library / FOLDER / f"IMG_{i}.JPG")
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\nanalysis_auto_enabled: false\nthumbnail_queue_enabled: false\n")
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app) as c:
        c.post(f"/api/catalog/index/{FOLDER}")
        db = main.app.state.catalog.db
        person = db.create_cluster("Dana")
        with db._connect() as conn:
            for i in range(2):
                conn.execute("INSERT INTO photo_faces (filepath, face_index, px, embedding, cluster_id) "
                             "VALUES (?, 0, 100, ?, ?)", (f"{FOLDER}/IMG_{i}.JPG", _vec(1, i).tobytes(), person))
        faces = c.get(f"/api/faces/people/{person}/faces").json()
        assert faces["name"] == "Dana" and len(faces["faces"]) == 2
        assert c.get("/api/faces/people/999/faces").status_code == 404

        res = c.post("/api/faces/move", json={"faces": [{"filepath": f"{FOLDER}/IMG_1.JPG"}], "name": "Eve"})
        assert res.status_code == 200 and res.json()["moved"] == 1
        assert c.post("/api/faces/move", json={"faces": []}).status_code == 400
        res = c.post("/api/faces/remove", json={"faces": [{"filepath": f"{FOLDER}/IMG_0.JPG"}]})
        assert res.json()["removed"] == 1

        res = c.post("/api/catalog/batch", json={
            "filepaths": [f"{FOLDER}/IMG_0.JPG", f"{FOLDER}/IMG_1.JPG"], "rating": 3, "add_tags": ["Picnic"]})
        assert res.json()["photos"] == 2
        photos = {p["filepath"]: p for p in c.get(f"/api/catalog/photos/{FOLDER}").json()["photos"]}
        assert photos[f"{FOLDER}/IMG_1.JPG"]["rating"] == 3
        assert "Picnic" in photos[f"{FOLDER}/IMG_0.JPG"]["tags"]
