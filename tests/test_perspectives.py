"""
Perspectives: a year seen as Family or as School. A perspective is a name and
its people (exact names or "Prefix*"); it picks the shoots its people are most
of, faceless frames of those shoots included. Shared by the household; the one a
device opens on is that device's.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from catalog.perspectives import folder_people, folders_of, matches, suggestions  # noqa: E402
from catalog.service import CatalogService  # noqa: E402

BAND = "2025/2025-10-13"
DINNER = "2025/2025-11-27"
HIKE = "2025/2025-07-04"


def test_names_match_exactly_or_by_prefix_without_case():
    assert matches("BAND Alex", ["BAND*"])
    assert matches("robin", ["Robin"])
    assert not matches("Robin2", ["Robin"])
    assert not matches("Alex", ["BAND*", ""])


def _people(spec):
    """{folder: [(n photos, [names])]} -> rows."""
    rows = []
    for folder, groups in spec.items():
        k = 0
        for n, names in groups:
            for _ in range(n):
                k += 1
                rows += [(folder, f"IMG_{k:04d}", name) for name in names]
    return folder_people(rows)


SCHOOL = {"id": 1, "name": "School", "people": ["BAND*"]}
FAMILY = {"id": 2, "name": "Family", "people": ["Robin", "Sam"]}


def test_a_shoot_belongs_where_its_people_are_most_of_it():
    people = _people({
        # The family's child is in the band: in a third as many band photos.
        BAND: [(30, ["BAND Alex"]), (8, ["Robin"]), (3, ["Robin", "BAND Jo"])],
        DINNER: [(8, ["Robin", "Sam"]), (1, ["BAND Alex"])],
        HIKE: [(10, ["Robin"]), (9, ["BAND Kim"])],      # both: a friend along on a family hike
    })
    both = [SCHOOL, FAMILY]
    assert folders_of(SCHOOL, both, people) == {BAND, HIKE}
    assert folders_of(FAMILY, both, people) == {DINNER, HIKE}


def test_a_shoot_with_too_few_of_them_is_not_theirs():
    people = _people({DINNER: [(1, ["Sam"])]})
    assert folders_of(FAMILY, [FAMILY], people) == set()


def test_suggestions_from_shared_prefixes_and_the_most_photographed():
    names = ["Robin", "BAND Alex", "Noah", "BAND-Pat L.", "Dojo-Rae", "Sam", "Dojo-#2", "Harbor-Lena"]
    out = suggestions(names, [])
    assert {"name": "BAND", "people": ["BAND*"]} in out
    assert {"name": "Dojo", "people": ["Dojo*"]} in out
    family = next(s for s in out if s["name"] == "Family")
    assert family["people"] == ["Robin", "Noah", "Sam", "Harbor-Lena"]
    # Not offered again once a perspective has them.
    assert all(s["name"] != "BAND" for s in suggestions(names, [{"people": ["BAND*"]}]))


@pytest.fixture()
def lib(tmp_path):
    db = CatalogDB(str(tmp_path / "c.db"))
    clusters = {name: db.create_cluster(name) for name in ("Robin", "BAND Alex")}
    photos = []
    for folder, day, n, who in [(BAND, "2025:10:13", 6, "BAND Alex"), (DINNER, "2025:11:27", 4, "Robin")]:
        for i in range(n):
            r = PhotoRecord(filepath=f"{folder}/IMG_{i:03d}.JPG", filename=f"IMG_{i:03d}.JPG", folder=folder,
                            extension=".jpg", base_name=f"IMG_{i:03d}", capture_date=f"{day} {10 + i}:00:00")
            db.upsert_photo(r)
            photos.append(r)
            # The last frame of each shoot has nobody in it: the food, the stage.
            if i < n - 1:
                with db._connect() as conn:
                    conn.execute("INSERT INTO photo_faces (filepath, face_index, w, h, px, embedding, cluster_id, quality) "
                                 "VALUES (?, 0, 0.1, 0.1, 120, ?, ?, 1)",
                                 (r.filepath, np.ones(8, np.float32).tobytes(), clusters[who]))
    return db, CatalogService(db, str(tmp_path))


def test_a_year_through_a_perspective_keeps_its_shoots_faceless_frames_and_all(lib):
    db, service = lib
    school = db.save_perspective("School", ["BAND*"])
    family = db.save_perspective("Family", ["Robin"])
    assert service.year_review(2025, [], "p", school)["summary"]["photos"] == 6
    assert service.year_review(2025, [], "p", family)["summary"]["photos"] == 4
    assert service.year_review(2025, [], "p", 0)["summary"]["photos"] == 10


def test_a_frame_only_of_people_outside_the_perspective_is_left_out(lib):
    """A school friend at the family dinner: the shoot is still the family's, but
    a frame of only the friend is not what the family's year is of, and the
    friend is not one of its people. A frame of both stays."""
    db, service = lib
    family = db.save_perspective("Family", ["Robin"])
    robin, alex = db.clusters_named("Robin")[0], db.clusters_named("BAND Alex")[0]
    for i, who in ((10, [alex]), (11, [robin, alex])):
        r = PhotoRecord(filepath=f"{DINNER}/IMG_{i:03d}.JPG", filename=f"IMG_{i:03d}.JPG", folder=DINNER,
                        extension=".jpg", base_name=f"IMG_{i:03d}", capture_date=f"2025:11:27 2{i - 10}:00:00")
        db.upsert_photo(r)
        with db._connect() as conn:
            for index, cluster in enumerate(who):
                conn.execute("INSERT INTO photo_faces (filepath, face_index, w, h, px, embedding, cluster_id, quality) "
                             "VALUES (?, ?, 0.1, 0.1, 120, ?, ?, 1)",
                             (r.filepath, index, np.ones(8, np.float32).tobytes(), cluster))

    # Nobody named: a stranger's clear face, and a faint face far in the back.
    for i, (quality, px) in ((12, (0.8, 120)), (13, (0.05, 20))):
        r = PhotoRecord(filepath=f"{DINNER}/IMG_{i:03d}.JPG", filename=f"IMG_{i:03d}.JPG", folder=DINNER,
                        extension=".jpg", base_name=f"IMG_{i:03d}", capture_date=f"2025:11:27 2{i - 10}:00:00")
        db.upsert_photo(r)
        with db._connect() as conn:
            conn.execute("INSERT INTO photo_faces (filepath, face_index, w, h, px, embedding, cluster_id, quality) "
                         "VALUES (?, 0, 0.1, 0.1, ?, ?, 0, ?)",
                         (r.filepath, px, np.ones(8, np.float32).tobytes(), quality))

    page = service.year_review(2025, [], "p", family)
    shown = {p.record.base_name for p in page["top"]}
    # The 4 of the dinner, the frame of both, and the one with a faint face.
    assert page["summary"]["photos"] == 6
    assert not {"IMG_010", "IMG_012"} & shown
    assert [p["name"] for p in page["people"]] == ["Robin"]
    # Seen whole, the friend and the stranger are there.
    assert service.year_review(2025, [], "p", 0)["summary"]["photos"] == 14


def test_perspectives_are_shared_and_each_device_has_its_own_default(lib):
    db, _ = lib
    pid = db.save_perspective("Family", ["Robin"])
    db.save_perspective("Family and friends", ["Robin", "Noah"], pid)
    assert db.list_perspectives() == [{"id": pid, "name": "Family and friends", "people": ["Robin", "Noah"]}]
    db.set_owner_setting("device:a", "perspective", str(pid))
    assert db.owner_setting("device:a", "perspective") == str(pid)
    assert db.owner_setting("device:b", "perspective", "0") == "0"
    assert db.delete_perspective(pid) and db.list_perspectives() == []


def test_people_come_from_faces_and_from_name_tags(lib):
    """A frame tagged with a person's name counts as them, face found or not."""
    db, _ = lib
    db.add_tag_to_photos([f"{BAND}/IMG_005.JPG"], "robin")
    rows = db.people_in_photos()
    assert (BAND, "IMG_000", "BAND Alex") in rows and (DINNER, "IMG_000", "Robin") in rows
    assert (BAND, "IMG_005", "Robin") in rows
