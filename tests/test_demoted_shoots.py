"""
Shoots one device finds less important: a whole folder, or one camera's photos
in it. Each device keeps its own; nothing is deleted. They sink in search and in
the folder list, stay out of On this day and Year in review for that device, and
a demoted camera's frames are not a folder's cover (the folder's grid leaves
them out too, in the page).
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from catalog.service import CatalogService  # noqa: E402

SCHOOL = "2025/2025-05-22"
HOME = "2025/2025-05-23"


def _rec(folder, name, camera, when):
    base, ext = name.rsplit(".", 1)
    return PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder, extension="." + ext.lower(),
                       base_name=base, camera=camera, capture_date=when)


@pytest.fixture()
def lib(tmp_path):
    db = CatalogDB(str(tmp_path / "c.db"))
    for r in [
        _rec(SCHOOL, "MK2A0001.JPG", "Canon EOS R6m2", "2025:05:22 10:00:00"),
        _rec(SCHOOL, "MK2A0002.JPG", "Canon EOS R6m2", "2025:05:22 11:00:00"),
        _rec(SCHOOL, "R0000001.JPG", "GR II", "2025:05:22 12:00:00"),
        _rec(SCHOOL, "IMG_0001.JPG", "", "2025:05:22 13:00:00"),
        _rec(HOME, "R0000002.JPG", "GR II", "2025:05:23 18:00:00"),
    ]:
        db.upsert_photo(r)
    return db, CatalogService(db, str(tmp_path))


def test_each_device_keeps_its_own(lib):
    db, _ = lib
    db.set_demoted("device:a", SCHOOL, "GR II")
    db.set_demoted("device:b", HOME)
    assert db.demoted_shoots("device:a") == {(SCHOOL, "GR II")}
    assert db.demoted_shoots("device:b") == {(HOME, "*")}
    assert db.demoted_shoots("local") == set()
    db.set_demoted("device:a", SCHOOL, "GR II", demoted=False)
    assert db.demoted_shoots("device:a") == set()


def test_a_folder_whole_or_one_camera_in_it(lib):
    db, service = lib
    db.set_demoted("p", SCHOOL, "GR II")
    db.set_demoted("p", SCHOOL, "")
    kept = sorted(p.record.filename for p in service._photos_for("p"))
    assert kept == ["MK2A0001.JPG", "MK2A0002.JPG", "R0000002.JPG"]
    db.set_demoted("p", HOME)
    assert sorted(p.record.filename for p in service._photos_for("p")) == ["MK2A0001.JPG", "MK2A0002.JPG"]


def test_on_this_day_and_the_year_leave_them_out_for_that_device_only(lib):
    db, service = lib
    db.set_demoted("p", SCHOOL)
    today = date(2026, 5, 22)
    mine = service.on_this_day(today, "p")
    others = service.on_this_day(today, "q")
    assert all(p.record.folder != SCHOOL for y in mine for p in y["photos"])
    assert any(p.record.folder == SCHOOL for y in others for p in y["photos"])
    review = service.year_review(2025, [], "p")
    assert review["summary"]["photos"] == 1
    assert service.year_review(2025, [], "q")["summary"]["photos"] == 5
    assert service.photo_years("p") == [{"year": 2025, "photos": 1}]


def test_a_cover_is_not_a_demoted_cameras_frame(lib):
    from places.store import build_cards

    db, _ = lib
    db.set_flag(f"{SCHOOL}/R0000001.JPG", 1)          # the pick: the cover, until demoted
    cover = lambda shoots=frozenset(): next(c["cover"] for c in build_cards(db, demoted=shoots)["cards"]
                                           if c["folder"] == SCHOOL)
    assert cover() == f"{SCHOOL}/R0000001.JPG"
    assert not cover(frozenset({(SCHOOL, "GR II")})).endswith("R0000001.JPG")
    # A folder demoted whole keeps its cover; so does one whose every frame is demoted.
    assert cover(frozenset({(SCHOOL, "*")})) == f"{SCHOOL}/R0000001.JPG"
    every = frozenset({(SCHOOL, "GR II"), (SCHOOL, "Canon EOS R6m2"), (SCHOOL, "")})
    assert cover(every) == f"{SCHOOL}/R0000001.JPG"


def test_on_this_day_is_kept_until_what_was_demoted_changes(lib):
    db, service = lib
    today = date(2026, 5, 22)
    first = service.on_this_day(today, "p")
    assert service.on_this_day(today, "p") is first
    db.set_demoted("p", SCHOOL)
    assert service.on_this_day(today, "p") is not first


def test_forgetting_a_folder_forgets_its_demotion(lib):
    db, _ = lib
    db.set_demoted("p", SCHOOL)
    db.forget_folder_tree("2025")
    assert db.demoted_shoots("p") == set()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    library = tmp_path / "lib"
    library.mkdir()
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app) as c:
        db = main.app.state.catalog.db
        for r in [_rec(SCHOOL, "MK2A0001.JPG", "Canon EOS R6m2", "2025:05:22 10:00:00"),
                  _rec(SCHOOL, "R0000001.JPG", "GR II", "2025:05:22 12:00:00"),
                  _rec(HOME, "R0000002.JPG", "GR II", "2025:05:23 18:00:00")]:
            db.upsert_photo(r)
        yield c


def test_through_the_api_demoted_photos_come_last_in_a_search(client):
    res = client.put("/api/library/demoted", json={"folder": SCHOOL, "camera": "GR II"})
    assert res.status_code == 200
    assert res.json()["shoots"] == [{"folder": SCHOOL, "camera": "GR II"}]
    assert client.get("/api/library/demoted").json()["shoots"] == [{"folder": SCHOOL, "camera": "GR II"}]

    found = client.get("/api/catalog/search", params={"q": "date:2025", "sort": "date", "desc": "true"}).json()
    assert found["total"] == 3 and found["demoted"] == 1
    assert found["photos"][-1]["filename"] == "R0000001.JPG"

    client.put("/api/library/demoted", json={"folder": SCHOOL, "camera": "GR II", "demoted": False})
    assert client.get("/api/catalog/search", params={"q": "date:2025"}).json()["demoted"] == 0


def test_on_this_day_asked_twice_through_the_api(client):
    """The kept answer is not the one turned into rows the first time."""
    first = client.get("/api/library/memories", params={"day": "2026-05-22"}).json()
    again = client.get("/api/library/memories", params={"day": "2026-05-22"}).json()
    assert first == again and first["years"]
