"""
A photograph's time where it was taken, when its camera's clock was on another
zone (catalog/clock.py): an R6 Mark II left on +10:00 after a trip, a DJI on UTC
abroad, a Ricoh that says nothing about its clock.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

pytest.importorskip("tzdata")

from catalog import clock  # noqa: E402
from catalog.database import CatalogDB, PhotoRecord  # noqa: E402

HOME = "America/Los_Angeles"


def _rec(folder, name, camera, when):
    base, ext = name.rsplit(".", 1)
    return PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder, extension="." + ext.lower(),
                       base_name=base, camera=camera, capture_date=when, file_size=100)


@pytest.fixture()
def db(tmp_path):
    return CatalogDB(str(tmp_path / "c.db"))


def test_what_a_file_says_about_its_clock():
    assert clock.recorded_offset({"OffsetTimeOriginal": "+10:00", "TimeZone": "-07:00"}) == "+10:00"
    assert clock.recorded_offset({"TimeZone": "-07:00"}) == "-07:00"
    # DJI writes the zone only on XMP's CreateDate.
    assert clock.recorded_offset({"CreateDate": "2026:08:23 04:55:20+00:00"}) == "+00:00"
    assert clock.recorded_offset({"CreateDate": "2026:10:17 19:29:45"}) == ""
    assert clock.recorded_offset({}) == ""


def test_a_clock_left_on_a_trips_time_at_home():
    # 02:55 by a clock on +10 was 08:55 the day before in California.
    assert clock.local_time("2025:03:08 02:55:10", "+10:00", where_zone=HOME) == "2025:03:07 08:55:10"
    # In summer California is -7.
    assert clock.local_time("2026:05:02 01:25:00", "+10:00", where_zone=HOME) == "2026:05:01 08:25:00"


def test_summer_time_left_on_in_winter_is_an_hour():
    assert clock.local_time("2024:12:04 17:46:00", "-07:00", where_zone=HOME) == "2024:12:04 16:46:00"


def test_a_clock_already_on_the_time_there_is_left_alone():
    assert clock.local_time("2026:07:20 16:53:00", "-07:00", where_zone=HOME) == ""
    # No zone recorded and none given: the clock is taken to be on local time.
    assert clock.local_time("2026:08:22 12:45:00", "", where_zone=HOME) == ""
    # Nowhere known: nothing to work out.
    assert clock.local_time("2025:03:08 02:55:10", "+10:00") == ""
    assert clock.local_time("0000:00:00 00:00:00", "+10:00", where_zone=HOME) == ""


def test_a_named_clock_zone_wins_over_the_file():
    assert clock.local_time("2026:08:23 02:33:48", "", clock_zone="UTC",
                            where_zone="Europe/Rome") == "2026:08:23 04:33:48"


def _library(db):
    for r in [
        _rec("2025/2025-03-08", "MK2A0001.CR3", "Canon EOS R6m2", "2025:03:08 02:55:10"),
        # The camera's JPEG twin carries no camera and no zone: it goes by the raw.
        _rec("2025/2025-03-08", "MK2A0001.JPG", "", "2025:03:08 02:55:10"),
        _rec("2026/2026-08-22", "R0000001.DNG", "GR II", "2026:08:22 20:40:00"),
        _rec("2026/2026-08-22", "DJI_0001.JPG", "PP-101", "2026:08:22 05:12:15"),
    ]:
        db.upsert_photo(r)
    clock.record(db, [("2025/2025-03-08/MK2A0001.CR3", 100, "+10:00"),
                      ("2025/2025-03-08/MK2A0001.JPG", 100, ""),
                      ("2026/2026-08-22/R0000001.DNG", 100, ""),
                      ("2026/2026-08-22/DJI_0001.JPG", 100, "+00:00")])


def test_the_library_put_right_from_home_and_a_rule(db):
    _library(db)
    clock.set_home_zone(db, HOME)
    clock.recompute(db)
    dates = {r.filename: r.capture_date for r in db.get_photos_in_folder("2025/2025-03-08")}
    assert dates == {"MK2A0001.CR3": "2025:03:07 08:55:10", "MK2A0001.JPG": "2025:03:07 08:55:10"}
    # The DJI keeps UTC; without a rule it is read as at home.
    dji = next(r for r in db.get_photos_in_folder("2026/2026-08-22") if r.camera == "PP-101")
    assert dji.capture_date == "2026:08:21 22:12:15"
    # It was abroad: its frames are that place's time; the Ricoh at home is untouched.
    clock.add_rule(db, "PP-101", "2026-08-01", "2026-08-31", where_zone="Europe/Rome")
    clock.recompute(db)
    shown = {r.camera: r.capture_date for r in db.get_photos_in_folder("2026/2026-08-22")}
    assert shown == {"PP-101": "2026:08:22 07:12:15", "GR II": "2026:08:22 20:40:00"}


def test_the_clock_reading_is_kept(db):
    _library(db)
    clock.set_home_zone(db, HOME)
    clock.recompute(db)
    record = db.get_photo("2025/2025-03-08/MK2A0001.CR3")
    assert record.camera_date == "2025:03:08 02:55:10"
    # Written back, the record keeps what the clock said; worked out again, the same.
    db.upsert_photo(record)
    with db._connect() as conn:
        assert conn.execute("SELECT capture_date FROM photos WHERE filepath = ?",
                            (record.filepath,)).fetchone()[0] == "2025:03:08 02:55:10"
    assert clock.recompute(db) == 0


def test_removing_the_rule_and_home_undoes_it(db):
    _library(db)
    clock.set_home_zone(db, HOME)
    rule = clock.add_rule(db, "PP-101", where_zone="Europe/Rome")
    clock.recompute(db)
    clock.remove_rule(db, rule)
    clock.set_home_zone(db, "")
    clock.recompute(db)
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM photos WHERE local_date != ''").fetchone()[0] == 0


@dataclass
class _Town:
    zone: str


class _Towns:
    """A town list that knows one place: Rome."""

    def nearest(self, lat, lon):
        return _Town("Europe/Rome")


def test_a_folders_place_says_where_it_was(db):
    # Rome with the camera still on California summer time.
    db.upsert_photo(_rec("2024/2024-11-30", "MK2A0100.CR3", "Canon EOS R6m2", "2024:11:30 18:27:00"))
    clock.record(db, [("2024/2024-11-30/MK2A0100.CR3", 100, "-07:00")])
    with db._connect() as conn:
        conn.execute("INSERT INTO folder_notes (folder, place, lat, lon) VALUES (?, 'Rome', 41.9, 12.5)",
                     ("2024/2024-11-30",))
    clock.set_home_zone(db, HOME)
    clock.recompute(db, _Towns())
    assert db.get_photo("2024/2024-11-30/MK2A0100.CR3").capture_date == "2024:12:01 02:27:00"


def test_the_folders_that_moved_and_their_cards(db):
    from places.store import build_cards

    _library(db)
    clock.set_home_zone(db, HOME)
    clock.recompute(db)
    moved = clock.moved_days(db)
    assert moved["photos"] == 2                   # the R6m2 pair is one photograph, and the DJI
    days = {f["folder"]: [(m["from"], m["to"]) for m in f["moves"]] for f in moved["folders"]}
    assert days["2025/2025-03-08"] == [("2025-03-08", "2025-03-07")]
    # The folder named for the 17th shows the day it was.
    card = next(c for c in build_cards(db)["cards"] if c["folder"] == "2025/2025-03-08")
    assert card["date_from"] == card["date_to"] == "2025-03-07"


def test_reading_waits_for_each_file_once(db):
    _library(db)
    assert clock.pending(db) == []
    with db._connect() as conn:
        conn.execute("UPDATE photos SET file_size = 200 WHERE filename = 'DJI_0001.JPG'")
    assert [r["filepath"] for r in clock.pending(db)] == ["2026/2026-08-22/DJI_0001.JPG"]


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
        _library(main.app.state.catalog.db)
        yield c


def test_through_the_api(client):
    assert client.put("/api/library/clocks/home", json={"zone": "Mars/Olympus"}).status_code == 400
    state = client.put("/api/library/clocks/home", json={"zone": HOME}).json()
    assert state["home_zone"] == HOME and state["home_source"] == "set"
    assert state["moved"]["photos"] == 2
    r6 = next(c for c in state["cameras"] if c["camera"] == "Canon EOS R6m2")
    assert [x["offset"] for x in r6["recorded"]] == ["+10:00"]

    state = client.post("/api/library/clocks/rules", json={
        "camera": "PP-101", "date_from": "2026-08-01", "date_to": "2026-08-31",
        "where_zone": "Europe/Rome"}).json()
    assert state["moved"]["photos"] == 1          # the DJI frame stays on the 22nd abroad
    found = client.get("/api/catalog/search", params={"q": "date:2025-03-07"}).json()
    assert found["total"] == 1

    rule = state["rules"][0]["id"]
    assert client.delete(f"/api/library/clocks/rules/{rule}").status_code == 200
    assert client.post("/api/library/clocks/rules", json={"camera": "PP-101"}).status_code == 400


def test_a_folder_whose_camera_lost_its_clock_takes_the_day_given(db):
    """An E-P1 whose battery ran out: every date zero, the folder filed under
    2009-01-01. The day given on its card dates them, in file order."""
    from places.store import build_cards, save_note

    folder = "2009/2009-01-01"
    for name in ("P1017212.JPG", "P1017211.ORF", "P1017211.JPG", "P1017213.JPG"):
        db.upsert_photo(_rec(folder, name, "E-P1", "0000:00:00 00:00:00"))
    db.upsert_photo(_rec(folder, "IMG_0001.JPG", "GR II", "2020:06:20 09:00:00"))   # a real clock: kept
    save_note(db, folder, {"title": "Summer at home", "taken_on": "2020-06-15"})
    clock.recompute(db)
    dates = {r.filename: r.capture_date for r in db.get_photos_in_folder(folder)}
    assert dates == {"P1017211.ORF": "2020:06:15 12:00:00", "P1017211.JPG": "2020:06:15 12:00:00",
                     "P1017212.JPG": "2020:06:15 12:00:01", "P1017213.JPG": "2020:06:15 12:00:02",
                     "IMG_0001.JPG": "2020:06:20 09:00:00"}
    card = next(c for c in build_cards(db)["cards"] if c["folder"] == folder)
    assert card["date_from"] == "2020-06-15" and card["note"]["taken_on"] == "2020-06-15"
    # Taken back off, the photos are undated again.
    save_note(db, folder, {"title": "Summer at home", "taken_on": ""})
    clock.recompute(db)
    assert {r.capture_date for r in db.get_photos_in_folder(folder) if r.camera == "E-P1"} == {"0000:00:00 00:00:00"}


def test_a_taken_on_that_is_not_a_day_is_not_kept(db):
    from places.store import _clean_note

    assert _clean_note({"taken_on": "2020-13-40"})["taken_on"] == ""
    assert _clean_note({"taken_on": "1850-01-01"})["taken_on"] == ""
    assert _clean_note({"taken_on": "2020-06-15"})["taken_on"] == "2020-06-15"
