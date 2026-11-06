"""
Exploring the library (catalog/explore.py): collections by scene, the timeline,
and Rediscover.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog import explore  # noqa: E402
from catalog.database import PhotoRecord  # noqa: E402
from catalog.search import Photo  # noqa: E402

TODAY = date(2026, 11, 3)
FLOOR, CLEAR = 0.025, 0.035


def _photo(day, name, time="12:00:00", folder=None, **kw):
    folder = folder or day[:4] + "/" + day
    stamp = day.replace("-", ":") + " " + time if time else ""
    path = f"{folder}/{name}"
    record = PhotoRecord(filepath=path, filename=name, folder=folder,
                         extension=".jpg", base_name=Path(name).stem, capture_date=stamp)
    return Photo(record=record, date=day, filepaths=[path], **kw)


# ── Collections ─────────────────────────────────────────────────────

def _cake_library():
    photos = [_photo(f"{y}-05-02", f"cake{y}.jpg") for y in (2001, 2010, 2020, 2024)]
    photos.append(_photo("2024-05-02", "reject.jpg", flag=-1))
    photos.append(_photo("2024-07-01", "beach.jpg"))
    scores = {p.record.filepath: 0.06 for p in photos if "cake" in p.record.filename or "reject" in p.record.filename}
    scores["2024/2024-07-01/beach.jpg"] = 0.01
    return photos, scores


def test_a_collection_the_library_holds_is_offered_with_covers_from_different_years():
    photos, scores = _cake_library()
    shelf = explore.shelf(photos, {"birthdays": scores}, FLOOR, CLEAR)
    assert [c["key"] for c in shelf] == ["birthdays"]
    entry = shelf[0]
    assert entry["total"] == 4 and entry["years"] == 4      # the reject is not one of them
    assert entry["query"] == 'shows:"a birthday cake with candles"'
    assert len({p.date[:4] for p in entry["covers"]}) == len(entry["covers"]) == explore.COVERS


def test_a_collection_with_only_weak_matches_is_not_offered():
    photos, _ = _cake_library()
    weak = {p.record.filepath: 0.03 for p in photos}    # over the floor, never clear
    assert explore.shelf(photos, {"snow": weak}, FLOOR, CLEAR) == []


def test_a_collection_opens_a_year_at_a_time_most_recent_first():
    photos, scores = _cake_library()
    years = explore.by_year(photos, scores, FLOOR)
    assert [y["year"] for y in years] == [2024, 2020, 2010, 2001]
    assert [p.record.filename for p in years[0]["photos"]] == ["cake2024.jpg"]


def test_within_a_year_the_better_match_and_the_better_frame_come_first():
    a = _photo("2024-12-25", "tree.jpg", "09:00:00")
    b = _photo("2024-12-25", "tree_pick.jpg", "11:00:00", flag=1)
    c = _photo("2024-12-25", "far.jpg", "13:00:00")
    scores = {a.record.filepath: 0.05, b.record.filepath: 0.05, c.record.filepath: 0.026}
    year = explore.by_year([a, b, c], scores, FLOOR, per_year=1)[0]
    assert year["total"] == 3
    assert [p.record.filename for p in year["photos"]] == ["tree_pick.jpg"]


def test_a_match_far_from_the_best_one_is_left_out_even_over_the_floor():
    real = _photo("2015-06-06", "bride.jpg")
    parade = _photo("2019-07-04", "parade.jpg")
    scores = {real.record.filepath: 0.09, parade.record.filepath: 0.03}   # both over FLOOR
    found = explore.members([real, parade], scores, FLOOR)
    assert [p.record.filename for _, p in found] == ["bride.jpg"]


def test_a_photo_said_not_to_belong_is_left_out_of_that_collection_only():
    a = _photo("2015-06-06", "a.jpg")
    b = _photo("2016-06-06", "b.jpg")
    scores = {a.record.filepath: 0.06, b.record.filepath: 0.06}
    hidden = frozenset({a.record.filepath})
    assert [p.record.filename for _, p in explore.members([a, b], scores, FLOOR, hidden)] == ["b.jpg"]
    assert len(explore.members([a, b], scores, FLOOR)) == 2


# ── The timeline ────────────────────────────────────────────────────

def test_the_timeline_is_years_then_months_most_recent_first():
    photos = [_photo("2024-07-01", "a.jpg"), _photo("2024-07-20", "b.jpg"),
              _photo("2024-03-05", "c.jpg"), _photo("1998-12-24", "d.jpg"),
              _photo("", "undated.jpg", time="", folder="misc")]
    years = explore.timeline(photos)
    assert [y["year"] for y in years] == [2024, 1998]
    assert [m["month"] for m in years[0]["months"]] == ["2024-07", "2024-03"]
    july = years[0]["months"][0]
    assert (july["total"], july["days"]) == (2, 2)
    assert {s["folder"] for s in july["shoots"]} == {"2024/2024-07-01", "2024/2024-07-20"}
    assert years[0]["total"] == 3


def test_a_months_picks_are_its_best_and_never_a_reject():
    photos = [_photo("2024-07-01", f"p{i}.jpg", f"1{i}:00:00", rating=i) for i in range(8)]
    photos.append(_photo("2024-07-01", "reject.jpg", "09:00:00", flag=-1, rating=5))
    month = explore.timeline(photos, per_month=3)[0]["months"][0]
    assert month["total"] == 9
    assert [p.record.filename for p in month["photos"]] == ["p5.jpg", "p6.jpg", "p7.jpg"]
    assert month["shoots"][0]["cover"].record.filename == "p7.jpg"


def test_a_years_picks_take_one_frame_a_day_first():
    holiday = [_photo("2024-08-01", f"h{i}.jpg", f"1{i}:00:00", rating=5) for i in range(6)]
    other = _photo("2024-02-01", "feb.jpg", rating=1)
    year = explore.timeline(holiday + [other], per_year=2)[0]
    assert sorted(p.date for p in year["photos"]) == ["2024-02-01", "2024-08-01"]


# ── Rediscover ──────────────────────────────────────────────────────

def _old_library():
    good = [_photo(f"{y}-0{m}-1{d}", f"g{y}{m}{d}.jpg", rating=4)
            for y in (2005, 2012, 2019) for m in (3, 6) for d in range(4)]
    recent = [_photo("2026-06-01", "recent.jpg", rating=5)]
    return good + recent


def test_rediscover_offers_old_good_frames_one_a_day_and_is_the_same_all_day():
    photos = _old_library()
    first = explore.rediscover(photos, {}, TODAY, "seed", count=6)
    again = explore.rediscover(photos, {}, TODAY, "seed", count=6)
    picked = [p.record.filename for p in first["photos"]]
    assert picked == [p.record.filename for p in again["photos"]]
    assert len(picked) == 6 and "recent.jpg" not in picked
    assert len({p.date for p in first["photos"]}) == 6
    other = explore.rediscover(photos, {}, TODAY, "another", count=6)
    assert [p.record.filename for p in other["photos"]] != picked or len(photos) <= 6


def test_rediscover_leaves_out_what_was_looked_at_lately():
    photos = _old_library()
    seen = {p.record.filepath: "2026-10-01T10:00:00" for p in photos[:-1] if p.date.startswith("2005")}
    long_ago = {p.record.filepath: "2024-01-01T10:00:00" for p in photos if p.date.startswith("2012")}
    found = explore.rediscover(photos, {**seen, **long_ago}, TODAY, "x", count=20)
    years = {p.date[:4] for p in found["photos"]}
    assert "2005" not in years and "2012" in years


def test_a_good_day_is_one_with_enough_photographs():
    day = [_photo("2010-04-04", f"d{i}.jpg", f"1{i}:00:00") for i in range(explore.DAY_MIN_PHOTOS)]
    lone = [_photo("2011-01-01", "lone.jpg", rating=5)]
    found = explore.rediscover(day + lone, {}, TODAY, "s")
    assert found["day"]["date"] == "2010-04-04"
    assert found["day"]["ago"] == 16 and found["day"]["total"] == explore.DAY_MIN_PHOTOS


def test_nothing_old_enough_means_nothing_to_rediscover():
    found = explore.rediscover([_photo("2026-10-01", "new.jpg", rating=5)], {}, TODAY, "s")
    assert found == {"photos": [], "day": None}


# ── Through the API ─────────────────────────────────────────────────

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

API_FOLDER = "2019/2019-03-06"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    library = tmp_path / "lib"
    (library / API_FOLDER).mkdir(parents=True)
    for i in range(explore.DAY_MIN_PHOTOS):
        Image.fromarray(np.full((60, 80, 3), 40 * i, np.uint8), "RGB").save(
            library / API_FOLDER / f"IMG_{i}.JPG", format="JPEG")
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
        assert c.post(f"/api/catalog/index/{API_FOLDER}").status_code == 200
        yield c


def test_the_timeline_and_rediscover_through_the_api(client):
    years = client.get("/api/library/timeline").json()["years"]
    assert [y["year"] for y in years] == [2019]
    month = years[0]["months"][0]
    assert month["month"] == "2019-03" and month["total"] == explore.DAY_MIN_PHOTOS
    assert month["shoots"][0]["folder"] == API_FOLDER and "filepath" in month["shoots"][0]["cover"]

    found = client.get("/api/library/rediscover", params={"day": "2026-11-03"}).json()
    assert len(found["photos"]) == 1                      # one a day
    assert found["day"]["date"] == "2019-03-06" and found["day"]["folder"] == API_FOLDER

    # Looked at just now: not offered again for a while.
    shown = found["photos"][0]["filepath"]
    assert client.post("/api/library/viewed", json={"filepath": shown}).json()["ok"]
    again = client.get("/api/library/rediscover", params={"day": "2026-11-03"}).json()
    assert shown not in [p["filepath"] for p in again["photos"]]


def test_collections_say_when_smart_search_is_not_set_up(client, monkeypatch):
    from analysis import clip

    monkeypatch.setattr(clip, "available", lambda: False)
    assert client.get("/api/library/collections").json() == {"available": False, "collections": []}
    assert client.get("/api/library/collections/beach").status_code == 404


def test_a_collection_through_the_api(client, monkeypatch):
    from analysis import clip

    paths = [f"{API_FOLDER}/IMG_{i}.JPG" for i in range(explore.DAY_MIN_PHOTOS)]
    monkeypatch.setattr(clip, "available", lambda: True)
    monkeypatch.setattr(clip, "scores", lambda db, text: (
        {p: 0.06 for p in paths} if "beach" in text else {p: 0.0 for p in paths}))
    shelf = client.get("/api/library/collections").json()
    assert shelf["available"] and [c["key"] for c in shelf["collections"]] == ["beach"]
    assert shelf["collections"][0]["covers"][0]["folder"] == API_FOLDER
    beach = client.get("/api/library/collections/beach").json()
    assert beach["title"] == "The beach" and beach["total"] == explore.DAY_MIN_PHOTOS
    assert [y["year"] for y in beach["years"]] == [2019]


def test_saying_not_this_keeps_a_photo_out_of_a_collection(client, monkeypatch):
    from analysis import clip

    paths = [f"{API_FOLDER}/IMG_{i}.JPG" for i in range(explore.DAY_MIN_PHOTOS)]
    monkeypatch.setattr(clip, "available", lambda: True)
    monkeypatch.setattr(clip, "scores", lambda db, text: (
        {p: 0.06 for p in paths} if "beach" in text else {p: 0.0 for p in paths}))
    assert client.post("/api/library/collections/beach/hide", json={"filepath": paths[0]}).status_code == 200
    beach = client.get("/api/library/collections/beach").json()
    shown = [p["filepath"] for y in beach["years"] for p in y["photos"]]
    assert paths[0] not in shown and beach["total"] == len(paths) - 1
    assert client.post("/api/library/collections/nope/hide", json={"filepath": paths[0]}).status_code == 404
