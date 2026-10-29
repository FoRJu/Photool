"""
Folder cards, places and the map: GPS reading, offline place names, folder
notes and the cards the Library draws.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from places import gazetteer, gps, store  # noqa: E402

HAS_EXIFTOOL = shutil.which("exiftool") is not None

# A few towns in GeoNames' own layout: id, name, ascii name, alternates, lat,
# lon, class, code, country, cc2, admin1, admin2, admin3, admin4, population.
_TOWNS = [
    ("1", "Seattle", "Seattle", 47.6062, -122.3321, "US", "WA", 737015),
    ("2", "Bellevue", "Bellevue", 47.6101, -122.2015, "US", "WA", 151854),
    ("3", "Paris", "Paris", 48.8534, 2.3488, "FR", "11", 2138551),
    ("4", "Portland", "Portland", 45.5234, -122.6762, "US", "OR", 652503),
    ("5", "Portland", "Portland", 43.6615, -70.2553, "US", "ME", 68408),
    ("6", "Zürich", "Zurich", 47.3667, 8.55, "CH", "ZH", 341730),
    ("8", "Paris 16 Passy", "Paris 16 Passy", 48.8637, 2.2769, "FR", "11", 159386),
    ("9", "Cambridge", "Cambridge", 42.3751, -71.1056, "US", "MA", 118403),
    ("10", "Cambridge", "Cambridge", 52.2053, 0.1218, "GB", "ENG", 145818),
    ("7", "Greenwood", "Greenwood", 47.6909, -122.3554, "US", "WA", 9999, "PPLX"),
]


def write_names(models_dir: Path) -> None:
    folder = gazetteer.data_dir(models_dir)
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for gid, name, ascii_name, lat, lon, cc, a1, pop, *code in _TOWNS:
        lines.append("\t".join([gid, name, ascii_name, "", str(lat), str(lon), "P", (code or ["PPL"])[0],
                                cc, "", a1, "", "", "", str(pop), "", "0", "tz", "2024-01-01"]))
    (folder / "cities5000.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "admin1CodesASCII.txt").write_text(
        "US.MA\tMassachusetts\tMassachusetts\t6\nGB.ENG\tEngland\tEngland\t7\nUS.WA\tWashington\tWashington\t1\nUS.OR\tOregon\tOregon\t2\n"
        "US.ME\tMaine\tMaine\t3\nFR.11\tÎle-de-France\tIle-de-France\t4\nCH.ZH\tZurich\tZurich\t5\n",
        encoding="utf-8")
    (folder / "countryInfo.txt").write_text(
        "#ISO\tISO3\tISO-Numeric\tfips\tCountry\n"
        "US\tUSA\t840\tUS\tUnited States\nPH\tPHL\t608\tRP\tPhilippines\nFR\tFRA\t250\tFR\tFrance\nCH\tCHE\t756\tSZ\tSwitzerland\n",
        encoding="utf-8")


@pytest.fixture()
def names(tmp_path):
    write_names(tmp_path / "models")
    return gazetteer.Gazetteer(gazetteer.data_dir(tmp_path / "models"))


def _photo(db, folder, name, **kw):
    db.upsert_photo(PhotoRecord(
        filepath=f"{folder}/{name}", filename=name, folder=folder,
        extension=Path(name).suffix.lower(), base_name=Path(name).stem, **kw))


# ── Reading GPS ──────────────────────────────────────────────────────

class TestParse:
    def test_signed_coordinates_are_kept(self):
        out = gps.parse(json.dumps([
            {"SourceFile": "D:/Photos/a.jpg", "GPSLatitude": -33.86, "GPSLongitude": 151.21},
        ]))
        assert list(out.values()) == [(-33.86, 151.21)]

    def test_no_fix_and_null_island_are_not_places(self):
        out = gps.parse(json.dumps([
            {"SourceFile": "a.jpg"},
            {"SourceFile": "b.jpg", "GPSLatitude": 0, "GPSLongitude": 0},
            {"SourceFile": "c.jpg", "GPSLatitude": "47 deg", "GPSLongitude": 1.0},
            {"SourceFile": "d.jpg", "GPSLatitude": 95.0, "GPSLongitude": 1.0},
        ]))
        assert out == {}

    def test_garbage_is_nothing(self):
        assert gps.parse("not json") == {}


# ── Place names ──────────────────────────────────────────────────────

class TestGazetteer:
    def test_nearest_town(self, names):
        place = names.nearest(47.62, -122.35)
        assert place.name == "Seattle"
        assert place.km < 3

    def test_a_section_of_a_town_is_not_the_answer(self, names):
        # Closer to Greenwood (a Seattle neighbourhood) than to Seattle's centre.
        assert names.nearest(47.69, -122.355).name == "Seattle"

    def test_a_district_listed_as_a_town_gives_way_to_its_city(self, names):
        # The Eiffel Tower: nearer Passy's centre than Paris's, and in Paris.
        place = names.nearest(48.8584, 2.2945)
        assert place.name == "Paris"
        # A real town next to a big one keeps its own name.
        assert names.nearest(47.611, -122.20).name == "Bellevue"

    def test_labels_say_the_country_only_abroad(self, names):
        assert names.nearest(47.62, -122.35).label("US") == "Seattle, Washington"
        assert names.nearest(48.86, 2.35).label("US") == "Paris, France"

    def test_search_puts_the_bigger_town_first_and_a_comma_narrows(self, names):
        assert [p.region for p in names.search("portland")] == ["Oregon", "Maine"]
        assert [p.region for p in names.search("Portland, Maine")] == ["Maine"]
        assert [p.name for p in names.search("zurich")] == ["Zürich"]
        assert names.search("   ") == []

    def test_home_country_comes_first_among_same_names(self, names):
        assert [p.country_code for p in names.search("Cambridge")] == ["GB", "US"]
        assert [p.country_code for p in names.search("Cambridge", prefer="US")] == ["US", "GB"]

    def test_not_fetched_means_none(self, tmp_path):
        assert gazetteer.get(tmp_path / "nothing") is None


# ── Cards ────────────────────────────────────────────────────────────

class TestCards:
    def test_dates_come_from_the_folder_name_and_junk_clocks_are_ignored(self):
        assert store.date_in_name("2026/2026-07-04") == "2026-07-04"
        assert store.date_in_name("Export/graduation_2026_UW") is None
        assert store.date_in_name("trip 20250315") == "2025-03-15"
        assert store._capture_day("0000:00:00 00:00:00") is None
        assert store._capture_day("2024:05:04 10:00:00") == "2024-05-04"

    def test_cover_is_a_pick_then_stars_then_the_middle_and_never_a_reject(self):
        rows = [("a", 0, 0), ("b", 0, 0), ("c", 0, 0)]
        assert store._pick_cover(rows) == "b"
        assert store._pick_cover([("a", 0, 3), ("b", 1, 0), ("c", 0, 5)]) == "b"
        assert store._pick_cover([("a", -1, 5), ("b", 0, 2)]) == "b"

    def test_a_card_says_who_what_when_and_where(self, tmp_path, names):
        db = CatalogDB(str(tmp_path / "c.db"))
        folder = "2026/2026-07-04"
        for i in range(4):
            _photo(db, folder, f"IMG_{i}.JPG", capture_date="2026:07:04 10:00:00", rating=4 if i == 2 else 0)
        _photo(db, "Export/party", "x.jpg")
        db.set_photo_tags(f"{folder}/IMG_0.JPG", ["Fireworks"])
        db.set_photo_tags(f"{folder}/IMG_1.JPG", ["Fireworks", "Alice", "bob"])
        alice_tag = db.create_tag("Alice")
        cluster = db.create_cluster("Alice", alice_tag)
        bob = db.create_cluster("Bob")
        with db._connect() as conn:
            conn.execute("INSERT INTO photo_faces (filepath, face_index, cluster_id) VALUES (?, 0, ?)",
                         (f"{folder}/IMG_3.JPG", bob))
            conn.execute("INSERT INTO photo_faces (filepath, face_index, cluster_id) VALUES (?, 0, ?)",
                         (f"{folder}/IMG_2.JPG", cluster))
        store.record_gps(db, [(f"{folder}/IMG_{i}.JPG", 47.61, -122.33, 0) for i in range(3)]
                         + [(f"{folder}/IMG_3.JPG", None, None, 0)])

        result = store.build_cards(db, names)
        card = next(c for c in result["cards"] if c["folder"] == folder)
        assert result["home_country"] == "US"
        assert card["date_from"] == card["date_to"] == "2026-07-04"
        assert card["count"] == 4
        assert card["cover"] == f"{folder}/IMG_2.JPG"
        assert card["place"]["label"] == "Seattle, Washington"
        assert card["place"]["source"] == "gps"
        assert {p["name"] for p in card["people"]} == {"Alice", "Bob"}
        assert [t["name"] for t in card["tags"]] == ["Fireworks"]
        other = next(c for c in result["cards"] if c["folder"] == "Export/party")
        assert other["place"] is None and other["date_from"] is None

    def test_gps_pending_skips_what_was_read(self, tmp_path):
        db = CatalogDB(str(tmp_path / "c.db"))
        _photo(db, "f", "a.jpg", file_size=10)
        _photo(db, "f", "b.jpg", file_size=20)
        store.record_gps(db, [("f/a.jpg", None, None, 10)])
        assert [r["filepath"] for r in store.gps_pending(db)] == ["f/b.jpg"]
        assert store.gps_counts(db) == {"photos": 2, "read": 1, "unread": 1, "found": 0}


# ── Notes ────────────────────────────────────────────────────────────

class TestNotes:
    def test_a_note_wins_and_survives_a_new_catalog(self, tmp_path, names):
        library = tmp_path / "lib"
        (library / "2024/2024-05-04").mkdir(parents=True)
        db = CatalogDB(str(tmp_path / "c.db"))
        _photo(db, "2024/2024-05-04", "a.jpg")
        _photo(db, "2024/2024-05-04", "b.jpg")
        store.save_note(db, "2024/2024-05-04", {
            "title": "Graduation", "subtitle": "UW, the whole family",
            "cover": "2024/2024-05-04/a.jpg", "place": "Paris, France", "lat": 48.85, "lon": 2.35,
        }, str(library))
        card = store.build_cards(db, names)["cards"][0]
        assert card["note"]["title"] == "Graduation"
        assert card["cover"] == "2024/2024-05-04/a.jpg"
        assert card["place"] == {"label": "Paris, France", "lat": 48.85, "lon": 2.35,
                                 "source": "you", "towns": [], "country_code": "FR"}
        assert (library / "2024/2024-05-04" / store.NOTE_FILE).is_file()

        fresh = CatalogDB(str(tmp_path / "fresh.db"))
        _photo(fresh, "2024/2024-05-04", "a.jpg")
        assert store.adopt_sidecars(fresh, str(library), ["2024/2024-05-04"]) == 1
        assert store.notes(fresh)["2024/2024-05-04"]["subtitle"] == "UW, the whole family"

    def test_clearing_every_field_removes_the_note_and_its_file(self, tmp_path):
        library = tmp_path / "lib"
        (library / "f").mkdir(parents=True)
        db = CatalogDB(str(tmp_path / "c.db"))
        store.save_note(db, "f", {"title": "x"}, str(library))
        store.save_note(db, "f", {}, str(library))
        assert store.notes(db) == {}
        assert not (library / "f" / store.NOTE_FILE).exists()

    def test_a_place_without_coordinates_is_dropped(self, tmp_path):
        db = CatalogDB(str(tmp_path / "c.db"))
        note = store.save_note(db, "f", {"title": "t", "place": "Somewhere"})
        assert note["place"] == "" and note["lat"] is None

    def test_a_cover_that_is_gone_falls_back(self, tmp_path):
        db = CatalogDB(str(tmp_path / "c.db"))
        _photo(db, "f", "a.jpg")
        store.save_note(db, "f", {"cover": "f/deleted.jpg"})
        assert store.build_cards(db)["cards"][0]["cover"] == "f/a.jpg"


# ── API ──────────────────────────────────────────────────────────────

FOLDER = "2025/2025-03-06"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((40, 60, 3), 80 * i, dtype=np.uint8), "RGB").save(
            library / FOLDER / f"IMG_{i}.JPG", format="JPEG")
    models = tmp_path / "models"
    write_names(models)
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        f"models_dir: '{models}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
        "thumbnail_queue_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main
    from places import routes

    routes._adopted.clear()
    with TestClient(main.app) as c:
        assert c.post(f"/api/catalog/index/{FOLDER}").status_code == 200
        c.library = library
        yield c


def _wait_auto():
    from places import routes

    for _ in range(400):
        if routes.auto_idle():
            return
        time.sleep(0.05)
    raise AssertionError("the background read did not finish")


def _gps_photo(path, lat=47.6062, lon=-122.3321):
    subprocess.run(["exiftool", "-q", "-overwrite_original",
                    f"-GPSLatitude={abs(lat)}", f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
                    f"-GPSLongitude={abs(lon)}", f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
                    str(path)], check=True)


def _wait(client):
    for _ in range(200):
        job = client.get("/api/places/status").json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the scan did not finish")


class TestApi:
    def test_cards_notes_covers_and_search(self, client):
        data = client.get("/api/library/cards").json()
        assert data["gazetteer"] is True
        assert [c["folder"] for c in data["cards"]] == [FOLDER]
        assert data["cards"][0]["date_from"] == "2025-03-06"

        found = client.get("/api/places/search", params={"q": "seat", "home": "US"}).json()["places"]
        assert found[0]["label"] == "Seattle, Washington"

        res = client.put(f"/api/library/cards/{FOLDER}", json={
            "title": "Spring", "place": found[0]["label"], "lat": found[0]["lat"], "lon": found[0]["lon"]})
        assert res.status_code == 200
        card = client.get("/api/library/cards").json()["cards"][0]
        assert card["note"]["title"] == "Spring"
        assert card["place"]["source"] == "you"
        assert (client.library / FOLDER / ".photool-folder.json").is_file()

        covers = client.get(f"/api/library/cards/{FOLDER}/covers").json()["filepaths"]
        assert len(covers) == 3
        assert client.put("/api/library/cards/nope", json={"title": "x"}).status_code == 404

    def test_a_note_can_be_written_while_the_folder_is_still_indexing(self, client):
        # Opened for the first time: on disk, not yet in the catalog.
        later = "2025/2025-04-01"
        (client.library / later).mkdir(parents=True)
        shutil.copy(client.library / FOLDER / "IMG_1.JPG", client.library / later / "IMG_9.JPG")
        assert later not in [c["folder"] for c in client.get("/api/library/cards").json()["cards"]]

        res = client.put(f"/api/library/cards/{later}", json={"title": "Kite day"})
        assert res.status_code == 200 and res.json()["note"]["title"] == "Kite day"
        assert (client.library / later / ".photool-folder.json").is_file()

        assert client.post(f"/api/catalog/index/{later}").status_code == 200
        cards = {c["folder"]: c for c in client.get("/api/library/cards").json()["cards"]}
        assert cards[later]["note"]["title"] == "Kite day"

    @pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool is not installed")
    def test_a_scan_reads_where_photos_were_taken(self, client):
        _wait_auto()
        _gps_photo(client.library / FOLDER / "IMG_1.JPG")
        # Indexing read them already, without a position; forget that so the
        # whole-library pass has something to do.
        db = client.app.state.catalog.db
        with db._connect() as conn:
            conn.execute("DELETE FROM photo_gps")
        assert client.post("/api/places/scan", json={}).json()["started"] is True
        job = _wait(client)
        assert job["status"] == "completed", job
        assert job["found"] == 1 and job["done"] == 3
        status = client.get("/api/places/status").json()
        assert status["gps"]["unread"] == 0 and status["gps"]["found"] == 1
        card = client.get("/api/library/cards").json()["cards"][0]
        assert card["place"]["label"] == "Seattle, Washington"
        assert card["place"]["lat"] == pytest.approx(47.6062, abs=1e-3)
        assert card["place"]["lon"] == pytest.approx(-122.3321, abs=1e-3)
        # Nothing left to read: a second scan is a no-op.
        client.post("/api/places/scan", json={})
        assert _wait(client)["done"] == 0


def test_a_folder_listing_leaves_out_sidecars(client):
    (client.library / FOLDER / "IMG_0.JPG.xmp").write_text("<x/>")
    (client.library / FOLDER / "IMG_1.xmp").write_text("<x/>")
    names = [f["name"] for f in client.get(f"/api/library/files/{FOLDER}").json()["files"]]
    assert names == ["IMG_0.JPG", "IMG_1.JPG", "IMG_2.JPG"]


class TestHomeAndMany:
    def test_home_makes_unplaced_and_nearby_folders_home(self, client):
        assert client.get("/api/places/home").json()["home"] is None
        assert client.get("/api/library/cards").json()["cards"][0]["at_home"] is False
        res = client.put("/api/places/home", json={"place": "Seattle, Washington", "lat": 47.6062, "lon": -122.3321})
        assert res.json()["home"]["place"] == "Seattle, Washington"
        data = client.get("/api/library/cards").json()
        assert data["home"]["lat"] == 47.6062
        assert data["cards"][0]["at_home"] is True              # no place: home
        assert (client.library / ".photool-library.json").is_file()

        client.put(f"/api/library/cards/{FOLDER}", json={"place": "Paris, France", "lat": 48.85, "lon": 2.35})
        assert client.get("/api/library/cards").json()["cards"][0]["at_home"] is False
        client.put(f"/api/library/cards/{FOLDER}", json={"place": "Seattle", "lat": 47.61, "lon": -122.33})
        assert client.get("/api/library/cards").json()["cards"][0]["at_home"] is True

        # Home decides which country is home, so places at home name the state.
        assert data["home_country"] == "US"
        found = client.get("/api/places/search", params={"q": "cambridge", "home": "us"}).json()["places"]
        assert found[0]["label"] == "Cambridge, Massachusetts"

        client.put("/api/places/home", json={})
        assert client.get("/api/places/home").json()["home"] is None

    def test_one_place_for_many_folders_keeps_their_titles(self, client):
        client.put(f"/api/library/cards/{FOLDER}", json={"title": "Spring"})
        res = client.put("/api/library/places", json={
            "folders": [FOLDER, "not/a/folder"], "place": "Paris, France", "lat": 48.85, "lon": 2.35})
        assert res.json()["changed"] == 1
        card = client.get("/api/library/cards").json()["cards"][0]
        assert card["note"]["title"] == "Spring"
        assert card["place"]["label"] == "Paris, France"
        recent = client.get("/api/library/cards").json()["recent_places"]
        assert recent == [{"place": "Paris, France", "lat": 48.85, "lon": 2.35, "folders": 1}]

        client.put("/api/library/places", json={"folders": [FOLDER]})
        card = client.get("/api/library/cards").json()["cards"][0]
        assert card["place"] is None and card["note"]["title"] == "Spring"

    def test_home_comes_back_with_a_new_catalog(self, tmp_path):
        library = tmp_path / "lib"
        library.mkdir()
        db = CatalogDB(str(tmp_path / "a.db"))
        store.set_home(db, {"place": "Seattle", "lat": 47.6, "lon": -122.3}, str(library))
        fresh = CatalogDB(str(tmp_path / "b.db"))
        assert store.adopt_home(fresh, str(library))["place"] == "Seattle"
        assert store.get_home(fresh)["lat"] == 47.6


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool is not installed")
def test_newly_indexed_photos_are_read_on_their_own(client):
    _wait_auto()
    folder = "2025/2025-04-01"
    (client.library / folder).mkdir(parents=True)
    photo = client.library / folder / "IMG_9.JPG"
    Image.fromarray(np.full((40, 60, 3), 120, dtype=np.uint8), "RGB").save(photo, format="JPEG")
    _gps_photo(photo, 48.8584, 2.2945)
    assert client.post(f"/api/catalog/index/{folder}").status_code == 200
    _wait_auto()
    card = next(c for c in client.get("/api/library/cards").json()["cards"] if c["folder"] == folder)
    assert card["place"]["source"] == "gps"
    assert card["place"]["label"].startswith("Paris, ")
    # Opening the folder again is cached and reads nothing.
    assert client.post(f"/api/catalog/index/{folder}").json().get("cached") is True
    from places import routes
    assert routes.auto_idle()
