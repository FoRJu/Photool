"""
Searching the library with a line of text: the language on its own, then
through the HTTP route and the library.search command on a real catalog.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import PhotoRecord  # noqa: E402
from catalog.search import SearchError, describe, gather, parse, photo_date, search  # noqa: E402


def rec(filepath, **kw):
    folder, filename = filepath.rsplit("/", 1)
    base, ext = filename.rsplit(".", 1)
    return PhotoRecord(filepath=filepath, filename=filename, folder=folder,
                       extension="." + ext.lower(), base_name=base, **kw)


LIBRARY = [
    # A CR3+JPG pair: rated on the RAW, tagged on the JPG.
    rec("2026/2026-04-05/MK2A0001.CR3", rating=4, camera="Canon EOS R6m2",
        lens="RF24-105mm F4 L IS USM", focal_length="24 mm", aperture="f/4.0",
        shutter_speed="1/200", iso="6400", capture_date="2026:04:05 10:00:00"),
    rec("2026/2026-04-05/MK2A0001.JPG", camera="Canon EOS R6m2",
        lens="RF24-105mm F4 L IS USM", focal_length="24 mm", aperture="f/4.0",
        shutter_speed="1/200", iso="6400", capture_date="2026:04:05 10:00:00"),
    # A Ricoh DNG with no EXIF date: the folder says when.
    rec("2025/2025-12-24/R0010001.DNG", flag=1, color_label=1, camera="GR II",
        lens="GR LENS", focal_length="18.3 mm", aperture="f/2.8", shutter_speed="2", iso="200"),
    # An old 5D JPG, rejected, in a folder with no date in it.
    rec("trips/hawaii/IMG_9000.JPG", flag=-1, camera="Canon EOS 5D Mark II",
        lens="EF70-200mm f/4L USM", focal_length="200.0 mm", aperture="f/5.6",
        shutter_speed="1/1000", iso="100", capture_date="2019:07:01 08:00:00"),
    # Nothing known about it at all.
    rec("misc/scan_001.JPG"),
]
TAGS = {"2026/2026-04-05/MK2A0001.JPG": ["School Play"], "trips/hawaii/IMG_9000.JPG": ["family"]}
EDITED = {"2026/2026-04-05/MK2A0001.CR3"}
BASKET = {"misc/scan_001.JPG"}


def found(query, **kw):
    photos = gather(LIBRARY, TAGS, EDITED, BASKET)
    return sorted(p.record.filename for p in search(photos, query, **kw))


class TestTheLanguage:
    def test_an_empty_query_finds_everything_once(self):
        # Four photos: the CR3+JPG pair is one.
        assert found("") == ["IMG_9000.JPG", "MK2A0001.JPG", "R0010001.DNG", "scan_001.JPG"]

    @pytest.mark.parametrize("query, expected", [
        ("rating:4", ["MK2A0001.JPG"]),
        ("rating:>=1", ["MK2A0001.JPG"]),
        ("rating:0", ["IMG_9000.JPG", "R0010001.DNG", "scan_001.JPG"]),
        ("stars:3..5", ["MK2A0001.JPG"]),
        ("flag:pick", ["R0010001.DNG"]),
        ("flag:reject", ["IMG_9000.JPG"]),
        ("flag:pick,reject", ["IMG_9000.JPG", "R0010001.DNG"]),
        ("label:red", ["R0010001.DNG"]),
        ("colour:none", ["IMG_9000.JPG", "MK2A0001.JPG", "scan_001.JPG"]),
    ])
    def test_cull_state(self, query, expected):
        assert found(query) == expected

    @pytest.mark.parametrize("query, expected", [
        ("iso:>1600", ["MK2A0001.JPG"]),
        ("iso:100,200", ["IMG_9000.JPG", "R0010001.DNG"]),
        ("f:<=2.8", ["R0010001.DNG"]),
        ("aperture:4", ["MK2A0001.JPG"]),
        ("focal:18..24", ["MK2A0001.JPG", "R0010001.DNG"]),
        ("focal:>=100", ["IMG_9000.JPG"]),
        ("shutter:<1/500", ["IMG_9000.JPG"]),
        ("shutter:>=1", ["R0010001.DNG"]),
        ("camera:r6", ["MK2A0001.JPG"]),
        ("camera:\"5d mark\"", ["IMG_9000.JPG"]),
        ("lens:70-200", ["IMG_9000.JPG"]),
    ])
    def test_exposure_data(self, query, expected):
        assert found(query) == expected

    def test_a_photo_without_the_data_is_not_found_by_a_number(self):
        assert "scan_001.JPG" not in found("iso:<100000")
        assert "scan_001.JPG" in found("-iso:>0")

    @pytest.mark.parametrize("query, expected", [
        ("date:2026", ["MK2A0001.JPG"]),
        ("date:2026-04", ["MK2A0001.JPG"]),
        ("date:2026-04-05", ["MK2A0001.JPG"]),
        ("date:2025-12", ["R0010001.DNG"]),          # from the folder name
        ("date:>2025", ["MK2A0001.JPG"]),
        ("date:>=2025-12-24", ["MK2A0001.JPG", "R0010001.DNG"]),
        ("date:<2020", ["IMG_9000.JPG"]),
        ("date:2019..2025", ["IMG_9000.JPG", "R0010001.DNG"]),
        ("date:2025..", ["MK2A0001.JPG", "R0010001.DNG"]),
    ])
    def test_dates(self, query, expected):
        assert found(query) == expected

    @pytest.mark.parametrize("query, expected", [
        ('tag:"school play"', ["MK2A0001.JPG"]),
        ("tag:FAMILY", ["IMG_9000.JPG"]),
        ("-tag:family", ["MK2A0001.JPG", "R0010001.DNG", "scan_001.JPG"]),
        ("folder:hawaii", ["IMG_9000.JPG"]),
        ("type:raw", ["MK2A0001.JPG", "R0010001.DNG"]),
        ("type:cr3", ["MK2A0001.JPG"]),
        ("type:jpg -type:raw", ["IMG_9000.JPG", "scan_001.JPG"]),
        ("is:edited", ["MK2A0001.JPG"]),
        ("is:unreviewed", ["scan_001.JPG"]),
        ("is:basket", ["scan_001.JPG"]),
        ("is:pick,reject", ["IMG_9000.JPG", "R0010001.DNG"]),
    ])
    def test_tags_places_and_state(self, query, expected):
        assert found(query) == expected

    def test_plain_words_look_everywhere(self):
        assert found("hawaii") == ["IMG_9000.JPG"]       # folder
        assert found("gr") == ["R0010001.DNG"]           # camera and lens
        assert found("school") == ["MK2A0001.JPG"]       # a tag
        assert found("scan") == ["scan_001.JPG"]         # filename
        assert found("-canon") == ["R0010001.DNG", "scan_001.JPG"]

    def test_terms_all_hold(self):
        assert found("camera:canon iso:>1000 is:edited") == ["MK2A0001.JPG"]
        assert found("camera:canon flag:pick") == []

    def test_the_pair_is_shown_as_its_jpeg_with_the_pairs_state(self):
        photos = search(gather(LIBRARY, TAGS, EDITED, BASKET), "camera:r6")
        assert len(photos) == 1
        pair = photos[0]
        assert pair.record.extension == ".jpg"
        assert pair.rating == 4 and pair.edited and pair.tags == {"School Play"}
        assert sorted(pair.filepaths) == ["2026/2026-04-05/MK2A0001.CR3", "2026/2026-04-05/MK2A0001.JPG"]

    def test_sorting(self):
        photos = gather(LIBRARY, TAGS, EDITED, BASKET)
        assert [p.record.filename for p in search(photos, "", sort="date")][:2] == ["MK2A0001.JPG", "R0010001.DNG"]
        assert [p.record.filename for p in search(photos, "", sort="rating")][0] == "MK2A0001.JPG"

    @pytest.mark.parametrize("query, says", [
        ("lense:35", "Unknown field 'lense:'"),
        ("rating:many", "not a number"),
        ("flag:maybe", "flag is one of"),
        ("date:April", "not a date"),
        ("is:lovely", "is: is one of"),
        ('tag:"open', "quote is not closed"),
        ("iso:", "needs a value"),
    ])
    def test_a_query_it_cannot_read_says_why(self, query, says):
        with pytest.raises(SearchError, match=says):
            parse(query)

    def test_the_query_is_described_back(self):
        assert describe('stars:>=3 -tag:"school play" garden') == ["rating:>=3", "-tag:school play", "garden"]


class TestDates:
    @pytest.mark.parametrize("capture, folder, expected", [
        ("2026:04:05 10:00:00", "x", "2026-04-05"),
        ("0000:00:00 00:00:00", "2024/2024-01-18", "2024-01-18"),
        ("", "2024/06/15", "2024-06-15"),
        ("", "photos/20230102_party", "2023-01-02"),
        ("", "vacation/hawaii", ""),
        ("", "2025/IMG_20251301", ""),       # month 13 is not a date
    ])
    def test_the_date_comes_from_exif_or_the_folder(self, capture, folder, expected):
        assert photo_date(capture, folder) == expected


# ── Through the app ──────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient

    library = tmp_path / "lib"
    for folder, name in (("2025/2025-03-06", "IMG_0001.JPG"), ("2025/2025-03-06", "IMG_0002.JPG"),
                         ("2024/2024-07-01", "IMG_0100.JPG")):
        (library / folder).mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((40, 60, 3), 90, np.uint8)).save(library / folder / name)
    (tmp_path / "config.yaml").write_text(
        # Single-quoted: YAML takes the backslashes of a Windows path literally there.
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app, base_url="http://localhost:8000") as c:
        for folder in ("2025/2025-03-06", "2024/2024-07-01"):
            assert c.post(f"/api/catalog/index/{folder}").status_code == 200
        yield c


class TestThroughTheApp:
    def test_the_route_returns_grid_rows_across_folders(self, client):
        client.put("/api/catalog/rate", json={"filepath": "2024/2024-07-01/IMG_0100.JPG", "rating": 5})
        res = client.get("/api/catalog/search", params={"q": "rating:5"}).json()
        assert res["total"] == 1
        row = res["photos"][0]
        assert row["filepath"] == "2024/2024-07-01/IMG_0100.JPG"
        assert row["folder"] == "2024/2024-07-01" and row["rating"] == 5
        assert {"filename", "extension", "tags", "edit", "edited"} <= set(row)
        assert client.get("/api/catalog/search", params={"q": "date:2025"}).json()["total"] == 2

    def test_a_bad_query_is_a_400_that_says_why(self, client):
        res = client.get("/api/catalog/search", params={"q": "lense:35"})
        assert res.status_code == 400
        assert "Unknown field" in res.json()["detail"]

    def test_the_command(self, client):
        out = client.post("/api/commands/library.search",
                          json={"query": "date:2025 -is:edited", "sort": "filename", "desc": False}).json()
        assert [p["filename"] for p in out["photos"]] == ["IMG_0001.JPG", "IMG_0002.JPG"]
        assert out["understood"] == ["date:2025", "-is:edited"]
        bad = client.post("/api/commands/library.search", json={"query": "flag:maybe"})
        assert bad.status_code == 400
