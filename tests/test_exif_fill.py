"""
Filling in the camera data a catalog row is missing (catalog/exif_fill.py), and
the batch EXIF read it uses.
"""

import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog import exif_fill  # noqa: E402
from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from thumbnails import extract  # noqa: E402

HAS_EXIFTOOL = shutil.which("exiftool") is not None
FOLDER = "2026/2026-10-17"


@pytest.fixture()
def library(tmp_path):
    root = tmp_path / "lib"
    (root / FOLDER).mkdir(parents=True)
    db = CatalogDB(str(tmp_path / "catalog.db"))
    return root, db


def _photo(root, db, name, camera="", data=b"x"):
    path = root / FOLDER / name
    path.write_bytes(data)
    db.upsert_photo(PhotoRecord(
        filepath=f"{FOLDER}/{name}", filename=name, folder=FOLDER,
        extension=path.suffix.lower(), base_name=path.stem,
        file_size=path.stat().st_size, camera=camera))
    return path


class TestFill:
    @pytest.fixture()
    def opened(self, monkeypatch):
        """The files exiftool is asked about; each answers from EXIF below."""
        asked = []
        exif = {"A.CR3": {"camera": "Canon EOS R6m2", "lens": "RF24-105mm", "iso": "800",
                          "date": "2026:10:17 15:50:51"},
                "SCAN.JPG": {"camera": ""}}

        def batch(sources, fast=False):
            asked.extend(Path(s).name for s in sources)
            return {str(s): exif.get(Path(s).name, {}) for s in sources}

        monkeypatch.setattr(extract, "get_exif_batch", batch)
        return asked

    def test_blank_rows_are_filled_and_known_ones_left_alone(self, library, opened):
        root, db = library
        _photo(root, db, "A.CR3")
        _photo(root, db, "B.CR3", camera="GR II")
        job = exif_fill.run(root, db)
        assert job["status"] == "completed" and job["filled"] == 1
        assert opened == ["A.CR3"]
        a = db.get_photo(f"{FOLDER}/A.CR3")
        assert (a.camera, a.lens, a.iso, a.capture_date) == (
            "Canon EOS R6m2", "RF24-105mm", "800", "2026:10:17 15:50:51")
        assert db.get_photo(f"{FOLDER}/B.CR3").camera == "GR II"

    def test_a_photo_with_no_exif_is_looked_at_once(self, library, opened):
        root, db = library
        _photo(root, db, "SCAN.JPG")
        exif_fill.run(root, db)
        exif_fill.run(root, db)
        assert opened == ["SCAN.JPG"]
        assert exif_fill.counts(db) == {"photos": 1, "known": 0, "read": 1}

    def test_a_file_replaced_on_disk_is_read_again(self, library, opened):
        root, db = library
        path = _photo(root, db, "SCAN.JPG")
        exif_fill.run(root, db)
        path.write_bytes(b"a longer file")
        with db._connect() as conn:          # what indexing records of the new file
            conn.execute("UPDATE photos SET file_size = ?", (path.stat().st_size,))
        assert [r["filepath"] for r in exif_fill.pending(db)] == [f"{FOLDER}/SCAN.JPG"]

    def test_a_photo_gone_from_disk_is_left_for_later(self, library, opened):
        root, db = library
        _photo(root, db, "A.CR3").unlink()
        exif_fill.run(root, db)
        assert opened == [] and len(exif_fill.pending(db)) == 1


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool is not installed")
def test_one_unreadable_file_does_not_lose_the_batch(tmp_path):
    exif = Image.Exif()
    exif[0x0110] = "GR II"                      # Model
    good = []
    for name in ("R1.JPG", "R2.JPG"):
        Image.new("RGB", (8, 8)).save(tmp_path / name, exif=exif)
        good.append(tmp_path / name)
    bad = tmp_path / "R0.JPG"
    bad.write_bytes(b"not a picture")
    gone = tmp_path / "R9.JPG"                  # never written: exiftool skips it
    found = extract.get_exif_batch([bad, gone, *good], fast=True)
    assert [found[str(p)]["camera"] for p in good] == ["GR II", "GR II"]
    assert found[str(bad)]["camera"] == ""
