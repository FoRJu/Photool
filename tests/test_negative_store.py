"""
Where a negative's description is kept: its own catalog table and one sidecar
attribute, apart from the edit.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB
from catalog.service import NEGATIVE_ATTR, CatalogService
from catalog.xmp import read_xmp

D = {"kind": "colour", "base": [0.75, 0.45, 0.25], "base_from": "border",
     "range": [1.2, 1.1, 1.0], "contrast": 10, "balance": 80}
FILE = "2024/06/15/IMG_0001.CR3"


def _setup(tmp: Path):
    lib = tmp / "library"
    folder = lib / "2024" / "06" / "15"
    folder.mkdir(parents=True)
    for name in ("IMG_0001.CR3", "IMG_0001.JPG", "IMG_0002.CR3"):
        (folder / name).write_bytes(b"fake " * 100)
    service = CatalogService(CatalogDB(str(tmp / "catalog.db")), str(lib))
    service.index_folder("2024/06/15", [".cr3", ".jpg"])
    return service, lib


def test_set_and_read_back(tmp_path):
    service, lib = _setup(tmp_path)
    stored = service.set_negative(FILE, D)
    assert stored["base"] == D["base"] and stored["contrast"] == 10
    assert service.negative_of(FILE) == stored
    assert json.loads(read_xmp(lib / FILE).develop[NEGATIVE_ATTR]) == stored
    assert service.negative_of("2024/06/15/IMG_0002.CR3") is None


def test_a_pair_answers_for_either_file(tmp_path):
    service, _ = _setup(tmp_path)
    service.set_negative("2024/06/15/IMG_0001.JPG", D)
    assert service.db.frame_negative("2024/06/15", "IMG_0001")["kind"] == "colour"
    assert service.db.frame_negative("2024/06/15", "IMG_0002") is None


def test_clearing_the_edit_leaves_the_negative(tmp_path):
    service, lib = _setup(tmp_path)
    service.set_negative(FILE, D)
    service.set_photo_edit(FILE, {"exposure": 0.5})
    develop = read_xmp(lib / FILE).develop
    assert NEGATIVE_ATTR in develop and develop["crs:Exposure2012"] == "+0.50"
    service.clear_photo_edit(FILE)
    assert service.negative_of(FILE) is not None
    develop = read_xmp(lib / FILE).develop
    assert NEGATIVE_ATTR in develop and "crs:Exposure2012" not in develop
    assert service.photo_edit(FILE) == {}


def test_clearing_the_negative_leaves_the_edit(tmp_path):
    service, lib = _setup(tmp_path)
    service.set_photo_edit(FILE, {"exposure": 0.5})
    service.set_negative(FILE, D)
    service.clear_negative(FILE)
    assert service.negative_of(FILE) is None
    develop = read_xmp(lib / FILE).develop
    assert NEGATIVE_ATTR not in develop and develop["crs:Exposure2012"] == "+0.50"
    assert service.photo_edit(FILE) == {"exposure": 0.5}


def test_a_rebuilt_catalogue_has_the_negative(tmp_path):
    service, lib = _setup(tmp_path)
    stored = service.set_negative(FILE, D)
    fresh = CatalogService(CatalogDB(str(tmp_path / "rebuilt.db")), str(lib))
    fresh.index_folder("2024/06/15", [".cr3", ".jpg"])
    assert fresh.negative_of(FILE) == stored
    # and the negative alone is not taken for an edit
    assert fresh.photo_edit(FILE) == {}
