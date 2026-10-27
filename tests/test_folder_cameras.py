"""
A day shot on two cameras in one folder: the folder's photos narrowed to one
camera (pair-aware, with "" for the ones no camera is known for), and a search
typed while the folder is open kept to that folder.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from catalog.search import gather  # noqa: E402
from catalog.service import CatalogService  # noqa: E402

DAY = "2026/2026-05-22"
OTHER = "2026/2026-05-23"


def _rec(folder, name, camera=""):
    base, ext = name.rsplit(".", 1)
    return PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder, extension="." + ext.lower(),
                       base_name=base, camera=camera, capture_date="2026:05:22 10:00:00")


@pytest.fixture()
def lib(tmp_path):
    db = CatalogDB(str(tmp_path / "c.db"))
    for r in [
        _rec(DAY, "MK2A0001.CR3", "Canon EOS R6m2"), _rec(DAY, "MK2A0001.JPG", "Canon EOS R6m2"),
        # The camera's JPEG lost its EXIF; its raw twin says which camera it was.
        _rec(DAY, "MK2A0002.CR3", "Canon EOS R6m2"), _rec(DAY, "MK2A0002.JPG", ""),
        _rec(DAY, "R0000001.JPG", "GR II"), _rec(DAY, "R0000002.JPG", "GR II"),
        _rec(DAY, "IMG_0001.JPG", ""),
        _rec(OTHER, "R0000003.JPG", "GR II"),
    ]:
        db.upsert_photo(r)
    return db, CatalogService(db, str(tmp_path))


def names(records):
    return sorted(r.filename for r in records)


def test_one_cameras_frames_pair_and_all(lib):
    db, _ = lib
    r6 = db.get_photos_in_folder(DAY, camera_filter="Canon EOS R6m2")
    assert names(r6) == ["MK2A0001.CR3", "MK2A0001.JPG", "MK2A0002.CR3", "MK2A0002.JPG"]
    assert names(db.get_photos_in_folder(DAY, camera_filter="GR II")) == ["R0000001.JPG", "R0000002.JPG"]


def test_the_frames_no_camera_is_known_for(lib):
    db, _ = lib
    assert names(db.get_photos_in_folder(DAY, camera_filter="")) == ["IMG_0001.JPG"]


def test_no_camera_filter_is_every_frame(lib):
    db, _ = lib
    assert len(db.get_photos_in_folder(DAY)) == 7


def test_a_pair_searches_by_whichever_half_knows_its_camera(lib):
    db, _ = lib
    photos = gather(db.get_photos_in_folder(DAY), {}, set(), set())
    by_base = {p.record.base_name: p.record.camera for p in photos}
    assert by_base["MK2A0002"] == "Canon EOS R6m2"


def test_a_search_in_a_folder_stays_in_it(lib):
    _db, service = lib
    everywhere = service.search("camera:gr")
    assert len(everywhere) == 3
    here = service.search("camera:gr", folder=DAY)
    assert sorted(p.record.filename for p in here) == ["R0000001.JPG", "R0000002.JPG"]
    assert len(service.search("camera:r6", folder=DAY)) == 2


def test_only_the_pairs_a_folder_search_found(lib):
    db, _ = lib
    found = db.get_photos_in_folder(DAY, base_names=["MK2A0001", "R0000002"])
    assert names(found) == ["MK2A0001.CR3", "MK2A0001.JPG", "R0000002.JPG"]
    assert db.get_photos_in_folder(DAY, base_names=[]) == []


def test_bulk_actions_carry_the_camera_and_the_search(lib):
    """The grid's filters as the bulk endpoints read them: an action on "these"
    must act on the camera and the search on screen, not the whole folder."""
    from api.routes import PhotoFilterParams, _filter_kwargs

    db, _ = lib
    kw = _filter_kwargs(PhotoFilterParams(camera="GR II", bases=["R0000001", "MK2A0001"], pair_mode="all"))
    assert names(db.get_photos_in_folder(DAY, **kw)) == ["R0000001.JPG"]


def test_demoted_cameras_left_out_pair_and_all(lib):
    """The cameras demoted in a folder are not on its grid, and so not in what
    an action on "these" or the bursts reach: pair-aware, "" for no camera."""
    db, _ = lib
    assert names(db.get_photos_in_folder(DAY, hide_cameras=["Canon EOS R6m2"])) == [
        "IMG_0001.JPG", "R0000001.JPG", "R0000002.JPG"]
    assert names(db.get_photos_in_folder(DAY, hide_cameras=["GR II", ""])) == [
        "MK2A0001.CR3", "MK2A0001.JPG", "MK2A0002.CR3", "MK2A0002.JPG"]
    assert len(db.get_photos_in_folder(DAY, hide_cameras=[])) == 7


def test_bulk_actions_leave_out_the_hidden_cameras(lib):
    from api.routes import PhotoFilterParams, _filter_kwargs

    db, _ = lib
    kw = _filter_kwargs(PhotoFilterParams(hide_cameras=["GR II", "Canon EOS R6m2"], pair_mode="all"))
    assert names(db.get_photos_in_folder(DAY, **kw)) == ["IMG_0001.JPG"]
