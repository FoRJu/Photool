"""
The Library's whole-library picks (the timeline, the years, On this day) are
kept until the library changes in a way they read, not for a few minutes. The
catalog counts those changes itself (triggers), so an import, a rating or a
date counts and a thumbnail or a re-read of the same EXIF does not. After a
change the answer already made is given at once and made again behind it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from catalog.service import CatalogService  # noqa: E402

FOLDER = "2024/2024-08-10"


def _rec(name, when):
    base, ext = name.rsplit(".", 1)
    return PhotoRecord(filepath=f"{FOLDER}/{name}", filename=name, folder=FOLDER, extension="." + ext.lower(),
                       base_name=base, camera="GR II", capture_date=when)


@pytest.fixture()
def lib(tmp_path):
    db = CatalogDB(str(tmp_path / "c.db"))
    db.upsert_photo(_rec("R0000001.JPG", "2024:08:10 10:00:00"))
    db.upsert_photo(_rec("R0000002.JPG", "2024:08:10 11:00:00"))
    service = CatalogService(db, str(tmp_path))
    # The watcher is not wanted here: the test redoes picks itself.
    service._start_redoer = lambda: None
    return db, service


def test_what_the_picks_read_is_counted_and_nothing_else(lib):
    db, _ = lib
    start = db.change_count()
    db.set_rating(f"{FOLDER}/R0000001.JPG", 4)
    assert db.change_count() == start + 1
    db.set_rating(f"{FOLDER}/R0000001.JPG", 4)          # the same again: no change
    assert db.change_count() == start + 1
    db.upsert_photo(_rec("R0000002.JPG", "2024:08:10 11:00:00"))   # read again, the same
    assert db.change_count() == start + 1
    db.upsert_photo(_rec("R0000003.JPG", "2024:08:11 09:00:00"))   # an import
    assert db.change_count() == start + 2
    db.set_flag(f"{FOLDER}/R0000002.JPG", -1)
    assert db.change_count() == start + 3


def test_a_pick_is_kept_until_the_library_changes(lib):
    db, service = lib
    calls = []

    def work():
        calls.append(1)
        return len(calls)

    assert service._picked("test", "local", None, work) == 1
    assert service._picked("test", "local", None, work) == 1
    assert len(calls) == 1
    # Changed: the answer made is still given at once ...
    db.set_rating(f"{FOLDER}/R0000001.JPG", 5)
    assert service._picked("test", "local", None, work) == 1
    assert len(calls) == 1
    # ... and made again behind it, once.
    assert service.redo_stale_picks() == 1
    assert service._picked("test", "local", None, work) == 2
    assert service.redo_stale_picks() == 0


def test_the_timeline_shows_a_new_import_once_made_again(lib):
    db, service = lib
    months = [m["month"] for y in service.timeline() for m in y["months"]]
    assert months == ["2024-08"]
    db.upsert_photo(_rec("R0000009.JPG", "2024:09:01 09:00:00"))
    service.redo_stale_picks()
    months = [m["month"] for y in service.timeline() for m in y["months"]]
    assert months == ["2024-09", "2024-08"]


def test_demoting_a_shoot_is_seen_straight_away(lib):
    db, service = lib
    assert service.timeline()
    db.set_demoted("local", FOLDER)
    assert service.timeline() == []
