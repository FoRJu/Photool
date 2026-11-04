"""
Bringing an old collection into the library (importer/archive_plan.py, the
rules, and tools/archive_import.py, the plan and the copy). Every folder name
here is made up, in the shapes such collections have.
"""

import csv
import struct
import sys
from datetime import date, datetime
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from importer.archive_plan import (  # noqa: E402
    Rules, SourceFile, SourceFolder, decide, find_date, plan_folders,
)


# ── Dates in names ──────────────────────────────────────────────────

@pytest.mark.parametrize("name, not_after, value, rest", [
    ("2010-02-21 Picnic", None, "2010-02-21", "Picnic"),
    ("Scan 20080314", None, "2008-03-14", "Scan"),
    ("scan 2010 0612", None, "2010-06-12", "scan"),
    ("Recital I 09182004", None, "2004-09-18", "Recital I"),
    ("Camera 08142005", None, "2005-08-14", "Camera"),
    ("11-03-08", None, "2008-11-03", ""),
    ("20080305-20080512", None, "2008-03-05", ""),
    ("200210 Lakeside", None, "2002-10", "Lakeside"),
    ("200210Hill walk 1", None, "2002-10", "Hill walk 1"),
    ("OLD TOWN 052005", None, "2005-05", "OLD TOWN"),
    ("Coast road Feb 2007", None, "2007-02", "Coast road"),
    ("Kite flying Mar08", None, "2008-03", "Kite flying"),
    ("Coast trip Aug 2006", None, "2006-08", "Coast trip"),
    ("2006 MARCH GARDEN IN BLOOM", None, "2006-03", "GARDEN IN BLOOM"),
    ("2006 五月底 公园", None, "2006-05", "公园"),
    ("Canyon2004 II", None, "2004", "Canyon II"),
    ("2002around new year", None, "2002", "around new year"),
    ("Canyon 0603", date(2003, 6, 30), "2003-06", "Canyon"),
])
def test_dates_in_folder_names(name, not_after, value, rest):
    found = find_date(name, not_after)
    assert found is not None and found.value == value and found.rest == rest


def test_a_date_after_the_files_were_saved_is_not_the_date():
    # A backup made in August 2007 of scans saved in 2006 does not date them.
    assert find_date("20070815 Backup of an old disk", date(2006, 7, 1)) is None
    assert find_date("Harbour") is None and find_date("10234") is None


# ── One folder ─────────────────────────────────────────────────────

def _files(n, ext=".jpg", taken=None, make="Canon", saved=date(2010, 3, 1)):
    return [SourceFile(name=f"IMG_{i:04d}{ext}", size=1000 + i, modified=saved, fingerprint=f"{ext}{i}{taken}",
                       taken=taken, make=make if taken else "") for i in range(n)]


def _folder(rel, files, other=0):
    return SourceFolder(rel=rel, media=files, other=other)


DAY = datetime(2010, 2, 21, 10)


@pytest.mark.parametrize("rel, dest", [
    ("2012/2012-03-10", "2012/2012-03-10"),
    ("Photo/2010/2010-02-21 Picnic", "2010/2010-02-21 Picnic"),
    ("Photo/2012/07/16 party", "2012/2012-07-16 party"),
    ("Photo/2012/07/21/101CANON", "2012/2012-07-21"),
    ("Photo/2011/Camera 2011-06-12", "2011/2011-06-12 Camera"),
])
def test_day_folders_join_the_library_s_own(rel, dest):
    row = decide(_folder(rel, _files(3)))
    assert (row.action, row.dest) == ("merge", dest)


def test_a_scan_is_dated_by_its_folder_and_its_rolls_are_one_trip():
    scans = _files(5, ".psd", saved=date(2006, 9, 20))
    row = decide(_folder("Scanned film/Coast trip Aug 2006/10234", scans))
    assert (row.action, row.kind, row.dest, row.set_date) == (
        "archive", "scan", "Archive/2006/2006-08 Coast trip", "yes")


def test_a_scan_with_no_date_anywhere_is_asked_about():
    row = decide(_folder("Scanned film/Wedding/Roll 1", _files(4, ".tif", saved=date(2004, 1, 9))))
    assert row.dest == "Archive/Undated/Wedding" and row.review == "yes"
    assert "2004-01" in row.notes


def test_a_camera_dump_is_filed_by_day_and_a_named_event_kept_whole():
    dump = decide(_folder("Backups/Card backup/DCIM/101CANON", _files(5, taken=DAY)))
    assert (dump.action, dump.dest) == ("split", "Archive/{year}/{date}")
    event = decide(_folder("Backups/SD400/Harbour walk Sept 2008",
                           _files(5, taken=datetime(2008, 9, 13, 15))))
    assert (event.action, event.dest) == ("archive", "Archive/2008/2008-09-13 Harbour walk")


def test_a_range_in_the_day_tree_goes_to_the_day_folders():
    files = _files(2, taken=datetime(2008, 3, 5)) + _files(2, ".cr2", taken=datetime(2008, 5, 1))
    row = decide(_folder("Photo/2008/20080305-20080512", files))
    assert (row.action, row.dest) == ("split", "{year}/{year}-{month}-{day}")


def test_caches_and_document_folders_are_left_out():
    assert decide(_folder("old/iPod Photo Cache/F00", _files(3))).action == "skip"
    assert decide(_folder("misc", _files(1), other=20)).action == "skip"


def test_a_collection_s_own_rules_come_from_its_file(tmp_path):
    path = tmp_path / "archive-rules.txt"
    path.write_text("# this collection's own\nskip: work files\nskip: music*\n"
                    "generic: from the old laptop\nstar: picks for the album\n", encoding="utf-8")
    rules = Rules.read(path)
    assert decide(_folder("Drive/Work files/run 1", _files(3, ".tif")), rules=rules).action == "skip"
    assert decide(_folder("Drive/Music library", _files(1)), rules=rules).action == "skip"
    assert decide(_folder("Drive/Work files/run 1", _files(3, ".tif"))).action != "skip"
    named = decide(_folder("Harbour 2005/From the old laptop", _files(2, ".tif")), rules=rules)
    assert named.dest == "Archive/2005/2005 Harbour"
    assert decide(_folder("Picks for the album", _files(2)), rules=rules).action == "star"


def test_copies_are_taken_once_from_the_best_dated_folder():
    a = _folder("Disk A/Scanned film/200210 Lakeside", _files(3, ".tif", saved=date(2002, 10, 20)))
    b = _folder("Photos/Albums/200210 Lakeside", _files(3, ".tif", saved=date(2002, 10, 20)))
    rows = {r.source: r for r in plan_folders([a, b], {})}
    kept = [r for r in rows.values() if r.action != "skip"]
    assert len(kept) == 1 and kept[0].new == 3
    other = next(r for r in rows.values() if r.action == "skip")
    assert other.duplicates == 3 and "elsewhere" in other.notes


# ── Plan and apply, end to end ───────────────────────────────────────

def _jpeg(path: Path, taken: datetime = None, colour=(120, 80, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), colour)
    exif = Image.Exif()
    if taken:
        exif[271] = "Canon"
        exif.get_ifd(0x8769)[36867] = taken.strftime("%Y:%m:%d %H:%M:%S")
    img.save(path, "JPEG", exif=exif.tobytes())


def _psd(path: Path, rgb=(200, 40, 10)):
    w, h = 16, 12
    path.parent.mkdir(parents=True, exist_ok=True)
    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, h, w, 8, 3)
    path.write_bytes(head + struct.pack(">III", 0, 0, 0) + struct.pack(">H", 0)
                     + b"".join(bytes([c]) * (w * h) for c in rgb))


class FakeServer:
    def __init__(self):
        self.indexed, self.rated = [], []

    def reachable(self):
        return True

    def index(self, folder):
        self.indexed.append(folder)

    def rate(self, path, stars):
        self.rated.append((path, stars))


@pytest.fixture()
def world(tmp_path, monkeypatch):
    from config import Config
    import tools.archive_import as tool

    monkeypatch.setattr(tool, "_exiftool", lambda: None)       # read dates with Pillow
    src, lib = tmp_path / "old", tmp_path / "lib"
    # Already by day, and the library already has that day under a name.
    _jpeg(src / "2012/2012-08-04/IMG_0001.JPG", datetime(2012, 8, 4, 9))
    _jpeg(src / "2012/2012-08-04/IMG_0002.JPG", datetime(2012, 8, 4, 9, 5), (10, 10, 10))
    _jpeg(lib / "2012/2012-08-04 Beach/IMG_0002.JPG", datetime(2012, 8, 4, 9, 5), (10, 10, 10))
    # A camera dump, and the same card copied into a backup.
    for root in ("Backups/Card backup/DCIM/100CANON", "My Pictures/Card backup/DCIM/100CANON"):
        _jpeg(src / root / "IMG_0100.JPG", datetime(2006, 4, 8, 12), (1, 2, 3))
        _jpeg(src / root / "IMG_0101.JPG", datetime(2006, 4, 9, 12), (4, 5, 6))
    # Film scans: a scanned JPEG and its edited Photoshop document, same name.
    scans = src / "Scanned film/200210 Lakeside"
    _jpeg(scans / "scan0001.jpg", None, (50, 60, 70))
    _psd(scans / "scan0001.psd")
    (scans / "Thumbs.db").write_bytes(b"x")
    # Starred copies of a dump photo, and one found nowhere else.
    starred = src / "old/Starred Photos"
    starred.mkdir(parents=True)
    (starred / "IMG_0100.JPG").write_bytes((src / "Backups/Card backup/DCIM/100CANON/IMG_0100.JPG").read_bytes())
    _jpeg(starred / "only_here.jpg", datetime(2007, 1, 1), (9, 9, 9))
    # An application's cache.
    _jpeg(src / "old/iPod Photo Cache/F01/thumb1.jpg", None)
    config = Config(library_root=str(lib), db_path=str(tmp_path / "c.db"),
                    thumbnail_cache=str(tmp_path / "t"), folder_pattern="{year}/{year}-{month}-{day}")
    return src, lib, config, tool


def _plan(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        assert fh.readline().startswith("# source:")
        return {r["source"]: r for r in csv.DictReader(fh)}


def test_plan_then_apply(world, tmp_path):
    src, lib, config, tool = world
    plan = tmp_path / "archive-plan.csv"
    tool.run_plan(src, plan, config, say=lambda *_: None)
    rows = _plan(plan)
    assert rows["2012/2012-08-04"]["action"] == "merge"
    assert rows["2012/2012-08-04"]["in_library"] == "1"
    assert rows["Scanned film/200210 Lakeside"]["dest"] == "Archive/2002/2002-10 Lakeside"
    assert rows["Scanned film/200210 Lakeside"]["other_files"] == "1"
    dumps = [r for k, r in rows.items() if k.endswith("100CANON")]
    assert sorted(r["action"] for r in dumps) == ["skip", "split"]
    assert rows["old/iPod Photo Cache/F01"]["action"] == "skip"
    assert rows["old/Starred Photos"]["action"] == "star"

    # A dry run copies nothing.
    tool.run_apply(plan, config, dry_run=True, server=FakeServer(), say=lambda *_: None)
    assert not (lib / "Archive").exists()

    server = FakeServer()
    out = tool.run_apply(plan, config, server=server, say=lambda *_: None)
    assert out["failed"] == 0
    # The day joins the folder the library already has for it; the photo it has is not copied again.
    assert sorted(p.name for p in (lib / "2012/2012-08-04 Beach").iterdir()) == ["IMG_0001.JPG", "IMG_0002.JPG"]
    # The dump is filed by day, once.
    assert (lib / "Archive/2006/2006-04-08/IMG_0100.JPG").is_file()
    assert (lib / "Archive/2006/2006-04-09/IMG_0101.JPG").is_file()
    assert len(list(lib.rglob("IMG_0100.JPG"))) == 1
    # The scan and its edit are two photographs, not a pair; the cache file is left behind.
    scans = lib / "Archive/2002/2002-10 Lakeside"
    assert sorted(p.name for p in scans.iterdir()) == ["scan0001-psd.psd", "scan0001.jpg"]
    # Copies are exact.
    assert (scans / "scan0001.jpg").read_bytes() == (src / "Scanned film/200210 Lakeside/scan0001.jpg").read_bytes()
    # Favourites: the original gets stars, the one found nowhere else is copied and starred.
    rated = dict(server.rated)
    assert rated["Archive/2006/2006-04-08/IMG_0100.JPG"] == 4
    assert rated["Archive/Starred/only_here.jpg"] == 4
    assert (lib / "Archive/Starred/only_here.jpg").is_file()
    assert "Archive/2002/2002-10 Lakeside" in server.indexed
    assert not list(lib.rglob("thumb1.jpg"))

    # Run again: nothing is copied twice.
    again = tool.run_apply(plan, config, server=FakeServer(), say=lambda *_: None)
    assert again["copied"] == 0 and again["failed"] == 0
    log = list(csv.DictReader(open(tmp_path / "archive-log.csv", encoding="utf-8-sig")))
    assert {r["what"] for r in log} <= {"exists", "in library", "duplicate", "star"}


def test_a_date_edited_in_dest_is_the_one_used(world, tmp_path, monkeypatch):
    """The date column is for reading: Excel rewrites dates in it when it saves.
    A scan is dated by the start of its folder's name."""
    src, lib, config, tool = world
    plan = tmp_path / "archive-plan.csv"
    tool.run_plan(src, plan, config, say=lambda *_: None)
    text = plan.read_text(encoding="utf-8-sig").replace("Archive/2002/2002-10 Lakeside",
                                                        "Archive/2002/2002-10-14 Lakeside walk")
    text = text.replace(",2002-10,", ",Oct-02,")                  # what Excel does to the date column
    plan.write_text(text, encoding="utf-8-sig")
    calls = []
    monkeypatch.setattr(tool, "_exiftool", lambda: "exiftool")
    monkeypatch.setattr(tool.subprocess, "run", lambda args, **k: calls.append(args) or
                        type("R", (), {"stdout": b"[]", "returncode": 0})())
    tool.run_apply(plan, config, server=FakeServer(), say=lambda *_: None)
    stamp = [a for a in calls if any(str(x).startswith("-DateTimeOriginal=") for x in a)]
    assert stamp and "-DateTimeOriginal=2002:10:14 12:00:00" in stamp[0]
    assert (lib / "Archive/2002/2002-10-14 Lakeside walk/scan0001.jpg").is_file()
