"""
When a photograph was taken, in the time of the place it was taken.

A camera files its pictures by its own clock, and the clock is only as right as
whoever set it last. A camera set to another zone for a trip and never set
back files months of pictures taken at home under the wrong day. A DJI pocket
camera keeps UTC, so an evening far east of Greenwich lands on the day after. A Ricoh says nothing about its clock
at all.

What a photograph's date should be is the wall-clock time where it was taken.
That takes two things:

* **The moment.** What the clock read, and the zone the clock was on: what the
  file says (``OffsetTimeOriginal``, or DJI's ``+00:00``), or what a person
  said about that camera over those days (a ``camera_clocks`` rule). A clock
  with no zone is taken to be on local time already, and left alone.
* **The place.** A rule's zone for that camera and those days (the DJI was
  abroad while the Ricoh was at home, in one folder); else the folder's place
  (typed on its card, or its GPS); else home.

The clock's own reading stays in ``photos.capture_date``; the time worked out
here goes to ``photos.local_date`` when it differs, and every reader of a
photo's date reads that (CatalogDB._row_to_record). Nothing is written to the
files, and nothing moves: a folder named for the day the clock said keeps its
name, and its card shows the day it really was.
"""

import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones
except ImportError:                 # pragma: no cover - Python < 3.9
    ZoneInfo = None

_RAW = "%Y:%m:%d %H:%M:%S"
_OFFSET = re.compile(r"^([+-])(\d\d):?(\d\d)$")


# ── Zones ────────────────────────────────────────────────────────────

def zone(name: str):
    """A tz database zone by name, or None for an unknown or empty one."""
    if not name or ZoneInfo is None:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def zone_names() -> list[str]:
    if ZoneInfo is None:
        return []
    try:
        return sorted(n for n in available_timezones() if "/" in n and not n.startswith(("Etc/", "SystemV/")))
    except Exception:               # no tz database at all
        return []


def offset_zone(text: str):
    """``-07:00`` / ``+0000`` / ``Z`` as a fixed zone, or None."""
    text = (text or "").strip()
    if text in ("Z", "z"):
        return timezone.utc
    m = _OFFSET.match(text)
    if not m:
        return None
    minutes = int(m.group(2)) * 60 + int(m.group(3))
    if minutes > 14 * 60:
        return None
    return timezone(timedelta(minutes=-minutes if m.group(1) == "-" else minutes))


def recorded_offset(entry: dict) -> str:
    """The zone an exiftool entry says the camera's clock was on, ``""`` for
    none: the EXIF offsets, Canon's own setting, or the offset on XMP's
    CreateDate (DJI writes ``+00:00`` there and nowhere else)."""
    for key in ("OffsetTimeOriginal", "OffsetTime", "TimeZone"):
        value = str(entry.get(key) or "").strip()
        if offset_zone(value) is not None:
            return value
    created = str(entry.get("CreateDate") or "")
    m = re.search(r"([+-]\d\d:\d\d|Z)$", created)
    return m.group(1) if m else ""


# ── Settings ─────────────────────────────────────────────────────────

def _setting(db, key: str) -> str:
    with db._connect() as conn:
        row = conn.execute("SELECT value FROM library_settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else ""


def home_zone(db, gazetteer=None) -> tuple[str, str]:
    """Home's zone, and where it came from: ``"set"`` (chosen by name), ``"home"``
    (the home place's town) or ``""`` (not known)."""
    chosen = _setting(db, "home_zone")
    if zone(chosen) is not None:
        return chosen, "set"
    if gazetteer is not None:
        from places.store import get_home

        home = get_home(db)
        if home:
            near = gazetteer.nearest(float(home["lat"]), float(home["lon"]))
            if near is not None and zone(near.zone) is not None:
                return near.zone, "home"
    return "", ""


def set_home_zone(db, name: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with db._connect() as conn:
        if name:
            conn.execute(
                "INSERT INTO library_settings (key, value, updated_at) VALUES ('home_zone', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (name, now))
        else:
            conn.execute("DELETE FROM library_settings WHERE key = 'home_zone'")


def rules(db) -> list[dict]:
    with db._connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM camera_clocks ORDER BY camera, date_from, id")]


def add_rule(db, camera: str, date_from: str = "", date_to: str = "",
             clock_zone: str = "", where_zone: str = "") -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with db._connect() as conn:
        cur = conn.execute(
            "INSERT INTO camera_clocks (camera, date_from, date_to, clock_zone, where_zone, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)", (camera, date_from, date_to, clock_zone, where_zone, now))
        return int(cur.lastrowid)


def remove_rule(db, rule_id: int) -> bool:
    with db._connect() as conn:
        return conn.execute("DELETE FROM camera_clocks WHERE id = ?", (int(rule_id),)).rowcount > 0


def _rule_for(found: list[dict], camera: str, day: str) -> Optional[dict]:
    """The latest rule for this camera that covers this clock day."""
    best = None
    for r in found:
        if r["camera"] != camera:
            continue
        if r["date_from"] and day < r["date_from"]:
            continue
        if r["date_to"] and day > r["date_to"]:
            continue
        if best is None or r["id"] > best["id"]:
            best = r
    return best


# ── Where a folder was ───────────────────────────────────────────────

def folder_zones(db, gazetteer) -> dict[str, str]:
    """The zone of each folder that has a place: typed on its card, else the
    middle of its GPS."""
    if gazetteer is None:
        return {}
    with db._connect() as conn:
        typed = {r["folder"]: (r["lat"], r["lon"]) for r in conn.execute(
            "SELECT folder, lat, lon FROM folder_notes WHERE lat IS NOT NULL AND lon IS NOT NULL")}
        gps = {r["folder"]: (r["lat"], r["lon"]) for r in conn.execute(
            "SELECT p.folder, AVG(g.lat) AS lat, AVG(g.lon) AS lon FROM photo_gps g "
            "JOIN photos p ON p.filepath = g.filepath WHERE g.lat IS NOT NULL GROUP BY p.folder")}
    out = {}
    for folder, (lat, lon) in {**gps, **typed}.items():
        near = gazetteer.nearest(float(lat), float(lon))
        if near is not None and zone(near.zone) is not None:
            out[folder] = near.zone
    return out


# ── The time ─────────────────────────────────────────────────────────

def local_time(raw: str, offset: str = "", clock_zone: str = "", where_zone: str = "") -> str:
    """What the clock read, ``raw`` (``YYYY:MM:DD HH:MM:SS``), as the time where
    it was taken; ``""`` when that is the same, or cannot be known.

    The clock was on ``clock_zone`` when one is named, else on ``offset`` (what
    the file said); with neither it is taken to be on local time already."""
    try:
        read = datetime.strptime((raw or "")[:19], _RAW)
    except ValueError:
        return ""
    clock = zone(clock_zone) or offset_zone(offset)
    where = zone(where_zone)
    if clock is None or where is None:
        return ""
    # A named zone needs the clock reading placed in it (its summer time
    # included); a fixed offset is the same either way.
    local = read.replace(tzinfo=clock).astimezone(where).replace(tzinfo=None)
    out = local.strftime(_RAW)
    return "" if out == raw[:19] else out


def _no_clock(raw: str) -> bool:
    """A clock reading that is no time at all: empty, or zeros."""
    try:
        return datetime.strptime((raw or "")[:19], _RAW).year < 1900
    except ValueError:
        return True


def recompute(db, gazetteer=None, folder: Optional[str] = None) -> int:
    """Work out ``local_date`` for every photograph (or one folder's) again.
    Returns how many rows changed.

    A photograph whose clock gave no time at all takes its folder's ``taken_on``
    day, when one was given, at noon and a second apart in file order so the
    folder keeps its sequence."""
    home, _ = home_zone(db, gazetteer)
    places = folder_zones(db, gazetteer)
    found = rules(db)
    with db._connect() as conn:
        taken_on = {r["folder"]: r["taken_on"] for r in conn.execute(
            "SELECT folder, taken_on FROM folder_notes WHERE taken_on != ''")}
    sql = ("SELECT p.filepath, p.folder, p.camera, p.base_name, p.capture_date, p.local_date, "
           "c.clock_offset FROM photos p LEFT JOIN photo_clock c ON c.filepath = p.filepath")
    args: tuple = ()
    if folder is not None:
        sql += " WHERE p.folder = ?"
        args = (folder,)
    with db._connect() as conn:
        rows = conn.execute(sql, args).fetchall()

    # A pair is one photograph: a camera's JPEG that lost its EXIF goes by its
    # raw twin's camera and zone, as the grid does.
    twin_camera: dict[tuple, str] = {}
    twin_offset: dict[tuple, str] = {}
    for r in rows:
        key = (r["folder"], r["base_name"])
        if (r["camera"] or "").strip() and key not in twin_camera:
            twin_camera[key] = r["camera"].strip()
        if r["clock_offset"] and key not in twin_offset:
            twin_offset[key] = r["clock_offset"]

    # The undated, in file order, a pair as one.
    undated: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda r: (r["folder"], r["base_name"] or "")):
        if r["folder"] in taken_on and _no_clock(r["capture_date"]):
            names = undated.setdefault(r["folder"], [])
            if not names or names[-1] != r["base_name"]:
                names.append(r["base_name"])
    order = {(f, b): i for f, names in undated.items() for i, b in enumerate(names)}

    changes = []
    for r in rows:
        key = (r["folder"], r["base_name"])
        if key in order:
            day = datetime.strptime(taken_on[r["folder"]], "%Y-%m-%d") + timedelta(hours=12, seconds=order[key])
            new = day.strftime(_RAW)
            if new != (r["local_date"] or ""):
                changes.append((new, r["filepath"]))
            continue
        camera = (r["camera"] or "").strip() or twin_camera.get(key, "")
        raw = r["capture_date"] or ""
        rule = _rule_for(found, camera, raw[:10].replace(":", "-")) if camera else None
        where = (rule and rule["where_zone"]) or places.get(r["folder"]) or home
        new = local_time(raw, r["clock_offset"] or twin_offset.get(key, ""),
                         (rule and rule["clock_zone"]) or "", where)
        if new != (r["local_date"] or ""):
            changes.append((new, r["filepath"]))
    if changes:
        with db._connect() as conn:
            conn.executemany("UPDATE photos SET local_date = ? WHERE filepath = ?", changes)
    return len(changes)


def moved_days(db) -> dict:
    """The folders whose photographs were taken on another day than their
    camera's clock said: how many, and on which days."""
    with db._connect() as conn:
        # A pair is one photograph, and goes by whichever half knows its camera.
        rows = conn.execute(
            "SELECT folder, camera, clock_day, day, COUNT(DISTINCT base_name) AS n FROM ("
            "  SELECT p.folder, p.base_name, substr(p.capture_date, 1, 10) AS clock_day, "
            "  substr(p.local_date, 1, 10) AS day, "
            "  (SELECT MAX(t.camera) FROM photos t WHERE t.folder = p.folder "
            "   AND t.base_name = p.base_name) AS camera "
            "  FROM photos p WHERE p.local_date != '' "
            "  AND substr(p.local_date, 1, 10) != substr(p.capture_date, 1, 10)) "
            "GROUP BY folder, camera, clock_day, day ORDER BY folder DESC").fetchall()
    folders: dict[str, dict] = {}
    for r in rows:
        f = folders.setdefault(r["folder"], {"folder": r["folder"], "photos": 0, "moves": []})
        f["photos"] += r["n"]
        f["moves"].append({"camera": r["camera"], "from": r["clock_day"].replace(":", "-"),
                           "to": r["day"].replace(":", "-"), "photos": r["n"]})
    return {"photos": sum(f["photos"] for f in folders.values()), "folders": list(folders.values())}


def cameras(db) -> list[dict]:
    """Each camera, with how many photographs and the zones its files say its
    clock was on (with the clock days each was seen)."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT p.camera, COALESCE(c.clock_offset, '') AS clock_offset, COUNT(*) AS n, "
            "MIN(substr(p.capture_date, 1, 10)) AS first, MAX(substr(p.capture_date, 1, 10)) AS last "
            "FROM photos p LEFT JOIN photo_clock c ON c.filepath = p.filepath "
            "WHERE p.camera != '' GROUP BY p.camera, clock_offset ORDER BY p.camera").fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        cam = out.setdefault(r["camera"], {"camera": r["camera"], "photos": 0, "recorded": []})
        cam["photos"] += r["n"]
        cam["recorded"].append({"offset": r["clock_offset"], "photos": r["n"],
                                "first": (r["first"] or "").replace(":", "-"),
                                "last": (r["last"] or "").replace(":", "-")})
    return sorted(out.values(), key=lambda c: -c["photos"])


# The town list, for folder places and home (set at startup: configure).
_gazetteer_fn = None


def configure(gazetteer_fn) -> None:
    global _gazetteer_fn
    _gazetteer_fn = gazetteer_fn


def _gazetteer():
    try:
        return _gazetteer_fn() if _gazetteer_fn else None
    except Exception:
        logger.exception("The town list could not be loaded")
        return None


def refresh(db, folder: Optional[str] = None) -> int:
    """:func:`recompute` with the town list: after a place, home, a rule or a
    photograph's date changed."""
    if not getattr(db, "writable", True):
        return 0
    return recompute(db, _gazetteer(), folder)


# ── Reading what each file says ──────────────────────────────────────

_TAGS = ["OffsetTimeOriginal", "OffsetTime", "TimeZone", "XMP:CreateDate"]

_lock = threading.Lock()
_stop = threading.Event()
_job: dict = {"status": "idle", "done": 0, "total": 0, "folder": "", "error": ""}
_wanted = False
_thread = None


def job() -> dict:
    return dict(_job)


def pending(db, folder=None) -> list[dict]:
    """Photographs whose file has not been read for its clock as it is now."""
    sql = ("SELECT p.filepath, p.folder, p.file_size FROM photos p "
           "LEFT JOIN photo_clock c ON c.filepath = p.filepath "
           "WHERE c.filepath IS NULL OR c.source_size != p.file_size")
    args: tuple = ()
    if folder is not None:
        sql += " AND p.folder = ?"
        args = (folder,)
    sql += " ORDER BY p.folder DESC, p.filepath"
    with db._connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def record(db, rows: list[tuple]) -> None:
    """Store ``(filepath, size, offset)`` reads."""
    now = datetime.now().isoformat(timespec="seconds")
    with db._connect() as conn:
        conn.executemany(
            "INSERT INTO photo_clock (filepath, clock_offset, source_size, read_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(filepath) DO UPDATE SET clock_offset = excluded.clock_offset, "
            "source_size = excluded.source_size, read_at = excluded.read_at",
            [(fp, offset, int(size or 0), now) for fp, size, offset in rows])


def run(library_root, db, stop_event: Optional[threading.Event] = None) -> dict:
    """Read every pending file, a folder at a time, and work out the folder's
    times as each is done."""
    import background as gate
    from thumbnails.extract import _batch_extract_exif

    items = pending(db)
    groups: dict = {}
    for row in items:
        groups.setdefault(row["folder"], []).append(row)
    _job.update(status="running", total=len(items), done=0, error="", started_at=time.time())
    root = Path(library_root)
    gazetteer = _gazetteer() if items else None
    try:
        for name, group in groups.items():
            if stop_event is not None and stop_event.is_set():
                break
            # Background work: a folder at a time, out of the way of someone
            # using the app (see background.py).
            if not gate.wait_turn("Camera clocks", stop_event and stop_event.is_set):
                break
            _job["folder"] = name
            paths = {root / r["filepath"]: r for r in group}
            present = [p for p in paths if p.is_file()]
            found = _batch_extract_exif(present, _TAGS, fast=True) if present else {}
            record(db, [(paths[p]["filepath"], paths[p]["file_size"], recorded_offset(found.get(str(p), {})))
                        for p in present])
            recompute(db, gazetteer, name)
            _job["done"] += len(group)
        _job["status"] = "stopped" if stop_event is not None and stop_event.is_set() else "completed"
    except Exception as exc:
        logger.exception("Reading camera clocks failed")
        _job.update(status="failed", error=str(exc))
    finally:
        _job.update(folder="", finished_at=time.time())
    return job()


def wake(library_root, db) -> None:
    """Read whatever is pending, in the background. Returns at once."""
    global _thread, _wanted
    if not getattr(db, "writable", True):
        return
    with _lock:
        _wanted = True
        if _thread is not None:
            return
        _stop.clear()
        _thread = threading.Thread(target=_worker, args=(library_root, db),
                                   name="camera-clocks", daemon=True)
        _thread.start()


def _worker(library_root, db) -> None:
    global _thread, _wanted
    while True:
        with _lock:
            if not _wanted or _stop.is_set():
                _thread = None
                return
            _wanted = False
        run(library_root, db, stop_event=_stop)


def stop() -> None:
    _stop.set()
