"""
GPS rows, folder notes, and the cards the Library draws from them.

A folder is named the way the importer files it, ``2026/2026-07-04``, which
says when and nothing else. A card says the rest from what the catalog already
knows: the best frame in it, the days it covers, where it was (from GPS, or
from a place someone typed), who is in it (named faces and tags) and what it
was (the other tags). Anything a person writes on the card wins over what was
worked out, and is kept twice: in the catalog, and in a small file inside the
folder, so it survives the catalog the way XMP sidecars do.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

NOTE_FILE = ".photool-folder.json"
LIBRARY_FILE = ".photool-library.json"

# A place this close to home is home: GPS wanders a little around a house, and
# a typed town name and a fix in the back garden are the same answer.
HOME_KM = 3.0
NOTE_FIELDS = ("title", "subtitle", "cover", "place", "lat", "lon", "taken_on", "kind")

# What kind of shoot a folder is, which decides how it is culled (the library's
# culling guide, .photomanager/culling.md): a school or other event is culled to
# share, so fewer frames get through, and a family day keeps more of them.
SHOOT_KINDS = ("school", "family")

# How many coordinates per folder go into naming the towns it covers. A trip
# folder of two thousand frames does not need two thousand nearest-town
# searches to say it was in three towns.
_TOWN_SAMPLES = 60

_DATE_IN_NAME = re.compile(r"(?<!\d)((?:19|20)\d\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")


# ── GPS rows ─────────────────────────────────────────────────────────

def gps_pending(db, folder: Optional[str] = None) -> list[dict]:
    """Photographs whose coordinates have not been read, with their folder."""
    sql = ("SELECT p.filepath, p.folder, p.file_size FROM photos p "
           "LEFT JOIN photo_gps g ON g.filepath = p.filepath "
           "WHERE (g.filepath IS NULL OR g.source_size != p.file_size)")
    args: tuple = ()
    if folder is not None:
        sql += " AND p.folder = ?"
        args = (folder,)
    sql += " ORDER BY p.folder DESC, p.filepath"
    with db._connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def gps_counts(db) -> dict:
    with db._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN g.filepath IS NOT NULL AND g.source_size = p.file_size THEN 1 ELSE 0 END) AS read, "
            "SUM(CASE WHEN g.lat IS NOT NULL THEN 1 ELSE 0 END) AS found "
            "FROM photos p LEFT JOIN photo_gps g ON g.filepath = p.filepath"
        ).fetchone()
    total, read, found = int(row["total"] or 0), int(row["read"] or 0), int(row["found"] or 0)
    return {"photos": total, "read": read, "unread": total - read, "found": found}


def record_gps(db, rows: list[tuple]) -> int:
    """Store ``(filepath, lat|None, lon|None, size)`` rows; returns how many."""
    if not rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    with db._connect() as conn:
        conn.executemany(
            "INSERT INTO photo_gps (filepath, lat, lon, source_size, read_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(filepath) DO UPDATE SET lat = excluded.lat, lon = excluded.lon, "
            "source_size = excluded.source_size, read_at = excluded.read_at",
            [(fp, lat, lon, int(size or 0), now) for fp, lat, lon, size in rows],
        )
    return len(rows)


# ── Folder notes ─────────────────────────────────────────────────────

def _clean_note(fields: dict) -> dict:
    note = {
        "title": str(fields.get("title") or "").strip()[:200],
        "subtitle": str(fields.get("subtitle") or "").strip()[:500],
        "cover": str(fields.get("cover") or "").strip(),
        "place": str(fields.get("place") or "").strip()[:200],
        "lat": None, "lon": None,
        # The day its photographs were taken, for the ones whose camera clock
        # gave none (a camera that lost its battery): catalog/clock.py.
        "taken_on": "",
        "kind": str(fields.get("kind") or "").strip() if fields.get("kind") in SHOOT_KINDS else "",
    }
    taken = str(fields.get("taken_on") or "").strip()
    try:
        if taken and 1900 <= datetime.strptime(taken, "%Y-%m-%d").year <= 2100:
            note["taken_on"] = taken
    except ValueError:
        pass
    try:
        lat, lon = fields.get("lat"), fields.get("lon")
        if lat is not None and lon is not None:
            lat, lon = float(lat), float(lon)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                note["lat"], note["lon"] = lat, lon
    except (TypeError, ValueError):
        pass
    if note["lat"] is None:
        note["place"] = ""      # a place is a point on the map, or nothing
    return note


def notes(db) -> dict[str, dict]:
    with db._connect() as conn:
        rows = conn.execute("SELECT * FROM folder_notes").fetchall()
    return {r["folder"]: {k: r[k] for k in NOTE_FIELDS} for r in rows}


def save_note(db, folder: str, fields: dict, library_root: Optional[str] = None) -> dict:
    """Replace a folder's note. Every field empty removes it."""
    note = _clean_note(fields)
    empty = not any(note[k] for k in ("title", "subtitle", "cover", "place", "taken_on", "kind"))
    with db._connect() as conn:
        if empty:
            conn.execute("DELETE FROM folder_notes WHERE folder = ?", (folder,))
        else:
            conn.execute(
                "INSERT INTO folder_notes (folder, title, subtitle, cover, place, lat, lon, taken_on, kind, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(folder) DO UPDATE SET "
                "title = excluded.title, subtitle = excluded.subtitle, cover = excluded.cover, "
                "place = excluded.place, lat = excluded.lat, lon = excluded.lon, "
                "taken_on = excluded.taken_on, kind = excluded.kind, updated_at = excluded.updated_at",
                (folder, note["title"], note["subtitle"], note["cover"], note["place"],
                 note["lat"], note["lon"], note["taken_on"], note["kind"],
                 datetime.now().isoformat(timespec="seconds")),
            )
    if library_root:
        write_sidecar(library_root, folder, None if empty else note)
    return note


def write_sidecar(library_root: str, folder: str, note: Optional[dict]) -> bool:
    """The durable copy. A failure to write it never fails the save."""
    path = Path(library_root) / folder / NOTE_FILE
    try:
        if note is None:
            if path.exists():
                path.unlink()
            return True
        path.write_text(json.dumps(note, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except OSError:
        return False


def adopt_sidecars(db, library_root: str, folders: list[str]) -> int:
    """Take in notes found beside photographs for folders the catalog has none for."""
    have = notes(db)
    adopted = 0
    for folder in folders:
        if folder in have:
            continue
        path = Path(library_root) / folder / NOTE_FILE
        try:
            if not path.is_file():
                continue
            fields = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(fields, dict):
            save_note(db, folder, fields)       # no library_root: the file is already there
            adopted += 1
    return adopted


def set_place_many(db, folders: list[str], place: dict, library_root: Optional[str] = None) -> int:
    """Give many folders one place (or clear it), keeping the rest of each note."""
    have = notes(db)
    lat, lon = (place or {}).get("lat"), (place or {}).get("lon")
    changed = 0
    for folder in folders:
        note = dict(have.get(folder) or {})
        note.update(place=(place or {}).get("place", ""), lat=lat, lon=lon)
        save_note(db, folder, note, library_root)
        changed += 1
    return changed


def recent_places(db, limit: int = 8) -> list[dict]:
    """Places typed on folders, newest first, one entry per place."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT place, lat, lon, MAX(updated_at) AS at, COUNT(*) AS folders FROM folder_notes "
            "WHERE lat IS NOT NULL AND place != '' GROUP BY place ORDER BY at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [{"place": r["place"], "lat": r["lat"], "lon": r["lon"], "folders": r["folders"]} for r in rows]


# ── Home ─────────────────────────────────────────────────────────────

def get_home(db) -> Optional[dict]:
    with db._connect() as conn:
        row = conn.execute("SELECT value FROM library_settings WHERE key = 'home'").fetchone()
    if not row:
        return None
    try:
        home = json.loads(row["value"])
    except ValueError:
        return None
    return home if isinstance(home, dict) and home.get("lat") is not None else None


def set_home(db, place: Optional[dict], library_root: Optional[str] = None) -> Optional[dict]:
    """Where home is, or ``None`` to forget it. Kept beside the photographs too."""
    home = None
    if place and place.get("lat") is not None and place.get("lon") is not None:
        clean = _clean_note({"place": place.get("place", ""), "lat": place["lat"], "lon": place["lon"]})
        if clean["lat"] is not None:
            home = {"place": clean["place"], "lat": clean["lat"], "lon": clean["lon"]}
    with db._connect() as conn:
        if home is None:
            conn.execute("DELETE FROM library_settings WHERE key = 'home'")
        else:
            conn.execute(
                "INSERT INTO library_settings (key, value, updated_at) VALUES ('home', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (json.dumps(home, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
            )
    if library_root:
        path = Path(library_root) / LIBRARY_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        if home is None:
            data.pop("home", None)
        else:
            data["home"] = home
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
    return home


def adopt_home(db, library_root: str) -> Optional[dict]:
    """Home from the file beside the photographs, when the catalog has none."""
    if get_home(db) is not None:
        return None
    try:
        data = json.loads((Path(library_root) / LIBRARY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("home"), dict):
        return set_home(db, data["home"])
    return None


def km_between(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    import math

    la1, la2 = math.radians(a_lat), math.radians(b_lat)
    d = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin(math.radians(b_lon - a_lon) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, d)))


# ── Cards ────────────────────────────────────────────────────────────

def _capture_day(value: str) -> Optional[str]:
    """``2026:07:04 10:11:12`` (EXIF) or ISO to ``2026-07-04``; junk to ``None``."""
    if not value or len(value) < 10:
        return None
    day = value[:10].replace(":", "-")
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return None
    return day if 1990 <= parsed.year <= 2100 else None


def date_in_name(folder: str) -> Optional[str]:
    """The last date written in a folder's path, ``2026/2026-07-04`` to ``2026-07-04``."""
    found = _DATE_IN_NAME.findall(folder or "")
    if not found:
        return None
    y, m, d = found[-1]
    try:
        return datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _pick_cover(rows: list[tuple]) -> str:
    """The frame that stands for a folder: a pick, else the best rated, else the middle.

    Rejects are never chosen while anything else is there.
    """
    if not rows:
        return ""
    kept = [r for r in rows if r[1] != -1] or rows
    best = max(kept, key=lambda r: (r[1] == 1, r[2]))
    if best[1] == 1 or best[2] > 0:
        return best[0]
    return kept[len(kept) // 2][0]


def build_cards(db, gazetteer=None, demoted: frozenset = frozenset()) -> dict:
    """Everything the folder overview and the map draw, in one pass.

    ``demoted`` is the ``{(folder, camera)}`` the viewer finds less important:
    a cover worked out for them is not one of those cameras' frames, as long as
    the folder has any other."""
    with db._connect() as conn:
        photo_rows = conn.execute(
            "SELECT folder, filepath, base_name, flag, rating, color_label, camera, local_date, "
            "COALESCE(NULLIF(local_date, ''), capture_date) AS capture_date "
            "FROM photos ORDER BY folder, filepath"
        ).fetchall()
        people_rows = conn.execute(
            "SELECT p.folder, c.name, c.tag_id, COUNT(DISTINCT p.base_name) AS n "
            "FROM photo_faces f JOIN person_clusters c ON c.id = f.cluster_id AND c.name != '' "
            "JOIN photos p ON p.filepath = f.filepath GROUP BY p.folder, c.id"
        ).fetchall()
        person_tags = {r[0] for r in conn.execute(
            "SELECT DISTINCT tag_id FROM person_clusters WHERE tag_id > 0").fetchall()}
        # A tag spelled like a named person is that person, linked or not: a
        # name typed as a tag before the face was named is still a name.
        person_names = {r[0].casefold(): r[0] for r in conn.execute(
            "SELECT DISTINCT name FROM person_clusters WHERE name != ''").fetchall()}
        tag_rows = conn.execute(
            "SELECT p.folder, t.id, t.name, COUNT(DISTINCT p.base_name) AS n "
            "FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            "JOIN photos p ON p.filepath = pt.filepath GROUP BY p.folder, t.id"
        ).fetchall()
        gps_rows = conn.execute(
            "SELECT p.folder, g.lat, g.lon FROM photo_gps g JOIN photos p ON p.filepath = g.filepath "
            "WHERE g.lat IS NOT NULL"
        ).fetchall()
    folder_notes = notes(db)

    by_folder: dict[str, list] = defaultdict(list)
    for r in photo_rows:
        by_folder[r["folder"]].append(r)

    people: dict[str, Counter] = defaultdict(Counter)
    for r in people_rows:
        people[r["folder"]][r["name"]] = max(people[r["folder"]][r["name"]], int(r["n"]))
    tags: dict[str, Counter] = defaultdict(Counter)
    for r in tag_rows:
        if r["id"] in person_tags or r["name"].casefold() in person_names:
            # A named person's tag: the same person, counted where tagging found
            # them even when the face scan did not, and spelled as their face is.
            name = person_names.get(r["name"].casefold(), r["name"])
            people[r["folder"]][name] = max(people[r["folder"]][name], int(r["n"]))
        else:
            tags[r["folder"]][r["name"]] += int(r["n"])
    coords: dict[str, list] = defaultdict(list)
    for r in gps_rows:
        coords[r["folder"]].append((float(r["lat"]), float(r["lon"])))

    # Name places first, so "abroad" can be decided against the country most
    # of the library was taken in.
    places: dict[str, dict] = {}
    countries: Counter = Counter()
    for folder in by_folder:
        note = folder_notes.get(folder) or {}
        if note.get("lat") is not None:
            place = {"label": note.get("place") or "", "lat": note["lat"], "lon": note["lon"],
                     "source": "you", "towns": [], "country_code": ""}
            if gazetteer is not None:
                near = gazetteer.nearest(note["lat"], note["lon"])
                if near:
                    place["country_code"] = near.country_code
                    countries[near.country_code] += len(by_folder[folder])
            places[folder] = place
            continue
        points = coords.get(folder)
        if not points:
            continue
        lat = statistics.median(p[0] for p in points)
        lon = statistics.median(p[1] for p in points)
        place = {"label": "", "lat": lat, "lon": lon, "source": "gps", "towns": [],
                 "country_code": "", "_near": [], "photos": len(points)}
        if gazetteer is not None:
            step = max(1, len(points) // _TOWN_SAMPLES)
            near = [gazetteer.nearest(a, b) for a, b in points[::step]]
            near = [n for n in near if n is not None]
            place["_near"] = near
            if near:
                code = Counter(n.country_code for n in near).most_common(1)[0][0]
                place["country_code"] = code
                countries[code] += len(points)
        places[folder] = place

    home = countries.most_common(1)[0][0] if countries else ""
    # A home that was set says which country is home, whatever the GPS says.
    home_place = get_home(db)
    if home_place is not None and gazetteer is not None:
        near = gazetteer.nearest(home_place["lat"], home_place["lon"])
        if near is not None:
            home = near.country_code
    for place in places.values():
        near = place.pop("_near", None)
        if near:
            towns = Counter(n.label(home) for n in near)
            place["towns"] = [name for name, _ in towns.most_common(4)]
            place["label"] = place["towns"][0]

    cards = []
    for folder, rows in by_folder.items():
        note = folder_notes.get(folder) or {}
        days = sorted(d for d in (_capture_day(r["capture_date"]) for r in rows) if d)
        named_day = date_in_name(folder)
        # A folder named for a day is that day, even when a few frames carry a
        # camera clock that was never set: the name is what the importer chose.
        # Not when its times were put right (catalog/clock.py): the name is
        # then the day a wrong clock said.
        if any(r["local_date"] for r in rows):
            named_day = None
        if named_day:
            date_from = date_to = named_day
            if days and days[0] >= named_day:
                date_to = max(named_day, days[-1])
        else:
            date_from, date_to = (days[0], days[-1]) if days else (None, None)
        bases = {r["base_name"] or r["filepath"] for r in rows}
        decided = {r["base_name"] or r["filepath"] for r in rows
                   if r["rating"] > 0 or r["flag"] != 0 or r["color_label"] != 0}
        cover_rows = [(r["filepath"], int(r["flag"] or 0), int(r["rating"] or 0)) for r in rows]
        cover = note.get("cover") or ""
        if cover and not any(r[0] == cover for r in cover_rows):
            cover = ""                      # the chosen frame was moved or deleted
        hidden = {c for f, c in demoted if f == folder and c != "*"}
        if hidden and not cover:
            # A pair goes by whichever half knows its camera, as the grid does.
            pair_camera: dict[str, str] = {}
            for r in rows:
                camera = (r["camera"] or "").strip()
                if camera and not pair_camera.get(r["base_name"]):
                    pair_camera[r["base_name"]] = camera
            shown = {r["filepath"] for r in rows if pair_camera.get(r["base_name"], "") not in hidden}
            cover_rows = [r for r in cover_rows if r[0] in shown] or cover_rows
        cards.append({
            "folder": folder,
            "count": len(bases),
            "picks": len({r["base_name"] for r in rows if r["flag"] == 1}),
            "rejects": len({r["base_name"] for r in rows if r["flag"] == -1}),
            "decided": len(decided),
            "date_from": date_from,
            "date_to": date_to,
            "cover": cover or _pick_cover(cover_rows),
            "note": {k: note.get(k) for k in NOTE_FIELDS} if note else None,
            "place": places.get(folder),
            "people": [{"name": n, "photos": c} for n, c in people[folder].most_common(8)],
            "tags": [{"name": n, "photos": c} for n, c in tags[folder].most_common(6)],
        })
    cards.sort(key=lambda c: (c["date_from"] or "", c["folder"]), reverse=True)

    # Home: a folder with no place counts as home, and one placed at home says
    # so, so the card can stay quiet about it and the map can give home one pin.
    for card in cards:
        place = card["place"]
        if home_place is None:
            card["at_home"] = False
        elif place is None:
            card["at_home"] = True
        else:
            card["at_home"] = km_between(place["lat"], place["lon"],
                                         home_place["lat"], home_place["lon"]) <= HOME_KM
    return {"cards": cards, "home_country": home, "home": home_place,
            "recent_places": recent_places(db)}


def cover_candidates(db, folder: str, limit: int = 24) -> list[str]:
    """Frames worth offering as a folder's cover: picks and stars first."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT filepath, base_name FROM photos WHERE folder = ? AND flag != -1 "
            "ORDER BY (flag = 1) DESC, rating DESC, filepath",
            (folder,),
        ).fetchall()
    seen, out = set(), []
    for r in rows:
        key = r["base_name"] or r["filepath"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r["filepath"])
    if len(out) > limit:
        # Keep the decided frames at the front and spread the rest across the day.
        head = out[: limit // 2]
        rest = out[limit // 2:]
        step = len(rest) / (limit - len(head))
        out = head + [rest[int(i * step)] for i in range(limit - len(head))]
    return out
