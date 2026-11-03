"""
Other ways into a library of decades: by what is in the pictures, along time,
and back to what has not been looked at in a while.

All three read the library as search sees it (pairs merged, :mod:`catalog.search`)
and pick frames the way On this day does (:mod:`catalog.memories`): picked, then
rated, then developed, never a reject, and a burst of one moment once.

**Collections** are scenes a family library is full of — birthdays, Christmas,
the beach, snow — found by smart search (analysis/clip.py) from a fixed set of
descriptions, so nobody has to know what to type. A collection is shown only
when the library clearly holds it, and opens a year at a time: the birthdays,
every year.

**The timeline** is the library a month at a time, each month its best few
frames and its shoots, for scrolling decades instead of paging days.

**Rediscover** offers good frames from more than a year ago that this person
has not opened up close for half a year, and one good day, picked at random —
the same for a whole day, different tomorrow, or now with Shuffle.
"""

import random
import re
from datetime import date, datetime, timedelta
from typing import Optional

from catalog.memories import _moment, _score

# ── Collections ───────────────────────────────────────────────────────

# (key, title, what the pictures show). Worded the way smart search reads best:
# a short description of the picture, not a keyword.
COLLECTIONS = [
    ("birthdays", "Birthdays", "a birthday cake with candles"),
    ("christmas", "Christmas", "a christmas tree with presents"),
    ("halloween", "Halloween", "children in halloween costumes"),
    ("beach", "The beach", "people on a sandy beach by the sea"),
    ("snow", "Snow", "people playing in the snow"),
    ("pool", "Swimming", "children swimming in a pool"),
    ("mountains", "Mountains", "hiking in the mountains"),
    ("babies", "Babies", "a newborn baby"),
    ("playground", "Playgrounds", "kids at a playground"),
    ("stage", "On stage", "a performance on a stage"),
    ("sports", "Sports", "children playing a sport"),
    ("graduation", "Graduations", "a graduation ceremony with caps and gowns"),
    ("wedding", "Weddings", "a wedding"),
    ("dogs", "Dogs", "a dog"),
    ("cats", "Cats", "a cat"),
    ("food", "Meals together", "a family meal around a table"),
    ("fireworks", "Fireworks", "fireworks in the night sky"),
    ("garden", "Gardens & flowers", "flowers in a garden"),
]
BY_KEY = {k: (k, t, q) for k, t, q in COLLECTIONS}

# A collection needs this many photographs, and its best one this clear a
# match (search.SHOWS_CLEAR), to be offered: a library with no snow in it has
# no Snow shelf made of its nearest blurs.
MIN_MEMBERS = 3
COVERS = 4
PER_YEAR = 12


def _year(photo) -> Optional[int]:
    d = photo.date or ""
    return int(d[:4]) if d[:4].isdigit() else None


def _relevance(photo, scores: dict) -> float:
    values = [scores[f] for f in photo.filepaths if f in scores]
    return max(values) if values else -1.0


def members(photos: list, scores: dict, floor: float) -> list:
    """``[(relevance, photo)]`` of what a collection holds, most relevant first.
    Rejects are left out."""
    out = []
    for photo in photos:
        if photo.flag == -1:
            continue
        rel = _relevance(photo, scores)
        if rel >= floor:
            out.append((rel, photo))
    out.sort(key=lambda t: (-t[0], t[1].record.filepath))
    return out


def _worth(rel: float, photo) -> float:
    """How good a frame is to show for a collection: what it shows first (a
    hundredth of relevance is a lot), then what was made of it."""
    return 400 * rel + _score(photo)


def shelf(photos: list, scores_by_key: dict, floor: float, clear: float) -> list:
    """One entry per collection the library holds: ``{key, title, query, total,
    years, covers}``, the biggest first. ``covers`` are search Photos, one per
    year from the years it is best in."""
    out = []
    for key, title, text in COLLECTIONS:
        found = members(photos, scores_by_key.get(key) or {}, floor)
        if len(found) < MIN_MEMBERS or found[0][0] < clear:
            continue
        best_by_year: dict = {}
        for rel, photo in found:
            year = _year(photo)
            worth = _worth(rel, photo)
            if year is not None and (year not in best_by_year or worth > best_by_year[year][0]):
                best_by_year[year] = (worth, photo)
        ranked = sorted(best_by_year.values(), key=lambda t: -t[0])
        covers = [p for _, p in ranked[:COVERS]] or [p for _, p in found[:COVERS]]
        out.append({
            "key": key, "title": title, "query": f'shows:"{text}"',
            "total": len(found), "years": len(best_by_year), "covers": covers,
        })
    out.sort(key=lambda c: -c["total"])
    return out


def _spread(pairs: list, limit: int, minutes: int = 3) -> list:
    """The best ``limit`` of ``[(worth, photo)]``, a few minutes apart where the
    time is known, in the order they were taken."""
    chosen, times = [], []
    gap = timedelta(minutes=minutes)
    for _, photo in sorted(pairs, key=lambda t: (-t[0], t[1].record.filepath)):
        when = _moment(photo)
        if when is not None and any(abs(when - t) < gap for t in times):
            continue
        chosen.append(photo)
        if when is not None:
            times.append(when)
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda p: (p.record.capture_date or "", p.record.filepath))


def by_year(photos: list, scores: dict, floor: float, per_year: int = PER_YEAR) -> list:
    """A collection a year at a time, the most recent first: ``[{year, total,
    photos}]``. A frame with no date comes last, under year 0."""
    years: dict = {}
    for rel, photo in members(photos, scores, floor):
        years.setdefault(_year(photo) or 0, []).append((_worth(rel, photo), photo))
    return [{"year": y, "total": len(years[y]), "photos": _spread(years[y], per_year)}
            for y in sorted(years, key=lambda y: (y == 0, -y))]


# ── The timeline ──────────────────────────────────────────────────────

PER_MONTH = 6
YEAR_PICKS = 8
MONTH_SHOOTS = 6


def timeline(photos: list, per_month: int = PER_MONTH, per_year: int = YEAR_PICKS) -> list:
    """Every year with photographs, the most recent first: ``{year, total,
    photos, months}``; each month ``{month, total, days, photos, shoots}``, where
    a shoot is ``{folder, total, cover}`` (its best frame that month). Frames
    with no date are left out; rejects are counted but never shown."""
    by_month: dict = {}
    for photo in photos:
        if not photo.date or not re.match(r"^\d{4}-\d\d", photo.date):
            continue
        by_month.setdefault(photo.date[:7], []).append(photo)
    years: dict = {}
    for month in sorted(by_month, reverse=True):
        found = by_month[month]
        shown = [(_score(p), p) for p in found if p.flag != -1]
        folders: dict = {}
        for worth, photo in shown:
            entry = folders.setdefault(photo.record.folder, {"total": 0, "best": None})
            entry["total"] += 1
            if entry["best"] is None or worth > entry["best"][0]:
                entry["best"] = (worth, photo)
        shoots = sorted(folders.items(), key=lambda kv: (-kv[1]["total"], kv[0]))[:MONTH_SHOOTS]
        year = years.setdefault(int(month[:4]), {"year": int(month[:4]), "total": 0, "months": [], "_all": []})
        year["total"] += len(found)
        year["_all"].extend(shown)
        year["months"].append({
            "month": month,
            "total": len(found),
            "days": len({p.date for p in found}),
            "photos": _spread(shown, per_month, minutes=10),
            "shoots": [{"folder": f, "total": e["total"], "cover": e["best"][1]} for f, e in shoots],
        })
    out = []
    for y in sorted(years, reverse=True):
        entry = years[y]
        everything = entry.pop("_all")
        # The year's picks: one per day at most first, so a summer holiday does
        # not fill the year.
        days: dict = {}
        for worth, photo in everything:
            if photo.date not in days or worth > days[photo.date][0]:
                days[photo.date] = (worth, photo)
        entry["photos"] = _spread(list(days.values()), per_year, minutes=10)
        out.append(entry)
    return out


# ── Rediscover ────────────────────────────────────────────────────────

REDISCOVER = 12
MIN_AGE_DAYS = 365       # nothing from this past year: that is not rediscovering
QUIET_DAYS = 180         # nor anything opened up close in the last half year
DAY_MIN_PHOTOS = 6
DAY_PICKS = 24


def _good(photo) -> bool:
    return photo.flag == 1 or photo.rating >= 3 or bool(photo.edited)


def _age_days(photo, today: date) -> Optional[int]:
    try:
        return (today - date.fromisoformat(photo.date)).days
    except (TypeError, ValueError):
        return None


def _seen_lately(photo, viewed: dict, today: date, quiet_days: int) -> bool:
    for f in photo.filepaths:
        stamp = viewed.get(f)
        if not stamp:
            continue
        try:
            if (today - datetime.fromisoformat(stamp).date()).days < quiet_days:
                return True
        except ValueError:
            continue
    return False


def rediscover(photos: list, viewed: dict, today: date, seed: str, count: int = REDISCOVER,
               min_age: int = MIN_AGE_DAYS, quiet_days: int = QUIET_DAYS) -> dict:
    """``{photos, day}``: ``count`` good frames from more than ``min_age`` days ago
    that were not opened in the last ``quiet_days`` (``viewed`` is ``{filepath:
    iso time}``), one per day at most, older and better ones likelier; and one
    good day at random, ``{date, ago, folder, total, photos}`` or None. The same
    ``seed`` gives the same answer."""
    rng = random.Random(seed)
    old = []
    for photo in photos:
        if photo.flag == -1:
            continue
        age = _age_days(photo, today)
        if age is not None and age >= min_age:
            old.append((age, photo))

    fresh = [(age, p) for age, p in old if not _seen_lately(p, viewed, today, quiet_days)]
    good = [(age, p) for age, p in fresh if _good(p)]
    # A library hardly rated yet still has something to offer: everything old,
    # with what was rated or worked on still far likelier.
    pool = good if len(good) >= count * 2 else fresh
    keyed = []
    for age, photo in pool:
        weight = (1.0 + _score(photo)) * (1.0 + age / 3650.0)
        keyed.append((rng.random() ** (1.0 / weight), photo))
    keyed.sort(key=lambda t: (-t[0], t[1].record.filepath))
    chosen, days = [], set()
    for _, photo in keyed:
        if photo.date in days:
            continue
        days.add(photo.date)
        chosen.append(photo)
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda p: (p.date, p.record.filepath))

    # One good day: any day with enough photographs, the better days likelier.
    by_day: dict = {}
    for _, photo in old:
        by_day.setdefault(photo.date, []).append(photo)
    candidates = sorted(d for d, ps in by_day.items() if len(ps) >= DAY_MIN_PHOTOS)
    day = None
    if candidates:
        weights = [1.0 + sum(_score(p) for p in by_day[d]) ** 0.5 for d in candidates]
        picked = rng.choices(candidates, weights=weights, k=1)[0]
        found = by_day[picked]
        folders: dict = {}
        for photo in found:
            folders[photo.record.folder] = folders.get(photo.record.folder, 0) + 1
        day = {
            "date": picked,
            "ago": today.year - int(picked[:4]),
            "folder": max(folders, key=folders.get),
            "total": len(found),
            "photos": _spread([(_score(p), p) for p in found], DAY_PICKS),
        }
    return {"photos": chosen, "day": day}
