"""
A year in review: one page that says what a year of photographs was — how many,
when, who was in them, where they were taken, and the frames worth showing
again, with a collage of the best of them to keep.

The frames are picked the way "suggest best" picks a folder's, because most of a
library is never rated: what was picked, starred or developed first, then the
people named in it at their clearest, then sharp for its own shoot; a soft frame
or blown highlights count against it, and a reject is never shown. A year is
many days out with a camera, and the busiest would otherwise fill the page, so
each day gets a share of the frames that grows with the square root of how much
was taken: a day of two thousand frames gets more than an afternoon of fifty,
but not forty times more. Days rather than folders, because one day is
sometimes filed in two (the camera's JPEGs in one, its raw files in another). Frames are kept apart in time, or by look where the time is unknown, so a
burst is one frame. Where smart
search has described both pictures, two that show the same scene are one frame
too, however far apart they were taken: the same people posing again by the
same cars twenty minutes later is not a second moment.
"""

import bisect
import math
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Optional

from analysis.hints import compute_hints
from catalog.memories import _moment
from faces.timeline import face_score

# Frames on the page, and how many of them make the collage.
TOP = 24
COLLAGE = 9
# Within one shoot, frames at least this far apart in time; where the time is
# unknown, at least this many bits apart in their 64-bit difference hash.
SPREAD_MINUTES = 10
SPREAD_BITS = 12
# Two pictures whose smart-search descriptions are this alike show the same
# scene. Measured on a year's picks: the same people re-posed at one event
# 0.83-0.87, different moments of the year a median of 0.55 and at most 0.81.
LOOK_SAME = 0.82
# A day this small gets a frame only when the bigger ones leave room.
SMALL_SHOOT = 5
PEOPLE = 12
MOMENTS = 8


def years(photos: list) -> list[dict]:
    """``[{year, photos}]``, the most recent first, for every year with dated
    photographs (rejects not counted)."""
    count = Counter(p.date[:4] for p in photos if p.date and p.flag != -1)
    return [{"year": int(y), "photos": n} for y, n in sorted(count.items(), reverse=True)]


def _score(photo, sharp_rank: float, hint: dict, people: float) -> float:
    return (40 * (photo.flag == 1) + 10 * photo.rating + 6 * photo.edited
            + 10 * min(people, 2.0) + 10 * sharp_rank
            - 15 * bool(hint.get("soft_frame")) - 10 * bool(hint.get("blown_highlights")))


def _hash_bits(value: str) -> Optional[int]:
    try:
        return int(value, 16) if value else None
    except ValueError:
        return None


def _apart(photo, kept: list, stats: dict, looks: Optional[dict] = None) -> bool:
    """Whether ``photo`` is far enough from every frame already kept."""
    when = _moment(photo)
    bits = _hash_bits((stats.get(photo.record.filepath) or {}).get("dhash") or "")
    look = (looks or {}).get(photo.record.filepath)
    for other in kept:
        other_look = (looks or {}).get(other.record.filepath)
        if (look is not None and other_look is not None and look.shape == other_look.shape
                and float(look @ other_look) >= LOOK_SAME):
            return False
        other_when = _moment(other)
        if when is not None and other_when is not None:
            if abs(when - other_when) < timedelta(minutes=SPREAD_MINUTES):
                return False
            continue
        other_bits = _hash_bits((stats.get(other.record.filepath) or {}).get("dhash") or "")
        if bits is not None and other_bits is not None and bin(bits ^ other_bits).count("1") < SPREAD_BITS:
            return False
    return True


def _shares(sizes: dict, total: int) -> dict:
    """Frames per day: by the square root of its size, at least one for each
    day that is not small while there is room, and never more than it has."""
    if not sizes:
        return {}
    weight = {f: math.sqrt(n) for f, n in sizes.items()}
    whole = sum(weight.values())
    share = {f: min(sizes[f], int(total * w / whole)) for f, w in weight.items()}
    # The rest go to the days that lost most to rounding, biggest first; a
    # small day only when every bigger one has its frame.
    order = sorted(sizes, key=lambda f: (sizes[f] < SMALL_SHOOT, -(total * weight[f] / whole - share[f]), -sizes[f], f))
    left = total - sum(share.values())
    while left > 0:
        moved = False
        for f in order:
            if left <= 0:
                break
            if share[f] < sizes[f]:
                share[f] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return share


def top_frames(photos: list, stats: dict, people_score: dict, total: int = TOP,
               looks: Optional[dict] = None) -> list:
    """The year's best ``total`` frames, ``[(score, photo)]`` in the order taken.
    ``looks`` maps a photo's shown filepath to its unit smart-search vector."""
    # Sharp and soft are judged against the frame's own shoot, as the grid's
    # hints are; the frames are shared out by day.
    by_folder: dict[str, list] = defaultdict(list)
    for photo in photos:
        if photo.flag != -1:
            by_folder[photo.record.folder].append(photo)

    by_day: dict[str, list] = defaultdict(list)
    for folder, items in by_folder.items():
        rows = {p.record.filepath: stats[p.record.filepath] for p in items if p.record.filepath in stats}
        hints = compute_hints(rows)["hints"]
        sharp = sorted(float(r.get("sharpness") or 0.0) for r in rows.values())

        def rank(value: float) -> float:
            if len(sharp) < 2:
                return 0.5
            return bisect.bisect_left(sharp, value) / (len(sharp) - 1)

        for p in items:
            fp = p.record.filepath
            row = stats.get(fp) or {}
            people = max((people_score.get(f, 0.0) for f in p.filepaths), default=0.0)
            by_day[p.date].append((_score(p, rank(float(row.get("sharpness") or 0.0)) if row else 0.3,
                                          hints.get(fp) or {}, people), p))
    for scored in by_day.values():
        scored.sort(key=lambda t: (-t[0], t[1].record.filepath))

    share = _shares({d: len(v) for d, v in by_day.items()}, total)
    chosen, kept = [], []
    # The busiest days choose first, so a duplicate falls out of the quieter.
    for day in sorted(by_day, key=lambda d: (-len(by_day[d]), d)):
        mine = 0
        for score, photo in by_day[day]:
            if mine >= share.get(day, 0):
                break
            if _apart(photo, kept, stats, looks):
                kept.append(photo)
                chosen.append((score, photo))
                mine += 1
    chosen.sort(key=lambda t: (t[1].date or "", t[1].record.capture_date or "", t[1].record.filepath))
    return chosen


def collage_picks(chosen: list, count: int = COLLAGE) -> list:
    """The collage's frames out of the year's best: the highest scoring, one per
    day first so the collage is the year rather than its busiest weekend, in
    the order taken."""
    best = sorted(chosen, key=lambda t: (-t[0], t[1].record.filepath))
    picked, days = [], set()
    for score, photo in best:
        if photo.date not in days:
            picked.append((score, photo))
            days.add(photo.date)
        if len(picked) >= count:
            break
    for item in best:
        if len(picked) >= count:
            break
        if item not in picked:
            picked.append(item)
    picked.sort(key=lambda t: (t[1].date or "", t[1].record.capture_date or "", t[1].record.filepath))
    return [p for _, p in picked]


def people_of(faces: list[dict], in_year: dict) -> list[dict]:
    """The named people of the year, the most photographed first, each with
    their best face of it. ``faces`` are named people's faces (with cluster_id,
    name, quality, px, x/y/w/h); ``in_year`` maps a filepath to its photo."""
    seen: dict[int, set] = defaultdict(set)
    best: dict[int, tuple] = {}
    names: dict[int, str] = {}
    for face in faces:
        photo = in_year.get(face["filepath"])
        if photo is None or photo.flag == -1:
            continue
        cid = int(face["cluster_id"])
        names[cid] = face["name"]
        seen[cid].add((photo.record.folder, photo.record.base_name))
        score = face_score({**face, "rating": photo.rating, "flag": photo.flag})
        if cid not in best or score > best[cid][0]:
            best[cid] = (score, face)
    out = []
    for cid in sorted(seen, key=lambda c: (-len(seen[c]), names[c].casefold())):
        face = best[cid][1]
        out.append({"id": cid, "name": names[cid], "photos": len(seen[cid]),
                    "filepath": face["filepath"], "face_index": int(face["face_index"])})
    return out[:PEOPLE]


def _card_year(card: dict) -> str:
    return (card.get("date_from") or "")[:4]


def moments_and_places(cards: list[dict], year: int, counts: dict) -> tuple[list, list]:
    """The year's biggest shoots, and the places away from home it was taken in.
    ``counts`` is the year's photographs per folder."""
    mine = [c for c in cards if c["folder"] in counts]
    moments = []
    for card in sorted(mine, key=lambda c: (-counts[c["folder"]], c["folder"]))[:MOMENTS]:
        note = card.get("note") or {}
        place = card.get("place") or {}
        moments.append({
            "folder": card["folder"],
            "title": note.get("title") or "",
            "date_from": card.get("date_from"), "date_to": card.get("date_to"),
            "photos": counts[card["folder"]],
            "cover": card.get("cover") or "",
            "place": place.get("label") or "" if not card.get("at_home") else "",
            "people": [p["name"] for p in card.get("people") or []][:4],
        })
    places: dict[str, dict] = {}
    for card in mine:
        place = card.get("place") or {}
        label = place.get("label") or ""
        if not label or card.get("at_home"):
            continue
        entry = places.setdefault(label, {"label": label, "photos": 0, "folders": [], "cover": "",
                                          "lat": place.get("lat"), "lon": place.get("lon"), "_best": -1})
        entry["photos"] += counts[card["folder"]]
        entry["folders"].append(card["folder"])
        if counts[card["folder"]] > entry["_best"]:
            entry["_best"] = counts[card["folder"]]
            entry["cover"] = card.get("cover") or ""
    out = sorted(places.values(), key=lambda p: (-p["photos"], p["label"]))
    for p in out:
        p.pop("_best")
    return moments, out


def summary(photos: list) -> dict:
    """The year in numbers."""
    days = Counter(p.date for p in photos)
    day_folders: dict[str, Counter] = defaultdict(Counter)
    for p in photos:
        day_folders[p.date][p.record.folder] += 1
    months = [0] * 12
    for p in photos:
        months[int(p.date[5:7]) - 1] += 1
    cameras = Counter(p.record.camera.strip() for p in photos if (p.record.camera or "").strip())
    busiest = max(days, key=lambda d: (days[d], d)) if days else ""
    return {
        "photos": len(photos),
        "days": len(days),
        "shoots": len({p.record.folder for p in photos}),
        "months": months,
        "busiest_day": {"date": busiest, "photos": days[busiest],
                        "folder": day_folders[busiest].most_common(1)[0][0]} if busiest else None,
        "cameras": [{"name": n, "photos": c} for n, c in cameras.most_common(3)],
        "picked": sum(p.flag == 1 for p in photos),
        "rated": sum(p.rating > 0 for p in photos),
        "edited": sum(bool(p.edited) for p in photos),
    }


def review(photos: list, year: int, stats: dict, faces: list[dict], cards: list[dict],
           top: int = TOP, looks: Optional[dict] = None) -> dict:
    """Everything the page shows for ``year``: ``{year, summary, top, collage,
    people, moments, places}`` (frames as Photo objects). ``looks`` maps any of
    a photo's filepaths to its smart-search vector, where it has one."""
    mine = [p for p in photos if p.date[:4] == str(year) and p.flag != -1]
    in_year = {fp: p for p in mine for fp in p.filepaths}
    people_score: dict[str, float] = defaultdict(float)
    for face in faces:
        if face["filepath"] in in_year:
            people_score[face["filepath"]] += face_score(face)
    shown_looks = {}
    for p in mine:
        vector = next(((looks or {}).get(f) for f in p.filepaths if (looks or {}).get(f) is not None), None)
        if vector is not None:
            shown_looks[p.record.filepath] = vector
    chosen = top_frames(mine, stats, people_score, top, shown_looks)
    counts = Counter(p.record.folder for p in mine)
    moments, places = moments_and_places(cards, year, counts)
    return {
        "year": year,
        "summary": summary(mine),
        "top": [p for _, p in chosen],
        "collage": collage_picks(chosen),
        "people": people_of(faces, in_year),
        "moments": moments,
        "places": places,
    }
