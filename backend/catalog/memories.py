"""
On this day: the photographs taken on today's date in earlier years.

A day is found by the date search already gives a photograph (its EXIF date, or
the date in its folder's name), so a frame with no EXIF still has its day. When
a year has nothing on the day itself, the nearest day within a few either side
stands in, so a birthday shot the evening before is not missed.

A year shows its best few frames, not its first: what was picked, rated or
developed first, never a reject, and spread over the day — a burst of thirty
frames of one moment is one memory, so frames are kept a few minutes apart.
"""

import re
from datetime import date, datetime, timedelta
from typing import Optional

# Days either side of the date a year may stand in with, when it has nothing on it.
WINDOW = 3
# Frames shown for one year at most, and how far apart in time they are kept.
PER_YEAR = 8
SPREAD_MINUTES = 3


def _score(photo) -> float:
    """What makes a frame worth showing again: chosen, then rated, then worked on."""
    return (15 * (photo.flag == 1) + 10 * photo.rating + 6 * photo.edited
            + 2 * bool(photo.tags))


def _moment(photo) -> Optional[datetime]:
    m = re.match(r"^(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d)", photo.record.capture_date or "")
    if not m:
        return None
    try:
        return datetime(*map(int, m.groups()))
    except ValueError:
        return None


def _shift(day: date, year: int) -> Optional[date]:
    try:
        return day.replace(year=year)
    except ValueError:                  # 29 February in a year without one
        return day.replace(year=year, day=28)


def _pick(photos: list, limit: int) -> list:
    """The best ``limit`` frames, at least SPREAD_MINUTES apart where the time is
    known, in the order they were taken."""
    chosen, times = [], []
    gap = timedelta(minutes=SPREAD_MINUTES)
    for photo in sorted(photos, key=lambda p: (-_score(p), p.record.filepath)):
        when = _moment(photo)
        if when is not None and any(abs(when - t) < gap for t in times):
            continue
        chosen.append(photo)
        if when is not None:
            times.append(when)
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda p: (p.record.capture_date or "", p.record.filepath))


def on_this_day(photos: list, today: date, per_year: int = PER_YEAR, window: int = WINDOW) -> list:
    """``[{year, ago, date, total, folder, photos}]``, the most recent year first,
    for each earlier year with photographs on ``today``'s date (or near it)."""
    by_day: dict = {}
    for photo in photos:
        if photo.flag == -1 or not photo.date:
            continue
        by_day.setdefault(photo.date, []).append(photo)
    years = sorted({int(d[:4]) for d in by_day if d[:4].isdigit() and int(d[:4]) < today.year},
                   reverse=True)
    out = []
    for year in years:
        target = _shift(today, year)
        found, day = None, None
        for offset in sorted(range(-window, window + 1), key=abs):
            day = target + timedelta(days=offset)
            found = by_day.get(day.isoformat())
            if found:
                break
        if not found:
            continue
        folders: dict = {}
        for photo in found:
            folders[photo.record.folder] = folders.get(photo.record.folder, 0) + 1
        out.append({
            "year": year,
            "ago": today.year - year,
            "date": day.isoformat(),
            "exact": day == target,
            "total": len(found),
            "folder": max(folders, key=folders.get),
            "photos": _pick(found, per_year),
        })
    return out
