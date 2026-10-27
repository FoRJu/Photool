"""
On this day (catalog/memories.py): earlier years' best frames from today's date.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import PhotoRecord  # noqa: E402
from catalog.memories import on_this_day  # noqa: E402
from catalog.search import Photo  # noqa: E402

TODAY = date(2026, 10, 18)


def _photo(day, name, time="12:00:00", **kw):
    folder = day[:4] + "/" + day
    stamp = day.replace("-", ":") + " " + time if time else ""
    record = PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder,
                         extension=".jpg", base_name=Path(name).stem, capture_date=stamp)
    return Photo(record=record, date=day, **kw)


def test_each_earlier_year_on_the_day_most_recent_first():
    photos = [_photo("2024-10-18", "a.jpg"), _photo("2021-10-18", "b.jpg"),
              _photo("2024-10-25", "far.jpg"), _photo("2026-10-18", "today.jpg")]
    years = on_this_day(photos, TODAY)
    assert [(y["year"], y["ago"], y["exact"]) for y in years] == [(2024, 2, True), (2021, 5, True)]
    assert [p.record.filename for p in years[0]["photos"]] == ["a.jpg"]


def test_a_year_with_nothing_on_the_day_takes_the_nearest_day_and_says_so():
    years = on_this_day([_photo("2024-10-20", "a.jpg"), _photo("2024-10-15", "b.jpg")], TODAY)
    assert [(y["date"], y["exact"]) for y in years] == [("2024-10-20", False)]


def test_rejects_are_never_shown_and_picks_come_first():
    photos = [_photo("2024-10-18", "reject.jpg", "10:00:00", flag=-1),
              _photo("2024-10-18", "plain.jpg", "11:00:00"),
              _photo("2024-10-18", "pick.jpg", "12:00:00", flag=1),
              _photo("2024-10-18", "star.jpg", "13:00:00", rating=3)]
    year = on_this_day(photos, TODAY, per_year=2)[0]
    assert year["total"] == 3
    # The best two, shown in the order they were taken.
    assert [p.record.filename for p in year["photos"]] == ["pick.jpg", "star.jpg"]


def test_a_burst_is_one_memory():
    burst = [_photo("2024-10-18", f"b{i}.jpg", f"12:00:0{i}", rating=5 - i) for i in range(5)]
    later = _photo("2024-10-18", "later.jpg", "15:00:00")
    year = on_this_day(burst + [later], TODAY)[0]
    assert [p.record.filename for p in year["photos"]] == ["b0.jpg", "later.jpg"]


def test_the_29th_of_february_finds_the_28th_in_other_years():
    years = on_this_day([_photo("2025-02-28", "a.jpg")], date(2028, 2, 29))
    assert [(y["date"], y["exact"]) for y in years] == [("2025-02-28", True)]
