"""
Year in review: the year in numbers, its best frames shared out by day and kept
apart, the collage's picks and layout, its people, shoots and places away.
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import PhotoRecord  # noqa: E402
from catalog.review import (LOOK_SAME, _shares, collage_picks, moments_and_places, people_of,  # noqa: E402
                            review, summary, top_frames, years)
from catalog.search import gather  # noqa: E402
from export.collage import render, rows  # noqa: E402


def _rec(folder, name, when="", rating=0, flag=0, camera="Cam", ext=".JPG"):
    base = name.rsplit(".", 1)[0]
    return PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder, extension=ext,
                       base_name=base, capture_date=when, rating=rating, flag=flag, camera=camera)


def _photos(records, edited=()):
    return gather(records, {}, set(edited), set())


def _day(folder, day, n, start_minute=0, step=15, **kw):
    """``n`` frames on ``day`` (YYYY:MM:DD), ``step`` minutes apart."""
    out = []
    for i in range(n):
        m = start_minute + i * step
        out.append(_rec(folder, f"{folder[-5:]}_{start_minute:03d}_{i:03d}.JPG",
                        f"{day} {8 + m // 60:02d}:{m % 60:02d}:00", **kw))
    return out


def test_years_most_recent_first_without_rejects():
    photos = _photos([_rec("2024/2024-01-01", "a.JPG", "2024:01:01 10:00:00"),
                      _rec("2025/2025-01-01", "b.JPG", "2025:01:01 10:00:00"),
                      _rec("2025/2025-01-01", "c.JPG", "2025:01:01 11:00:00", flag=-1),
                      _rec("misc", "d.JPG")])
    assert years(photos) == [{"year": 2025, "photos": 1}, {"year": 2024, "photos": 1}]


def test_shares_grow_with_the_square_root_and_fill_the_total():
    share = _shares({"big": 400, "mid": 100, "small": 3}, 12)
    assert sum(share.values()) == 12
    assert share["big"] > share["mid"] > 0
    assert share["big"] < 12 * 400 / 503          # less than by plain proportion
    assert _shares({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}   # never more than a day has


def test_rated_and_picked_frames_come_first():
    recs = _day("2025/2025-03-01", "2025:03:01", 30)
    recs[7].rating = 5
    recs[19].flag = 1
    photos = _photos(recs)
    chosen = top_frames(photos, {}, {}, total=2)
    assert {p.record.filename for _, p in chosen} == {recs[7].filename, recs[19].filename}


def test_a_burst_is_one_frame():
    recs = _day("2025/2025-03-01", "2025:03:01", 20, step=0)   # all in the same minute
    recs += _day("2025/2025-03-01", "2025:03:01", 1, start_minute=120)
    chosen = top_frames(_photos(recs), {}, {}, total=5)
    assert len(chosen) == 2


def test_frames_without_a_time_are_kept_apart_by_look():
    recs = [_rec("2025/2025-04-02", f"n{i}.JPG") for i in range(3)]
    stats = {r.filepath: {"sharpness": 100.0, "clip_high": 0.0, "dhash": h}
             for r, h in zip(recs, ["ffff0000ffff0000", "ffff0000ffff0001", "0000ffff0000ffff"])}
    chosen = top_frames(_photos(recs), stats, {}, total=3)
    assert len(chosen) == 2


def test_one_scene_described_alike_is_one_frame():
    recs = _day("2025/2025-05-05", "2025:05:05", 3, step=30)
    a = np.zeros(8, np.float32); a[0] = 1
    b = np.zeros(8, np.float32); b[0] = LOOK_SAME + 0.05; b[1] = np.sqrt(1 - b[0] ** 2)
    c = np.zeros(8, np.float32); c[2] = 1
    looks = {recs[0].filepath: a, recs[1].filepath: b, recs[2].filepath: c}
    chosen = top_frames(_photos(recs), {}, {}, total=3, looks=looks)
    assert len(chosen) == 2


def test_one_day_in_two_folders_is_one_day():
    """The camera's JPEGs filed under one folder and the same day's raws under
    another share one day's frames, and the collage takes one of them."""
    recs = _day("2025/2025-10-12", "2025:10:13", 40) + _day("2025/2025-10-13", "2025:10:13", 40, start_minute=5)
    recs += _day("2025/2025-06-01", "2025:06:01", 40)
    chosen = top_frames(_photos(recs), {}, {}, total=6)
    days = [p.date for _, p in chosen]
    # 80 frames that day against 40: by the square root, 4 and 2 (as two
    # folders of 40 it would have been 2, 2 and 2).
    assert days.count("2025-10-13") == 4 and days.count("2025-06-01") == 2
    picks = collage_picks(chosen, count=2)
    assert sorted(p.date for p in picks) == ["2025-06-01", "2025-10-13"]


def test_the_year_in_numbers():
    recs = _day("2025/2025-02-01", "2025:02:01", 3) + _day("2025/2025-07-04", "2025:07:04", 5, camera="Other")
    recs[0].rating = 2
    s = summary(_photos(recs))
    assert s["photos"] == 8 and s["days"] == 2 and s["shoots"] == 2
    assert s["months"][1] == 3 and s["months"][6] == 5
    assert s["busiest_day"] == {"date": "2025-07-04", "photos": 5, "folder": "2025/2025-07-04"}
    assert s["cameras"][0] == {"name": "Other", "photos": 5}
    assert s["rated"] == 1


def test_people_most_photographed_first_with_their_best_face():
    recs = _day("2025/2025-02-01", "2025:02:01", 3)
    photos = _photos(recs)
    in_year = {p.record.filepath: p for p in photos}
    face = lambda i, cid, name, px: {"filepath": recs[i].filepath, "face_index": 0, "cluster_id": cid, "name": name,
                                     "px": px, "quality": 1.0, "x": 0, "y": 0, "w": .1, "h": .1}
    faces = [face(0, 1, "Ann", 50), face(1, 1, "Ann", 220), face(2, 2, "Bo", 100),
             {**face(0, 3, "Old", 100), "filepath": "2019/x.JPG"}]
    out = people_of(faces, in_year)
    assert [(p["name"], p["photos"]) for p in out] == [("Ann", 2), ("Bo", 1)]
    assert out[0]["filepath"] == recs[1].filepath


def test_shoots_and_places_away_from_home():
    cards = [
        {"folder": "2025/a", "date_from": "2025-03-01", "cover": "2025/a/1.JPG", "at_home": False,
         "place": {"label": "Rome, Italy", "lat": 41.9, "lon": 12.5}, "note": {"title": "Rome"}, "people": []},
        {"folder": "2025/b", "date_from": "2025-03-02", "cover": "2025/b/1.JPG", "at_home": False,
         "place": {"label": "Rome, Italy", "lat": 41.9, "lon": 12.5}, "note": None, "people": [{"name": "Ann", "photos": 3}]},
        {"folder": "2025/c", "date_from": "2025-05-01", "cover": "", "at_home": True,
         "place": {"label": "Home Town"}, "note": None, "people": []},
        {"folder": "2024/z", "date_from": "2024-05-01", "cover": "", "at_home": False,
         "place": {"label": "Elsewhere"}, "note": None, "people": []},
    ]
    moments, places = moments_and_places(cards, 2025, {"2025/a": 10, "2025/b": 30, "2025/c": 5})
    assert [m["folder"] for m in moments] == ["2025/b", "2025/a", "2025/c"]
    assert moments[1]["title"] == "Rome" and moments[2]["place"] == ""
    assert [(p["label"], p["photos"], p["cover"]) for p in places] == [("Rome, Italy", 40, "2025/b/1.JPG")]


def test_review_puts_it_together_and_leaves_other_years_and_rejects_out():
    recs = _day("2025/2025-02-01", "2025:02:01", 10) + _day("2024/2024-02-01", "2024:02:01", 10)
    recs[0].flag = -1
    out = review(_photos(recs), 2025, {}, [], [])
    assert out["summary"]["photos"] == 9
    assert all(p.date.startswith("2025") and p.flag != -1 for p in out["top"])
    assert 0 < len(out["collage"]) <= len(out["top"])


def test_collage_rows_fill_the_width_and_keep_every_frame():
    aspects = [1.5, 0.67, 1.5, 1.5, 1.0, 0.75, 1.5, 1.33, 1.5]
    layout = rows(aspects, 1000, 10)
    assert sorted(k for r in layout for k in r) == list(range(9))
    for r in layout:
        h = (1000 - 10 * (len(r) - 1)) / sum(aspects[k] for k in r)
        assert 80 < h < 700


def test_collage_renders_a_titled_jpeg():
    images = [Image.new("RGB", (300 if i % 2 else 200, 200), (40 * i, 100, 150)) for i in range(5)]
    jpeg = render(images, "2025", "5 photos", width=1200)
    out = Image.open(io.BytesIO(jpeg))
    assert out.format == "JPEG" and out.width == 1200 and out.height > 300
    # The title band is the background colour at the left margin's edge.
    assert out.getpixel((10, 10))[0] > 240
