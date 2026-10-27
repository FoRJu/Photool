"""
A person's timeline: one picture of them for every month (or year) they were
photographed, oldest first — a child growing up, played as a slideshow.

Each period shows its best face of the person, not its first: a large, clear
face, on a frame the person picked or rated, never a reject. A photograph's
date is the one search gives it (its EXIF date, or the date in its folder's
name); a frame with neither has no place on a timeline and is only counted.
"""

from typing import Optional

from catalog.search import photo_date

PERIODS = ("month", "year")

# A face this wide (in the frame it was found in) is as good as any for size: a
# bigger one is not a better picture, only a closer one.
FULL_PX = 240


def face_score(face: dict) -> float:
    """How good a picture of the person one face makes: its clearness and its
    size first, then what the photographer thought of the frame."""
    quality = float(face["quality"] if face.get("quality") is not None else 1.0)
    size = min(float(face.get("px") or 0.0), FULL_PX) / FULL_PX
    return (quality * size
            + 0.3 * (face.get("flag") == 1)
            + 0.08 * int(face.get("rating") or 0))


def timeline(faces: list[dict], period: str = "month") -> dict:
    """``{periods: [...], undated}`` from one person's faces (rows of
    ``photo_faces`` joined to their photo's capture_date, folder, rating, flag).

    Each period is ``{period, date, filepath, face_index, x, y, w, h, photos}``:
    its best face and how many photos of the person it holds.
    """
    width = 7 if period == "month" else 4
    best: dict[str, tuple] = {}
    photos: dict[str, set] = {}
    undated: set = set()
    for face in faces:
        if face.get("flag") == -1:
            continue
        day = photo_date(face.get("capture_date") or "", face.get("folder") or "")
        if not day:
            undated.add(face["filepath"])
            continue
        key = day[:width]
        photos.setdefault(key, set()).add(face["filepath"])
        score = face_score(face)
        held: Optional[tuple] = best.get(key)
        # Ties go to the earlier frame, so the pick does not move between loads.
        if held is None or score > held[0] or (score == held[0] and (day, face["filepath"]) < held[1:3]):
            best[key] = (score, day, face["filepath"], face)
    periods = []
    for key in sorted(best):
        _score, day, filepath, face = best[key]
        periods.append({
            "period": key,
            "date": day,
            "filepath": filepath,
            "face_index": int(face["face_index"]),
            "x": float(face["x"]), "y": float(face["y"]),
            "w": float(face["w"]), "h": float(face["h"]),
            "photos": len(photos[key]),
        })
    return {"periods": periods, "undated": len(undated)}
