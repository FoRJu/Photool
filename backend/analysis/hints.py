"""Quality hints — soft frame, blown highlights.

A hint is an opinion offered while the user is deciding, never a decision. Nothing
here rejects, ranks, or hides a photo; it marks a frame so the eye goes to it
before the thumb is dismissed.

The two hints are deliberately different in kind:

* **Soft frame** is *relative*. Sharpness is measured from a 400px thumbnail, so
  the number only means something next to other 400px thumbnails of the same
  shoot — the same rule that makes dHash grouping folder-scoped. A handheld shot
  at f/1.8 and a tripod landscape are not comparable; three frames from the same
  burst are. So the folder's median sharpness is the reference, and a frame well
  below it is flagged.
* **Blown highlights** is *absolute*. A fraction of pixels pinned at white is a
  fact about the file, not about its neighbours, so it needs no baseline.

Hints are computed when they are asked for and never stored. The thresholds will
be tuned as the library is looked at; storing the interpretation would freeze it
and make a change require re-measuring every photo. The measurements themselves
are already stored, and they are what is expensive to produce.
"""

from __future__ import annotations

from typing import Iterable

# A frame is soft when its sharpness falls below this fraction of its peers'
# median (see PEER_SECONDS below). Set 2026-10-21 on 30 frames judged by eye at
# 1:1 from four culled folders (11 soft, 19 sharp): at 0.32 it caught 9 of the
# soft ones and none of the sharp ones; at 0.40 it also marked 7 of the sharp
# ones; against the folder's median instead, as it once was, every sharp one.
SOFT_RATIO = 0.32

# A frame has blown highlights when this fraction of its pixels are pinned at
# white. Deliberately conservative — the flag should be rare enough that seeing
# it means something. Same five folders: clip_high runs p50 = 0.004, p90 = 0.046,
# p99 = 0.16, so 5% of the frame at white catches ~9% of frames and means a real
# patch of the picture is gone, not one specular pixel. At 2% it flags a fifth of
# the library, most of it harmless.
BLOWN_CLIP = 0.05

# Below this many measured frames in a folder there is no usable median, so the
# relative hint is withheld. The absolute one still applies.
MIN_FOLDER_SAMPLE = 12

# A frame is judged against its *peers*, not the whole folder: the frames of the
# same camera taken within PEER_SECONDS of it at a focal length within PEER_FOCAL
# of its own. A distant subject at 105 mm holds less fine detail than a street at
# 30 mm, so against the folder every telephoto frame looked soft: on four culled
# folders (2026-10-21) the folder-relative flag marked 548 frames, 22% of which
# the person kept, and of a sample nine in twelve were sharp. A frame with fewer
# than MIN_PEERS peers gets no relative verdict.
PEER_SECONDS = 30
PEER_FOCAL = 1.25
MIN_PEERS = 2
# ...except a frame so far below the whole folder that no subject explains it:
# camera shake, a frame taken by accident. Set from the same folders.
ACCIDENT_RATIO = 0.10

HINTS = ("soft_frame", "blown_highlights")


def median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    n = len(values)
    if not n:
        return None
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _measured(row: dict) -> bool:
    """A row is usable when it carries the two numbers the hints read."""
    return bool(row) and row.get("sharpness") is not None and row.get("clip_high") is not None


def folder_baseline(rows: Iterable[dict]) -> dict:
    """The reference a folder's frames are judged against.

    Returns ``{"measured": int, "median_sharpness": float | None}``. The median
    is None when the sample is too small to describe the folder; callers should
    still apply absolute hints in that case.
    """
    sharp = [float(r["sharpness"]) for r in rows if _measured(r) and float(r["sharpness"]) > 0]
    med = median(sharp)
    if len(sharp) < MIN_FOLDER_SAMPLE:
        med = None
    return {"measured": len(sharp), "median_sharpness": med}


def photo_hints(row: dict, median_sharpness: float | None,
                peer_sharpness: float | None = None) -> dict:
    """Hints for one frame. Keys are always present; a value is True or False.

    ``peer_sharpness`` is the median of the frame's peers (see :func:`peers`), or
    None when it has too few to be judged against."""
    hints = dict.fromkeys(HINTS, False)
    if not _measured(row):
        return hints

    sharpness = float(row["sharpness"])
    if peer_sharpness and peer_sharpness > 0 and sharpness / peer_sharpness < SOFT_RATIO:
        hints["soft_frame"] = True
    if median_sharpness and median_sharpness > 0 and sharpness / median_sharpness < ACCIDENT_RATIO:
        hints["soft_frame"] = True
    if float(row["clip_high"]) > BLOWN_CLIP:
        hints["blown_highlights"] = True
    return hints


def _focal(text) -> float | None:
    import re

    m = re.match(r"\s*([\d.]+)", str(text or ""))
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def peers(records) -> dict[str, list[str]]:
    """Each frame's peers among ``records`` (catalog records): the same camera,
    within PEER_SECONDS, at a focal length within PEER_FOCAL. A frame whose time
    cannot be read has none."""
    from datetime import datetime

    frames = []
    for r in records:
        try:
            when = datetime.strptime((r.capture_date or "")[:19], "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
        frames.append((when, r.camera or "", _focal(r.focal_length), r.filepath))
    # By time, camera and file only: two frames of one second can differ in
    # whether their focal length is known, and None does not sort against a number.
    frames.sort(key=lambda f: (f[0], f[1], f[3]))
    out: dict[str, list[str]] = {f[3]: [] for f in frames}
    start = 0
    for i, (when, camera, focal, path) in enumerate(frames):
        while (when - frames[start][0]).total_seconds() > PEER_SECONDS:
            start += 1
        for j in range(start, i):
            other = frames[j]
            if other[1] != camera:
                continue
            if (focal is None) != (other[2] is None):
                continue
            if focal and other[2] and max(focal, other[2]) / min(focal, other[2]) > PEER_FOCAL:
                continue
            out[path].append(other[3])
            out[other[3]].append(path)
    return out


def peer_sharpness(rows: dict[str, dict], peer_map: dict[str, list[str]]) -> dict[str, float]:
    """The median sharpness of each frame's measured peers and the frame itself,
    for the frames with at least MIN_PEERS measured peers."""
    out = {}
    for path, others in peer_map.items():
        if not _measured(rows.get(path)):
            continue
        values = [float(rows[o]["sharpness"]) for o in others if _measured(rows.get(o))]
        if len(values) >= MIN_PEERS:
            out[path] = median(values + [float(rows[path]["sharpness"])])
    return out


def compute_hints(rows: dict[str, dict], baseline: dict | None = None,
                  peer_medians: dict[str, float] | None = None) -> dict:
    """Classify a folder's frames.

    ``rows`` maps filepath to a stats row, as returned by
    ``CatalogDB.list_photo_stats_for_folder``. Returns the reference used plus
    hints for every frame that has usable measurements — frames that were never
    measured are absent, which is how the caller tells "no problems found" apart
    from "not looked at yet".

    ``baseline`` is the folder these frames belong to. Pass it when ``rows`` is a
    filtered subset: what counts as sharp here is a property of the folder, not
    of whatever filter happens to be on the screen. Computed from ``rows`` when
    omitted.
    """
    if baseline is None:
        baseline = folder_baseline(rows.values())
    med = baseline["median_sharpness"]
    peer_medians = peer_medians or {}
    hints = {
        path: photo_hints(row, med, peer_medians.get(path))
        for path, row in rows.items()
        if _measured(row)
    }
    return {
        "measured": len(hints),
        "median_sharpness": med,
        "hints": hints,
    }
