"""
Matching a set — the small, bounded corrections that make a batch of photos
look like they were shot by one camera under one light.

The measurements stored by :mod:`analysis.service` describe each frame as the
camera rendered it. At export the pixels about to be written are measured the
same way, and the difference against the set's median is what gets corrected: a
brightness shift in stops and a white-balance gain per channel. Both are clamped
to a nudge, so a frame that was deliberately exposed differently stays itself.

The target is a median rather than a mean on purpose: one blown frame at the
back of a school hall should not move the whole set.
"""

import math
from statistics import median
from typing import Iterable, Optional

from PIL import Image

# How far a correction may reach. Beyond these the change stops being "make the
# set consistent" and becomes "re-edit this photo", which is not what the
# toggle promises.
#
# Both limits are in the domain the measurements are in: an EV shift is a log2
# ratio of measured brightness, so 0.5 means at most 1.41x as bright, and the
# white-balance limit is a 12% gain on a channel. Measuring and correcting in
# the same domain is what makes the loop close — a correction expressed in
# scene light would only move the measured numbers by less than half as much.
MAX_EV_SHIFT = 0.5
MAX_WB_DELTA = 0.12

# Below these the correction is invisible, and applying it only costs a pass
# through the pixels.
MIN_EV_SHIFT = 0.05
MIN_WB_DELTA = 0.02


def set_target(rows: Iterable[Optional[dict]]) -> Optional[dict]:
    """The median brightness and white balance of the photos in a set.

    ``rows`` are stored measurement rows (or ``None`` for members that have not
    been measured yet); anything unusable is left out rather than guessed at.
    Returns ``None`` when fewer than two members can anchor the set — with one
    frame there is nothing to match it to.
    """
    lums: list[float] = []
    reds: list[float] = []
    blues: list[float] = []
    for row in rows:
        if not row:
            continue
        try:
            lum = float(row.get("lum_p50") or 0.0)
            wb_r = float(row.get("wb_r") or 0.0)
            wb_b = float(row.get("wb_b") or 0.0)
        except (TypeError, ValueError):
            continue
        if lum <= 0 or wb_r <= 0 or wb_b <= 0:
            continue
        lums.append(lum)
        reds.append(wb_r)
        blues.append(wb_b)

    if len(lums) < 2:
        return None
    return {
        "count": len(lums),
        "lum": median(lums),
        "wb_r": median(reds),
        "wb_b": median(blues),
    }


def correction(
    current: dict,
    target: dict,
    strength: float = 1.0,
) -> Optional[dict]:
    """The nudge that moves one frame's measurements towards the set's target.

    ``current`` is measured on the pixels about to be written, not on the
    thumbnail, so the correction accounts for whatever the RAW decoder actually
    produced instead of what was expected of it. Returns ``None`` when the frame
    already matches, or when strength is zero — the caller then writes the file
    untouched.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return None
    try:
        lum = float(current.get("lum_p50") or 0.0)
        wb_r = float(current.get("wb_r") or 0.0)
        wb_b = float(current.get("wb_b") or 0.0)
    except (TypeError, ValueError):
        return None
    if lum <= 0 or wb_r <= 0 or wb_b <= 0:
        return None

    ev = _clamp(math.log2(target["lum"] / lum), MAX_EV_SHIFT) * strength
    if abs(ev) < MIN_EV_SHIFT:
        ev = 0.0

    gain_r = 1.0 + (_clamp(target["wb_r"] / wb_r - 1.0, MAX_WB_DELTA)) * strength
    gain_b = 1.0 + (_clamp(target["wb_b"] / wb_b - 1.0, MAX_WB_DELTA)) * strength
    if abs(gain_r - 1.0) < MIN_WB_DELTA:
        gain_r = 1.0
    if abs(gain_b - 1.0) < MIN_WB_DELTA:
        gain_b = 1.0

    if ev == 0.0 and gain_r == 1.0 and gain_b == 1.0:
        return None
    return {"ev": ev, "wb_r": gain_r, "wb_b": gain_b}


def apply_correction(image: Image.Image, change: dict) -> Image.Image:
    """Apply a brightness and white-balance nudge to an RGB image.

    Both are multiplications, so each channel is one curve: scale the value,
    clip at white. A single ``point`` pass keeps this cheap on a full-size
    frame, and it is the same operation whether the pixels came from a RAW file
    or a JPEG — which is what lets one code path match a set of both.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    light = 2.0 ** change.get("ev", 0.0)
    # One curve per band, flattened: Pillow wants 256 values per band as a single
    # sequence, not three nested ones. Green is included even though matching
    # never moves it, because a personal white-balance nudge does — and a
    # composition of the two arrives here as one dictionary.
    table: list[int] = []
    for gain in (
        light * change.get("wb_r", 1.0),
        light * change.get("wb_g", 1.0),
        light * change.get("wb_b", 1.0),
    ):
        table.extend(_channel_table(gain))
    return image.point(table)


def _channel_table(gain: float) -> list[int]:
    """256-entry curve for one channel under a multiplicative gain."""
    if gain == 1.0:
        return list(range(256))
    return [round(min(255.0, value * gain)) for value in range(256)]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
