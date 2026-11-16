"""
Scanned negatives, turned back into the scene they recorded.

A scan's value ``v`` is how much light the film let through. Film answers exposure
with density, ``density = base + gamma * log10(exposure)``, ``gamma`` about 0.6 for
negative film, so per channel::

    scene_c = K * (base_c / v_c) ** q_c          q_c about 1 / gamma

A division by the film base and a power. Three things follow:

1. **The scan's encoding hides in the power.** A linear scan read as gamma-encoded, or
   the reverse, raises every ``v`` to some power ``g``, and the result to ``g * q``:
   the same picture with more or less contrast. So the Contrast control covers a
   wrong guess about the file, and the colour is not broken by it.
2. **The three layers do not have the same gamma** as a scanner sees them, so equal
   ``q`` leaves a cast that changes from shadow to highlight. Each channel's density
   range is measured and the three evened out (Layer balance), within ``SPREAD``, so
   a scene that really was all one colour (a sunset) is not forced to grey.
3. **The result is scene-linear light with headroom**, which is what ``LinearFrame.rgb``
   is: everything after it needs no change.

Being a negative is a fact about the photograph, not part of its edit. Its
*description* (below) is kept apart from the edit, and the inversion happens where
the file is decoded (export.engine), so every consumer of a decoded frame gets the
scene.

The description, the only thing stored:

    kind       "colour" or "bw"
    base       three floats in (0, 1]: the film base, in the decode's linear values
    base_from  "border", "picture", "picked" or "roll": how it was found (shown only)
    range      three floats: each channel's density from the base to the densest part
    contrast   -100..100, default 0
    balance    0..100, default 100
"""

import math
from typing import Optional

import numpy as np

KINDS = ("colour", "bw")
BASE_FROM = ("border", "picture", "picked", "roll")

# Nothing in a scan is quite zero; this keeps the division finite (1 in 16 bits).
FLOOR = 2.0 ** -16
# A negative's density answers log exposure with about this slope.
FILM_GAMMA = 0.60
# A negative's density range over its base is seldom more. A larger measured range
# means the file was read with too steep a curve (point 1 above): the contrast is
# held to the most film would give, and the Contrast slider does the rest.
MAX_RANGE = 1.8
# No layer is stretched or squeezed by more than this against the others, so a frame
# that is truly all one colour is not forced to grey.
SPREAD = 1.25
# The frame's densest part lands just under display white. At full balance every
# channel reaches it together, so white is neutral; the base comes out at K in all
# three, neutral at any balance.
WHITE = 0.9
# Dust and the scanner's holder are near zero in the scan, near infinity in the scene.
CEILING = 8.0

# measure(): the size it works at, so it answers the same at any decode size.
MEASURE_SIZE = 768
# Without a crop, the picture is taken to be the frame inset by this on every side.
INSET = 0.06
# Border pixels: flat (std in a 5 x 5 window under this fraction of the mean), not
# clipped (every channel under CLIPPED: bare light, sprocket holes), at least as bright
# as the picture's thinnest part (luma at or above BRIGHT_ENOUGH of the area's
# 99.5th percentile: the holder is darker), and at least ENOUGH of the frame.
FLAT = 0.03
CLIPPED = 0.97
BRIGHT_ENOUGH = 0.98
ENOUGH = 0.004
# Range held to this, in density.
RANGE_LIMITS = (0.3, 3.5)
# Mean spread of the three channels' densities under which the film is black and white.
BW_SPREAD = 0.03
# base_at(): the patch's radius, as a fraction of the long side.
PATCH = 0.01

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

BARE_LIGHT = ("That is bare light, not film. Click the film's own border, "
              "or the gap between two frames.")


def _three(value) -> Optional[list]:
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(out) != 3 or not all(math.isfinite(v) for v in out):
        return None
    return out


def clean(description) -> Optional[dict]:
    """A description with its values clamped and defaults filled; ``None`` when
    ``kind`` or ``base`` is missing or bad."""
    if not isinstance(description, dict):
        return None
    kind = description.get("kind")
    if kind not in KINDS:
        return None
    base = _three(description.get("base"))
    if base is None or any(v <= 0 for v in base):
        return None
    base = [min(1.0, max(FLOOR, v)) for v in base]
    rng = _three(description.get("range"))
    if rng is None:
        rng = [1.0, 1.0, 1.0]
    rng = [min(RANGE_LIMITS[1], max(RANGE_LIMITS[0], v)) for v in rng]
    base_from = description.get("base_from")
    if base_from not in BASE_FROM:
        base_from = "picture"

    def number(key, lo, hi, default):
        try:
            v = float(description.get(key, default))
        except (TypeError, ValueError):
            return default
        return default if not math.isfinite(v) else min(hi, max(lo, v))

    return {
        "kind": kind,
        "base": [round(v, 6) for v in base],
        "base_from": base_from,
        "range": [round(v, 4) for v in rng],
        "contrast": number("contrast", -100.0, 100.0, 0.0),
        "balance": number("balance", 0.0, 100.0, 100.0),
    }


def _powers(d: dict) -> tuple:
    """(q per channel, K)."""
    rng = np.asarray(d["range"], dtype=np.float64)
    m = float(rng.mean())
    q0 = min(1.0, MAX_RANGE / m) / FILM_GAMMA * 2.0 ** (d["contrast"] / 100.0)
    if d["kind"] == "bw":
        q = np.full(3, q0)
    else:
        r = np.clip(m / rng, 1.0 / SPREAD, SPREAD)
        q = q0 * (1.0 + (d["balance"] / 100.0) * (r - 1.0))
    k = WHITE * 10.0 ** (-q0 * m)
    return q.astype(np.float32), np.float32(k)


def positive(rgb: np.ndarray, description: dict) -> np.ndarray:
    """The scene, (H, W, 3) float32 scene-linear, from the scan as decoded."""
    d = clean(description)
    if d is None:
        return np.asarray(rgb, dtype=np.float32)
    from adjust.render import in_strips

    q, k = _powers(d)
    base = np.asarray(d["base"], dtype=np.float32)
    bw = d["kind"] == "bw"

    def one(strip):
        v = np.maximum(strip.astype(np.float32, copy=False), np.float32(FLOOR))
        # (base / v) ** q, as exp(q * log(base / v)): one pass, no temporaries of note.
        out = np.log(base / v)
        out *= q
        np.exp(out, out=out)
        out *= k
        np.minimum(out, np.float32(CEILING), out=out)
        if bw:
            y = out @ LUMA
            out = np.repeat(y[..., None], 3, axis=2)
        return out

    return in_strips(one, np.asarray(rgb))


def _small(rgb: np.ndarray) -> np.ndarray:
    """The scan at MEASURE_SIZE on its long side, median-filtered 3 x 3, so dust and
    grain do not set a percentile and the answer is the same at any decode size."""
    import cv2

    rgb = np.asarray(rgb, dtype=np.float32)
    h, w = rgb.shape[:2]
    scale = MEASURE_SIZE / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return cv2.medianBlur(np.ascontiguousarray(rgb, dtype=np.float32), 3)


def _area_mask(shape: tuple, area: Optional[tuple]) -> np.ndarray:
    h, w = shape[:2]
    if area is None:
        area = (INSET, INSET, 1.0 - INSET, 1.0 - INSET)
    left, top, right, bottom = [float(v) for v in area]
    x0, x1 = int(round(left * w)), int(round(right * w))
    y0, y1 = int(round(top * h)), int(round(bottom * h))
    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _border_base(small: np.ndarray, inside: np.ndarray) -> Optional[list]:
    import cv2

    outside = ~inside
    if not outside.any():
        return None
    luma = small @ LUMA
    mean = cv2.blur(luma, (5, 5), borderType=cv2.BORDER_REFLECT)
    sq = cv2.blur(luma * luma, (5, 5), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    flat = std < FLAT * np.maximum(mean, 1e-6)
    unclipped = (small < CLIPPED).all(axis=2)
    thinnest = float(np.percentile(luma[inside], 99.5))
    bright = luma >= BRIGHT_ENOUGH * thinnest
    keep = outside & flat & unclipped & bright
    if keep.sum() < ENOUGH * keep.size:
        return None
    return [float(np.median(small[..., c][keep])) for c in range(3)]


def measure(rgb: np.ndarray, kind: Optional[str] = None, area: Optional[tuple] = None,
            base=None, base_from: Optional[str] = None) -> dict:
    """The description of a negative, from the scan as decoded. ``area`` is
    ``(left, top, right, bottom)`` fractions, the picture (the edit's crop); a given
    ``base`` (picked, or from the roll) is kept with the ``base_from`` it came with."""
    small = _small(rgb)
    inside = _area_mask(small.shape, area)
    picture = small[inside]
    given = _three(base) if base is not None else None
    if given is not None and all(v > 0 for v in given):
        found, found_from = given, base_from if base_from in BASE_FROM else "picked"
    else:
        found = _border_base(small, inside)
        found_from = "border"
        if found is None:
            found = [float(np.percentile(picture[:, c], 99.9)) for c in range(3)]
            found_from = "picture"
    found = [min(1.0, max(FLOOR, v)) for v in found]
    base_arr = np.asarray(found, dtype=np.float32)
    p1 = np.array([np.percentile(picture[:, c], 1.0) for c in range(3)], dtype=np.float64)
    p1 = np.maximum(p1, FLOOR)
    rng = np.clip(np.log10(base_arr / p1), *RANGE_LIMITS)
    if kind not in KINDS:
        d = np.log10(base_arr / np.maximum(picture, FLOOR))
        spread = float(np.mean(d.max(axis=1) - d.min(axis=1)))
        kind = "bw" if spread < BW_SPREAD else "colour"
    return clean({
        "kind": kind,
        "base": found,
        "base_from": found_from,
        "range": [float(v) for v in rng],
        "contrast": 0.0,
        "balance": 100.0,
    })


def base_at(rgb: np.ndarray, u: float, v: float) -> list:
    """The film base at ``(u, v)`` (fractions of the frame), on the scan as decoded:
    each channel's median over a patch round the point. ``ValueError`` with a sentence
    the panel can show when the point is bare light."""
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    r = max(1, int(round(PATCH * max(h, w))))
    x = int(round(min(max(u, 0.0), 1.0) * (w - 1)))
    y = int(round(min(max(v, 0.0), 1.0) * (h - 1)))
    patch = np.asarray(rgb[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1], dtype=np.float32)
    found = [float(np.median(patch[..., c])) for c in range(3)]
    if max(found) >= CLIPPED:
        raise ValueError(BARE_LIGHT)
    if min(found) <= 0:
        raise ValueError("That is black in the scan: the holder, not film. Click the "
                         "film's own border, or the gap between two frames.")
    return found
