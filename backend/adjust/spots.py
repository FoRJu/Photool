"""
Spot removal: dust on the sensor, a blemish, a wire across the sky.

An edit's ``spots`` is a list of circles, each painted over from another part
of the picture:

  x, y      the spot's centre, fractions of the whole frame (before any crop or
            turn, as masks are: adjust.masks)
  r         its radius, a share of the frame's longer side
  sx, sy    the centre of the circle it is painted from
  mode      ``heal`` (default): the source's texture, brought to the light and
            colour around the spot, so a patch of sky from higher up comes in
            as the sky around the dust. ``clone``: the source as it is.
  feather   0..100, the share of the radius the soft edge takes (default 50)
  opacity   1..100 (default 100)

They are done first, on the linear frame, before any other control: the rest of
the edit sees a picture without the dust, as in Lightroom. In order, so a later
spot may take from where an earlier one has healed.

Healing works in log light: the difference between the spot's surroundings and
the source's surroundings (a ring from r to 1.5 r) is smoothed into the circle
and added to the source, which carries the source's texture over and its
brightness and colour away.

A spot belongs to its frame, like the crop: dust sits at one place in one frame
(catalog.service keeps a frame's own spots when another's edit is put on it).
"""

import math
from typing import Callable, Optional

import numpy as np

SPOTS_KEY = "spots"
MAX_SPOTS = 200
MODES = ("heal", "clone")
RING = 1.5           # the surroundings a heal matches: out to this many radii
_EPS = 1e-4


def _num(value, low, high, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(high, max(low, v))


def normalize_spot(spot) -> Optional[dict]:
    if not isinstance(spot, dict):
        return None
    out = {
        "x": round(_num(spot.get("x"), 0.0, 1.0, math.nan), 5),
        "y": round(_num(spot.get("y"), 0.0, 1.0, math.nan), 5),
        "r": round(_num(spot.get("r"), 0.001, 0.2, 0.01), 5),
    }
    if math.isnan(out["x"]) or math.isnan(out["y"]):
        return None
    # A spot with no source takes one beside it: to the right, or the left at
    # the frame's edge. The develop view asks for a better one
    # (find_source) before it adds a spot.
    side = 3.0 * out["r"]
    default_sx = out["x"] + side if out["x"] + side <= 1.0 else out["x"] - side
    out["sx"] = round(_num(spot.get("sx"), 0.0, 1.0, default_sx), 5)
    out["sy"] = round(_num(spot.get("sy"), 0.0, 1.0, out["y"]), 5)
    out["mode"] = spot.get("mode") if spot.get("mode") in MODES else "heal"
    feather = int(round(_num(spot.get("feather"), 0.0, 100.0, 50.0)))
    opacity = int(round(_num(spot.get("opacity"), 1.0, 100.0, 100.0)))
    if feather != 50:
        out["feather"] = feather
    if opacity != 100:
        out["opacity"] = opacity
    return out


def normalize_spots(spots) -> Optional[list]:
    if isinstance(spots, str):
        import json

        try:
            spots = json.loads(spots)
        except ValueError:
            return None
    if not isinstance(spots, (list, tuple)):
        return None
    out = [s for s in (normalize_spot(s) for s in spots[:MAX_SPOTS]) if s]
    return out or None


def to_text(spots: Optional[list]) -> str:
    import json

    return json.dumps(spots or [], separators=(",", ":"), sort_keys=True)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def heal_patch(dst: np.ndarray, src: np.ndarray, radius: float, mode: str = "heal",
               feather: int = 50, opacity: int = 100) -> np.ndarray:
    """One spot on two equal square boxes of linear RGB centred on the spot and
    on its source: the spot's box with the circle painted over."""
    import cv2

    n = dst.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.hypot(xx - c, yy - c)
    if mode == "clone":
        painted = src
    else:
        log_dst = np.log(np.maximum(dst, 0.0) + _EPS)
        log_src = np.log(np.maximum(src, 0.0) + _EPS)
        ring = ((d > radius) & (d <= RING * radius)).astype(np.float32)
        sigma = max(0.8, radius * 0.6)
        spread = cv2.GaussianBlur(ring, (0, 0), sigma)
        diff = np.empty_like(log_dst)
        for ch in range(3):
            diff[..., ch] = cv2.GaussianBlur((log_dst[..., ch] - log_src[..., ch]) * ring, (0, 0), sigma)
        diff /= np.maximum(spread, 1e-6)[..., None]
        painted = np.exp(log_src + diff) - _EPS
    inner = radius * (1.0 - feather / 100.0)
    alpha = (1.0 - _smoothstep(inner, radius, d)) * (opacity / 100.0)
    return (dst * (1.0 - alpha[..., None]) + painted * alpha[..., None]).astype(np.float32)


def apply_spots(rgb: np.ndarray, spots: list, window: tuple = (0.0, 0.0, 1.0, 1.0),
                fetch: Optional[Callable] = None) -> np.ndarray:
    """``rgb`` (the part ``window`` of the whole frame) with each spot painted
    over, as a new array.

    ``fetch(left, top, size)`` returns a ``size`` square of the whole frame at
    this scale, from whole-frame pixel ``(left, top)`` (edge pixels repeated
    beyond it). By default it reads ``rgb`` itself, healed so far, which is all a
    whole frame needs; a loupe's patch passes one reading the full decode, since
    a spot's source is seldom inside the patch.
    """
    out = rgb.astype(np.float32, copy=True)
    h, w = out.shape[:2]
    left, top, right, bottom = window
    full_w = w / max(right - left, 1e-9)
    full_h = h / max(bottom - top, 1e-9)
    ox, oy = left * full_w, top * full_h          # the window's corner, whole-frame px
    unit = max(full_w, full_h)

    def own(x0, y0, size):
        xs = np.clip(np.arange(x0, x0 + size) - round(ox), 0, w - 1)
        ys = np.clip(np.arange(y0, y0 + size) - round(oy), 0, h - 1)
        return out[np.ix_(ys, xs)]

    read = fetch or own
    for spot in spots:
        radius = max(spot["r"] * unit, 1.0)
        half = int(math.ceil(RING * radius)) + 1
        size = 2 * half + 1
        # The pixel each position falls in.
        cx, cy = math.floor(spot["x"] * full_w), math.floor(spot["y"] * full_h)
        # Not on these pixels at all: nothing to do (a loupe looking elsewhere).
        if (cx + half < ox or cx - half >= ox + w or cy + half < oy or cy - half >= oy + h):
            continue
        sx, sy = math.floor(spot["sx"] * full_w), math.floor(spot["sy"] * full_h)
        dst = own(cx - half, cy - half, size) if fetch is None else read(cx - half, cy - half, size)
        src = read(sx - half, sy - half, size)
        healed = heal_patch(dst, src, radius, spot.get("mode", "heal"),
                            spot.get("feather", 50), spot.get("opacity", 100))
        # Back onto the window, where the two overlap.
        x0, y0 = cx - half - round(ox), cy - half - round(oy)
        a0, b0 = max(0, -x0), max(0, -y0)
        a1, b1 = min(size, w - x0), min(size, h - y0)
        if a0 < a1 and b0 < b1:
            out[y0 + b0:y0 + b1, x0 + a0:x0 + a1] = healed[b0:b1, a0:a1]
    return out


def find_source(rgb: np.ndarray, x: float, y: float, r: float) -> tuple:
    """A good place to paint a spot at ``(x, y)`` from, on a whole frame of
    linear RGB: a circle nearby whose surroundings look most like the spot's,
    in texture rather than brightness (a heal brings the brightness over).
    Returns ``(sx, sy)`` as fractions."""
    h, w = rgb.shape[:2]
    unit = max(w, h)
    radius = max(r * unit, 1.5)
    half = int(math.ceil(RING * radius)) + 1
    size = 2 * half + 1
    luma = np.log(np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0) + _EPS)

    def box(cx, cy):
        xs = np.clip(np.arange(cx - half, cx + half + 1), 0, w - 1)
        ys = np.clip(np.arange(cy - half, cy + half + 1), 0, h - 1)
        return luma[np.ix_(ys, xs)]

    c = half
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx - c, yy - c)
    ring = (d > radius) & (d <= RING * radius)
    inside = d <= radius
    cx, cy = x * w, y * h
    target = box(round(cx), round(cy))
    t_ring = target[ring] - target[ring].mean()
    t_tex = float(target[ring].std())

    best, best_score = None, math.inf
    for dist in (2.4, 3.2, 4.5, 6.5):
        for k in range(24):
            a = 2 * math.pi * k / 24
            px, py = cx + math.cos(a) * dist * radius, cy + math.sin(a) * dist * radius
            # The whole circle and its ring inside the frame.
            if not (half <= px <= w - 1 - half and half <= py <= h - 1 - half):
                continue
            cand = box(round(px), round(py))
            ring_vals = cand[ring] - cand[ring].mean()
            score = float(np.mean((ring_vals - t_ring) ** 2))
            # A source with a speck of its own, or busier than the spot's
            # surroundings, would paint that in.
            score += 2.0 * abs(float(cand[inside].std()) - t_tex) ** 2
            score *= 1.0 + 0.04 * dist                       # nearer is likelier alike
            if score < best_score:
                best, best_score = (px, py), score
    if best is None:                                         # a spot too big to place
        side = 3.0 * r
        return (min(1.0, x + side) if x + side <= 1.0 else max(0.0, x - side), y)
    return (round(best[0] / w, 5), round(best[1] / h, 5))
