"""
Local adjustments: a gradient or a brush that changes part of the picture.

An edit's ``masks`` is a list of them, each with its own few controls:

  linear   full effect on one side of a line, fading to none across a band:
           ``x1, y1`` is where the effect is whole, ``x2, y2`` where it has gone.
           A darker sky is one dragged down from the top. ``invert`` turns
           it round: none at ``x1, y1``, whole at ``x2, y2`` and beyond.
  radial   an ellipse — centre ``cx, cy``, radii ``rx, ry``, turned by ``angle``
           degrees — whole inside and fading to none at its edge over ``feather``
           (0..100, the share of the radius the fade takes). ``invert`` puts the
           effect outside instead: a vignette around a face, or light on it.
  brush    nothing until painted: its ``strokes`` alone say where it reaches.
           ``invert`` for everything but where it is painted: the whole frame,
           less what the strokes cover (an erasing stroke gives it back).
  subject  where the picture's subject is, and ``sky`` where its sky is, as a
  sky      model finds them in the frame (adjust.segment); ``invert`` for
           everything else (the background, the land).
  people   everyone in the frame, and ``faces`` their faces (adjust.segment);
  faces    ``person`` (1, 2, … counted from the left) for one person only, and
           ``invert`` for everything else.

Any of them carries ``strokes``, laid over its shape in the order they were
painted: each a list of ``points`` along the hand's path, a radius ``size`` (a
share of the frame's longer side, from 0.0005 up: a hair's brush), a
``feather`` (0..100, the share of the radius the soft edge takes), a ``flow``
(1..100, how much one stroke lays down) and ``erase``. A stroke adds to the
mask, an erasing one takes away from it — from a radial too, so light on a face
need not reach the sky beside it.

Any of them can be cropped by a ``section``, a rectangle of the frame it reaches
inside: ``x1, y1, x2, y2`` as fractions of the whole frame, and ``feather``
(0..100, the share of the shorter side its edges fade in over, so 100 fades from
the middle). The section has the last word: outside it the mask reaches nothing,
an inverted brush and its strokes included. A section of the whole frame with no
feather limits nothing: it is a section switched on, waiting to be dragged.

Any of them can be narrowed by a ``range`` of the picture, as Lightroom's range
masks are: only where the picture is within a band of brightness, or near a
colour. The range is judged on the picture as it is shown before the edit, so
moving a mask's own exposure does not move what it reaches.

  {"kind": "luminance", "lo", "hi", "smooth"}   brightness 0..100; ``smooth``
      (0..100) how far it fades out beyond the band
  {"kind": "color", "cb", "cr", "lum", "width"}  the colour sampled (its blue-
      and red-difference, -0.5..0.5, and brightness 0..100) and how wide a range
      of colours round it is taken (0..100)

Every position is a fraction of the *whole* frame as the camera made it (0..1
across and down, before any crop or turn), so cropping or levelling afterwards
does not move a mask off what it was drawn on. A radius is a fraction of the
frame's width (``rx``) and height (``ry``).

The controls are the ones a mask is for, on the global sliders' scales:
exposure in stops (±4), saturation on -200..200 (it adds to the global
slider's, so it can undo a picture at -100 and go as far again), and
contrast, highlights, shadows, temperature and clarity on -100..100. They are performed where the global
controls are (adjust.render): exposure and temperature in linear light,
contrast, highlights and shadows as a change of the tone curve on the frame's
base, clarity on its detail, saturation on the finished colour, each scaled per
pixel by the mask's weight. So a mask's +1 EV is the same stop as the
exposure slider's, where the mask is whole.

A mask with every control at zero is kept: it is drawn, and its controls are
about to be moved. It changes no pixel.
"""

import json
import math
from typing import Optional

import numpy as np

# The controls a mask carries, and their ranges.
MASK_SLIDERS = ("exposure", "contrast", "highlights", "shadows", "temperature",
                "saturation", "clarity")
MAX_MASK_EXPOSURE = 4.0
# Saturation adds to the global slider's (adjust.render), so a gradient reaches
# twice as far: +200 over a picture at -100 is +100 where it is whole.
MAX_MASK_SATURATION = 200.0
MAX_MASKS = 16
KINDS = ("linear", "radial", "brush", "subject", "sky", "people", "faces")
# The kinds a model selects (adjust.segment.KINDS; not imported, so this module
# needs no OpenCV).
CHOSEN = ("subject", "sky", "people", "faces")
# The kinds that can be narrowed to one person, and how many people that may name.
OF_PEOPLE = ("people", "faces")
MAX_PERSON = 64
MAX_STROKES = 200
# A stroke longer than this is thinned (every other point, and so on): the
# hand's path is sampled far more finely than a soft brush needs.
MAX_STROKE_POINTS = 600
# A position may sit off the frame (a gradient begun above the top edge), but
# not anywhere at all.
_POSITION = (-2.0, 3.0)
# The smallest brush, as a share of the frame's longer side: fine enough to
# correct a few pixels of a face or an edge by hand.
MIN_STROKE_SIZE = 0.0005
# The smallest side of a section, as a share of the frame.
_SECTION_MIN = 0.01


def _num(value, low: float, high: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(high, max(low, v))


def _normalize_stroke(stroke) -> Optional[dict]:
    if not isinstance(stroke, dict) or not isinstance(stroke.get("points"), (list, tuple)):
        return None
    points = []
    for point in stroke["points"]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = (_num(v, *_POSITION, math.nan) for v in point)
        if math.isnan(x) or math.isnan(y):
            continue
        points.append([round(x, 4), round(y, 4)])
    if not points:
        return None
    stride = math.ceil(len(points) / MAX_STROKE_POINTS)
    if stride > 1:
        points = points[::stride] + ([points[-1]] if (len(points) - 1) % stride else [])
    out = {
        "points": points,
        "size": round(_num(stroke.get("size"), MIN_STROKE_SIZE, 0.5, 0.03), 4),
        "feather": int(round(_num(stroke.get("feather"), 0.0, 100.0, 50.0))),
        "flow": int(round(_num(stroke.get("flow"), 1.0, 100.0, 100.0))),
    }
    if stroke.get("erase"):
        out["erase"] = True
    return out


def _normalize_strokes(strokes) -> list:
    if not isinstance(strokes, (list, tuple)):
        return []
    return [s for s in (_normalize_stroke(s) for s in strokes[-MAX_STROKES:]) if s]


def normalize_mask(mask) -> Optional[dict]:
    """One mask, cleaned: known fields only, each in range. ``None`` when it
    is not a mask this app draws."""
    if not isinstance(mask, dict) or mask.get("kind") not in KINDS:
        return None
    kind = mask["kind"]
    out: dict = {"kind": kind}
    if kind == "linear":
        for key, default in (("x1", 0.5), ("y1", 0.0), ("x2", 0.5), ("y2", 0.5)):
            out[key] = round(_num(mask.get(key), *_POSITION, default), 5)
        if out["x1"] == out["x2"] and out["y1"] == out["y2"]:
            out["y2"] = round(out["y1"] + 0.01, 5)       # a band, not a point
        if mask.get("invert"):
            out["invert"] = True
    elif kind == "radial":
        for key, default in (("cx", 0.5), ("cy", 0.5)):
            out[key] = round(_num(mask.get(key), *_POSITION, default), 5)
        for key, default in (("rx", 0.2), ("ry", 0.2)):
            out[key] = round(_num(mask.get(key), 0.005, 3.0, default), 5)
        out["angle"] = round(_num(mask.get("angle"), -180.0, 180.0, 0.0), 2)
        out["feather"] = int(round(_num(mask.get("feather"), 0.0, 100.0, 50.0)))
        out["invert"] = bool(mask.get("invert"))
    elif kind == "brush":
        if mask.get("invert"):
            out["invert"] = True
    elif kind in CHOSEN:
        out["invert"] = bool(mask.get("invert"))
        if kind in OF_PEOPLE:
            person = int(_num(mask.get("person"), 0, MAX_PERSON, 0))
            if person:
                out["person"] = person
    strokes = _normalize_strokes(mask.get("strokes"))
    if strokes:
        out["strokes"] = strokes
    rng = normalize_range(mask.get("range"))
    if rng:
        out["range"] = rng
    sec = normalize_section(mask.get("section"))
    if sec:
        out["section"] = sec
    for key in MASK_SLIDERS:
        if key == "exposure":
            value = round(_num(mask.get(key), -MAX_MASK_EXPOSURE, MAX_MASK_EXPOSURE, 0.0), 3)
        elif key == "saturation":
            value = int(round(_num(mask.get(key), -MAX_MASK_SATURATION, MAX_MASK_SATURATION, 0.0)))
        else:
            value = int(round(_num(mask.get(key), -100.0, 100.0, 0.0)))
        if value:
            out[key] = value
    return out


def normalize_range(rng) -> Optional[dict]:
    """A mask's range, cleaned, or ``None`` when it narrows nothing."""
    if not isinstance(rng, dict):
        return None
    if rng.get("kind") == "luminance":
        lo = int(round(_num(rng.get("lo"), 0.0, 100.0, 0.0)))
        hi = int(round(_num(rng.get("hi"), 0.0, 100.0, 100.0)))
        lo, hi = min(lo, hi), max(lo, hi)
        if lo <= 0 and hi >= 100:
            return None
        return {"kind": "luminance", "lo": lo, "hi": hi,
                "smooth": int(round(_num(rng.get("smooth"), 0.0, 100.0, 30.0)))}
    if rng.get("kind") == "color":
        cb, cr = (_num(rng.get(k), -0.5, 0.5, math.nan) for k in ("cb", "cr"))
        if math.isnan(cb) or math.isnan(cr):
            return None
        return {"kind": "color", "cb": round(cb, 4), "cr": round(cr, 4),
                "lum": int(round(_num(rng.get("lum"), 0.0, 100.0, 50.0))),
                "width": int(round(_num(rng.get("width"), 0.0, 100.0, 40.0)))}
    return None


def _section_span(low: float, high: float) -> tuple:
    """Two edges of a section, inside the frame and at least a hundredth apart."""
    low = min(1.0 - _SECTION_MIN, max(0.0, low))
    return low, min(1.0, max(low + _SECTION_MIN, high))


def normalize_section(section) -> Optional[dict]:
    """A mask's section, cleaned, or ``None`` when it is not a section at all.

    A section that is given is kept, even one of the whole frame with no
    feather: that is a section switched on, waiting to be dragged."""
    if not isinstance(section, dict):
        return None
    left = _num(section.get("x1"), 0.0, 1.0, 0.0)
    top = _num(section.get("y1"), 0.0, 1.0, 0.0)
    right = _num(section.get("x2"), 0.0, 1.0, 1.0)
    bottom = _num(section.get("y2"), 0.0, 1.0, 1.0)
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top
    left, right = _section_span(left, right)
    top, bottom = _section_span(top, bottom)
    feather = int(round(_num(section.get("feather"), 0.0, 100.0, 0.0)))
    return {"x1": round(left, 5), "y1": round(top, 5),
            "x2": round(right, 5), "y2": round(bottom, 5), "feather": feather}


# BT.709, as LUMA: brightness, and the blue and red differences that place a colour.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_CB_SCALE, _CR_SCALE = 1.8556, 1.5748
_DARK = 0.05


def colour_of(enc: np.ndarray):
    """(luma, cb, cr) of display-encoded colour, each (H, W) float32."""
    y = enc @ _LUMA
    return y, (enc[..., 2] - y) / _CB_SCALE, (enc[..., 0] - y) / _CR_SCALE


def range_weight(enc: np.ndarray, rng: dict) -> np.ndarray:
    """How much of a range each pixel of ``enc`` (H, W, 3 display-encoded 0..1,
    the picture before the edit) is in, 0..1, as (H, W) float32."""
    y, cb, cr = colour_of(enc)
    if rng["kind"] == "luminance":
        lum = y * 100.0
        soft = max(1.0, rng.get("smooth", 30) * 0.5)
        below = _smoothstep(rng["lo"] - soft, rng["lo"], lum) if rng["lo"] > 0 else 1.0
        above = 1.0 - _smoothstep(rng["hi"], rng["hi"] + soft, lum) if rng["hi"] < 100 else 1.0
        return np.ascontiguousarray(below * above, dtype=np.float32)
    # A colour: near the sampled one in its colour as a share of its brightness
    # (its chromaticity), and only a little in brightness, so a red shirt in shade
    # is still the red shirt. _DARK keeps the near-black, whose colour is noise,
    # from looking strongly coloured.
    y0 = rng.get("lum", 50) / 100.0
    k, k0 = 1.0 / (y + _DARK), 1.0 / (y0 + _DARK)
    d = np.hypot(cb * k - rng["cb"] * k0, cr * k - rng["cr"] * k0) + 0.15 * np.abs(y - y0)
    reach = 0.05 + 0.45 * rng.get("width", 40) / 100.0
    return np.ascontiguousarray(1.0 - _smoothstep(0.5 * reach, reach, d), dtype=np.float32)


def normalize_masks(masks) -> Optional[list]:
    """The masks of an edit, cleaned, or ``None`` for none."""
    if isinstance(masks, str):
        try:
            masks = json.loads(masks)
        except ValueError:
            return None
    if not isinstance(masks, (list, tuple)):
        return None
    out = [m for m in (normalize_mask(m) for m in masks[:MAX_MASKS]) if m]
    return out or None


def active(masks: Optional[list]) -> list:
    """The masks that change something: a control moved, and somewhere to
    act (a brush with nothing painted reaches no pixel; inverted, it reaches
    every one)."""
    return [m for m in masks or []
            if any(k in m for k in MASK_SLIDERS)
            and (m["kind"] != "brush" or m.get("invert")
                 or any(not s.get("erase") for s in m.get("strokes", ())))]


def to_text(masks: Optional[list]) -> str:
    """The masks as one compact line, for a sidecar attribute."""
    return json.dumps(masks or [], separators=(",", ":"), sort_keys=True)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def weight(mask: dict, shape: tuple, window: tuple = (0.0, 0.0, 1.0, 1.0),
           matte: Optional[np.ndarray] = None) -> np.ndarray:
    """How much of a mask each pixel gets, 0..1, as an (H, W) float32.

    ``shape`` is the (H, W) of the pixels being rendered and ``window`` which
    part of the whole frame they are, as (left, top, right, bottom) fractions —
    all of it for a preview or an export, a patch for the loupe. Distances are
    measured in the whole frame's pixels, so an ellipse is round where it was
    drawn round, and a band or a brush is as wide across a patch as across the
    frame.

    ``matte`` is the whole frame's selection for a subject or sky mask
    (adjust.segment.matte); without one such a mask reaches nothing, inverted
    or not, rather than everything.

    A ``section`` (crop) is multiplied over the result last, so the mask reaches
    only inside it, whatever its shape, strokes or inversion.
    """
    h, w = shape
    left, top, right, bottom = window
    full_w = w / max(right - left, 1e-9)
    full_h = h / max(bottom - top, 1e-9)
    xs = (left + (np.arange(w, dtype=np.float32) + 0.5) / w * (right - left)) * full_w
    ys = (top + (np.arange(h, dtype=np.float32) + 0.5) / h * (bottom - top)) * full_h

    if mask["kind"] == "brush" or (mask["kind"] in CHOSEN and matte is None):
        out = np.zeros((h, w), dtype=np.float32)
    elif mask["kind"] in CHOSEN:
        from adjust.segment import sample

        out = sample(matte, (h, w), window)
        if mask.get("invert"):
            out = 1.0 - out
        out = np.ascontiguousarray(out, dtype=np.float32)
    else:
        out = _shape(mask, xs, ys, full_w, full_h)
    # Strokes in the order they were painted: one lays its paint over what is
    # there, one erasing takes its share of what is there away.
    for stroke in mask.get("strokes") or ():
        hit = _coverage(stroke, xs, ys, full_w, full_h)
        if hit is None:
            continue
        rows, cols, cover = hit
        part = out[rows, cols]
        if stroke.get("erase"):
            part *= 1.0 - cover
        else:
            part += cover * (1.0 - part)
    # An inverted brush is everything but its paint, so its strokes mark what
    # to leave alone; any other kind is inverted before its strokes refine it.
    if mask["kind"] == "brush" and mask.get("invert"):
        out = 1.0 - out
    # A section is the last word: outside it the mask reaches nothing at all,
    # whatever its shape, strokes or inversion said.
    section = mask.get("section")
    if section:
        out = out * _section_weight(section, xs, ys, full_w, full_h)
    return out


def _section_weight(section: dict, xs: np.ndarray, ys: np.ndarray,
                    full_w: float, full_h: float) -> np.ndarray:
    """A section as a reach map: 1 inside its rectangle, 0 outside, and a band
    fading inwards from each edge over up to half the shorter side, so feather
    100 fades from the middle and peaks at 1 there."""
    x1, x2 = section["x1"] * full_w, section["x2"] * full_w
    y1, y2 = section["y1"] * full_h, section["y2"] * full_h
    fade = section.get("feather", 0) / 100.0 * min(x2 - x1, y2 - y1) / 2.0

    def axis(values, low, high):
        if fade <= 0.0:
            return ((values >= low) & (values <= high)).astype(np.float32)
        return (_smoothstep(low, low + fade, values)
                * (1.0 - _smoothstep(high - fade, high, values))).astype(np.float32)

    return np.outer(axis(ys, y1, y2), axis(xs, x1, x2)).astype(np.float32)


def _shape(mask: dict, xs: np.ndarray, ys: np.ndarray, full_w: float, full_h: float) -> np.ndarray:
    """A gradient's own weight, before any strokes."""
    X, Y = np.meshgrid(xs, ys)
    if mask["kind"] == "linear":
        x1, y1 = mask["x1"] * full_w, mask["y1"] * full_h
        dx, dy = mask["x2"] * full_w - x1, mask["y2"] * full_h - y1
        length2 = max(dx * dx + dy * dy, 1e-6)
        t = ((X - x1) * dx + (Y - y1) * dy) / length2
        ramp = _smoothstep(0.0, 1.0, t)
        return (ramp if mask.get("invert") else 1.0 - ramp).astype(np.float32)

    a = max(mask["rx"] * full_w, 1e-3)
    b = max(mask["ry"] * full_h, 1e-3)
    theta = math.radians(mask.get("angle", 0.0))
    c, s = math.cos(theta), math.sin(theta)
    u = X - mask["cx"] * full_w
    v = Y - mask["cy"] * full_h
    ur, vr = u * c + v * s, -u * s + v * c
    r = np.sqrt((ur / a) ** 2 + (vr / b) ** 2)
    feather = mask.get("feather", 50) / 100.0
    if feather <= 0.0:
        inside = (r <= 1.0).astype(np.float32)
    else:
        inside = 1.0 - _smoothstep(1.0 - feather, 1.0, r)
    out = 1.0 - inside if mask.get("invert") else inside
    return out.astype(np.float32)


def _coverage(stroke: dict, xs: np.ndarray, ys: np.ndarray, full_w: float, full_h: float):
    """How much of one stroke each pixel gets, 0..1, over the smallest box of
    pixels it reaches: ``(rows, cols, cover)``, or ``None`` when it misses
    these pixels. ``xs``/``ys`` are the pixel centres in the whole frame's
    pixels."""
    radius = max(stroke.get("size", 0.03) * max(full_w, full_h), 0.5)
    inner = radius * (1.0 - stroke.get("feather", 50) / 100.0)
    pts = np.array(stroke["points"], dtype=np.float64) * (full_w, full_h)
    # A point closer than a fifth of the radius to the last one kept adds
    # nothing a soft brush would show; the stroke's end is always kept.
    kept = [0]
    for i in range(1, len(pts)):
        if math.hypot(*(pts[i] - pts[kept[-1]])) >= radius * 0.2:
            kept.append(i)
    if kept[-1] != len(pts) - 1:
        kept.append(len(pts) - 1)
    pts = pts[kept]

    def span(axis, lo, hi):
        return (int(np.searchsorted(axis, lo - radius, "left")),
                int(np.searchsorted(axis, hi + radius, "right")))

    r0, r1 = span(ys, pts[:, 1].min(), pts[:, 1].max())
    c0, c1 = span(xs, pts[:, 0].min(), pts[:, 0].max())
    if r0 >= r1 or c0 >= c1:
        return None
    # Each pixel's distance to the path: the nearest of its segments, each
    # measured only over the box it can reach.
    dist2 = np.full((r1 - r0, c1 - c0), np.inf, dtype=np.float32)
    segments = list(zip(pts[:-1], pts[1:])) or [(pts[0], pts[0])]
    for a, b in segments:
        sr0, sr1 = span(ys, min(a[1], b[1]), max(a[1], b[1]))
        sc0, sc1 = span(xs, min(a[0], b[0]), max(a[0], b[0]))
        sr0, sr1 = max(sr0, r0), min(sr1, r1)
        sc0, sc1 = max(sc0, c0), min(sc1, c1)
        if sr0 >= sr1 or sc0 >= sc1:
            continue
        X = xs[sc0:sc1][None, :] - np.float32(a[0])
        Y = ys[sr0:sr1][:, None] - np.float32(a[1])
        d = b - a
        length2 = float(d @ d)
        if length2 > 1e-9:
            t = np.clip((X * np.float32(d[0]) + Y * np.float32(d[1])) / np.float32(length2), 0.0, 1.0)
            X = X - t * np.float32(d[0])
            Y = Y - t * np.float32(d[1])
        part = dist2[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0]
        np.minimum(part, X * X + Y * Y, out=part)
    dist = np.sqrt(dist2)
    if inner >= radius - 1e-6:
        cover = (dist <= radius).astype(np.float32)
    else:
        cover = 1.0 - _smoothstep(inner, radius, dist)
    return slice(r0, r1), slice(c0, c1), (cover * np.float32(stroke.get("flow", 100) / 100.0)).astype(np.float32)
