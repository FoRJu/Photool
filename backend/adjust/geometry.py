"""
Crop and level: the geometry half of an edit.

The tone controls change what a pixel looks like; these change which pixels are
kept. They are applied last, to the finished picture, so every path that draws
an edit — the develop preview, the library thumbnails and the export — frames
the photograph the same way without each of them knowing how.

An edit stores three things:

  angle                       degrees to turn the picture clockwise to level it
                              (-45..45), about its centre.
  crop_x, crop_y, crop_w, crop_h
                              the kept rectangle as fractions of the picture
                              (0..1), measured on the *whole* frame after the
                              turn. Fractions, not pixels, because the preview,
                              a thumbnail and a 24 MP export are different sizes
                              of the same picture.

A turned picture has empty corners. A crop that reaches into them would export
black wedges, so the rectangle is always pulled back inside the part of the frame
that is really picture: :func:`fit_box`. Levelling with no crop drawn keeps the
largest rectangle of the original shape that fits, which is what every editor
does and what a person levelling a horizon expects.
"""

import math
from typing import Optional

from PIL import Image

GEOMETRY_KEYS = ("angle", "crop_x", "crop_y", "crop_w", "crop_h")

MAX_ANGLE = 45.0
# Below these a control is doing nothing.
MIN_ANGLE = 0.05
MIN_SIDE = 0.02
_EPS = 1e-4


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_geometry(settings: Optional[dict]) -> dict:
    """The crop and angle an edit set describes, clamped, or ``{}`` for none.

    A crop needs all four numbers; three of them are not a rectangle and are
    dropped. A rectangle that is the whole frame is not a crop.
    """
    if not settings:
        return {}
    out: dict = {}

    angle = _number(settings.get("angle"))
    if angle is not None:
        angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))
        if abs(angle) >= MIN_ANGLE:
            out["angle"] = round(angle, 2)

    values = [_number(settings.get(k)) for k in ("crop_x", "crop_y", "crop_w", "crop_h")]
    if all(v is not None for v in values):
        x, y, w, h = values
        w = max(MIN_SIDE, min(1.0, w))
        h = max(MIN_SIDE, min(1.0, h))
        x = max(0.0, min(1.0 - w, x))
        y = max(0.0, min(1.0 - h, y))
        if not (x < _EPS and y < _EPS and w > 1 - _EPS and h > 1 - _EPS):
            out.update(
                crop_x=round(x, 5), crop_y=round(y, 5),
                crop_w=round(w, 5), crop_h=round(h, 5),
            )
    return out


def _rotated_fits(box: tuple, angle: float, width: int, height: int) -> bool:
    """True when every corner of ``box`` lies on picture after turning by ``angle``."""
    x, y, w, h = box
    cx, cy = width / 2.0, height / 2.0
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        dx, dy = px * width - cx, py * height - cy
        # Back to where that point was before the turn.
        sx = cos * dx + sin * dy
        sy = -sin * dx + cos * dy
        if abs(sx) > width / 2.0 + 0.5 or abs(sy) > height / 2.0 + 0.5:
            return False
    return True


def fit_box(box: tuple, angle: float, width: int, height: int) -> tuple:
    """Pull a crop rectangle back inside the picture that survives a turn.

    The rectangle keeps its shape and its centre where it can, and shrinks until
    its corners are on picture. One whose centre is itself off the picture is
    moved to the middle first. ``angle`` of 0 changes nothing.
    """
    x, y, w, h = box
    if abs(angle) < MIN_ANGLE or _rotated_fits(box, angle, width, height):
        return box

    def scaled(cx: float, cy: float, s: float) -> tuple:
        return (cx - w * s / 2.0, cy - h * s / 2.0, w * s, h * s)

    cx, cy = x + w / 2.0, y + h / 2.0
    for centre in ((cx, cy), (0.5, 0.5)):
        if not _rotated_fits(scaled(*centre, 0.01), angle, width, height):
            continue
        low, high = 0.01, 1.0
        for _ in range(30):
            mid = (low + high) / 2.0
            if _rotated_fits(scaled(*centre, mid), angle, width, height):
                low = mid
            else:
                high = mid
        return scaled(*centre, low)
    return (0.5 - w * 0.005, 0.5 - h * 0.005, w * 0.01, h * 0.01)


def box_of(settings: Optional[dict], width: int, height: int) -> tuple:
    """The kept rectangle ``(x, y, w, h)`` in fractions, after fitting to the turn."""
    geo = normalize_geometry(settings)
    box = (
        geo.get("crop_x", 0.0), geo.get("crop_y", 0.0),
        geo.get("crop_w", 1.0), geo.get("crop_h", 1.0),
    )
    return fit_box(box, geo.get("angle", 0.0), width, height)


def apply_geometry(image: Image.Image, settings: Optional[dict]) -> Image.Image:
    """Turn and crop a finished picture as an edit says; unchanged when it says nothing."""
    geo = normalize_geometry(settings)
    if not geo:
        return image
    width, height = image.size
    angle = geo.get("angle", 0.0)
    if angle:
        # PIL turns counter-clockwise; the stored angle is clockwise.
        image = image.rotate(-angle, resample=Image.BICUBIC, expand=False)
    x, y, w, h = box_of(geo, width, height)
    left = int(round(x * width))
    top = int(round(y * height))
    right = max(left + 1, min(width, int(round((x + w) * width))))
    bottom = max(top + 1, min(height, int(round((y + h) * height))))
    if (left, top, right, bottom) == (0, 0, width, height):
        return image
    return image.crop((left, top, right, bottom))


def source_point(settings: Optional[dict], u: float, v: float,
                 width: int, height: int) -> tuple:
    """Where a point of the *finished* picture came from, in the original frame.

    ``(u, v)`` are fractions of the cropped picture; the result is fractions of
    the uncropped, unturned frame. A loupe looks at the source pixels, so a click
    on a levelled, cropped preview has to be walked back through both steps.
    """
    geo = normalize_geometry(settings)
    x, y, w, h = box_of(geo, width, height)
    px = (x + max(0.0, min(1.0, u)) * w) * width
    py = (y + max(0.0, min(1.0, v)) * h) * height
    angle = geo.get("angle", 0.0)
    if angle:
        cx, cy = width / 2.0, height / 2.0
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        dx, dy = px - cx, py - cy
        px, py = cx + cos * dx + sin * dy, cy - sin * dx + cos * dy
    return (max(0.0, min(1.0, px / width)), max(0.0, min(1.0, py / height)))


def shown_point(settings: Optional[dict], sx: float, sy: float,
                width: int, height: int) -> tuple:
    """Where a point of the original frame lands in the *finished* picture: the
    way back of :func:`source_point`. Fractions of the cropped picture, outside
    0..1 when the crop left the point out."""
    geo = normalize_geometry(settings)
    x, y, w, h = box_of(geo, width, height)
    px, py = sx * width, sy * height
    angle = geo.get("angle", 0.0)
    if angle:
        cx, cy = width / 2.0, height / 2.0
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        dx, dy = px - cx, py - cy
        px, py = cx + cos * dx - sin * dy, cy + sin * dx + cos * dy
    return ((px / width - x) / w, (py / height - y) / h)


def _corners(box: tuple) -> list:
    x, y, w, h = box
    return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]


def _bounds(points: list) -> tuple:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    left, top = max(0.0, min(xs)), max(0.0, min(ys))
    right, bottom = min(1.0, max(xs)), min(1.0, max(ys))
    return (left, top, right - left, bottom - top)


def shown_box(settings: Optional[dict], box: tuple, width: int, height: int) -> Optional[tuple]:
    """A rectangle of the original frame (a face found in it) as it sits in the
    finished picture: the box around its turned corners, cut to the crop.
    ``None`` when the crop left it out."""
    left, top, w, h = _bounds([shown_point(settings, px, py, width, height) for px, py in _corners(box)])
    return None if w <= 0.002 or h <= 0.002 else (left, top, w, h)


def source_box(settings: Optional[dict], box: tuple, width: int, height: int) -> tuple:
    """A rectangle drawn on the finished picture as a rectangle of the original
    frame: the box around its corners walked back through crop and turn."""
    return _bounds([source_point(settings, px, py, width, height) for px, py in _corners(box)])
