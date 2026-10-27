"""
Upright: the transform and the level that straighten a frame's own lines.

Lightroom's Upright finds the lines in a picture and corrects the perspective so
that its verticals are vertical, its horizon level. This does the same, and
answers as the sliders: the ``tf_vertical`` tilt (adjust.lens) and the ``angle``
of the level (adjust.geometry), so what it did can be seen and changed.

  vertical   the lines within 25° of vertical (posts, door frames, a building's
             edges, people standing) made vertical: a tilt and a level, the tilt
             searched over the slider's whole range and the level worked out for
             each tilt.
  level      the lines within 15° of horizontal (a horizon, a stage edge, a
             table) made level: a level only.

The lines are found on the frame as the develop view shows it, its lens
corrected when the edit corrects it (OpenCV's line segment detector, with
lines through busy texture let go). Each line counts by its length, and lines that disagree with the
rest by more than a few degrees (a leaning flagpole, a diagonal railing) are let
go before the answer is settled. With too few lines it says so rather than
guessing.
"""

import math
from typing import Optional

import numpy as np

from adjust.lens import homography

VERTICAL_WITHIN = 25.0
LEVEL_WITHIN = 15.0
# The least length of line, as a share of the frame's height, worth asking: less
# is a crowd, a texture, a letter on a sign.
MIN_LINE = 0.05
# The most of the picture around a line that may be edges (a 25-pixel box) before
# the line is texture rather than a thing: leaves were 0.28, a building 0.19.
MAX_BUSY = 0.24
# How much line (all of it together, as a share of the frame's height) it takes to
# trust an answer.
MIN_EVIDENCE = 1.2
MAX_LEVEL = 15.0
# How far a line may disagree with the rest, in degrees, before it is let go.
_OUTLIER = (8.0, 3.0)
# In the tilt search, the most one line's disagreement can cost, in degrees.
_CAP = 4.0
# When the lines it set upright are believed: plenty of them (frame heights of
# line) agreeing loosely, or enough of them agreeing closely (degrees).
_PLENTY, _LOOSE = 3.0, 1.1
_TIGHT = 1.0


def find_lines(enc: np.ndarray) -> np.ndarray:
    """The straight lines in a display-encoded picture (H, W, 3, 0..1), as an
    (N, 4) array of their ends in pixels: x1, y1, x2, y2.

    OpenCV's line segment detector, not edges and a Hough transform: on a street
    with a tree in it the Hough transform found 261 "lines" in the leaves against
    16 on the building beside them, and they decided the answer. Lines that still
    run through busy texture (leaves, gravel, a crowd) are let go too.
    """
    import cv2

    h, w = enc.shape[:2]
    scale = min(1.0, 1200.0 / max(h, w))
    grey = (np.clip(enc @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0, 1) * 255).astype(np.uint8)
    if scale < 1.0:
        grey = cv2.resize(grey, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    found = cv2.createLineSegmentDetector().detect(grey)[0]
    if found is None:
        return np.zeros((0, 4), np.float64)
    lines = found.reshape(-1, 4).astype(np.float64)
    gh, gw = grey.shape
    long = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]) >= MIN_LINE * gh
    lines = lines[long]
    if len(lines):
        edges = cv2.Canny(grey, 60, 160)
        busy = cv2.boxFilter((edges > 0).astype(np.float32), -1, (25, 25))
        t = np.linspace(0.0, 1.0, 9)
        xs = np.clip((lines[:, [0]] + t * (lines[:, [2]] - lines[:, [0]])).astype(int), 0, gw - 1)
        ys = np.clip((lines[:, [1]] + t * (lines[:, [3]] - lines[:, [1]])).astype(int), 0, gh - 1)
        lines = lines[busy[ys, xs].mean(axis=1) <= MAX_BUSY]
    return lines / scale


def _normalised(lines: np.ndarray, w: int, h: int) -> np.ndarray:
    radius = math.hypot(w / 2.0, h / 2.0)
    out = lines.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - w / 2.0) / radius
    out[:, [1, 3]] = (out[:, [1, 3]] - h / 2.0) / radius
    return out


def _through(h_inv: Optional[np.ndarray], pts: np.ndarray) -> np.ndarray:
    """Points (N, 2) of the frame to the transformed picture."""
    if h_inv is None:
        return pts
    x, y = pts[:, 0], pts[:, 1]
    d = h_inv[2, 0] * x + h_inv[2, 1] * y + h_inv[2, 2]
    return np.stack([(h_inv[0, 0] * x + h_inv[0, 1] * y + h_inv[0, 2]) / d,
                     (h_inv[1, 0] * x + h_inv[1, 1] * y + h_inv[1, 2]) / d], axis=1)


def _settle(angles: np.ndarray, weights: np.ndarray) -> tuple:
    """The weighted mean of ``angles`` (degrees) once the lines far from it are
    let go, and the weighted mean squared disagreement that is left."""
    keep = np.ones(len(angles), bool)
    mean = float(np.average(angles, weights=weights))
    for limit in _OUTLIER:
        keep = np.abs(angles - mean) <= limit
        if weights[keep].sum() <= 0:
            return mean, float("inf"), keep
        mean = float(np.average(angles[keep], weights=weights[keep]))
    spread = float(np.average((angles[keep] - mean) ** 2, weights=weights[keep]))
    return mean, spread, keep


def _vertical_angles(lines: np.ndarray, h_inv) -> tuple:
    a = _through(h_inv, lines[:, :2])
    b = _through(h_inv, lines[:, 2:])
    dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    flip = dy < 0
    dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
    return np.degrees(np.arctan2(dx, dy)), np.hypot(dx, dy)


def upright(enc: np.ndarray, mode: str = "vertical") -> dict:
    """What straightens this picture's lines: ``{"found": True, "tf_vertical",
    "angle", "lines"}`` (vertical), ``{"found": True, "angle", "lines"}``
    (level), or ``{"found": False, "reason"}``."""
    h, w = enc.shape[:2]
    lines = find_lines(enc)
    if not len(lines):
        return {"found": False, "reason": "no straight lines in this picture"}
    n = _normalised(lines, w, h)
    dx, dy = n[:, 2] - n[:, 0], n[:, 3] - n[:, 1]
    height = h / math.hypot(w / 2.0, h / 2.0)

    if mode == "level":
        flip = dx < 0
        ang = np.degrees(np.arctan2(np.where(flip, -dy, dy), np.where(flip, -dx, dx)))
        near = np.abs(ang) <= LEVEL_WITHIN
        length = np.hypot(dx, dy)
        if length[near].sum() < MIN_EVIDENCE * height:
            return {"found": False, "reason": "too few level lines (a horizon, a stage edge) to go by"}
        mean, _, keep = _settle(ang[near], length[near])
        # Turned clockwise by the level, a line's slope rises by the same angle.
        angle = float(np.clip(-mean, -MAX_LEVEL, MAX_LEVEL))
        return {"found": True, "angle": round(angle, 2), "lines": int(keep.sum())}

    ang0 = np.degrees(np.arctan2(np.where(dy < 0, -dx, dx), np.abs(dy)))
    near = np.abs(ang0) <= VERTICAL_WITHIN
    length0 = np.hypot(dx, dy)
    if length0[near].sum() < MIN_EVIDENCE * height:
        return {"found": False, "reason": "too few vertical lines (posts, doors, a building's edges) to go by"}
    lines = n[near]

    def cost(v):
        hm = homography({"tf_vertical": v})
        h_inv = None if hm is None else np.linalg.inv(np.asarray(hm))
        ang, length = _vertical_angles(lines, h_inv)
        # Every tilt is judged on all the lines, each one's disagreement with the
        # rest capped (a line that is no vertical costs the same at any tilt), so
        # the costs of two tilts compare. The level is the lines' robust mean.
        mean = float(np.median(ang))
        for _ in range(3):
            near = np.abs(ang - mean) <= _CAP
            if length[near].sum() <= 0:
                break
            mean = float(np.average(ang[near], weights=length[near]))
        r2 = np.minimum((ang - mean) ** 2, _CAP ** 2)
        # A small pull towards no change, so a frame whose lines are as straight
        # one way as another is left as it is.
        return float(np.average(r2, weights=length)) + 5e-6 * v * v, mean

    grid = np.linspace(-100, 100, 81)
    best = min(grid, key=lambda v: cost(v)[0])
    fine = np.clip(np.arange(best - 2.5, best + 2.51, 0.25), -100, 100)
    best = float(min(fine, key=lambda v: cost(v)[0]))
    spread, mean = cost(best)
    hm = homography({"tf_vertical": best})
    ang, length = _vertical_angles(lines, None if hm is None else np.linalg.inv(np.asarray(hm)))
    _, rest, keep = _settle(ang, length)
    # Believed only when the lines it set upright agree with each other and there
    # is enough of them: a portrait's "verticals" are arms, folds and hair, which
    # no tilt sets upright together. And never at the end of the range, which is
    # the search running out rather than an answer.
    agreed = float(length[keep].sum()) / height
    spread_deg = math.sqrt(rest)
    # Measured on the library (lines from the segment detector): buildings, an
    # auditorium and a street of houses agree within 0.4 to 0.9° once set upright;
    # a portrait, a band group under palm trees and a guard huddle 1.3 to 1.7°,
    # their "verticals" being arms, trunks and flagpoles.
    believed = ((agreed >= _PLENTY and spread_deg <= _LOOSE)
                or (agreed >= MIN_EVIDENCE and spread_deg <= _TIGHT))
    if not believed or abs(best) >= 99:
        return {"found": False, "reason": "the vertical lines here do not agree on a correction "
                                          "(people and trees are not verticals)",
                "tf_vertical": int(round(best)), "agreed": round(agreed, 2), "spread": round(math.sqrt(rest), 2)}
    # Turned clockwise by the level, a vertical line leans by the same angle the
    # other way; the level that sets the lines upright is their mean lean.
    angle = float(np.clip(mean, -MAX_LEVEL, MAX_LEVEL))
    return {"found": True, "tf_vertical": int(round(best)), "angle": round(angle, 2),
            "lines": int(keep.sum()), "agreed": round(agreed, 2), "spread": round(math.sqrt(rest), 2)}
