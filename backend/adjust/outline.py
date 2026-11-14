"""
Lens blur with its edge taken from an outline (adjust.depth, adjust.objects).

The depth map is about 1000 px across and soft by ten or twenty of its pixels where one
thing stands in front of another. On a large frame that band is a hundred pixels wide, and
there an edge is given a distance between the near thing's and the far one's: the edge of
a person goes soft, a ring of the wall beside them stays sharp, and their colour is
smeared into the blur.

Segment Anything draws a far finer outline of one thing. With one, the map is split in
two: the near thing's depth (``inside``), carried out from where the map is sure it is the
thing, and the rest's (``outside``), carried in from where it is sure it is not. Each pixel
of the picture is then shared between the two by the outline, so an edge pixel that is half
person is half in the person's layer and half in the wall's, and not wholly in a layer
between them. Nothing in the photograph is invented: the blur is the same blur, cut where
the thing really ends.
"""

from typing import Optional

import numpy as np

# How much of a candidate must lie near the distance clicked for it to hold together.
_HOLDS = 0.8
_NEAR = 0.15
# Above this sigma a blur is worked at reduced size: it is smooth, and the cost of a
# Gaussian grows with its width.
_WORKING_SIGMA = 8.0


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    if sigma <= _WORKING_SIGMA:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    h, w = x.shape
    shrink = sigma / _WORKING_SIGMA
    small = (max(1, round(w / shrink)), max(1, round(h / shrink)))
    out = cv2.resize(x, small, interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), _WORKING_SIGMA * small[0] / max(w / shrink, 1e-6),
                           borderType=cv2.BORDER_REFLECT)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def _carried(depth: np.ndarray, sure: np.ndarray) -> np.ndarray:
    """``depth`` where ``sure``, and everywhere else carried out from there: a Gaussian's
    weighted mean of the sure pixels nearby, wider until every pixel has one."""
    weight = sure.astype(np.float32)
    out = np.where(sure, depth, 0.0).astype(np.float32)
    filled = sure.copy()
    held = depth.astype(np.float32) * weight
    sigma = 2.0
    longest = max(depth.shape)
    while not filled.all():
        num = _gauss(held, sigma)
        den = _gauss(weight, sigma)
        ok = ~filled & (den > 1e-6)
        out[ok] = num[ok] / den[ok]
        filled |= ok
        if sigma > 4 * longest:
            # Nothing can reach further: what is left takes the sure pixels' mean.
            out[~filled] = float(depth[sure].mean())
            break
        sigma *= 2.0
    return np.clip(out, 0.0, 1.0)


def sides(depth: np.ndarray, matte: np.ndarray) -> Optional[tuple]:
    """``(inside, outside)``: the outlined thing's depth and the rest's, each over the whole
    map, from ``depth`` (0 far to 1 near) and ``matte`` (the outline, 0..1, any size).
    ``None`` when the outline has no edge: nothing in it, or nothing out of it."""
    import cv2

    h, w = depth.shape
    if matte.shape != (h, w):
        matte = cv2.resize(matte.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    inner, outer = matte > 0.5, matte < 0.5
    if not inner.any() or not outer.any():
        return None
    # The band between the two, either side of the outline, is where the map cannot be trusted.
    r = max(3, round(max(h, w) / 50))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sure_in = cv2.erode(inner.astype(np.uint8), disc).astype(bool)
    sure_out = cv2.erode(outer.astype(np.uint8), disc).astype(bool)
    if not sure_in.any():
        sure_in = inner            # a thin thing: all of it
    if not sure_out.any():
        sure_out = outer
    return _carried(depth, sure_in), _carried(depth, sure_out)


def pick(logits: np.ndarray, scores: np.ndarray, depth: np.ndarray, at: float) -> int:
    """Which of SAM's candidates to take for one click: the largest that holds together at
    one distance (most of it within reach of ``at``, the map's value at the click), so a
    click on a shirt takes the person and not the person and the floor. ``depth`` is the
    map at the logits' size. The model's best when none holds together."""
    from adjust.objects import chosen

    best, size = None, 0
    for i, logit in enumerate(logits):
        inside = logit > 0
        n = int(inside.sum())
        if not n:
            continue
        near = float((np.abs(depth[inside] - at) <= _NEAR).mean())
        if near >= _HOLDS and n > size:
            best, size = i, n
    return best if best is not None else chosen(scores, None)
