"""
Match a look: make a frame look like a reference photograph.

Unlike a learned style (adjust.styles), which is learned from RAW+JPEG pairs of
one camera, the reference here is any picture: a frame from the library as it
was edited, or a file from elsewhere. Its look is read the way a printer would
match a proof, by tone and colour — what share of the picture sits at each
level of each channel — and handed back as ordinary controls, so the result can
be seen, changed and undone like any edit:

  curve_r, curve_g, curve_b   each channel's levels brought to the reference's,
                              which carries its contrast, its black and white
                              points (a faded, matte print), and its colour at
                              each level (a warm highlight, a teal shadow)
  saturation                  what is left to match after the curves
  mono, curve                 instead, for a reference in black and white

Pictures of different scenes have different contents, not just different
looks: a snowfield is bright because it is snow. So the match is of
percentiles, not whole histograms, and each curve is kept within reach of the
diagonal (MAX_SHIFT), which leaves the frame's own subject where it is. The
develop view scales the result by an amount (0..100) toward no change.
"""

from typing import Optional

import numpy as np

# Where the levels are compared: dense at the ends, where black and white
# points and fades live, sparse in the middle.
PERCENTILES = (0.5, 2, 6, 15, 30, 50, 70, 85, 94, 98, 99.5)
# The furthest a level is moved, out of 255: a look, not a different picture.
MAX_SHIFT = 70
# Below this mean saturation (0..1) a reference is black and white.
MONO_SATURATION = 0.04
MAX_SATURATION_CHANGE = 60


def _saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation of an 8-bit picture, 0..1, of its non-dark pixels."""
    x = rgb.reshape(-1, 3).astype(np.float32)
    hi, lo = x.max(axis=1), x.min(axis=1)
    lit = hi > 24
    if not lit.any():
        return 0.0
    return float(np.mean((hi[lit] - lo[lit]) / hi[lit]))


def _chroma(rgb: np.ndarray) -> float:
    """Mean distance from grey, 8-bit units: what the saturation slider scales."""
    x = rgb.reshape(-1, 3).astype(np.float32)
    luma = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return float(np.mean(np.abs(x - luma[:, None])))


def _curve(source: np.ndarray, target: np.ndarray) -> list:
    """Points taking ``source``'s levels to ``target``'s, at PERCENTILES."""
    xs = np.percentile(source, PERCENTILES)
    ys = np.percentile(target, PERCENTILES)
    # Beyond the last measured level, the same offset carried to the ends.
    xs = np.concatenate([[0.0], xs, [255.0]])
    ys = np.concatenate([[max(0.0, ys[0] - xs[1])], ys, [min(255.0, ys[-1] + 255.0 - xs[-2])]])
    ys = np.clip(ys, xs - MAX_SHIFT, xs + MAX_SHIFT)
    points, last_x = [], -1
    top = 0.0
    for x, y in zip(xs, ys):
        x = int(round(x))
        if x <= last_x:            # flat stretch of the source: one point for it
            continue
        top = max(top, float(y))   # never turning back down
        points.append([x, int(round(min(255.0, top)))])
        last_x = x
    return points


def _through(rgb: np.ndarray, curves: list) -> np.ndarray:
    out = np.empty(rgb.shape, np.float32)
    for c, points in enumerate(curves):
        px, py = zip(*points)
        out[..., c] = np.interp(rgb[..., c].astype(np.float32), px, py)
    return out


def match(frame: np.ndarray, reference: np.ndarray) -> dict:
    """The controls that bring ``frame`` (8-bit RGB, as the frame is now
    developed, before any curve or saturation) to the look of ``reference``
    (8-bit RGB, any size)."""
    if _saturation(reference) < MONO_SATURATION:
        grey_ref = reference.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        grey = frame.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return {"mono": 1, "curve": _curve(grey, grey_ref)}

    curves = [_curve(frame[..., c], reference[..., c]) for c in range(3)]
    after = _chroma(_through(frame, curves))
    wanted = _chroma(reference)
    out = {"curve_r": curves[0], "curve_g": curves[1], "curve_b": curves[2]}
    if after > 1.0:
        change = int(round((wanted / after - 1.0) * 100))
        change = max(-MAX_SATURATION_CHANGE, min(MAX_SATURATION_CHANGE, change))
        if abs(change) >= 3:
            out["saturation"] = change
    return out


def scaled(found: dict, amount: float) -> dict:
    """``found`` taken ``amount`` (0..1) of the way from no change."""
    out = {}
    for key, value in found.items():
        if key.startswith("curve"):
            out[key] = [[x, int(round(x + amount * (y - x)))] for x, y in value]
        elif key == "saturation":
            out[key] = int(round(value * amount))
        else:
            out[key] = value
    return out


def describe(found: dict) -> Optional[str]:
    if found.get("mono"):
        return "black and white, with the reference's tones"
    parts = ["the reference's tone and colour by channel"]
    if found.get("saturation"):
        parts.append(f"saturation {found['saturation']:+d}")
    return ", ".join(parts)
