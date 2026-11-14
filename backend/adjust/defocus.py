"""
Blur where a mask reaches (adjust.masks, its ``blur`` control, 0..100).

Not a smear: the picture is put out of focus as a lens would, through a disc in linear
light, so a bright point opens into a disc and the dark round it stays dark (adjust.depth
does the same by distance). Only what the mask reaches is spread: a person left out of a
background mask is not blurred into the background round them, which a plain blur of the
whole picture under the mask would do.

At 100 the disc is as wide as lens blur's widest (adjust.depth.MAX_BLUR of the frame's
longer side), measured on the whole photograph, so a preview, a loupe's patch and an
export blur alike.
"""

import json
from typing import Callable, Optional

import numpy as np


def wanted(masks: Optional[list]) -> list:
    """The masks that blur something."""
    from adjust.masks import active

    return [m for m in active(masks) if m.get("blur")]


def radius(amount: float, longest: float) -> float:
    """The disc's radius in pixels for ``amount`` (0..100) on a frame whose whole picture is
    ``longest`` pixels on its longer side."""
    from adjust.depth import MAX_BLUR

    return float(amount) / 100.0 * MAX_BLUR * longest


def reach(masks: Optional[list], longest: int) -> int:
    """How far, in pixels, the blur of ``masks`` can carry a point: what a part of a frame
    must be cut wider by, to be blurred like the whole."""
    most = max((m["blur"] for m in wanted(masks)), default=0)
    return int(np.ceil(radius(most, longest))) + 1 if most else 0


def pixels(rgb: np.ndarray, window: tuple, masks: list, weight: Callable) -> np.ndarray:
    """``rgb``, the part ``window`` of a frame, blurred where each of ``masks`` reaches.
    ``weight(mask)`` is the mask's weight over these pixels, or ``None`` (not here yet)."""
    from adjust.depth import _MIN_RADIUS, _layer

    h, w = rgb.shape[:2]
    left, top, right, bottom = window
    longest = max(w / max(right - left, 1e-9), h / max(bottom - top, 1e-9))
    out = rgb
    for m in wanted(masks):
        reach_here = weight(m)
        if reach_here is None or not reach_here.any():
            continue
        r = radius(m["blur"], longest)
        if r < _MIN_RADIUS:
            continue
        mine = np.ascontiguousarray(reach_here, dtype=np.float32)
        light, cover = _layer(out, mine, r)
        spread = light / np.maximum(cover, 1e-3)[..., None]
        out = (out + mine[..., None] * (spread - out)).astype(np.float32)
    return out


def defocused(frame, masks: list, weight: Callable):
    """``frame`` blurred where its masks say, as a frame of its own (sharing its mattes);
    ``frame`` itself when none blurs. ``weight(frame, mask)`` is a mask's weight over a
    frame's pixels (adjust.render.mask_reach). Kept on the frame for the masks that made it."""
    from adjust.render import LinearFrame

    masks = wanted(masks)
    if not masks:
        return frame
    key = json.dumps(masks, sort_keys=True)
    kept = getattr(frame, "_defocused", None)
    if kept is not None and kept[0] == key:
        return kept[1]
    rgb = pixels(frame.rgb, frame.window, masks, lambda m: weight(frame, m))
    if rgb is frame.rgb:
        return frame
    out = LinearFrame(rgb, look=frame.look, window=frame.window, scale=frame.scale,
                      anchor_of=frame.anchor_of or frame, lens=frame.lens, wb=frame.wb,
                      source_key=frame.source_key)
    out._mattes = frame._mattes
    frame._defocused = (key, out)
    return out
