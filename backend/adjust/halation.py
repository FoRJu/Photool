"""
Halation and bloom: bright light spreading as it does on film (the Effects section).

- **Halation.** Light goes through the emulsion, bounces off the back of the film base
  and exposes it again from behind, a little way out from where it came in. The
  red-sensitive layer lies deepest and takes most of it, so the spread light is
  red-orange: the fringe round lamps and neon on a stock with no anti-halation backing.
- **Bloom.** Light scattered in the lens and the emulsion: a wider, softer glow in the
  light's own colour, which takes the hard edge off every highlight.

Both are worked out in scene-linear light on the frame before its light and tone, and
added to it, never taken away: a lamp is not dimmed to pay for its glow. What goes above
white is rolled off by the display stage that follows, like any other light, and
Highlights, the curve, a style, black and white, the grain and the vignette all treat the
glow as part of the picture.

What counts as bright is judged on the light as exposed by the edit's own Exposure. A
mask's exposure, the white balance and a set's match are not counted: the glow knows none
of them.

The spread is measured on the whole photograph (its longer side), so a preview, an
export and a loupe's patch spread the light equally far across the picture. A loupe's
patch takes its glow from the preview frame (see ``pixels``): a lamp outside the patch
still lights it, and the glow is smooth enough to be sampled from a smaller picture.

Nothing here needs light above white to work, and nothing assumes there is none: a JPEG's
white (1.0) glows at every threshold.
"""

import json
from typing import Optional

import numpy as np

KEYS = ("halation", "halation_colour", "bloom", "glow_size", "glow_threshold")

# What counts as bright is the brightest of a pixel's three channels, not its luminance:
# film's layers are exposed each by its own colour, so a red neon sign (luminance a
# quarter of white's) still floods the red layer and glows. Luminance would leave it dark.
# Where the glow begins at Glow threshold 0, in linear light: a little under middle of
# the way to white, so lamps, bright windows and sky glow and a mid-grey wall does not.
# Each 100 of the slider doubles or halves it.
THRESHOLD = 0.45
# What a sensor clipped was brighter than it recorded, and a lamp on film is many stops
# over: the brightest light is counted this many times over as a source.
HOT = 2.5
# The spread's unit, as a fraction of the picture's longer side: 0.0025 of 36 mm is
# 90 µm, about what a 35 mm negative shows. Glow size moves it by 1.5 stops each way.
SCALE = 0.0025
SIZE_STOPS = 1.5
# The halo is the red-sensitive layer exposed again, so its source is one plane
# weighted to red.
HALO_SOURCE = np.array([0.60, 0.35, 0.05], dtype=np.float32)
# The halo's spread: a core and a tail, (weight, sigma in units of the scale).
HALO_SPREAD = ((0.65, 1.0), (0.35, 3.0))
# The halo's colour: film's red-orange at Halation colour 0, deep red at -100, nearly
# neutral at +100.
HALO_TINT = np.array([1.0, 0.28, 0.04], dtype=np.float32)
HALO_TINT_DEEP = np.array([1.0, 0.08, 0.0], dtype=np.float32)
HALO_TINT_PALE = np.array([1.0, 0.75, 0.55], dtype=np.float32)
# The halo's strength at Halation 100 (0.45 at first: at 40 the fringe was too faint to
# read on a preview); the slider is eased in by this power, so the
# first half of it is gentle and 30 to 40 reads as film.
HALO_STRENGTH = 0.6
# Bloom's spread, wider and softer than the halo, in the light's own colour.
BLOOM_SPREAD = ((0.6, 4.0), (0.4, 14.0))
# Bloom's strength at Bloom 100.
BLOOM_STRENGTH = 0.30
EASE = 1.5
# A Gaussian this wide or wider (in pixels) is worked out on a smaller picture: it is
# smooth, and the wide bloom on a full-size export is then a blur of a small one.
SHRINK_AT = 2.5


def wanted(clean: Optional[dict]) -> bool:
    """Whether the edit has any glow to lay on."""
    return bool(clean) and bool(clean.get("halation") or clean.get("bloom"))


def _longest(shape: tuple, window: tuple) -> float:
    """The longer side of the whole picture, in pixels of a frame of ``shape`` that is the
    part ``window`` of it (as adjust.defocus.pixels works it out)."""
    h, w = shape[:2]
    left, top, right, bottom = window
    return max(w / max(right - left, 1e-9), h / max(bottom - top, 1e-9))


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = (x - np.float32(lo)) * np.float32(1.0 / max(hi - lo, 1e-6))
    np.clip(t, 0.0, 1.0, out=t)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _shrunk(plane: np.ndarray, f: int) -> np.ndarray:
    """``plane`` made ``f`` times smaller by averaging (``plane`` itself at 1)."""
    import cv2

    if f <= 1:
        return np.ascontiguousarray(plane, dtype=np.float32)
    h, w = plane.shape[:2]
    return cv2.resize(np.ascontiguousarray(plane, dtype=np.float32),
                      (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)


def _gaussian(plane: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """``plane`` (H, W) or (H, W, C) blurred by a Gaussian of ``sx`` by ``sy`` pixels, at
    its own size. A wide one is worked out on a smaller copy: it is smooth, and the wide
    bloom on a full-size export is then a blur of a small picture."""
    import cv2

    if max(sx, sy) < 0.3:
        return plane
    h, w = plane.shape[:2]
    f = max(1, int(min(sx, sy) // SHRINK_AT))
    f = min(f, max(1, min(h, w) // 4))
    small = _shrunk(plane, f)
    sh, sw = small.shape[:2]
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sx * sw / w, sigmaY=sy * sh / h,
                             borderType=cv2.BORDER_REPLICATE)
    return small if f == 1 else cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _tint(colour: float) -> np.ndarray:
    c = float(np.clip(colour, -100.0, 100.0)) / 100.0
    end = HALO_TINT_DEEP if c < 0 else HALO_TINT_PALE
    return (HALO_TINT + abs(c) * (end - HALO_TINT)).astype(np.float32)


def _spreads(rgb: np.ndarray, window: tuple, clean: dict) -> dict:
    """What the glow's amounts and colour scale, for the edit's exposure, threshold and
    size: the source made small and, as they are asked for, its two spreads at that size."""
    exposure = float(clean.get("exposure") or 0.0)
    threshold = float(clean.get("glow_threshold") or 0.0)
    size = float(clean.get("glow_size") or 0.0)

    rgb = np.asarray(rgb, dtype=np.float32)
    # Only the exposure decides what is bright; the picture itself is not lifted here.
    y = rgb.max(axis=2) * np.float32(2.0 ** exposure)
    lo = THRESHOLD * 2.0 ** (threshold / 100.0)
    hi = min(2.0 * lo, max(1.0, lo * 1.1))
    wt = _smoothstep(lo, hi, y)
    wt *= 1.0 + (HOT - 1.0) * wt
    source = rgb * wt[..., None]

    s = SCALE * _longest(rgb.shape, window) * 2.0 ** (SIZE_STOPS * size / 100.0)
    # Every spread is at least the scale wide, so all of it is worked out on the source
    # made smaller by as much as the finest allows, and brought back to size once.
    h, w = rgb.shape[:2]
    f = max(1, min(int(s // SHRINK_AT), max(1, min(h, w) // 4)))
    small = _shrunk(source, f)
    sh, sw = small.shape[:2]
    return {"small": small, "k": (s * sw / w, s * sh / h), "size": (w, h)}


def _spread(spreads: dict, which: str) -> np.ndarray:
    if which not in spreads:
        small, (kx, ky) = spreads["small"], spreads["k"]
        if which == "halo":
            h_src = small @ HALO_SOURCE
            spreads[which] = sum(weight * _gaussian(h_src, k * kx, k * ky) for weight, k in HALO_SPREAD)
        else:
            spreads[which] = sum(weight * _gaussian(small, k * kx, k * ky) for weight, k in BLOOM_SPREAD)
    return spreads[which]


def _glow_of(spreads: dict, clean: dict) -> np.ndarray:
    h_amount = float(clean.get("halation") or 0.0)
    b_amount = float(clean.get("bloom") or 0.0)
    light = np.zeros_like(spreads["small"])
    if h_amount > 0:
        a_h = HALO_STRENGTH * (min(h_amount, 100.0) / 100.0) ** EASE
        light += (a_h * _spread(spreads, "halo"))[..., None] * _tint(clean.get("halation_colour") or 0.0)
    if b_amount > 0:
        a_b = BLOOM_STRENGTH * (min(b_amount, 100.0) / 100.0) ** EASE
        light += a_b * _spread(spreads, "bloom")
    w, h = spreads["size"]
    if light.shape[:2] != (h, w):
        import cv2

        light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
    return light.astype(np.float32)


def glow(rgb: np.ndarray, window: tuple, clean: dict) -> np.ndarray:
    """The light to be added to ``rgb`` (the part ``window`` of a picture, linear) for the
    edit ``clean``: the halo and the bloom, (H, W, 3) float32."""
    return _glow_of(_spreads(rgb, window, clean), clean)


def _key(clean: dict) -> str:
    return json.dumps([clean.get(k) for k in KEYS] + [clean.get("exposure", 0)])


def _kept(frame, clean: dict):
    """``(frame_out, glow)`` for ``frame``, made and kept on it if it is not there."""
    from adjust.render import LinearFrame

    key = _key(clean)
    kept = getattr(frame, "_lit", None)
    if kept is not None and kept[0] == key:
        return kept[1], kept[2]
    # Moving an amount or the colour only scales the spreads: kept by what makes them.
    spread_key = json.dumps([clean.get(k) for k in ("exposure", "glow_threshold", "glow_size")])
    kept = getattr(frame, "_spread", None)
    if kept is None or kept[0] != spread_key:
        kept = (spread_key, _spreads(frame.rgb, frame.window, clean))
        frame._spread = kept
    light = _glow_of(kept[1], clean)
    out = LinearFrame((frame.rgb + light).astype(np.float32), look=frame.look, window=frame.window,
                      scale=frame.scale, anchor_of=frame.anchor_of or frame, lens=frame.lens,
                      wb=frame.wb, source_key=frame.source_key)
    out._mattes = frame._mattes
    frame._lit = (key, out, light)
    return out, light


def lit(frame, clean: dict):
    """``frame`` with its glow on, as a frame of its own (sharing its mattes); ``frame``
    itself when the edit has none. Kept on the frame for the values that made it."""
    if not wanted(clean):
        return frame
    return _kept(frame, clean)[0]


def pixels(small, patch: np.ndarray, window: tuple, clean: dict) -> np.ndarray:
    """``patch`` (linear, the part ``window`` of the whole picture) with the glow of the
    frame ``small`` sampled over it and added: how a loupe's patch takes light from lamps
    that lie outside it."""
    import cv2

    if not wanted(clean):
        return patch
    light = _kept(small, clean)[1]
    h, w = patch.shape[:2]
    mh, mw = light.shape[:2]
    sl, st, sr, sb = small.window
    left, top, right, bottom = window
    # The patch's pixel centres as fractions of the whole picture, then as pixels of small.
    fx = left + (np.arange(w, dtype=np.float64) + 0.5) / w * (right - left)
    fy = top + (np.arange(h, dtype=np.float64) + 0.5) / h * (bottom - top)
    xs = ((fx - sl) / max(sr - sl, 1e-9) * mw - 0.5).astype(np.float32)
    ys = ((fy - st) / max(sb - st, 1e-9) * mh - 0.5).astype(np.float32)
    map_x = np.broadcast_to(xs[None, :], (h, w)).astype(np.float32)
    map_y = np.broadcast_to(ys[:, None], (h, w)).astype(np.float32)
    sampled = cv2.remap(np.ascontiguousarray(light), map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return (np.asarray(patch, dtype=np.float32) + sampled).astype(np.float32)
