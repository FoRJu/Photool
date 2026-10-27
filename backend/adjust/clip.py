"""
Clipped highlights, and the pink they turn into.

When a sensor clips, it clips one channel at a time. White balance then multiplies
the channels by different gains, so a pixel whose green has saturated still has a
red and a blue that went on climbing: the pixel comes out *redder and bluer than
the scene was*, which is magenta. Pull the exposure down and the cast gets darker
but stays: that is the pink in a flash-blown face, or on a sheet of music under a
stage light.

This module works in the **camera's own colour space**, before any conversion to
display primaries. That is the only place where "one channel clipped" is still
true: a colour matrix mixes a clipped channel into the other two, after which no
channel shows a clean plateau and none can be repaired on its own. Measured on a
Canon R6 II, whose red channel saturates below its green and blue, rebuilding in
sRGB left a clipped sheet of white paper salmon; the same rebuild in camera space
leaves it white.

For each pixel, only the channels that actually clipped are replaced. The ones
that did not still hold real data and are kept. The replacement takes:

  * its **colour** from nearby pixels that are bright but unclipped, which are
    most likely the same surface: the skin around a blown forehead, the paper
    around a blown patch of it. Dark neighbours are a different surface (the
    black music stand, the dark suit), and borrowing from them was what turned
    white paper beige. Where nothing bright is close, the colour is neutral.
  * its **brightness** from the data the pixel still has: the least brightness at
    which that colour agrees with every channel. A clipped channel only says
    "at least this much"; an unclipped one says "exactly this much". On the sheet
    of music, blue never clipped, so the paper is rebuilt as bright as blue says
    and the printed notes, which blue still records, come back with it.

A borrowed colour that would need the pixel to be far brighter than its clipped
channels show is the wrong colour (a warm neighbour lent to a white sheet), and
the least-saturated reading, neutral, is taken instead.

What this cannot do is invent detail where every channel clipped: there the
result is a smooth patch of the right colour. And it does not trust the file's
stated white level, because real sensors often saturate a little below it, which
is exactly the case where a threshold taken from the file misses the clipping.
"""

import numpy as np

from adjust.render import LUMA, _box, _resize_plane, in_strips, side_by_side

# A channel saturates where the data piles up at one value. A smooth bright
# surface has no pile — its brightest pixels are spread by noise — but a clipped
# plateau has a spike: this many times as many pixels in the very top sliver of
# the range as in the stretch just below it.
_SPIKE_TOP = 0.9995
_SPIKE_BAND = 0.99
_SPIKE_RATIO = 3.0
_SPIKE_MIN_SHARE = 1.5e-4

# A channel is clipped as it nears its saturation level.
_CLIP_START, _CLIP_FULL = 0.93, 0.985

# A pixel lends its colour when its brightest channel is at least this close to
# that channel's saturation level (and none has reached it). Bright neighbours are
# the same surface as the highlight they surround; dark ones are usually not.
_REFERENCE_LOW, _REFERENCE_HIGH = 0.30, 0.65
_FIELD_WORKING_SIZE = 512
_FIELD_RADII = (0.02, 0.06, 0.18, 0.5)  # of the working size, small to large
_FIELD_ENOUGH = 0.12

# How much brighter than its clipped channels a borrowed colour may claim the
# pixel to be before the colour is judged wrong and neutral is preferred.
_CONSISTENT, _INCONSISTENT = 1.15, 1.8

# The rebuilt brightness never claims more than this many times the level at
# which the colour would first have clipped.
_MAX_EXTRAPOLATION = 4.0
_FIELD_FLOOR = 0.05


def _smoothstep(edge0: float, edge1: float, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def saturation_levels(rgb: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    """Where each channel really saturates, in the frame's own units.

    ``nominal`` is the level the file's metadata implies. It is kept for a channel
    that shows no plateau, and replaced by the plateau where there is one.
    """
    levels = np.asarray(nominal, dtype=np.float64).copy()
    peaks = in_strips(lambda strip: strip.reshape(-1, 3).max(axis=0)[None], rgb).max(axis=0)
    # Each threshold in the frame's own precision, as a plain number compared
    # with a float32 channel is.
    tops = np.array([_SPIKE_TOP * float(p) for p in peaks], dtype=rgb.dtype)
    bands = np.array([_SPIKE_BAND * float(p) for p in peaks], dtype=rgb.dtype)

    def counted(strip):
        flat = strip.reshape(-1, 3)
        at_top = flat >= tops
        return np.stack([at_top.sum(axis=0), ((flat >= bands) & ~at_top).sum(axis=0)])[None]

    top_count, band_count = in_strips(counted, rgb).sum(axis=0)
    pixels = rgb.shape[0] * rgb.shape[1]
    for c in range(3):
        if peaks[c] <= 0.0:
            continue
        top = float(top_count[c]) / pixels
        band = float(band_count[c]) / pixels
        if top >= _SPIKE_MIN_SHARE and top > _SPIKE_RATIO * band:
            levels[c] = float(peaks[c])
    return levels


def channel_clip(rgb: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per channel: 0 well below its saturation level, 1 where it has reached it."""
    near = rgb / np.asarray(levels, dtype=np.float32)
    return _smoothstep(_CLIP_START, _CLIP_FULL, near).astype(np.float32)


def clip_weight(rgb: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """0 where no channel is near its saturation level, 1 where one has reached it."""
    return channel_clip(rgb, levels).max(axis=2)


def _colour_field(direction: np.ndarray, trust: np.ndarray) -> tuple[np.ndarray, float]:
    """The colour of the trusted pixels, carried into the places that have none.

    A normalised convolution at several scales: the nearest trusted colour where
    there is one close by, a wider neighbourhood where there is not, and neutral
    where there is nothing at all. Worked out on a small copy, which is returned
    with its scale; :func:`_sample` reads it at full-resolution positions.
    """
    h, w = trust.shape
    scale = min(1.0, _FIELD_WORKING_SIZE / float(max(h, w)))
    sw, sh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
    weight, *planes = side_by_side(
        lambda c: _resize_plane(trust if c is None else
                                in_strips(lambda d, t: d[..., c] * t, direction, trust),
                                (sw, sh), smooth=True),
        (None, 0, 1, 2))

    field = np.ones((sh, sw, 3), dtype=np.float32)           # neutral, in the end
    for fraction in reversed(_FIELD_RADII):                  # large first, small overrides
        radius = max(2, int(round(min(sw, sh) * fraction)))
        mass, *boxed = side_by_side(lambda p: _box(p, radius), [weight, *planes])
        level = np.clip(mass / _FIELD_ENOUGH, 0.0, 1.0)[..., None]
        local = np.stack(boxed, axis=-1) / np.maximum(mass, 1e-6)[..., None]
        field = field + level * (local - field)
    return field, (sw / float(w), sh / float(h))


def _sample(field: tuple, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """The small field read (bilinearly) at full-resolution pixel positions."""
    small, (fx, fy) = field
    sh, sw = small.shape[:2]
    x = np.clip((xs + 0.5) * fx - 0.5, 0.0, sw - 1.0)
    y = np.clip((ys + 0.5) * fy - 0.5, 0.0, sh - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), max(sw - 2, 0))
    y0 = np.minimum(np.floor(y).astype(np.int64), max(sh - 2, 0))
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    x1 = np.minimum(x0 + 1, sw - 1)
    y1 = np.minimum(y0 + 1, sh - 1)
    return (small[y0, x0] * (1 - tx) * (1 - ty) + small[y0, x1] * tx * (1 - ty)
            + small[y1, x0] * (1 - tx) * ty + small[y1, x1] * tx * ty)


def _unit_luma(colour: np.ndarray) -> np.ndarray:
    colour = np.maximum(colour, 0.0)
    colour = colour / np.maximum(colour @ LUMA, 1e-6)[..., None]
    return np.maximum(colour, _FIELD_FLOOR)


def _brightness(rgb, colour, clipped, levels):
    """The least brightness at which ``colour`` agrees with every channel, and
    how much of it only the unclipped channels ask for."""
    ratio = rgb / colour
    # A clipped channel is a lower bound, so it can only raise the answer; an
    # unclipped channel is a measurement.
    from_clipped = np.where(clipped > 0.5, ratio, 0.0).max(axis=-1)
    from_kept = np.where(clipped > 0.5, 0.0, ratio).max(axis=-1)
    least = np.maximum(from_clipped, from_kept)
    cap = _MAX_EXTRAPOLATION * (levels / colour).min(axis=-1)
    return np.minimum(least, cap), from_kept / np.maximum(from_clipped, 1e-6)


# Whole-frame passes are written channel by channel: a reduction over the last
# axis of an (H, W, 3) array is several times slower in NumPy, and these run on
# every pixel of every RAW opened.

def _trust(rgb: np.ndarray, levels: np.ndarray, untrusted: np.ndarray) -> np.ndarray:
    """How much each pixel may lend its colour: bright, and not itself rebuilt."""
    closeness = np.maximum(np.maximum(rgb[..., 0] * (1.0 / levels[0]), rgb[..., 1] * (1.0 / levels[1])),
                           rgb[..., 2] * (1.0 / levels[2]))
    trust = _smoothstep(_REFERENCE_LOW, _REFERENCE_HIGH, closeness).astype(np.float32)
    return trust * (1.0 - untrusted)


def _direction(rgb: np.ndarray) -> np.ndarray:
    luma = rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]
    return rgb * (1.0 / np.maximum(luma, 1e-6))[..., None]


def reconstruct_highlights(rgb: np.ndarray, nominal_levels, joint: bool = False) -> np.ndarray:
    """Rebuild the clipped channels of clipped pixels.

    ``rgb`` is linear light in the camera's own primaries with white balance
    applied; ``nominal_levels`` is the saturation level of each channel the file's
    metadata implies. Returns a new array, or ``rgb`` itself when nothing is
    clipped.

    ``joint`` is for a frame that could only be decoded after its colour matrix:
    there no channel can be trusted on its own once any has clipped, so a clipped
    pixel is rebuilt whole, all its channels treated as clipped together.

    A frame is mostly unclipped, so the per-pixel work is done only on the pixels
    near a channel's level; the colour field is still drawn from the whole frame.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    levels = saturation_levels(rgb, nominal_levels).astype(np.float32)
    start = _CLIP_START * levels
    near = (rgb[..., 0] >= start[0]) | (rgb[..., 1] >= start[1]) | (rgb[..., 2] >= start[2])
    ys, xs = np.nonzero(near)
    if len(ys) == 0:
        return rgb
    px = rgb[ys, xs]
    clipped = channel_clip(px, levels)
    if joint:
        clipped = np.repeat(clipped.max(axis=1, keepdims=True), 3, axis=1)
    weight = clipped.max(axis=1)
    if not float(weight.max()) > 0.0:
        return rgb

    untrusted = np.zeros(rgb.shape[:2], dtype=np.float32)
    untrusted[ys, xs] = weight
    direction = in_strips(_direction, rgb)
    trust = in_strips(lambda strip, kept: _trust(strip, levels, kept), rgb, untrusted)
    field = _colour_field(direction, trust)
    borrowed = _unit_luma(_sample(field, ys, xs))

    # Prefer the borrowed colour; fall back to neutral where it does not fit the
    # channels that kept their data.
    _, excess = _brightness(px, borrowed, clipped, levels)
    doubt = _smoothstep(_CONSISTENT, _INCONSISTENT, excess)[:, None]
    colour = _unit_luma(borrowed + doubt * (1.0 - borrowed))

    k, _ = _brightness(px, colour, clipped, levels)
    rebuilt = np.maximum(px, colour * k[:, None])
    fixed = px + clipped * (rebuilt - px)

    out = rgb.copy()
    out[ys, xs] = fixed

    # A pixel clipped in every channel has no data of its own to check a colour
    # against. Its nearest neighbours that kept even one channel have just been
    # rebuilt from that channel, so the colour is carried inwards from them: the
    # middle of a blown sheet of paper takes the paper's colour from its edges,
    # not from the warm stage around it.
    every = clipped.min(axis=1)
    if float(every.max()) > 0.0:
        # Only the pixels just rebuilt have changed, so only they are worked out
        # again: everywhere else ``out`` is ``rgb`` and nothing was untrusted.
        direction[ys, xs] = _direction(out[ys, xs])
        trust[ys, xs] = _trust(out[ys, xs], levels, every)
        field = _colour_field(direction, trust)
        inner = _unit_luma(_sample(field, ys, xs))
        k, _ = _brightness(px, inner, clipped, levels)
        inner_rebuilt = np.maximum(px, inner * k[:, None])
        out[ys, xs] = fixed + every[:, None] * (inner_rebuilt - fixed)
    return out
