"""
Sharpening: Lightroom's four controls, on the finished picture's brightness.

A RAW decode is soft. The camera sharpens its own JPEG and the camera look
(adjust.look) deliberately does not copy that, so a developed RAW needs its own
sharpening. It runs late, on display-encoded colour after the look, and on
brightness only:

* after noise reduction (which runs first, on the RAW data), so the grain it
  took out is not put back;
* on display-encoded values, where a step of brightness looks the same size in
  the shadows as in the highlights, so an edge halo is as visible in both and
  the controls mean the same everywhere;
* on brightness, not on each channel, so an edge does not get a coloured fringe.

The controls, stored so that 0 is Lightroom's default (the panel treats 0 as
"untouched" for every slider):

  sharpen          0..150  how much. 0 is off; Lightroom's default for RAW is 40
  sharpen_radius   -5..20  the size of the detail sharpened, as tenths of a pixel
                           added to 1.0 px: -5 is 0.5 px, 20 is 3.0 px
  sharpen_detail   -25..75 Lightroom's Detail 0..100, less 25. Low holds the
                           strong edges back (fewer halos), high sharpens them
                           as hard as fine texture
  sharpen_masking  0..100  keep the sharpening to edges: 0 everywhere, higher
                           leaves smooth areas (skin, sky) alone

The radius is in pixels of the whole photograph. A picture made smaller than
that (the preview, a web-sized export) has its radius scaled with it, but never
below :data:`MIN_SIGMA` of its own pixels, so sharpening still shows on a small
picture rather than being averaged away. That makes a small picture a little
crisper than a full-size one shrunk, which is what output sharpening is for.
"""

import math

import numpy as np

SHARPEN_KEYS = ("sharpen", "sharpen_radius", "sharpen_detail", "sharpen_masking")

# Lightroom's defaults, which the stored 0 of radius and detail stand for.
DEFAULT_RADIUS = 1.0
DEFAULT_DETAIL = 25.0

# The finest blur used, in the picture's own pixels. At 0.5 a 1600 px preview
# showed little of it (fine detail +13% at amount 40 where the loupe showed +50%);
# at 0.7 it is +22%, which is what a 2048 px export to share wants too.
MIN_SIGMA = 0.7
# Amount 100 adds the detail back one and a half times over: an unsharp mask at
# gain 1.5, about as crisp as Lightroom's 100.
GAIN_PER_100 = 1.5
# Detail as the largest edge contrast (display units) sharpened in full: past it
# the boost levels off, which is what stops a halo. Detail 0 holds edges to a
# few levels; 100 lets every edge through.
_DETAIL_LIMIT_LOW = 0.02
_DETAIL_LIMIT_HIGH = 0.5
# Masking: the edge strength (display units of change across one blur radius)
# below which sharpening is left out, at masking 100.
_MASK_EDGE = 0.08

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def radius_px(stored: float) -> float:
    return DEFAULT_RADIUS + float(stored) / 10.0


def detail_of(stored: float) -> float:
    return DEFAULT_DETAIL + float(stored)


def _gaussian(plane: np.ndarray, sigma: float) -> np.ndarray:
    """A separable Gaussian blur with edges repeated, in float32."""
    reach = max(1, int(math.ceil(3.0 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-reach, reach + 1, dtype=np.float32) / sigma) ** 2)
    taps /= taps.sum()
    h, w = plane.shape
    out = plane
    for axis in (0, 1):
        pad = [(reach, reach) if a == axis else (0, 0) for a in (0, 1)]
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, t in enumerate(taps):
            acc += t * (padded[i:i + h] if axis == 0 else padded[:, i:i + w])
        out = acc
    return out


def sharpen(enc: np.ndarray, settings: dict, scale: float = 1.0) -> np.ndarray:
    """``enc`` (H, W, 3 display-encoded, 0..1) sharpened by ``settings``.

    ``scale`` is this picture's pixels per pixel of the whole photograph (1.0
    for a full-size render or a 1:1 loupe patch).
    """
    amount = float(settings.get("sharpen", 0))
    if amount <= 0:
        return enc
    sigma = max(MIN_SIGMA, radius_px(settings.get("sharpen_radius", 0)) * float(scale))
    detail = min(100.0, max(0.0, detail_of(settings.get("sharpen_detail", 0)))) / 100.0
    masking = float(settings.get("sharpen_masking", 0)) / 100.0

    luma = enc @ _LUMA
    blurred = _gaussian(luma, sigma)
    high = luma - blurred
    limit = _DETAIL_LIMIT_LOW + (_DETAIL_LIMIT_HIGH - _DETAIL_LIMIT_LOW) * detail ** 1.5
    high = limit * np.tanh(high / limit)
    boost = np.float32(GAIN_PER_100 * amount / 100.0) * high

    if masking > 0:
        # Edge strength from the blurred picture, so noise and fine texture do
        # not count as edges: the change across one radius.
        gy, gx = np.gradient(blurred)
        edge = np.hypot(gx, gy) * np.float32(2.0 * sigma)
        low = _MASK_EDGE * masking
        t = np.clip((edge - 0.5 * low) / max(low, 1e-6), 0.0, 1.0)
        boost = boost * (t * t * (3.0 - 2.0 * t))

    return np.clip(enc + boost[..., None], 0.0, 1.0).astype(np.float32)
