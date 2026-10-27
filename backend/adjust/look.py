"""
The camera's look, learned from the file itself.

A Canon file carries two renderings of one exposure: the sensor data, and the
JPEG the camera made from it. The JPEG is not the sensor data with a gamma on it.
It is the sensor data after the camera's white balance, its colour matrix, its
Picture Style tone curve and saturation, its highlight and shadow handling, and
often its Auto Lighting Optimizer. A flat decode of the RAW has none of that, so
its defaults sit a long way from what the photographer saw on the back of the
camera.

This module measures that rendering instead of guessing at it, with the model a
camera's own image processor uses:

  * a **colour matrix** from the decode's linear light into the camera's
    rendering primaries. This is where the Picture Style's saturation and most of
    its hue character live.
  * one **tone curve applied to each channel** of that, for every brightness the
    sensor recorded, the value the camera wrote.

Applying the curve per channel, rather than to brightness alone with the colour
ratios carried along, is what a camera does, and it matters: a bright saturated
colour rolls off towards white the way it does in the camera's JPEG, a dark one
gains a little saturation, and a change of exposure moves a colour along the same
path the camera would have moved it. Measured on Canon R6 II and 5D Mark II
frames against their own JPEGs, this model halves the brightness error of the
earlier one (a brightness-only curve and a colour matrix on top) and cuts the
colour error by a third.

The camera's rendering is a property of the camera and its Picture Style, not of
one frame, so a fit is better made from many frames than from one: a stage scene
of black clothes and warm wood holds too little colour to pin a matrix down on its
own. :func:`frame_samples` takes what one frame can teach, :func:`fit_look` fits
any number of them together, and the caller decides what to pool
(``export.engine`` pools by camera model and Picture Style).

Two details make the comparison fair. The camera's JPEG is usually corrected for
lens distortion and the RAW is not, so the two disagree by tens of pixels at the
edges of a zoom's wide end; the JPEG is registered onto the RAW's geometry (a
scale, a shift and two radial terms) before anything is compared. And only flat
areas are compared, after a little smoothing, so that sharpening, noise reduction
and the half-pixel disagreements left after registration do not masquerade as tone.

The result is a *starting point*, not an overwrite. The look is applied last, on
the display side of every edit, so exposure, highlights, shadows and the rest
still work on the RAW's linear light with all of its headroom. Above the level
where the camera clipped, the curve stays at white; the data above it is exactly
what a highlight pull brings back into range.

What this does not do: it is a global rendering. A camera that lifts one region
and darkens another (the Auto Lighting Optimizer) is matched on average, not
region by region — on these frames a local term was tried and measured no better
— and the camera's sharpening is not reproduced.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from adjust.render import LUMA, MID_GREY, _box, _shoulder, decode, encode, in_strips, resize_linear

# The curve is sampled in stops from middle grey, over the range any sensor
# records, finely enough that interpolating it adds nothing visible.
STOPS = np.linspace(-14.0, 8.0, 2201, dtype=np.float64)
_LOG_MID = float(np.log2(MID_GREY))
_FLOOR = 2.0 ** -16

# Both pictures are compared at about this size. Large enough for the lens
# registration to find features at the edges of the frame, small enough to be
# quick; the comparison is statistical, so a larger copy would not answer better.
FIT_SIZE = 768

# What one frame contributes to a fit, at most.
SAMPLES_PER_FRAME = 4000

# The curve is the median of the camera's value in bins this wide, and a bin needs
# this many samples to be believed.
_BIN_STOPS = 0.125
_BIN_MIN = 12

# Below this the sensor is recording its own noise; those samples say nothing
# about the curve. The curve falls to black beneath the last bin that has data.
_NOISE_FLOOR_STOPS = -9.0

# A channel the camera wrote at (or very near) 0 or 255 says where the camera
# clipped, not what it would have done. It places the ends of the curve, but it
# cannot be taken back through the curve, so the matrix is fitted without it.
_CHANNEL_LOW, _CHANNEL_HIGH = 0.015, 0.985

# The flattest share of the frame that is compared.
_FLAT_SHARE = 0.6

# A fit this far from the camera's picture, on average, did not find the same
# frame: the two were cropped differently, or the preview is not of this exposure.
# The caller then falls back to the flat render instead of imposing a wrong look.
MAX_LUMA_ERROR = 0.07
MAX_ASPECT_MISMATCH = 0.03
# Nor did one that leaves most of the picture's variation unexplained. A small
# average error alone is not enough: an unrelated picture smoothed flat is matched
# by a flat curve to within a few levels.
MAX_UNEXPLAINED = 0.5

# A camera JPEG whose colour is this close to none was written by a
# black-and-white image control.
_MONOCHROME_CHROMA = 0.004

# The colour matrix is kept to something a camera could plausibly do. The lower
# bound is zero because a camera can: a black-and-white image control writes a
# JPEG with no colour at all.
_SINGULAR_MAX = 3.0
_ITERATIONS = 10
# An exposure edit this many stops or more is shown with the pool's tone curve
# rather than the frame's own (see curve_for).
EXPOSURE_BLEND_STOPS = 1.0

# Gauss-Newton steps on the picture error after the alternation, at most.
_REFINE_STEPS = 12
# A gentle pull towards "no change" for a fit made from a single frame, which may
# not hold enough colour to decide the matrix on its own. A pooled fit has the
# variety and needs none.
SINGLE_FRAME_RIDGE = 0.02

# Lens registration: a grid of patches is matched between the two pictures.
_REG_GRID = (6, 9)
_REG_PATCH = 0.09                  # half-size of a patch, of the short side
_REG_MIN_PATCHES = 12
_REG_MIN_SHIFT = 0.5               # pixels at FIT_SIZE; below this it is left alone


@dataclass
class CameraLook:
    curve: np.ndarray      # float32 over STOPS: the display value the camera wrote
    matrix: np.ndarray     # 3x3 float32: decode's linear light → camera's primaries
    white_stops: float     # stops from grey at which the camera reached white
    error: float           # mean |luma| disagreement with the camera's picture
    frames: int = 1        # how many frames the fit was made from
    # The camera's curve over all the frames of its pool, when this look's own
    # curve is one frame's: what an exposure edit moves towards (see apply_look).
    pool_curve: Optional[np.ndarray] = None
    # The camera's correction of the lens's vignetting, in stops at r^2 and r^4
    # (r from the centre, 1 at the corner); see correct_lens.
    vignette: tuple = (0.0, 0.0)
    # What the matrix and the curve leave of the Picture Style's hue character: a
    # smooth change of chroma and of hue as a function of hue (see fit_hue), shape
    # (3, HUE_TERMS). None for a look without one.
    hue: Optional[np.ndarray] = None


@dataclass
class FrameSamples:
    """What one frame teaches about its camera's rendering: pairs of the decode's
    linear colour and the colour the camera wrote for it."""
    linear: np.ndarray     # (N, 3) float32
    camera: np.ndarray     # (N, 3) float32, display-encoded 0..1
    # The whole frame, small (about PROBE_WIDTH pixels across): what a look is
    # finally judged on. The samples above are flat areas spread over the tonal
    # range, which is what a look is *fitted* to; a look can fit them and still
    # draw the picture wrongly (a pool's colour on a frame much brighter than the
    # pool's, measured: 20% worse on the samples, 37% worse on the picture).
    # Kept for the frame being judged only; not stored with a pool.
    probe_linear: Optional[np.ndarray] = None    # (h, w, 3) float32
    probe_camera: Optional[np.ndarray] = None    # (h, w, 3) float32, NaN where the JPEG is not
    # What kind of scene this is, as far as the camera's tone treatment cares
    # (see scene_signature). Stored with a pool, unlike the probe.
    scene: Optional[np.ndarray] = None           # (SCENE_TERMS,) float32


# How wide the whole-frame probe is.
PROBE_WIDTH = 128

# ── Scenes ───────────────────────────────────────────────────────────
#
# One camera and style does not mean one tone curve. Auto Lighting Optimizer lifts a
# dark, contrasty scene far more than a flat, bright one, and Highlight Tone Priority
# reshapes the top of the curve by how much highlight there is. A pool that holds a
# garden at noon and a stage in spotlights has a curve that is neither. So a frame's
# tone, where it must come from other frames (an exposure edit moves towards "what the
# camera would have done"), is taken from the frames most like it in the way that
# matters: how its brightness is spread, with the exposure taken out, and how
# colourful it is.

SCENE_BINS = 20
SCENE_TERMS = SCENE_BINS + 2
_SCENE_RANGE = (-6.0, 4.0)       # stops from the picture's median brightness
SCENE_NEIGHBOURS = 8
# Frames of one bracket sit within about 0.1 of each other and a different scene is
# 0.4 or more away (a garden against a stage, measured on R6 II frames).
SCENE_RADIUS = 0.3


def scene_signature(probe_linear: Optional[np.ndarray], probe_camera: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """How a picture's brightness is spread (a histogram of stops from its median, so
    that exposure drops out) and how coloured it is, from the whole-frame probe."""
    if probe_linear is None:
        return None
    lin = probe_linear.reshape(-1, 3)
    if probe_camera is not None:
        ok = np.isfinite(probe_camera.reshape(-1, 3)).all(axis=1)
        lin = lin[ok]
    if len(lin) < 64:
        return None
    luma = np.maximum(lin @ LUMA, 1e-6)
    stops = np.log2(luma / np.median(luma))
    hist, _ = np.histogram(np.clip(stops, *_SCENE_RANGE), bins=SCENE_BINS, range=_SCENE_RANGE)
    chroma = np.clip(lin / luma[:, None], 0.0, 4.0).mean(axis=0)
    return np.concatenate([hist / hist.sum(), chroma[[0, 2]]]).astype(np.float32)


def scene_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """How unlike two scenes are; 0 for the same, about 1 for quite different, ``inf``
    when either is unknown."""
    if a is None or b is None or len(a) != SCENE_TERMS or len(b) != SCENE_TERMS:
        return float("inf")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(0.5 * np.abs(a[:SCENE_BINS] - b[:SCENE_BINS]).sum() + np.abs(a[SCENE_BINS:] - b[SCENE_BINS:]).sum())


def nearest_scenes(others: Sequence["FrameSamples"], target: "FrameSamples",
                   radius: float = SCENE_RADIUS, limit: int = SCENE_NEIGHBOURS) -> Optional[list]:
    """The frames of ``others`` of the same kind of scene as ``target`` (closest first,
    at most ``limit``): none of them for a scene the pool has not met. ``None``
    when scenes are not known (a pool saved before they were kept, a frame
    without a probe), which callers treat as "use the whole pool"."""
    if not np.isfinite(scene_distance(target.scene, target.scene)):
        return None
    dist = [(scene_distance(s.scene, target.scene), i) for i, s in enumerate(others)]
    known = [d for d in dist if np.isfinite(d[0])]
    if not known:
        return None
    return [others[i] for d, i in sorted(known)[:limit] if d <= radius]


def scene_curve(matrix: np.ndarray, samples: Sequence["FrameSamples"]) -> Optional[np.ndarray]:
    """The tone curve of ``samples`` taken together, through ``matrix``."""
    samples = [s for s in samples if s is not None and len(s.linear)]
    if not samples:
        return None
    curve = _fit_curve(np.concatenate([s.linear for s in samples]).astype(np.float64),
                       np.concatenate([s.camera for s in samples]).astype(np.float64),
                       np.asarray(matrix, dtype=np.float64))
    return None if curve is None else curve.astype(np.float32)


def _probe(small: np.ndarray, cam: np.ndarray):
    """``small`` and its aligned JPEG reduced to about PROBE_WIDTH pixels across. A
    block that is partly outside the JPEG is NaN in the camera copy."""
    h, w = small.shape[:2]
    k = max(1, w // PROBE_WIDTH)
    hh, ww = (h // k) * k, (w // k) * k

    def blocks(a):
        return a[:hh, :ww].reshape(hh // k, k, ww // k, k, 3).mean(axis=(1, 3)).astype(np.float32)

    return blocks(small), blocks(cam)


def probe_error(look: "CameraLook", samples: FrameSamples) -> float:
    """Mean |difference| over every channel between ``look``'s picture of the
    whole frame and the camera's. ``inf`` when the samples carry no probe."""
    if samples.probe_linear is None or samples.probe_camera is None:
        return float("inf")
    valid = np.isfinite(samples.probe_camera).all(axis=2)
    if not valid.any():
        return float("inf")
    shown = apply_look(samples.probe_linear, look)
    return float(np.mean(np.abs(shown - samples.probe_camera)[valid]))


# ── Applying a look ──────────────────────────────────────────────────

def _to_stops(x: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(x, _FLOOR)) - _LOG_MID


def _curve_of(curve: np.ndarray, linear: np.ndarray) -> np.ndarray:
    return np.interp(_to_stops(linear), STOPS, curve).astype(np.float32)


def curve_for(look: CameraLook, exposure: float = 0.0) -> np.ndarray:
    """The tone curve to show a frame with after an exposure edit of
    ``exposure`` stops.

    A frame's own curve is what the camera did at the exposure the frame was
    shot at, and it is exact there. The camera does not keep one curve for every
    exposure: Auto Lighting Optimizer lifts the mid-tones of a darker frame more
    (on an R6 II bracket, by 0.08 of full scale at two stops under). So the further
    an edit moves the exposure, the less this frame's curve says about what the
    camera would have done, and the more its pool, which has seen the camera at
    many exposures, does: the curve moves from the frame's own to the pool's over
    the first stop of an edit.
    """
    if look.pool_curve is None or exposure == 0.0:
        return look.curve
    t = min(1.0, abs(float(exposure)) / EXPOSURE_BLEND_STOPS)
    return (look.curve + (look.pool_curve - look.curve) * t).astype(np.float32)


def apply_look(rgb: np.ndarray, look: CameraLook, amount: float = 1.0,
               exposure: float = 0.0) -> np.ndarray:
    """Linear light to display-encoded colour, through the camera's rendering.

    ``amount`` 1.0 is the camera's look and 0.0 is the plain decode, with a
    straight blend between them, so the look can be dialled back rather than
    taken or left. ``exposure`` is the edit's exposure change in stops, which
    chooses the curve (:func:`curve_for`).
    """
    amount = float(min(1.0, max(0.0, amount)))
    curve = curve_for(look, exposure)
    hue = look.hue * amount if look.hue is not None else None

    def camera_of(strip):
        camera = _curve_of(curve, strip @ look.matrix.T)
        if hue is not None:
            camera = _shift_hue(np.clip(camera, 0.0, 1.0), hue)
        return camera

    rgb = np.asarray(rgb, dtype=np.float32)
    camera = in_strips(camera_of, rgb)
    if amount < 1.0:
        plain = encode(_shoulder(rgb))
        camera = plain + (camera - plain) * amount
    return np.clip(camera, 0.0, 1.0).astype(np.float32)



# ── Hue character ────────────────────────────────────────────────────
#
# A matrix is linear: it can make every colour more saturated, or turn all of them
# by one angle, but not turn greens towards yellow and leave reds alone. A Picture
# Style does exactly that (Canon's Neutral and Standard pull foliage yellower and
# paler than the primaries alone would), and on a frame full of leaves, measured on
# R6 II frames, it was the largest error left once brightness and the lens were
# right. So after the curve there is one more small stage, in OkLab: chroma is
# multiplied, and hue turned, by amounts that vary smoothly with hue (a few Fourier
# terms, so it cannot invent detail), and fade out towards neutral so greys stay
# grey.

HUE_HARMONICS = 3
HUE_TERMS = 2 * (2 * HUE_HARMONICS + 1)
_HUE_MID_L = 0.65
_HUE_RIDGE = 0.002              # of the sample count, on every term
_HUE_MIN_CHROMA = 0.03          # OkLab chroma below which a colour carries no hue
_HUE_FADE = 0.03                # chroma at which the correction is at half strength
_HUE_MAX_LOG_GAIN = 0.5
_HUE_MAX_TURN = 0.45            # radians
_HUE_MIN_SAMPLES = 300
_FRAME_HUE_RIDGE = 0.02

_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32)
_TO_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32)
_FROM_LAB = np.linalg.inv(_TO_LAB.astype(np.float64)).astype(np.float32)
_FROM_LMS = np.linalg.inv(_TO_LMS.astype(np.float64)).astype(np.float32)


# The sRGB transfer function by table: the hue stage runs over every pixel of an
# export, and a power function per channel costs more than the rest of it.
_TABLE_SIZE = 4096
_DECODE_GRID = np.linspace(0.0, 1.0, _TABLE_SIZE + 1, dtype=np.float32)
_DECODE_TABLE = decode(_DECODE_GRID).astype(np.float32)
_ENCODE_GRID = (np.arange(_TABLE_SIZE + 1, dtype=np.float32) / _TABLE_SIZE) ** 2
_ENCODE_TABLE = encode(_ENCODE_GRID).astype(np.float32)


def _lookup(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``table`` at positions ``x`` (in table steps, 0.._TABLE_SIZE), interpolated."""
    i = np.minimum(x.astype(np.int32), _TABLE_SIZE - 1)
    f = x - i.astype(np.float32)
    return table[i] * (1.0 - f) + table[i + 1] * f


def _decode_fast(v: np.ndarray) -> np.ndarray:
    return _lookup(_DECODE_TABLE, v * _TABLE_SIZE)


def _encode_fast(v: np.ndarray) -> np.ndarray:
    return _lookup(_ENCODE_TABLE, np.sqrt(v) * _TABLE_SIZE)


def _oklab(display: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(display, dtype=np.float32), 0.0, 1.0)
    return np.cbrt(_decode_fast(v) @ _TO_LMS.T) @ _TO_LAB.T


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ _FROM_LAB.T) ** 3
    return _encode_fast(np.clip(lms @ _FROM_LMS.T, 0.0, 1.0))


def _hue_columns(cos_h: np.ndarray, sin_h: np.ndarray, lightness: np.ndarray):
    """The basis terms one at a time (cos and sin of the harmonics by recurrence, and
    each again scaled by lightness), so a whole picture's worth is never held at once."""
    c, s = np.ones_like(cos_h), np.zeros_like(cos_h)
    scale = (lightness - _HUE_MID_L) * 2.0
    yield 0, np.ones_like(cos_h)
    c, s = cos_h, sin_h
    for k in range(1, HUE_HARMONICS + 1):
        yield 2 * k - 1, c
        yield 2 * k, s
        c, s = c * cos_h - s * sin_h, s * cos_h + c * sin_h
    # (the lightness-scaled copies are produced by the caller from these)


def _hue_basis(h: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    """All terms for a modest number of points (fitting)."""
    base = np.zeros(h.shape + (HUE_TERMS // 2,), dtype=np.float32)
    for i, col in _hue_columns(np.cos(h), np.sin(h), lightness):
        base[..., i] = col
    return np.concatenate([base, base * ((lightness - _HUE_MID_L) * 2.0)[..., None]], axis=-1)


def shift_hue(display: np.ndarray, hue: np.ndarray) -> np.ndarray:
    """``display`` (encoded 0..1) with its chroma and hue moved by ``hue``'s terms.

    A big picture is done in strips on several threads (adjust.render.in_strips)."""
    display = np.asarray(display, dtype=np.float32)
    if display.ndim != 3:
        return _shift_hue(display, hue)
    return in_strips(lambda strip: _shift_hue(strip, hue), display)


def _shift_hue(display: np.ndarray, hue: np.ndarray) -> np.ndarray:
    lab = _oklab(display)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    safe = np.maximum(chroma, 1e-9)
    cos_h, sin_h = a / safe, b / safe
    fade = chroma / (chroma + _HUE_FADE)
    scale = (L - _HUE_MID_L) * 2.0
    hue = np.asarray(hue, dtype=np.float32)
    half = HUE_TERMS // 2
    rows = [np.zeros_like(L) for _ in range(3)]
    for i, col in _hue_columns(cos_h, sin_h, L):
        for r in range(3):
            rows[r] += col * hue[r, i]
            rows[r] += col * scale * hue[r, half + i]
    gain = np.exp(rows[0] * fade)
    turn = rows[1] * fade
    # cos/sin of a small angle by series: |turn| stays under _HUE_MAX_TURN-ish
    t2 = turn * turn
    cos_t = 1.0 - t2 * (0.5 - t2 * (1.0 / 24.0 - t2 / 720.0))
    sin_t = turn * (1.0 - t2 * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 / 5040.0)))
    out = np.empty_like(lab)
    out[..., 0] = L * np.exp(rows[2] * fade)
    out[..., 1] = chroma * gain * (cos_h * cos_t - sin_h * sin_t)
    out[..., 2] = chroma * gain * (sin_h * cos_t + cos_h * sin_t)
    return _from_oklab(out)


def fit_hue(look: "CameraLook", samples: Sequence[FrameSamples],
            ridge: float = _HUE_RIDGE, on_top: bool = False) -> Optional[np.ndarray]:
    """The hue terms that carry what ``look`` draws towards what the camera drew,
    from samples; None when there is too little colour to say."""
    linear = np.concatenate([s.linear for s in samples])
    camera = np.concatenate([s.camera for s in samples])
    plain = CameraLook(curve=look.curve, matrix=look.matrix, white_stops=look.white_stops,
                       error=look.error, hue=look.hue if on_top else None)
    shown = apply_look(linear[None], plain)[0]
    usable = ((camera > _CHANNEL_LOW) & (camera < _CHANNEL_HIGH)).all(axis=1)
    lab_s, lab_c = _oklab(shown), _oklab(camera)
    c_s, c_c = np.hypot(lab_s[:, 1], lab_s[:, 2]), np.hypot(lab_c[:, 1], lab_c[:, 2])
    usable &= (c_s > _HUE_MIN_CHROMA) & (c_c > _HUE_MIN_CHROMA * 0.5)
    if int(usable.sum()) < _HUE_MIN_SAMPLES:
        return None
    h_s = np.arctan2(lab_s[usable, 2], lab_s[usable, 1])
    h_c = np.arctan2(lab_c[usable, 2], lab_c[usable, 1])
    turn = (h_c - h_s + np.pi) % (2 * np.pi) - np.pi
    gain = np.log(c_c[usable] / c_s[usable])
    fade = c_s[usable] / (c_s[usable] + _HUE_FADE)
    basis = _hue_basis(h_s, lab_s[usable, 0]).astype(np.float64) * fade[:, None]
    weight = c_s[usable].astype(np.float64)
    n = len(weight)
    gram = (basis * weight[:, None]).T @ basis + ridge * weight.sum() * np.eye(HUE_TERMS)
    bright = np.log(np.maximum(lab_c[usable, 0], 1e-3) / np.maximum(lab_s[usable, 0], 1e-3))
    out = np.zeros((3, HUE_TERMS), dtype=np.float32)
    for row, target, limit in ((0, gain, _HUE_MAX_LOG_GAIN), (1, turn, _HUE_MAX_TURN),
                               (2, bright, _HUE_MAX_LOG_GAIN)):
        target = np.clip(target, -2 * limit, 2 * limit)
        out[row] = np.linalg.solve(gram, (basis * weight[:, None]).T @ target)
    return out if not on_top or look.hue is None else look.hue + out


# ── The lens ─────────────────────────────────────────────────────────
#
# Most lenses are darker at the corners than in the middle, by a stop or more at
# the wide end of a zoom, and Canon's JPEG engine corrects it (with Digital Lens
# Optimizer on, whatever the Peripheral Illumination setting says). A RAW decode
# does not. Without the correction, one global tone curve is asked to fit a
# picture whose corners the camera brightened: on R6 II frames at 24 mm the
# rendering came out 2 levels of OkLab lightness too bright in the middle and 6
# too dark in the corners. The correction is learned from each frame's JPEG and
# applied to the linear frame itself, as a camera does, so every edit, crop and
# loupe works on the corrected picture.

# The corner gain is held to what lenses do, and to what a camera's own effect can
# add: a Ricoh GR's Retro Image Control darkens the corners of its JPEG by a stop or
# more, which makes the camera's net correction negative.
_VIGNETTE_RANGE = (-2.0, 2.5)
_VIGNETTE_ROUNDS = 3


def _radius(shape, box=(0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """Distance from the frame's centre, 1 at a corner, for the pixels of ``box``
    (fractions of the whole frame: left, top, right, bottom)."""
    h, w = shape
    left, top, right, bottom = box
    xs = left + (right - left) * (np.arange(w) + 0.5) / w
    ys = top + (bottom - top) * (np.arange(h) + 0.5) / h
    # The aspect is the whole frame's; a patch's own shape says nothing about it.
    aspect = (w / max(right - left, 1e-9)) / (h / max(bottom - top, 1e-9))
    dx = (xs[None, :] - 0.5) * 2.0 * aspect
    dy = (ys[:, None] - 0.5) * 2.0
    return (np.hypot(dx, dy) / np.hypot(aspect, 1.0)).astype(np.float32)


def lens_gain(shape, vignette, box=(0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """The brightening the camera applies at each pixel, for ``vignette``."""
    a, b = vignette
    r2 = _radius(shape, box) ** 2
    return np.exp2(a * r2 + b * r2 * r2).astype(np.float32)


def correct_lens(rgb: np.ndarray, look: Optional[CameraLook], box=(0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """``rgb`` with the camera's vignetting correction applied; ``rgb`` itself when
    there is none to apply. ``box`` places a patch within the whole frame."""
    if look is None or not any(look.vignette):
        return rgb
    return (rgb * lens_gain(rgb.shape[:2], look.vignette, box)[..., None]).astype(np.float32)


def fit_vignette(small: np.ndarray, camera: np.ndarray, look: CameraLook):
    """How much brighter, in stops, the camera made the picture with distance from
    the centre than ``look`` alone explains: ``(a, b)`` at r^2 and r^4.

    ``small`` is the frame at FIT_SIZE and ``camera`` its JPEG aligned to it."""
    shown = apply_look(small, look) @ LUMA
    seen = np.nan_to_num(camera) @ LUMA
    usable = (np.isfinite(camera).all(axis=2) & (seen > 0.05) & (seen < 0.9)
              & (shown > 0.05) & (shown < 0.9))
    if usable.sum() < 500:
        return 0.0, 0.0
    gap = _to_stops(_invert(look.curve, seen)) - _to_stops(_invert(look.curve, shown))
    r2 = _radius(small.shape[:2]) ** 2
    x = np.stack([r2[usable], r2[usable] ** 2], axis=1).astype(np.float64)
    y = gap[usable].astype(np.float64)
    keep = np.ones(len(y), bool)
    coef = np.zeros(2)
    for _ in range(3):                                   # a few outliers are edges
        xm, ym = x[keep].mean(axis=0), y[keep].mean()
        coef = np.linalg.lstsq(x[keep] - xm, y[keep] - ym, rcond=None)[0]
        residual = np.abs((y - ym) - (x - xm) @ coef)
        keep = residual < 3.0 * float(np.median(residual[keep])) + 1e-6
    return float(coef[0]), float(coef[1])


# The camera's correction is the same for every frame shot with one lens at one
# focal length and aperture, so frames are pooled per lens setting. A frame's own
# search is the witness only when there is no pool: on the brightest frame of a
# bracket it overshoots (+1.33 stops where the camera applied +1.0) or finds
# nothing at all, the largest single remaining error on those frames.
_LENS_MIN_FRAMES = 3
_LENS_MAX_SPREAD = 0.25        # stops at the corner, median absolute deviation
_LENS_R2 = np.linspace(0.1, 1.0, 10)


def pool_vignettes(found):
    """The correction common to ``found`` ((a, b) pairs from frames of one lens
    setting): the median gain curve, refitted as (a, b). ``None`` when there are
    too few frames, or they disagree too much to say what the lens does."""
    found = [(float(a), float(b)) for a, b in found if abs(a) + abs(b) > 1e-6]
    if len(found) < _LENS_MIN_FRAMES:
        return None
    curves = np.array([a * _LENS_R2 + b * _LENS_R2 ** 2 for a, b in found])
    median = np.median(curves, axis=0)
    if float(np.median(np.abs(curves[:, -1] - median[-1]))) > _LENS_MAX_SPREAD:
        return None
    x = np.stack([_LENS_R2, _LENS_R2 ** 2], axis=1)
    coef = np.linalg.lstsq(x, median, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def fit_with_lens(linear: np.ndarray, camera_rgb: np.ndarray, vignette=None):
    """What a frame teaches, with the camera's vignetting correction learned first.

    Returns ``(samples, look)`` taken from the corrected frame, the look carrying
    the correction; ``(None, None)`` when the JPEG is not this picture. The
    correction is kept only if it brings the look closer to the camera. The frame
    is reduced and its JPEG registered once; the rounds work on that copy.

    ``vignette`` is a correction already known (the one the camera applies for
    this lens setting, learned from many frames): it is used as it is, with no
    search and no test, since the camera's correction is a fixed function of the
    lens setting and a single frame is a worse witness to it than the pool.
    """
    if not _same_shape(linear, camera_rgb):
        return None, None
    small = resize_linear(np.ascontiguousarray(linear, dtype=np.float32), FIT_SIZE)
    camera = align_camera(small, camera_rgb)
    if vignette is not None:
        corrected = small * lens_gain(small.shape[:2], vignette)[..., None]
        samples = _samples_from(corrected, camera)
        look = fit_look([samples]) if samples is not None else None
        if not trustworthy(look, [samples] if samples is not None else []):
            return None, None
        look.vignette = (float(vignette[0]), float(vignette[1]))
        return samples, look
    samples = _samples_from(small, camera)
    if samples is None:
        return None, None
    look = fit_look([samples])
    if not trustworthy(look, [samples]):
        return None, None
    best = (samples, look, look_error(look, [samples], colour=True))
    vignette = np.zeros(2)
    current = look
    for _ in range(_VIGNETTE_ROUNDS):
        corrected = small * lens_gain(small.shape[:2], vignette)[..., None]
        vignette = vignette + np.array(fit_vignette(corrected, camera, current))
        if not _VIGNETTE_RANGE[0] <= float(vignette.sum()) <= _VIGNETTE_RANGE[1]:
            break
        corrected = small * lens_gain(small.shape[:2], vignette)[..., None]
        trial_samples = _samples_from(corrected, camera)
        trial = fit_look([trial_samples]) if trial_samples is not None else None
        if trial is None:
            break
        trial.vignette = (float(vignette[0]), float(vignette[1]))
        error = look_error(trial, [trial_samples], colour=True)
        if error < best[2]:
            best = (trial_samples, trial, error)
        current = trial
    return best[0], best[1]


# ── What one frame can teach ─────────────────────────────────────────

def _blur(plane: np.ndarray) -> np.ndarray:
    """A small smoothing: two passes of a 3x3 mean, about a 1-pixel Gaussian."""
    return _box(_box(plane, 1), 1)


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """How far the content of ``b`` sits from the same content in ``a`` (dy, dx),
    by phase correlation, with the sharpness of the match.

    Both patches are tapered to zero at their borders first. Cut square, the two
    patches share the same hard edges, and those edges match each other at zero
    shift better than smooth content matches itself at the true one.
    """
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.rfft2((a - a.mean()) * window)
    fb = np.fft.rfft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.irfft2(cross, s=a.shape)
    h, w = corr.shape
    y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def sub(m, p, n):
        d = m - 2.0 * p + n
        return 0.0 if d == 0 else 0.5 * (m - n) / d

    dy = sub(corr[(y - 1) % h, x], corr[y, x], corr[(y + 1) % h, x])
    dx = sub(corr[y, (x - 1) % w], corr[y, x], corr[y, (x + 1) % w])
    y = y if y < h // 2 else y - h
    x = x if x < w // 2 else x - w
    return y + dy, x + dx, float(corr.max())


def _lens_model(points: np.ndarray, centre: np.ndarray, radius: float, u: np.ndarray) -> np.ndarray:
    """Where a point of the RAW sits in the JPEG: a shift, a scale and two radial
    terms about the centre. Linear in its parameters, so it is solved directly."""
    d = (points - centre) / radius
    r2 = (d * d).sum(axis=1, keepdims=True)
    ox, oy, s, a1, a2 = u
    return centre + np.array([ox, oy]) + radius * (s * d + a1 * d * r2 + a2 * d * r2 * r2)


def _register(raw_luma: np.ndarray, cam_luma: np.ndarray) -> Optional[tuple]:
    """Fit the lens model that takes the RAW's geometry to the JPEG's, or ``None``
    when the two already agree or there is too little to match on."""
    h, w = raw_luma.shape
    half = max(8, int(round(min(h, w) * _REG_PATCH)))
    found, moved = [], []
    for fy in np.linspace(0.1, 0.9, _REG_GRID[0]):
        for fx in np.linspace(0.08, 0.92, _REG_GRID[1]):
            cy, cx = int(fy * h), int(fx * w)
            if cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w:
                continue
            pa = raw_luma[cy - half:cy + half, cx - half:cx + half]
            pb = cam_luma[cy - half:cy + half, cx - half:cx + half]
            if pa.std() < 0.02 or pb.std() < 0.02:
                continue
            dy, dx, peak = _phase_shift(pa, pb)
            if peak < 0.08 or abs(dy) > half / 2 or abs(dx) > half / 2:
                continue
            found.append((cx, cy))
            moved.append((cx + dx, cy + dy))
    if len(found) < _REG_MIN_PATCHES:
        return None
    p, q = np.asarray(found, float), np.asarray(moved, float)
    if np.median(np.linalg.norm(q - p, axis=1)) < _REG_MIN_SHIFT:
        return None

    centre, radius = np.array([w / 2.0, h / 2.0]), float(np.hypot(w / 2.0, h / 2.0))
    keep = np.ones(len(p), bool)
    u = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    for _ in range(3):
        d = (p[keep] - centre) / radius
        r2 = (d * d).sum(axis=1)
        n = int(keep.sum())
        a = np.zeros((2 * n, 5))
        a[0::2, 0] = 1.0
        a[1::2, 1] = 1.0
        for k, basis in enumerate((d, d * r2[:, None], d * (r2 * r2)[:, None])):
            a[0::2, 2 + k] = radius * basis[:, 0]
            a[1::2, 2 + k] = radius * basis[:, 1]
        rhs = (q[keep] - centre).reshape(-1)
        u = np.linalg.lstsq(a, rhs, rcond=None)[0]
        err = np.linalg.norm(_lens_model(p, centre, radius, u) - q, axis=1)
        keep = err < max(1.0, 3.0 * float(np.median(err)))
        if keep.sum() < _REG_MIN_PATCHES:
            return None
    return centre, radius, u


def _warp(image: np.ndarray, model: tuple) -> np.ndarray:
    """Resample ``image`` into the RAW's geometry; outside the JPEG is NaN."""
    centre, radius, u = model
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    src = _lens_model(pts, centre, radius, u)
    x, y = src[:, 0], src[:, 1]
    inside = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx = np.clip(x - x0, 0, 1)[:, None]
    fy = np.clip(y - y0, 0, 1)[:, None]
    flat = image.reshape(-1, image.shape[2])
    i00 = y0 * w + x0
    out = (flat[i00] * (1 - fx) * (1 - fy) + flat[i00 + 1] * fx * (1 - fy)
           + flat[i00 + w] * (1 - fx) * fy + flat[i00 + w + 1] * fx * fy)
    out[~inside] = np.nan
    return out.reshape(image.shape).astype(np.float32)


def align_camera(linear: np.ndarray, camera_rgb: np.ndarray) -> np.ndarray:
    """The camera's JPEG resized to ``linear`` and registered onto its geometry,
    as float 0..1; NaN where the JPEG does not reach (it was cropped by its lens
    correction)."""
    h, w = linear.shape[:2]
    cam = np.asarray(
        Image.fromarray(np.asarray(camera_rgb, dtype=np.uint8), "RGB").resize((w, h), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    model = _register(encode(np.clip(linear @ LUMA, 0.0, 1.0)), cam @ LUMA)
    return _warp(cam, model) if model is not None else cam


def frame_samples(linear: np.ndarray, camera_rgb: np.ndarray,
                  rng: Optional[np.random.Generator] = None) -> Optional[FrameSamples]:
    """Pairs of (decode, camera) colour from one frame and its JPEG.

    ``linear`` is the RAW as scene-linear float (H, W, 3); ``camera_rgb`` is the
    camera's JPEG as 8-bit RGB (H', W', 3), both upright. Returns ``None`` when
    the two cannot be the same picture (different shapes) or there is nothing to
    compare.
    """
    if not _same_shape(linear, camera_rgb):
        return None
    small = resize_linear(np.ascontiguousarray(linear, dtype=np.float32), FIT_SIZE)
    return _samples_from(small, align_camera(small, camera_rgb), rng)


def _same_shape(linear: np.ndarray, camera_rgb: np.ndarray) -> bool:
    h, w = linear.shape[:2]
    ch, cw = camera_rgb.shape[:2]
    return min(h, w, ch, cw) >= 8 and abs((w / h) - (cw / ch)) / (w / h) <= MAX_ASPECT_MISMATCH


def _samples_from(small: np.ndarray, cam: np.ndarray,
                  rng: Optional[np.random.Generator] = None) -> Optional[FrameSamples]:
    """frame_samples, from the frame at FIT_SIZE and its JPEG already aligned to it."""
    probe_linear, probe_camera = _probe(small, cam)
    valid = np.isfinite(cam).all(axis=2)
    cam = np.nan_to_num(cam)

    # A little smoothing in each picture's own domain, then only the flattest
    # part of the frame: tone is compared, not edges.
    lin = np.stack([_blur(small[..., c]) for c in range(3)], axis=-1)
    cam = np.stack([_blur(cam[..., c]) for c in range(3)], axis=-1)
    valid &= _blur(valid.astype(np.float32)) > 0.999
    luma = cam @ LUMA
    gy, gx = np.gradient(luma)
    grad = np.hypot(gx, gy)
    if valid.sum() < 64:
        return None
    flat = valid & (grad <= np.quantile(grad[valid], _FLAT_SHARE))

    lin, cam = lin[flat], cam[flat]
    if len(lin) < 64:
        return None
    # Spread over the tonal range: a stage is mostly dark, and the curve and the
    # matrix should not be decided by the darkness alone.
    rng = rng or np.random.default_rng(0)
    stops = _to_stops(lin @ LUMA)
    bins = np.clip(((stops - STOPS[0]) / 0.5).astype(int), 0, 200)
    per_bin = max(16, SAMPLES_PER_FRAME // max(1, len(np.unique(bins))))
    chosen = []
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if len(idx) > per_bin:
            idx = rng.choice(idx, per_bin, replace=False)
        chosen.append(idx)
    idx = np.concatenate(chosen)
    if len(idx) > SAMPLES_PER_FRAME:
        idx = rng.choice(idx, SAMPLES_PER_FRAME, replace=False)
    return FrameSamples(linear=lin[idx].astype(np.float32), camera=cam[idx].astype(np.float32),
                        probe_linear=probe_linear, probe_camera=probe_camera,
                        scene=scene_signature(probe_linear, probe_camera))


# ── Fitting a look ───────────────────────────────────────────────────

def _isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The non-decreasing sequence closest to ``values`` in the weighted
    least-squares sense (pool adjacent violators).

    A camera does not reverse tone, so the curve must not either — but forcing it
    with a running maximum lets one outlying bin lift everything above it. Pooling
    averages the disagreement between neighbours instead.
    """
    blocks: list[list[float]] = []            # [mean, weight, count]
    for v, w in zip(values, weights):
        blocks.append([float(v), float(w), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([np.full(int(c), m) for m, _, c in blocks])


def _binned_curve(stops: np.ndarray, camera: np.ndarray) -> Optional[np.ndarray]:
    """The camera's median value at each brightness, as a full monotone curve."""
    usable = (stops > _NOISE_FLOOR_STOPS) & (stops < STOPS[-1])
    stops, camera = stops[usable], camera[usable]
    if len(stops) == 0:
        return None
    index = np.floor((stops - _NOISE_FLOOR_STOPS) / _BIN_STOPS).astype(int)
    order = np.lexsort((camera, index))
    index, camera = index[order], camera[order]
    bins, start, counts = np.unique(index, return_index=True, return_counts=True)
    keep = counts >= _BIN_MIN
    if keep.sum() < 6:
        return None
    bins, start, counts = bins[keep], start[keep], counts[keep]
    medians = camera[start + counts // 2]
    centres = _NOISE_FLOOR_STOPS + (bins + 0.5) * _BIN_STOPS
    values = _isotonic(medians, counts.astype(float))
    if len(values) >= 5:                                   # a gentle smoothing of the bin noise
        values = np.convolve(np.pad(values, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
        values = np.maximum.accumulate(values)

    # Below the darkest bin the curve falls to black; above the brightest it eases
    # towards white (and stays there if the camera was already clipped).
    low_x, low_y = centres[0], values[0]
    high_x, high_y = centres[-1], values[-1]
    curve = np.interp(STOPS, centres, values)
    below = STOPS < low_x
    curve[below] = low_y * np.clip((STOPS[below] - STOPS[0]) / (low_x - STOPS[0]), 0.0, 1.0)
    above = STOPS > high_x
    curve[above] = high_y + (1.0 - high_y) * (1.0 - np.exp(-(STOPS[above] - high_x) * 0.8))
    return np.clip(np.maximum.accumulate(curve), 0.0, 1.0).astype(np.float32)


def _invert(curve: np.ndarray, value: np.ndarray) -> np.ndarray:
    """The linear value the curve takes to ``value`` (on its rising part)."""
    rising = curve.astype(np.float64) + np.arange(len(curve)) * 1e-9
    return MID_GREY * np.exp2(np.interp(value, rising, STOPS))


def _fit_curve(linear: np.ndarray, camera: np.ndarray, matrix: np.ndarray) -> Optional[np.ndarray]:
    # Every channel counts here, the clipped ones included: a bin where the camera
    # wrote white is how the curve learns where the camera's white is.
    mapped = linear @ matrix.T
    ok = mapped > 0
    return _binned_curve(_to_stops(mapped[ok]), camera[ok])


def fit_look(samples: Sequence[FrameSamples], ridge: Optional[float] = None) -> Optional[CameraLook]:
    """Fit one camera rendering to the samples of one or more frames.

    Alternates between the two halves of the model: with the matrix held, the
    curve is the camera's median value at each brightness over all channels;
    with the curve held, the camera's values are taken back through it to the
    linear light it must have started from, and the matrix is the least-squares
    map to that (in relative terms, so dark colours count as much as bright ones).
    """
    samples = [s for s in samples if s is not None and len(s.linear)]
    if not samples:
        return None
    linear = np.concatenate([s.linear for s in samples]).astype(np.float64)
    camera = np.concatenate([s.camera for s in samples]).astype(np.float64)
    if ridge is None:
        ridge = SINGLE_FRAME_RIDGE if len(samples) == 1 else 0.0

    if float(np.mean(np.abs(camera - (camera @ LUMA)[:, None]))) < _MONOCHROME_CHROMA:
        # A black-and-white image control: every channel is the same brightness.
        return _finish(np.outer(np.ones(3), LUMA), linear, camera, samples)

    prior = np.eye(3)
    matrix = prior.copy()
    for _ in range(_ITERATIONS):
        curve = _fit_curve(linear, camera, matrix)
        if curve is None:
            return None
        wanted = _invert(curve, camera)
        good = ((camera > _CHANNEL_LOW) & (camera < _CHANNEL_HIGH)).all(axis=1)
        if good.sum() < 50:
            return None
        a, t = linear[good], wanted[good]
        solved = np.zeros((3, 3))
        for c in range(3):
            weight = 1.0 / np.maximum(t[:, c], 1e-3)
            aw, tw = a * weight[:, None], t[:, c] * weight
            n = len(aw)
            solved[c] = np.linalg.solve(aw.T @ aw + ridge * n * np.eye(3),
                                        aw.T @ tw + ridge * n * prior[c])
        # The model's overall scale is the curve's business: a neutral keeps its
        # brightness through the matrix.
        scale = float(LUMA @ solved @ np.ones(3))
        if not np.isfinite(scale) or scale <= 1e-3:
            return None
        u, s, vt = np.linalg.svd(solved / scale)
        matrix = (u * np.clip(s, 0.0, _SINGULAR_MAX)) @ vt
    matrix = _refine(matrix, linear, camera, ridge, prior)
    return _finish(matrix, linear, camera, samples, hue=True)


def _refine(matrix, linear, camera, ridge, prior):
    """Polish the matrix against the error in the picture itself.

    The alternation above takes the camera's values back through the curve, and
    where the curve is nearly flat (the highlight shoulder, which Highlight Tone
    Priority makes long) a value lands anywhere along a wide stretch of linear
    light; on a bright frame that pulls the matrix well off. Here each step is
    Gauss-Newton on the display error instead, linearised where the model is now,
    so a flat stretch simply contributes little. The curve is refitted between
    steps, and a step is kept only if the picture error goes down.
    """
    usable = (camera > _CHANNEL_LOW) & (camera < _CHANNEL_HIGH)

    def error(m):
        curve = _fit_curve(linear, camera, m)
        if curve is None:
            return np.inf, None
        shown = _curve_of(curve, linear @ m.T)
        return float(np.mean(((shown - camera) ** 2)[usable])), curve

    best, curve = error(matrix)
    if curve is None:
        return matrix
    slope_stops = np.gradient(curve.astype(np.float64), STOPS)
    for _ in range(_REFINE_STEPS):
        mapped = np.maximum(linear @ matrix.T, _FLOOR)
        step = np.zeros((3, 3))
        for c in range(3):
            x = mapped[:, c]
            ok = usable[:, c]
            # display value per unit of linear light, at the model's current value
            gain = np.interp(_to_stops(x), STOPS, slope_stops) / (x * np.log(2.0))
            residual = camera[:, c] - _curve_of(curve, x)
            j = linear[ok] * gain[ok, None]
            n = int(ok.sum())
            step[c] = np.linalg.solve(j.T @ j + (ridge + 1e-6) * n * np.eye(3),
                                      j.T @ residual[ok] - ridge * n * (matrix[c] - prior[c]))
        candidate = matrix + step
        scale = float(LUMA @ candidate @ np.ones(3))
        if not np.isfinite(scale) or scale <= 1e-3:
            break
        u, s, vt = np.linalg.svd(candidate / scale)
        candidate = (u * np.clip(s, 0.0, _SINGULAR_MAX)) @ vt
        err, cand_curve = error(candidate)
        if not err < best:
            break
        matrix, best, curve = candidate, err, cand_curve
        slope_stops = np.gradient(curve.astype(np.float64), STOPS)
    return matrix


def _finish(matrix, linear, camera, samples, hue: bool = False) -> Optional[CameraLook]:
    curve = _fit_curve(linear, camera, matrix)
    if curve is None:
        return None
    look = CameraLook(curve=curve, matrix=np.asarray(matrix, dtype=np.float32),
                      white_stops=_white_stops(curve), error=0.0, frames=len(samples))
    if hue:
        look.hue = fit_hue(look, samples, _HUE_RIDGE * (10.0 if len(samples) == 1 else 1.0))
    look.error = look_error(look, samples)
    return look


def look_error(look: CameraLook, samples: Sequence[FrameSamples], colour: bool = False) -> float:
    """Mean |luma| disagreement between the look and the camera, on ``samples``;
    with ``colour``, the mean over every channel instead, so a difference in
    colour alone (a black-and-white style against a colour one) counts too."""
    linear = np.concatenate([s.linear for s in samples])
    camera = np.concatenate([s.camera for s in samples])
    shown = apply_look(linear[None], look)[0]
    if colour:
        return float(np.mean(np.abs(shown - camera)))
    return float(np.mean(np.abs(shown @ LUMA - camera @ LUMA)))


def trustworthy(look: Optional[CameraLook], samples: Sequence[FrameSamples]) -> bool:
    """Whether ``look`` really describes the camera that made ``samples``: close
    on average, and accounting for most of how the camera's picture varies."""
    if look is None:
        return False
    error = look_error(look, samples)
    camera = np.concatenate([s.camera for s in samples]) @ LUMA
    spread = float(np.mean(np.abs(camera - np.median(camera))))
    return error <= MAX_LUMA_ERROR and error <= MAX_UNEXPLAINED * spread


def with_colour_of(pooled: CameraLook, samples: FrameSamples,
                   scene_samples: Optional[Sequence[FrameSamples]] = None) -> Optional[CameraLook]:
    """``pooled``'s colour matrix, with a tone curve fitted to this frame alone.

    The colour belongs to the camera and its Picture Style and is better learned
    from many frames; the tone does not. Auto Lighting Optimizer lifts a dark or
    high-contrast frame more than a bright, flat one, and Highlight Tone Priority
    reshapes the top of the curve, so the camera's tone curve changes from frame to
    frame. Measured on R6 II exposure brackets, this pairing matched each frame's
    JPEG better than its own single-frame fit, and better than a pooled look. The
    pool's curve is kept beside it for exposure edits (:func:`curve_for`).
    """
    look = _finish(pooled.matrix.astype(np.float64), samples.linear.astype(np.float64),
                   samples.camera.astype(np.float64), [samples])
    if look is not None:
        look.frames = pooled.frames
        # What an exposure edit moves towards: the pool's curve, or, given the frames
        # most like this one, theirs.
        look.pool_curve = pooled.curve
        if scene_samples is not None:
            # [] is a scene the pool has not met: no other frame says what the camera
            # would do at another exposure here, so an edit keeps this frame's own
            # curve rather than borrow a mixture of other scenes'.
            look.pool_curve = (scene_curve(pooled.matrix, list(scene_samples) + [samples])
                               if scene_samples else None)
        look.hue = pooled.hue
        if pooled.hue is not None and _FRAME_HUE_RIDGE is not None:
            look.hue = fit_hue(look, [samples], _FRAME_HUE_RIDGE, on_top=True)
        look.error = look_error(look, [samples])
    return look


# ── Is a look drawn the way the camera draws? ────────────────────────
#
# Mean error against the camera's JPEG cannot see posterization. A curve that is
# flat across a stop of skin tones and steep above it draws a face as one flat
# patch with a hard edge, and still matches the camera's picture on average: on
# an R6 II stage frame at high ISO such a look scored 0.0160 on the whole-frame
# probe against 0.0171 for the smooth look fitted to the same frame, and was kept.
# Two measures see it: the curve's own shape, which costs nothing, and the
# picture's local contrast, tone band by tone band, against the camera's.

# The mid-tones, as display values, over which a camera's curve rises evenly.
_EVEN_RANGE = (0.1, 0.85)
_EVEN_STEP = 0.0625
_EVEN_WINDOW = 0.25             # stops
# Below this a curve has a plateau no camera draws. Of 529 kept looks, those
# outside one stage shoot sat at 0.4 to 0.7; 316, nearly all from that shoot, were
# under it, most at 0.000. This is what marks a look as off: a plateau that does no
# harm in a frame without those tones does it after an exposure edit moves them in.
EVEN_MIN = 0.25

# The local-contrast measure: tone bands of the camera's picture, and how many
# pixels a band needs to be judged.
_DETAIL_BANDS = np.arange(0.08, 0.92, 0.06)
_DETAIL_BAND_WIDTH = 0.06
_DETAIL_MIN_PIXELS = 2000
_DETAIL_MIN_GRADIENT = 0.004    # below this an edge is noise
# Spread, in stops, between the band where ours has the least local contrast
# relative to the camera's and the one where it has the most. Over 529 kept looks:
# a median of 0.38 for an even curve and 1.0 for one with a plateau, where the
# posterized patches are plain to see above this. Not a verdict on its own: a
# camera JPEG with heavy noise reduction (a harbour at night) is softer than any
# develop and reaches 1.4 with nothing posterized. It says whether a plateau shows.
BANDING_MAX = 0.5


def curve_evenness(curve: np.ndarray) -> Optional[float]:
    """How evenly ``curve`` rises through the mid-tones: its shallowest slope over a
    quarter stop, as a share of its median slope there. About 0.5 for a camera's
    curve; near 0 for one with a plateau. ``None`` when the curve spends too little
    of its range in the mid-tones to say."""
    x = np.arange(STOPS[0], STOPS[-1], _EVEN_STEP)
    y = np.interp(x, STOPS, np.asarray(curve, dtype=np.float64))
    k = int(round(_EVEN_WINDOW / _EVEN_STEP))
    slope = (y[k:] - y[:-k]) / _EVEN_WINDOW
    middle = (y[k:] + y[:-k]) / 2.0
    mid = (middle > _EVEN_RANGE[0]) & (middle < _EVEN_RANGE[1])
    if mid.sum() < 4:
        return None
    typical = float(np.median(slope[mid]))
    return float(slope[mid].min() / typical) if typical > 0 else None


def tone_detail(shown: np.ndarray, camera: np.ndarray) -> Optional[dict]:
    """How a picture's local contrast compares with the camera's, band by band of
    the camera's brightness.

    ``shown`` and ``camera`` are display-encoded (H, W, 3), the same size, NaN in
    ``camera`` where it does not reach. Both are smoothed a little first, so that
    tone is compared, not the camera's sharpening and noise. In each band, the
    median of log2(our gradient / the camera's) over its edges: 0 where ours has
    the camera's contrast, -1 where it has half. Returns the flattest and the
    steepest band (``(stops, band's lower edge)``) and ``banding``, the spread
    between them; ``None`` when no band has enough edges.
    """
    valid = np.isfinite(camera).all(axis=-1)

    def luma(a):
        return _box(_box(_box(np.asarray(a @ LUMA, dtype=np.float32), 1), 1), 1)

    ours, theirs = luma(np.nan_to_num(shown)), luma(np.nan_to_num(camera))
    g_ours = np.hypot(*np.gradient(ours))
    g_theirs = np.hypot(*np.gradient(theirs))
    valid &= _box(valid.astype(np.float32), 3) > 0.999
    edges = valid & (g_theirs > _DETAIL_MIN_GRADIENT)
    ratio = np.log2((g_ours[edges] + 1e-3) / (g_theirs[edges] + 1e-3))
    tone = theirs[edges]
    bands = []
    for low in _DETAIL_BANDS:
        inside = (tone >= low) & (tone < low + _DETAIL_BAND_WIDTH)
        if inside.sum() >= _DETAIL_MIN_PIXELS:
            bands.append((round(float(np.median(ratio[inside])), 3), round(float(low), 2)))
    if not bands:
        return None
    flattest, steepest = min(bands), max(bands)
    return {"flattest": flattest, "steepest": steepest,
            "banding": round(steepest[0] - flattest[0], 3)}


def fit_camera_look(linear: np.ndarray, camera_rgb: np.ndarray) -> Optional[CameraLook]:
    """Learn the camera's rendering from a single frame and its JPEG.

    Returns ``None`` when the two are not the same picture closely enough to
    trust.
    """
    samples = frame_samples(linear, camera_rgb)
    if samples is None:
        return None
    look = fit_look([samples])
    return look if trustworthy(look, [samples]) else None


def _white_stops(curve: np.ndarray) -> float:
    """Stops from grey at which the camera's curve reaches white."""
    reached = np.nonzero(curve >= 0.985)[0]
    return float(STOPS[reached[0]]) if len(reached) else float(STOPS[-1])
