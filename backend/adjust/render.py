"""
The develop pipeline: a photograph's edit, applied in linear light.

``adjust.develop`` describes an edit; this module performs it. The two used to
be one 8-bit pass over a finished picture, and that is why the controls felt
mild and why they could not rescue anything:

* Exposure multiplied the *encoded* values. One "stop" was really about two and
  a bit, and a pixel that reached 255 was gone, so brightening a frame blew its
  highlights while the shadows barely moved.
* Highlights and shadows were gains looked up by a pixel's own brightness. A
  pixel already at 255 cannot be pulled back to anything but flat grey, and a
  pixel at 3 lifted by half is still 4.
* Everything was rounded to 8 bits after each step, so there was nothing above
  white to bring back.

Here the frame is decoded to scene-linear float with headroom above white
(see :func:`export.engine.linear_base`), and an edit is applied in four stages:

  1. **Light.** Exposure is a multiplication in linear light, so +1 EV is one
     stop, as the label says. White balance rides in the same multiplication.
  2. **Tone.** The frame is split into a *base* (its large-scale brightness)
     and a *detail* layer (everything finer than that). Highlights, shadows,
     whites, blacks, contrast and the HDR squeeze reshape the base only, and
     the detail is put back on top. That is what lets a window be pulled down
     three stops without its frame going flat, and a shadow be opened without
     the picture turning to mud: the local contrast that makes a surface look
     like a surface is not touched by a control that moves a region's level.
  3. **Display.** Values above white are rolled off smoothly instead of being
     cut, then encoded to the display curve.
  4. **Look.** Vibrance, saturation, fade and split toning, on the finished
     tonality, where a photographer judges them.

Detail cannot be invented. A pixel the sensor clipped holds no information, so
lowering highlights shows a grey patch there, not a cloud. What is recovered is
everything the sensor kept that an 8-bit rendering would have thrown away.
"""

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from adjust.geometry import GEOMETRY_KEYS, apply_geometry
from adjust.lens import FRAME_KEYS, TRANSFORM_KEYS
from functools import lru_cache

from adjust.develop import (
    CURVE_KEYS,
    HSL_BANDS,
    HSL_CENTRES,
    HSL_SLIDERS,
    combine,
    normalize,
    point_of,
)

# Rec. 709 luma, shared with the rest of the app so "brightness" means the same
# thing in a measurement and in a control.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Middle grey in linear light, and its place on the stops axis. Every tonal
# control is defined in stops from here, which is what makes "shadows" mean the
# same part of the picture on a bright frame and a dark one.
MID_GREY = 0.18
LOG_MID = math.log2(MID_GREY)

# Luma below this is treated as black when taking logs. 2^-14 is below the
# noise floor of any camera this will see, and it keeps log2 finite.
_FLOOR = 2.0 ** -14

# ── Tone: how far each control reaches ───────────────────────────────
#
# Amounts are the stops moved at full deflection, at the part of the range the
# control owns. These are deliberately large. A fill-light of three stops is
# what opens a backlit face; a control that reached a third of a stop was the
# reason the old ones read as ineffective. The curve is kept monotonic below,
# so reaching far cannot turn the picture inside out.
SHADOW_STOPS = 3.5
HIGHLIGHT_STOPS = 2.8
WHITE_STOPS = 1.2
BLACK_STOPS = 1.4
CONTRAST_SPREAD = 0.45          # +100 stretches the stops about grey by this much
COMPRESS_SQUEEZE = 0.62         # +100 keeps this fraction less of the scene's range
COMPRESS_PIVOT = -0.8           # stops from grey that the squeeze holds still
COMPRESS_DETAIL = 0.35          # extra local contrast that comes with a squeeze
CLARITY_DETAIL = 0.9            # +100 clarity: detail gain in the mid-tones

# Smallest slope the tone curve may have, in output stops per input stop. Below
# this a stack of extreme controls would flatten (or invert) a range of the
# picture; at this value the worst case is a very flat region, never a reversal.
MIN_SLOPE = 0.08

# The base layer is a self-guided filter of log-luminance, computed on a copy
# this long on its long side. It is a smooth field, so a small copy is not a
# loss, and the cost of the filter is independent of the frame's size.
BASE_WORKING_SIZE = 512
# Edge threshold, in stops squared. Variation below roughly half a stop is
# smoothed into the base; a step bigger than that (a horizon, a window frame)
# is an edge and stays one, which is what keeps halos off the picture.
BASE_EPS = 0.25
BASE_RADIUS_FRACTION = 1.0 / 14.0

# Where the display rolloff starts, in linear light, and the most headroom it
# will make room for. Values up to the knee pass through untouched; the range
# from the knee up to the frame's own peak is folded into the space between the
# knee and white. A frame with nothing above white has nothing to fold, and is
# not touched at all.
SHOULDER_KNEE = 0.80
SHOULDER_MAX_PEAK = 4.0

# Dither amplitude, in 8-bit levels. Half a level either way breaks up the
# bands a lifted shadow would otherwise show, and is invisible on its own.
DITHER_LEVELS = 0.5


# ── The display curve ────────────────────────────────────────────────
#
# The Rec. 709 camera curve, which is what rawpy writes by default. Using the
# same curve for decoding and encoding means an unedited frame goes in and out
# unchanged, and a JPEG source round-trips exactly.

_OETF_KNEE = 0.018


# The curve is evaluated by interpolating a table, not computed per pixel: a
# power function over every channel of a 2-megapixel frame costs more than the
# rest of the edit together, and this table is exact to well under one 8-bit
# level everywhere (the nearest of 16384 steps is within 0.03 of one level even
# on the steep toe at the dark end).
_ENCODE_STEPS = 16384
_ENCODE_GRID = np.linspace(0.0, 1.0, _ENCODE_STEPS + 1, dtype=np.float64)
_ENCODE_TABLE = np.where(
    _ENCODE_GRID < _OETF_KNEE,
    4.5 * _ENCODE_GRID,
    1.099 * np.power(_ENCODE_GRID, 0.45) - 0.099,
).astype(np.float32)


def encode(linear: np.ndarray) -> np.ndarray:
    """Linear light 0..1 to display-encoded 0..1."""
    index = np.clip(linear * _ENCODE_STEPS + 0.5, 0.0, float(_ENCODE_STEPS)).astype(np.int32)
    return _ENCODE_TABLE[index]


def decode(encoded: np.ndarray) -> np.ndarray:
    """Display-encoded 0..1 to linear light."""
    y = np.clip(encoded, 0.0, 1.0)
    return np.where(
        y < 4.5 * _OETF_KNEE, y / 4.5, np.power((y + 0.099) / 1.099, 1.0 / 0.45)
    ).astype(np.float32)


_DECODE_LUT = decode(np.arange(256, dtype=np.float32) / 255.0)


def linear_from_image(image: Image.Image) -> np.ndarray:
    """An 8-bit picture as scene-linear float. No headroom: 255 is white."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return _DECODE_LUT[arr]


def to_image(encoded: np.ndarray, dither: bool = True) -> Image.Image:
    """Display-encoded float 0..1 to an 8-bit picture.

    Rounded once, here, with a little noise first: a shadow that was lifted by
    three stops spreads a handful of levels across a long gradient, and plain
    rounding turns that into visible steps.
    """
    if not dither:
        return Image.fromarray(in_strips(_rounded, encoded), "RGB")
    return Image.fromarray(in_strips(_dithered, encoded, _dither_noise(encoded.shape[:2])), "RGB")


def _rounded(encoded: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def _dithered(encoded: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(encoded * 255.0 + noise[..., None]), 0, 255).astype(np.uint8)


@lru_cache(maxsize=4)
def _dither_noise(shape: tuple) -> np.ndarray:
    """The dither for a picture of this size: fixed, so a preview does not
    shimmer, and kept, since every drag of a slider asks for the same one."""
    rng = np.random.default_rng(1)
    noise = rng.random(shape, dtype=np.float32)
    noise = (noise + rng.random(shape, dtype=np.float32) - 1.0) * np.float32(2.0 * DITHER_LEVELS)
    noise.setflags(write=False)
    return noise


# ── Strips ───────────────────────────────────────────────────────────
#
# Most of an edit is one sum done to every pixel on its own. NumPy does each step
# of it on one core and through main memory, so a big picture is cut into strips
# of rows, done on several threads (NumPy lets go of the interpreter lock), and
# each strip's working set stays in cache. On a 2048 px preview this took the
# camera look from 0.80 s to 0.16 s; strips four times the size were half as fast.

STRIP_PIXELS = 1 << 16
_STRIP_WORKERS = min(8, os.cpu_count() or 1)
_strip_pool: Optional[ThreadPoolExecutor] = None
_strip_pool_lock = threading.Lock()
_inside_strip = threading.local()


def in_strips(fn, *arrays: np.ndarray) -> np.ndarray:
    """``fn(*arrays)``, for an ``fn`` that treats every pixel on its own, done a
    strip of rows at a time on several threads.

    The arrays share their height (their first axis), and the result is the
    strips' results stacked, so it is exactly what ``fn`` would have returned.
    Called from inside a strip, it simply runs ``fn``.
    """
    height = arrays[0].shape[0]
    width = arrays[0].shape[1] if arrays[0].ndim > 1 else 1
    rows = max(1, STRIP_PIXELS // max(1, width))
    if height < rows * 2 or getattr(_inside_strip, "on", False):
        return fn(*arrays)
    return np.concatenate(_on_strip_threads(
        lambda start: fn(*(a[start:start + rows] for a in arrays)), range(0, height, rows)))


def side_by_side(fn, items) -> list:
    """``[fn(item) for item in items]``, the items done at once on the strip
    threads: for a few independent passes, each too small to be cut into strips."""
    items = list(items)
    if len(items) < 2 or getattr(_inside_strip, "on", False):
        return [fn(item) for item in items]
    return _on_strip_threads(fn, items)


def _on_strip_threads(fn, items) -> list:
    global _strip_pool
    with _strip_pool_lock:
        if _strip_pool is None:
            _strip_pool = ThreadPoolExecutor(max_workers=_STRIP_WORKERS,
                                             thread_name_prefix="strip")

    def one(item):
        _inside_strip.on = True
        try:
            return fn(item)
        finally:
            _inside_strip.on = False

    return list(_strip_pool.map(one, items))


# ── Filtering ────────────────────────────────────────────────────────

def _smoothstep(edge0: float, edge1: float, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box(arr: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1) square, in float64 so a long running sum stays exact."""
    r = int(radius)
    padded = np.pad(arr.astype(np.float64), r, mode="edge")
    cs = padded.cumsum(axis=0).cumsum(axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    d = 2 * r + 1
    h, w = arr.shape
    total = cs[d:d + h, d:d + w] - cs[:h, d:d + w] - cs[d:d + h, :w] + cs[:h, :w]
    return (total / (d * d)).astype(np.float32)


def _resize_plane(plane: np.ndarray, size: tuple[int, int], smooth: bool) -> np.ndarray:
    """Resize one float plane. ``size`` is (width, height)."""
    img = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), "F")
    resample = Image.BILINEAR if not smooth else Image.BOX
    return np.asarray(img.resize(size, resample), dtype=np.float32)


def smooth_base(log_luma: np.ndarray) -> np.ndarray:
    """The large-scale brightness of a frame, in stops, edge-preserving.

    A self-guided filter: in a flat region the output is the local mean, and
    where the local variance is high compared with ``BASE_EPS`` the output
    follows the input, so strong edges do not bleed across.

    The coefficients are solved on a small copy, which is what keeps the cost
    independent of the frame's size, but they are applied to the *full-size*
    guide. Upsampling the finished base instead would blur it across an edge,
    and the detail layer, which is the difference between the picture and its
    base, would then carry that blur back in as a bright or dark line along
    every strong edge once the base was compressed.
    """
    h, w = log_luma.shape
    scale = min(1.0, BASE_WORKING_SIZE / float(max(h, w)))
    sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    small = _resize_plane(log_luma, (sw, sh), smooth=True) if scale < 1.0 else log_luma

    radius = max(2, int(round(min(sw, sh) * BASE_RADIUS_FRACTION)))
    mean = _box(small, radius)
    var = np.maximum(_box(small * small, radius) - mean * mean, 0.0)
    a = var / (var + BASE_EPS)
    b = mean - a * mean
    mean_a = _box(a, radius)
    mean_b = _box(b, radius)

    if scale < 1.0:
        mean_a = _resize_plane(mean_a, (w, h), smooth=False)
        mean_b = _resize_plane(mean_b, (w, h), smooth=False)
    return mean_a * log_luma + mean_b


# ── A decoded frame ──────────────────────────────────────────────────

@dataclass
class LinearFrame:
    """A decoded picture as scene-linear float, with its layers kept.

    The base/detail split is the expensive part of an edit and depends only on
    the pixels, never on the sliders, so it is computed on first use and kept.
    Dragging a slider then costs a lookup and some multiplications.
    """

    rgb: np.ndarray                   # (H, W, 3) float32, 1.0 = display white
    # The camera's rendering of this frame, learned from its embedded JPEG
    # (adjust.look.CameraLook), or None for a file that has none to learn from.
    look: Optional[object] = None
    # Which part of the whole picture this frame is, as (left, top, right,
    # bottom) fractions: all of it, or a loupe's patch. Grain and the vignette
    # are laid over the whole picture, so a patch takes its own part of them.
    window: tuple = (0.0, 0.0, 1.0, 1.0)
    # This frame's pixels per pixel of the whole photograph: 1.0 at full size
    # (an export, a loupe's patch), less for a preview. Sharpening's radius is in
    # the photograph's pixels, so it is scaled by this (adjust.sharpen).
    scale: float = 1.0
    _layers: Optional[tuple] = field(default=None, repr=False)
    # The last light-and-tone result, by the settings that made it (see
    # _light_and_tone): the costly half of an edit, kept between slider moves.
    _toned: Optional[tuple] = field(default=None, repr=False)
    # A patch cut from a larger frame (the loupe) is anchored to a style by the
    # whole frame, not by itself, so the two agree (see adjust.styles.anchor).
    anchor_of: Optional[object] = field(default=None, repr=False)
    _anchors: dict = field(default_factory=dict, repr=False)
    # Each mask's weight over these pixels, by its shape and strokes: painting
    # is the costly part, and moving a mask's slider does not repaint it.
    _mask_weights: dict = field(default_factory=dict, repr=False)
    # The subject or sky as a model found it in this whole frame (adjust.segment).
    _mattes: dict = field(default_factory=dict, repr=False)
    # This frame with its spots removed, by the spots (see healed).
    _healed: Optional[tuple] = field(default=None, repr=False)
    # The same picture with its noise taken out, lined up pixel for pixel: a
    # callable ``(wait) -> (H, W, 3) or None`` (export.engine._with_denoiser),
    # None for a frame that has no such twin (a JPEG).
    denoiser: Optional[object] = field(default=None, repr=False)
    # This frame with the noise sliders applied, by their values (see noise_reduced).
    _reduced: Optional[tuple] = field(default=None, repr=False)
    # How the camera corrects this frame's lens distortion: a callable
    # ``() -> model or None`` (export.engine.lens_profile, adjust.lens), None for a
    # frame that cannot know.
    lens: Optional[object] = field(default=None, repr=False)
    # This frame with its lens corrected, by the model (see lens_corrected).
    _corrected: Optional[tuple] = field(default=None, repr=False)
    # The frame as shown before any edit, display-encoded: what a mask's range
    # is judged on (see shown).
    _shown: Optional[np.ndarray] = field(default=None, repr=False)
    # A smaller copy of this frame to draw while a slider is held (see draft_of).
    _draft: Optional[object] = field(default=None, repr=False)

    def shown(self) -> np.ndarray:
        """The frame as it stands, display-encoded float (H, W, 3), kept."""
        if self._shown is None:
            self._shown = _display(self, self.rgb, 0.0)
        return self._shown

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]

    def layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(log-luma, base, detail) in stops, from the unedited frame."""
        if self._layers is None:
            luma = self.rgb @ LUMA
            log_luma = np.log2(np.maximum(luma, _FLOOR)).astype(np.float32)
            base = smooth_base(log_luma)
            self._layers = (log_luma, base, log_luma - base)
        return self._layers

    def encoded(self) -> Image.Image:
        """The frame as it stands, as an 8-bit picture.

        The same display step an edit ends with, so that "no edit" and "an edit
        that changes nothing" are the same picture.
        """
        return to_image(_display(self, self.rgb, 0.0), dither=False)


def region_frame(small: "LinearFrame", region: np.ndarray, box: tuple) -> "LinearFrame":
    """A full-resolution patch that is toned like the frame it was cut from.

    The tone controls work on a base/detail split whose base is the picture's
    large-scale brightness. Computed on a patch alone, that base would be the
    patch's own brightness and a face would be toned as if it were the whole
    scene, so a loupe would disagree with the preview it is magnifying. The base
    here is the *whole frame's*, taken from ``small`` and stretched over the
    patch; only the detail layer comes from the full-resolution pixels.

    ``box`` is the patch as ``(left, top, right, bottom)`` fractions of the frame.
    """
    patch = LinearFrame(region)
    patch.look = small.look
    patch.window = tuple(box)
    patch.anchor_of = small.anchor_of or small
    _, base_small, _ = small.layers()
    sh, sw = base_small.shape
    left, top, right, bottom = box
    window = Image.fromarray(np.ascontiguousarray(base_small), "F").resize(
        (region.shape[1], region.shape[0]),
        Image.BILINEAR,
        box=(left * sw, top * sh, max(left * sw + 1, right * sw), max(top * sh + 1, bottom * sh)),
    )
    base = np.asarray(window, dtype=np.float32)
    luma = region @ LUMA
    log_luma = np.log2(np.maximum(luma, _FLOOR)).astype(np.float32)
    patch._layers = (log_luma, base, log_luma - base)
    return patch


# A draft is drawn at this fraction of the preview's size: a quarter of the
# pixels, so a quarter of the time, and sharp enough to judge a slider by while it
# moves. The full preview follows when the slider is let go.
DRAFT_SCALE = 0.5


def draft_of(frame: "LinearFrame") -> "LinearFrame":
    """``frame`` at DRAFT_SCALE of its size, to draw while a slider is held.

    Edited like the frame itself: it is anchored to it, as a loupe's patch is, so
    its mattes and its styles' anchors are the whole frame's; it has the same
    lens, and its noise-reduced twin is the frame's, shrunk. Kept on the frame."""
    if frame._draft is not None:
        return frame._draft
    w, h = frame.size
    rgb = resize_linear(frame.rgb, max(1, round(max(w, h) * DRAFT_SCALE)))
    denoiser = None
    if frame.denoiser is not None:
        shrunk = {}

        def denoiser(wait: bool = True):
            whole = frame.denoiser(wait)
            if whole is None:
                return None
            if shrunk.get("of") is not whole:
                shrunk.update({"of": whole, "rgb": resize_linear(whole, max(rgb.shape[:2]))})
            return shrunk["rgb"]

    frame._draft = LinearFrame(rgb, look=frame.look, window=frame.window,
                               scale=frame.scale * rgb.shape[1] / w,
                               anchor_of=frame.anchor_of or frame,
                               denoiser=denoiser, lens=frame.lens)
    return frame._draft


def healed(frame: "LinearFrame", spots: list, fetch=None) -> "LinearFrame":
    """``frame`` with its spots removed (adjust.spots), as a frame of its own
    whose layers are those of the healed pixels. Kept on the frame for the
    spots it was made with: a slider's move does not heal again."""
    from adjust.spots import apply_spots

    key = json.dumps(spots, sort_keys=True)
    if frame._healed is not None and frame._healed[0] == key:
        return frame._healed[1]
    out = LinearFrame(apply_spots(frame.rgb, spots, frame.window, fetch), look=frame.look,
                      window=frame.window, scale=frame.scale, anchor_of=frame.anchor_of)
    frame._healed = (key, out)
    return out


def noise_reduced(frame: "LinearFrame", luma: float, color: float, wait: bool = True) -> "LinearFrame":
    """``frame`` with ``luma`` and ``color`` (0..100) of its noise taken out
    (adjust.noise), as a frame of its own; ``frame`` itself when it has no
    noise-reduced twin, or (``wait`` False) not yet. Kept on the frame for the
    values it was made with."""
    if frame.denoiser is None:
        return frame
    key = (float(luma), float(color))
    if frame._reduced is not None and frame._reduced[0] == key:
        return frame._reduced[1]
    denoised = frame.denoiser(wait)
    if denoised is None:
        return frame
    from adjust.noise import blend

    out = LinearFrame(blend(frame.rgb, denoised, luma, color), look=frame.look,
                      window=frame.window, scale=frame.scale, anchor_of=frame.anchor_of or frame,
                      lens=frame.lens)
    frame._reduced = (key, out)
    return out


def lens_corrected(frame: "LinearFrame", profile: float = 0, amount: float = 0,
                   transform: Optional[dict] = None) -> "LinearFrame":
    """``frame`` with its lens distortion corrected (adjust.lens): as the camera
    corrects it when ``profile`` is set, and by ``amount`` (-100..100) by hand;
    then its perspective transformed by ``transform`` (the tf_* controls).
    ``frame`` itself when that changes nothing. A frame of its own, with the
    corrected geometry, so masks and spots are placed on the corrected picture.

    Only a whole frame is corrected here: a loupe's patch is cut from the
    corrected frame where it is made (export.engine.loupe_image), since the
    pixels it needs lie outside it."""
    if tuple(frame.window) != (0.0, 0.0, 1.0, 1.0):
        return frame
    from adjust.lens import model_for, warp

    camera = frame.lens() if profile and frame.lens is not None else None
    h, w = frame.rgb.shape[:2]
    model = model_for({"lens_profile": profile, "distortion": amount, **(transform or {})}, camera, w / h)
    if model is None:
        return frame
    key = json.dumps(model, sort_keys=True)
    if frame._corrected is not None and frame._corrected[0] == key:
        return frame._corrected[1]
    out = LinearFrame(warp(frame.rgb, model), look=frame.look, scale=frame.scale, lens=frame.lens)
    frame._corrected = (key, out)
    return out


def style_anchor(frame: "LinearFrame", style) -> float:
    """The exposure factor that anchors ``style`` on this frame, worked out once
    per frame and style from the unedited picture (adjust.styles.anchor)."""
    owner = frame.anchor_of or frame
    key = (style.id, id(style.look))
    factor = owner._anchors.get(key)
    if factor is None:
        from adjust.styles import anchor

        factor = anchor(owner.rgb, style, owner.look)
        owner._anchors[key] = factor
    return factor


def _display(frame: "LinearFrame", rgb: np.ndarray, flat: float, ev: float = 0.0,
             style: Optional[str] = None, style_fade: float = 0.0) -> np.ndarray:
    """Linear light to display-encoded colour: the camera's look when the frame
    has one, the plain decode with a highlight shoulder when it has not.

    ``flat`` is 0..100: how far to leave the camera's look behind. It has no
    effect on a frame that has no look. ``ev`` is the edit's exposure change,
    which the look uses to choose its tone curve.

    ``style`` is the id of a stored style (adjust.styles) to render through
    instead of the frame's own camera look; ``style_fade`` (0..100) fades it back
    towards the camera's look. A style that cannot be found is ignored, so an
    edit naming one this machine does not have still shows the picture.
    """
    if style:
        from adjust.styles import load, render

        found = load(style)
        if found is not None:
            styled = render(rgb, found, style_anchor(frame, found), exposure=ev)
            if style_fade > 0.0:
                own = _display(frame, rgb, 0.0, ev)
                styled = styled + (own - styled) * np.float32(min(100.0, style_fade) / 100.0)
            if flat > 0.0:
                plain = encode(_shoulder(rgb))
                styled = styled + (plain - styled) * np.float32(min(100.0, flat) / 100.0)
            return np.clip(styled, 0.0, 1.0).astype(np.float32)
    if frame.look is not None and flat < 100.0:
        from adjust.look import apply_look

        return apply_look(rgb, frame.look, 1.0 - flat / 100.0, exposure=ev)
    return encode(_shoulder(rgb))


def resize_linear(rgb: np.ndarray, max_res: int) -> np.ndarray:
    """Downscale a linear frame, in linear light, preserving aspect ratio.

    Averaging in linear light is also the physically right way to shrink a
    picture: a fine black-and-white pattern averages to the grey the eye sees,
    not to the darker grey that averaging encoded values gives.
    """
    h, w = rgb.shape[:2]
    if max_res <= 0 or (w <= max_res and h <= max_res):
        return rgb
    ratio = min(max_res / w, max_res / h)
    # Rounded, not truncated: 2048/6000 * 6000 is a hair under 2048 in floating
    # point, and a long side of 2047 is not the size that was asked for.
    size = (max(1, min(max_res, round(w * ratio))), max(1, min(max_res, round(h * ratio))))
    planes = side_by_side(
        lambda i: np.asarray(
            Image.fromarray(np.ascontiguousarray(rgb[..., i]), "F").resize(
                size, Image.LANCZOS
            ),
            dtype=np.float32,
        ),
        range(3),
    )
    return np.clip(np.stack(planes, axis=-1), 0.0, None)


# ── The tone curve ───────────────────────────────────────────────────

_CURVE_X = np.linspace(-14.0, 8.0, 2201, dtype=np.float64)


def tone_curve(settings: dict) -> np.ndarray:
    """Output stops for each input stop, as an array over ``_CURVE_X``.

    Built once per edit and indexed per pixel, so the shape a slider describes
    is exactly the shape every pixel gets, and it can be read in a test. Both
    axes are stops from middle grey.

    Each control adds stops over the part of the range it owns. They are summed
    and then forced monotonic, so a pile of extreme sliders can flatten a range
    but never reverse it.
    """
    shadows = settings.get("shadows", 0) / 100.0
    highlights = settings.get("highlights", 0) / 100.0
    whites = settings.get("whites", 0) / 100.0
    blacks = settings.get("blacks", 0) / 100.0
    contrast = settings.get("contrast", 0) / 100.0
    compress = settings.get("compress", 0) / 100.0

    x = _CURVE_X
    y = x.copy()

    if compress:
        # Squeeze the scene about a pivot just under grey, so overall brightness
        # holds and it is the extremes that come in.
        y = COMPRESS_PIVOT + (y - COMPRESS_PIVOT) * (1.0 - COMPRESS_SQUEEZE * compress)

    if shadows:
        # Owns everything darker than about a stop under grey. The extra factor
        # fades the control out in the last few stops above black, where the
        # sensor holds noise and not scene, so a lift opens the picture without
        # printing the noise floor.
        weight = (1.0 - _smoothstep(-5.0, 0.8, x)) * _smoothstep(-11.0, -7.5, x)
        y = y + shadows * SHADOW_STOPS * weight
    if highlights:
        y = y + highlights * HIGHLIGHT_STOPS * _smoothstep(-1.4, 2.4, x)
    if blacks:
        y = y + blacks * BLACK_STOPS * (1.0 - _smoothstep(-6.8, -3.0, x))
    if whites:
        y = y + whites * WHITE_STOPS * _smoothstep(0.6, 2.6, x)
    if contrast:
        y = y + (y - 0.0) * contrast * CONTRAST_SPREAD

    # Monotonic, with a minimum slope. Re-anchored afterwards so the fix-up
    # cannot drift the picture: middle grey lands where the controls put it.
    grey = float(np.interp(0.0, x, y))
    step = np.maximum(np.diff(y), MIN_SLOPE * np.diff(x))
    y = np.concatenate([[y[0]], y[0] + np.cumsum(step)])
    y = y + (grey - float(np.interp(0.0, x, y)))
    return y.astype(np.float32)


def _shoulder(v: np.ndarray) -> np.ndarray:
    """Fold the headroom above white into the top of the range, smoothly.

    The curve leaves everything up to the knee alone, leaves the knee with unit
    slope (no visible edge where the rolloff begins) and lands the frame's
    brightest value exactly on white. Its strength follows the headroom there
    is: with a peak of 1.0 it is the identity, so a JPEG's white stays 255 and an
    edit that changes nothing changes nothing, and it tightens smoothly as the
    peak rises, so there is no step between a frame that just exceeds white and
    one that does not.
    """
    peak = min(float(v.max()), SHOULDER_MAX_PEAK)
    if peak <= 1.0:
        return v
    span = 1.0 - SHOULDER_KNEE
    k = (peak - SHOULDER_KNEE) / span
    out = v.copy()
    over = out > SHOULDER_KNEE
    u = np.clip((out[over] - SHOULDER_KNEE) / (peak - SHOULDER_KNEE), 0.0, 1.0)
    out[over] = SHOULDER_KNEE + span * (1.0 - (1.0 - u) ** k)
    return out


# ── The look ─────────────────────────────────────────────────────────

def _hue_rgb(hue: float) -> np.ndarray:
    """A fully saturated colour for a hue in degrees, as RGB 0..1."""
    h = (hue % 360.0) / 60.0
    c = 1.0
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    table = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    return np.array(table[int(h) % 6], dtype=np.float32)


def _apply_look(enc: np.ndarray, s: dict, local_saturation: Optional[np.ndarray] = None) -> np.ndarray:
    """Vibrance, saturation, fade and split toning on display-encoded colour.

    ``local_saturation`` is the gradients' saturation over these pixels, (H, W)
    on the slider's scale divided by 100: added to the edit's own, so the
    global slider sets the base and a gradient moves it from there (the whole
    picture grey, one shirt still red), rather than one scaling the other.
    """
    if local_saturation is None:
        return in_strips(lambda strip: _look_of(strip, s), enc)
    return in_strips(lambda strip, local: _look_of(strip, s, local[..., None]), enc, local_saturation)


def _look_of(enc: np.ndarray, s: dict, local_saturation=0.0) -> np.ndarray:
    luma = (enc @ LUMA)[..., None]

    saturation = s.get("saturation", 0) / 100.0 + local_saturation
    vibrance = s.get("vibrance", 0) / 100.0
    if np.any(saturation) or vibrance:
        chroma = (enc.max(axis=2) - enc.min(axis=2))[..., None]
        # Vibrance weighs by how little colour a pixel already has, and eases off
        # on orange-red hues (R > G > B, softly) where skin lives, so a face does
        # not go orange while the sky and the foliage open up.
        low = 1.0 - np.clip(chroma * 1.6, 0.0, 1.0)
        skin = np.clip((enc[..., 0:1] - enc[..., 1:2]) * 4.0, 0.0, 1.0) * np.clip(
            (enc[..., 1:2] - enc[..., 2:3]) * 4.0, 0.0, 1.0
        )
        factor = 1.0 + saturation + vibrance * low * (1.0 - 0.45 * skin)
        enc = luma + (enc - luma) * np.clip(factor, 0.0, 4.0)

    fade = s.get("fade", 0) / 100.0
    if fade:
        lo = 0.10 * fade
        hi = 1.0 - 0.05 * fade
        enc = lo + enc * (hi - lo)

    for end in ("shadow", "highlight"):
        amount = s.get(f"{end}_sat", 0) / 100.0
        if not amount:
            continue
        tint = _hue_rgb(s.get(f"{end}_hue", 0))
        tint = tint - tint.mean()          # a colour shift, not a brightness one
        y = np.clip((enc @ LUMA)[..., None], 0.0, 1.0)
        weight = (1.0 - y) ** 1.5 if end == "shadow" else y ** 1.5
        enc = enc + weight * (0.30 * amount) * tint
    return np.clip(enc, 0.0, 1.0)


# ── Tone curve ───────────────────────────────────────────────────────

_CURVE_STEPS = 4096


def curve_table(points: list) -> np.ndarray:
    """A point curve as a lookup table over 0..1, monotone between its points.

    Fritsch–Carlson monotone cubic: smooth through every point like a spline,
    but it never overshoots, so a curve whose points all rise cannot fold back
    on itself and turn a highlight darker than the tone below it. Outside the
    first and last point the curve is flat, as Lightroom's is: moving the end
    point is how the black or white point is set.
    """
    xs = np.array([p[0] for p in points], dtype=np.float64) / 255.0
    ys = np.array([p[1] for p in points], dtype=np.float64) / 255.0
    grid = np.linspace(0.0, 1.0, _CURVE_STEPS + 1)
    if len(xs) == 2:
        return np.interp(grid, xs, ys).astype(np.float32)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.empty_like(xs)
    m[0], m[-1] = delta[0], delta[-1]
    m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    for i, d in enumerate(delta):
        if d == 0.0:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / d, m[i + 1] / d
        if a < 0.0:
            m[i] = a = 0.0
        if b < 0.0:
            m[i + 1] = b = 0.0
        r = a * a + b * b
        if r > 9.0:
            t = 3.0 / math.sqrt(r)
            m[i], m[i + 1] = t * a * d, t * b * d
    x = np.clip(grid, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    t = (x - xs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    out = ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
           + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _lookup(table: np.ndarray, values: np.ndarray) -> np.ndarray:
    index = np.clip(values * _CURVE_STEPS, 0.0, _CURVE_STEPS).astype(np.int32)
    return table[index]


def _apply_curves(enc: np.ndarray, s: dict) -> np.ndarray:
    """The RGB curve on all three channels, then each channel's own curve."""
    enc = np.clip(enc, 0.0, 1.0)
    if "curve" in s:
        enc = _lookup(curve_table(s["curve"]), enc)
    for channel, key in enumerate(("curve_r", "curve_g", "curve_b")):
        if key in s:
            enc[..., channel] = _lookup(curve_table(s[key]), enc[..., channel])
    return enc


# ── Colour mixer ─────────────────────────────────────────────────────
#
# Eight bands of hue, each with a hue shift, a saturation and a luminance. A
# pixel between two band centres takes a blend of both, in proportion to how
# close it is, so there is no edge in a gradient sky where one band hands over to
# the next. Grey has no hue, and the mixer leaves it alone.

MIXER_HUE_DEGREES = 30.0        # ±100 hue moves a band this far toward its neighbour
MIXER_LUM_STOPS = 1.2           # ±100 luminance, in stops, on a fully coloured pixel


# Hue is measured on the colour circle around the grey axis (the angle of a
# pixel's opponent colour, red at 0°, yellow 60°, green 120° …) rather than on
# the HSV hexagon. The two agree at the primaries, the secondaries and orange,
# and the circle is what a hue shift can turn a pixel around cheaply: one
# rotation about grey instead of a trip through HSV and back.
_SQRT3 = math.sqrt(3.0)
_INV_SQRT3 = np.float32(1.0 / _SQRT3)
_HUE_STEPS = 720                # band weights are looked up at half-degree steps


def _hue_of(enc: np.ndarray) -> np.ndarray:
    r, g, b = enc[..., 0], enc[..., 1], enc[..., 2]
    return np.degrees(np.arctan2(np.float32(_SQRT3) * (g - b), 2.0 * r - g - b)) % np.float32(360.0)


def _band_table(s: dict, kind: str) -> Optional[np.ndarray]:
    """One kind of mixer control by hue, at half-degree steps, blended between bands."""
    values = np.array([s.get(f"{kind}_{band}", 0) / 100.0 for band in HSL_BANDS], dtype=np.float64)
    if not values.any():
        return None
    degrees = np.arange(_HUE_STEPS) * (360.0 / _HUE_STEPS)
    return np.interp(degrees, HSL_CENTRES, values, period=360.0)


def _apply_mixer(enc: np.ndarray, s: dict) -> np.ndarray:
    """Per-band hue, saturation and luminance on display-encoded colour.

    Each step keeps the pixel's brightness unless brightness is what it is for.
    A hue shift is a turn about the grey axis with the pixel's luma put back, so
    red turned orange does not also turn lighter; saturation scales the colour's
    distance from its own grey, so a blue sky taken to grey is a dark grey and
    not a white one; luminance alone moves the level.

    The band weights come from the pixel's hue as it was, so moving a colour
    into the next band does not hand it to that band's sliders too.
    """
    enc = np.clip(enc, 0.0, 1.0)
    hue_index = (_hue_of(enc) * (_HUE_STEPS / 360.0)).astype(np.int32) % _HUE_STEPS
    luma = (enc @ LUMA)[..., None]

    shift = _band_table(s, "hue")
    if shift is not None:
        # The turn depends on the hue alone, so its sine and cosine are looked up
        # with the band weights rather than worked out for every pixel.
        theta = np.radians(MIXER_HUE_DEGREES * shift)
        cos = np.cos(theta).astype(np.float32)[hue_index][..., None]
        sin = np.sin(theta).astype(np.float32)[hue_index][..., None]
        r, g, b = enc[..., 0:1], enc[..., 1:2], enc[..., 2:3]
        cross = np.concatenate([g - b, b - r, r - g], axis=-1)
        cross *= -_INV_SQRT3 * sin                   # positive turns red toward yellow
        along = (r + g + b) * (1.0 / 3.0)            # n (n · v) for n = (1,1,1)/√3
        turned = enc * cos
        turned += cross
        turned += along * (1.0 - cos)
        turned_luma = (turned @ LUMA)[..., None]
        turned *= luma / np.maximum(turned_luma, 1e-6)
        enc = np.clip(turned, 0.0, 1.0, out=turned)

    gain = _band_table(s, "sat")
    if gain is not None:
        factor = (1.0 + gain).astype(np.float32)[hue_index][..., None]
        enc = luma + (enc - luma) * factor

    lum = _band_table(s, "lum")
    if lum is not None:
        # Weighed by how coloured the pixel is, so a near-grey wall beside a blue
        # sky is not darkened along with it.
        top = enc.max(axis=-1, keepdims=True)
        sat = (top - enc.min(axis=-1, keepdims=True)) / np.maximum(top, 1e-6)
        colour = np.clip(sat * (1.0 / 0.35), 0.0, 1.0)
        stops = (MIXER_LUM_STOPS * lum).astype(np.float32)[hue_index][..., None]
        enc = enc * np.exp2(stops * colour)
    return np.clip(enc, 0.0, 1.0).astype(np.float32)


# ── Black and white ──────────────────────────────────────────────────
#
# The grey starts from the pixel's own brightness, and the B&W mix then makes
# each band of colour lighter or darker in grey: what a red filter does to a
# blue sky on black-and-white film. The mix reaches a pixel in proportion to
# how coloured it is, so a grey wall stays the grey it was whatever the mix.

BW_MIX_STOPS = 1.6              # ±100 on a band, in stops, on a fully coloured pixel


def _to_mono(enc: np.ndarray, s: dict) -> np.ndarray:
    enc = np.clip(enc, 0.0, 1.0)
    grey = (enc @ LUMA)[..., None]
    mix = _band_table(s, "bw")
    if mix is not None:
        hue_index = (_hue_of(enc) * (_HUE_STEPS / 360.0)).astype(np.int32) % _HUE_STEPS
        top = enc.max(axis=-1, keepdims=True)
        sat = (top - enc.min(axis=-1, keepdims=True)) / np.maximum(top, 1e-6)
        colour = np.clip(sat * (1.0 / 0.35), 0.0, 1.0)
        stops = (BW_MIX_STOPS * mix).astype(np.float32)[hue_index][..., None]
        grey = grey * np.exp2(stops * colour)
    return np.repeat(np.clip(grey, 0.0, 1.0), 3, axis=-1).astype(np.float32)


# ── Effects on the finished picture ──────────────────────────────────
#
# Grain and the vignette belong to the picture as framed, so they are laid on
# after the crop. Both are defined over the whole picture rather than in pixels:
# grain is a fixed pattern of so many grains across the long edge, so a 1600 px
# preview, a 6000 px export and a 1:1 loupe show the same grain at the same
# viewing size, and the same grains every time the photo is drawn.

GRAINS_ACROSS = 1400            # grain_size 0: about a 35 mm film's grain on a print
GRAIN_SIZE_RANGE = 1.3          # ±100 size: this many stops coarser or finer
GRAIN_AMPLITUDE = 0.16          # grain 100, in display units, at mid-grey
GRAIN_SEED = 1729
VIGNETTE_REACH = 0.85           # ±100: how far the corners move toward black or white


@lru_cache(maxsize=4)
def _grain_field(across: int, down: int) -> np.ndarray:
    rng = np.random.default_rng(GRAIN_SEED)
    return rng.standard_normal((down, across), dtype=np.float32)


def _picture_size(frame_size: tuple, window: tuple) -> tuple:
    """The whole picture's size in pixels, from a frame and the part it is."""
    w, h = frame_size
    left, top, right, bottom = window
    return w / max(right - left, 1e-6), h / max(bottom - top, 1e-6)


def _grain(enc: np.ndarray, amount: float, size: float, window: tuple) -> np.ndarray:
    h, w = enc.shape[:2]
    full_w, full_h = _picture_size((w, h), window)
    grains = GRAINS_ACROSS * 2.0 ** (-GRAIN_SIZE_RANGE * size / 100.0)
    scale = grains / max(full_w, full_h)
    across, down = max(2, int(math.ceil(full_w * scale))), max(2, int(math.ceil(full_h * scale)))
    field_ = _grain_field(across, down)
    left, top, right, bottom = window
    noise = np.array(Image.fromarray(field_, "F").resize(
        (w, h), Image.BICUBIC,
        box=(left * across, top * down, right * across, bottom * down),
    ), dtype=np.float32)
    # Grain enlarged past a pixel is smoother than the field it came from; set
    # its strength back so a coarse grain is as strong as a fine one. Grain made
    # smaller than a pixel (a thumbnail) is left averaged out, as it would be on
    # a small print: there it is texture, not noise.
    spread = float(noise.std())
    if spread > 1e-6 and w >= (right - left) * across:
        noise /= spread
    luma = enc @ LUMA
    # Film grain shows most in the mid-tones and least in clean blacks and whites.
    weight = 0.35 + 0.65 * 4.0 * luma * (1.0 - luma)
    delta = (GRAIN_AMPLITUDE * amount / 100.0) * noise * weight
    return enc + delta[..., None]


def _vignette(enc: np.ndarray, amount: float, window: tuple) -> np.ndarray:
    h, w = enc.shape[:2]
    left, top, right, bottom = window
    xs = np.linspace(left, right, w, dtype=np.float32) * 2.0 - 1.0
    ys = np.linspace(top, bottom, h, dtype=np.float32) * 2.0 - 1.0
    # Measured on an ellipse that fits the picture, so the corners of a long
    # frame darken as much as those of a square one.
    r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / math.sqrt(2.0)
    t = np.clip((r - 0.35) / 0.65, 0.0, 1.0)
    falloff = (t * t * (3.0 - 2.0 * t))[..., None]
    a = VIGNETTE_REACH * amount / 100.0
    if a < 0:
        return enc * (1.0 + a * falloff)
    return enc + (1.0 - enc) * (a * falloff)


def _finish(img: Image.Image, s: dict, window: tuple) -> Image.Image:
    """Vignette and grain on the framed 8-bit picture."""
    enc = np.asarray(img, dtype=np.float32) / 255.0
    if s.get("vignette"):
        enc = _vignette(enc, s["vignette"], window)
    if s.get("grain"):
        enc = _grain(enc, s["grain"], s.get("grain_size", 0), window)
    return to_image(np.clip(enc, 0.0, 1.0))


# ── The pipeline ─────────────────────────────────────────────────────

def _linear_gain_of_encoded(change: Optional[dict]) -> Optional[dict]:
    """A correction measured on encoded values, as the same gain in linear light.

    Set matching measures brightness and white balance on 8-bit pixels and asks
    for a multiplication of those. A multiplication of encoded values is a power
    of that gain in linear light, so the request is translated and not
    reinterpreted: the frame lands where matching said it should.
    """
    if not change:
        return None
    k = 1.0 / 0.45
    return {
        "ev": float(change.get("ev", 0.0)) * k,
        "wb_r": float(change.get("wb_r", 1.0)) ** k,
        "wb_g": float(change.get("wb_g", 1.0)) ** k,
        "wb_b": float(change.get("wb_b", 1.0)) ** k,
    }


_TONAL_KEYS = ("highlights", "shadows", "whites", "blacks", "contrast", "compress", "clarity")
# Everything the first two stages read. The rest of an edit — curve, mixer,
# black and white, the look, effects — starts from their result.
_LIGHT_AND_TONE_KEYS = ("exposure", "temperature", "tint", "flat", "style", "style_fade",
                        "masks") + _TONAL_KEYS
# A gradient's controls that act through the tone curve on the base.
_MASK_TONE_KEYS = ("contrast", "highlights", "shadows")


def _mask_layers(frame: LinearFrame, masks: list) -> list:
    """Each gradient that changes something, with its weight over these pixels."""
    from adjust.masks import CHOSEN, MASK_SLIDERS, active, weight

    h, w = frame.rgb.shape[:2]
    kept = {}
    layers = []
    for m in active(masks):
        key = json.dumps({k: v for k, v in m.items() if k not in MASK_SLIDERS}, sort_keys=True)
        if key not in kept:
            kept[key] = frame._mask_weights.get(key)
            if kept[key] is None:
                matte = None
                if m["kind"] in CHOSEN:
                    from adjust.segment import whole_matte

                    matte = whole_matte(frame, m["kind"], m.get("person"))
                kept[key] = weight(m, (h, w), frame.window, matte=matte)
                if m.get("range") and kept[key].any():
                    from adjust.masks import range_weight

                    kept[key] = kept[key] * range_weight(frame.shown(), m["range"])
                if m["kind"] in CHOSEN and matte is None:
                    # Not kept: once the model is here, the next render selects.
                    layers.append((m, kept.pop(key)))
                    continue
        layers.append((m, kept[key]))
    frame._mask_weights = kept       # only the masks of the edit now in hand
    # A brush painted elsewhere in the frame reaches none of a loupe's patch.
    return [(m, w) for m, w in layers if w.any()]


def _light_and_tone(frame: LinearFrame, clean: dict, change: Optional[dict]) -> np.ndarray:
    """Stages 1–3 — light, tone and display — as display-encoded colour.

    The display stage is the costly one when a frame has a camera look (its
    curve and its hue character run over every pixel), and nothing after it
    feeds back into it. So the frame keeps its last result: dragging the tone
    curve, the mixer or a grain slider redoes only what comes after, and the
    camera look is worked out once until a light or tone control moves.
    """
    # A gradient's saturation acts later, with the look (_mask_saturation).
    key = (tuple((k, [{n: v for n, v in m.items() if n != "saturation"} for m in clean[k]]
                  if k == "masks" else clean[k]) for k in _LIGHT_AND_TONE_KEYS if k in clean),
           tuple(sorted(change.items())) if change else None)
    cached = frame._toned
    if cached is not None and cached[0] == key:
        return cached[1].copy()

    ev = change["ev"] if change else 0.0
    gains = np.array(
        [change["wb_r"], change["wb_g"], change["wb_b"]] if change else [1.0, 1.0, 1.0],
        dtype=np.float32,
    )
    rgb = frame.rgb * gains

    # Gradients (adjust.masks): each control scaled by the gradient's weight at
    # every pixel, in the stage where its global slider acts. Exposure is a map
    # of stops added to the edit's own; temperature a per-pixel white balance.
    layers = _mask_layers(frame, clean.get("masks"))
    ev_map = ev
    for mask, w in layers:
        if mask.get("exposure"):
            ev_map = ev_map + w * np.float32(mask["exposure"])
        if mask.get("temperature"):
            local = point_of({"temperature": mask["temperature"]}) or {}
            log_gain = np.log2(np.array([local.get("wb_r", 1.0), local.get("wb_g", 1.0),
                                         local.get("wb_b", 1.0)], dtype=np.float32))
            rgb = rgb * np.exp2(w[..., None] * log_gain)
    mask_tone = [(m, w) for m, w in layers
                 if any(m.get(k) for k in _MASK_TONE_KEYS) or m.get("clarity")]

    if any(key in clean for key in _TONAL_KEYS) or mask_tone:
        curve = tone_curve(clean)
        # Each gradient's own curve, and its clarity, in the order of mask_tone.
        local_curves = [tone_curve({k: mask[k] for k in _MASK_TONE_KEYS if mask.get(k)})
                        if any(mask.get(k) for k in _MASK_TONE_KEYS) else None
                        for mask, _ in mask_tone]
        local_clarity = [mask.get("clarity", 0) / 100.0 for mask, _ in mask_tone]
        compress_gain = np.float32(1.0 + COMPRESS_DETAIL * clean.get("compress", 0) / 100.0)
        clarity = clean.get("clarity", 0) / 100.0
        ev_is_map = isinstance(ev_map, np.ndarray)

        def toned(rgb, log_luma, base, detail, *maps):
            ev_here, weights = (maps[0], maps[1:]) if ev_is_map else (ev_map, maps)
            x = base + ev_here - LOG_MID
            stops = np.interp(x, _CURVE_X, curve) + LOG_MID
            for local, w in zip(local_curves, weights):
                # The gradient's own curve, as a change of stops where it is whole.
                if local is not None:
                    stops = stops + w * (np.interp(x, _CURVE_X, local) - x)

            detail_gain = compress_gain
            clarities = [(c, w) for c, w in zip(local_clarity, weights) if c]
            if clarity or clarities:
                # Clarity is a mid-tone control: weighted so the ends of the range
                # keep their own look and a bright sky does not turn crunchy.
                mid = np.exp(-((x + 0.5) / 2.6) ** 2).astype(np.float32)
                amount = np.float32(clarity)
                for local, w in clarities:
                    amount = amount + w * np.float32(local)
                detail_gain = detail_gain + CLARITY_DETAIL * amount * mid
            target = stops + detail * detail_gain
            # Gain from the unedited frame to the edited one, exposure included.
            return rgb * np.exp2(target - log_luma)[..., None]

        maps = ([ev_map] if ev_is_map else []) + [w for _, w in mask_tone]
        rgb = in_strips(toned, rgb, *frame.layers(), *maps)
    elif isinstance(ev_map, np.ndarray):
        rgb = rgb * np.exp2(ev_map)[..., None]
    elif ev:
        rgb = rgb * np.float32(2.0 ** ev)

    enc = _display(frame, rgb, clean.get("flat", 0), ev,
                   style=clean.get("style"), style_fade=clean.get("style_fade", 0))
    frame._toned = (key, enc)
    return enc.copy()


def _mask_saturation(frame: LinearFrame, masks: Optional[list]) -> Optional[np.ndarray]:
    """The gradients' saturation over the frame, (H, W) on the slider's scale
    divided by 100, or ``None`` when none of them moves it."""
    if not any(m.get("saturation") for m in masks or ()):
        return None
    out = None
    for mask, w in _mask_layers(frame, masks):
        if mask.get("saturation"):
            part = w * np.float32(mask["saturation"] / 100.0)
            out = part if out is None else out + part
    return out


def develop_frame(
    frame: LinearFrame,
    settings: Optional[dict],
    matched: Optional[dict] = None,
    noise_wait: bool = True,
) -> Image.Image:
    """Apply an edit set to a decoded frame and return the 8-bit result.

    ``matched`` is a set-matching correction (``analysis.harmonize.correction``)
    to be composed with the edit's own exposure and white balance; it was
    measured on 8-bit pixels and is translated to linear light here.
    ``noise_wait`` False draws the frame without its noise reduction while that
    is still being made, rather than waiting for it (the develop view).

    The frame is not modified. Returns the picture as it stands when the edit
    is empty.
    """
    clean = normalize(settings)
    if "masks" in clean:
        from adjust.masks import active

        # A gradient that has been drawn but changes nothing is no edit yet.
        if not active(clean["masks"]):
            del clean["masks"]
    if "nr_luma" in clean or "nr_color" in clean:
        # Before everything, the spots included: they are healed from clean pixels.
        frame = noise_reduced(frame, clean.pop("nr_luma", 0), clean.pop("nr_color", 0), noise_wait)
        if not clean and not matched:
            return frame.encoded()
    if any(k in clean for k in FRAME_KEYS):
        # After the noise, which is taken out on the sensor's own grid, and before
        # the spots, which are placed on the corrected picture: the lens, then the
        # perspective.
        transform = {k: clean.pop(k) for k in TRANSFORM_KEYS if k in clean}
        frame = lens_corrected(frame, clean.pop("lens_profile", 0), clean.pop("distortion", 0), transform)
        if not clean and not matched:
            return frame.encoded()
    if "spots" in clean:
        # First, as in Lightroom: everything after sees the picture without them.
        frame = healed(frame, clean.pop("spots"))
        if not clean and not matched:
            return frame.encoded()
    if not clean and not matched:
        return frame.encoded()
    if not matched and not any(k not in GEOMETRY_KEYS for k in clean):
        # Only framing: the picture is as the camera made it, turned and cropped.
        return apply_geometry(frame.encoded(), clean)

    change = combine(_linear_gain_of_encoded(matched), point_of(clean))
    enc = _light_and_tone(frame, clean, change)

    # After the tone, as in Lightroom: the curve, then the colour mixer, then the
    # look on top of both. A black-and-white picture is turned grey first, so its
    # curve shapes the grey exactly as drawn (and a channel's curve tones it),
    # and the colour mixer, which has no colour left to work on, is skipped.
    mono = bool(clean.get("mono"))
    if mono:
        enc = _to_mono(enc, clean)
    if any(key in clean for key in CURVE_KEYS):
        enc = _apply_curves(enc, clean)
    if not mono and any(key in clean for key in HSL_SLIDERS):
        enc = _apply_mixer(enc, clean)

    local_saturation = _mask_saturation(frame, clean.get("masks"))
    if local_saturation is not None or any(key in clean for key in
                                           ("vibrance", "saturation", "fade", "shadow_sat", "highlight_sat")):
        enc = _apply_look(enc, clean, local_saturation)

    if "sharpen" in clean:
        # Last on the colour, before framing and grain: it sharpens the picture
        # as finished, and the grain laid on after is not sharpened with it.
        from adjust.sharpen import sharpen

        enc = sharpen(enc, clean, frame.scale)

    picture = apply_geometry(to_image(enc), clean)
    if clean.get("grain") or clean.get("vignette"):
        picture = _finish(picture, clean, frame.window)
    return picture


# ── Auto ─────────────────────────────────────────────────────────────

AUTO_MEASURE_SIZE = 384
# Target for the frame's median, in linear light. A little under middle grey:
# most scenes average darker than the 18% a meter assumes, and aiming at grey
# itself leaves every frame looking washed out.
AUTO_MEDIAN_TARGET = 0.14
AUTO_MAX_EV = 2.5
AUTO_EV_DAMPING = 0.75
AUTO_EV_DEADBAND = 0.2


def _look_stops() -> np.ndarray:
    from adjust.look import STOPS

    return STOPS


def auto_develop(frame: LinearFrame) -> dict:
    """Suggest an edit set for a frame, measured from the frame itself.

    Exposure first, because every other control is defined relative to it, and
    then whatever the exposed frame still needs: highlights if the bright end
    runs past white, shadows if the dark end is buried, and the HDR squeeze when
    the scene spans more range than a picture can show.

    Deliberately conservative about exposure — it is moved most of the way to a
    sensible median, not all of it, and not at all inside a deadband — because a
    frame that is meant to be dark should stay dark. The rest is arithmetic on
    the frame's own histogram.
    """
    rgb = resize_linear(frame.rgb, AUTO_MEASURE_SIZE)
    luma = (rgb @ LUMA).reshape(-1)
    if luma.size < 4:
        return {}
    p1, p5, p50, p95, p99 = (float(v) for v in np.percentile(luma, [1, 5, 50, 95, 99]))
    p50 = max(p50, _FLOOR)

    out: dict = {}
    look = frame.look
    if look is not None:
        # The camera has already chosen where this frame sits, so exposure is
        # judged on the picture its look will show, and left alone unless that
        # picture is far from a sensible median. What the camera cannot do is
        # keep what is above its white, so that is where Auto looks.
        stops = np.log2(np.maximum(luma, _FLOOR)) - LOG_MID
        shown = float(np.median(np.interp(stops, _look_stops(), look.curve)))
        if not 0.30 <= shown <= 0.62:
            target = float(decode(np.array([0.42], dtype=np.float32))[0])
            now = float(decode(np.array([shown], dtype=np.float32))[0])
            ev = max(-AUTO_MAX_EV, min(AUTO_MAX_EV, math.log2(target / max(now, _FLOOR)) * 0.5))
        else:
            ev = 0.0
        if abs(ev) >= AUTO_EV_DEADBAND:
            out["exposure"] = round(ev * 20.0) / 20.0
        gain = 2.0 ** out.get("exposure", 0.0)
        over = math.log2(max(p99 * gain, _FLOOR)) - LOG_MID - look.white_stops
        if over > 0.0:
            out["highlights"] = -int(min(100.0, 35.0 + 45.0 * over))
        dark = float(np.interp(math.log2(max(p5 * gain, _FLOOR)) - LOG_MID,
                               _look_stops(), look.curve))
        if dark < 0.12:
            out["shadows"] = int(min(85.0, (0.12 - dark) * 520.0))
        span = math.log2(max(p99 * gain, _FLOOR) / max(p1 * gain, _FLOOR))
        if span > 10.5:
            out["compress"] = int(min(70.0, (span - 10.5) * 14.0))
        return normalize(out)

    ev = math.log2(AUTO_MEDIAN_TARGET / p50) * AUTO_EV_DAMPING
    ev = max(-AUTO_MAX_EV, min(AUTO_MAX_EV, ev))
    if abs(ev) >= AUTO_EV_DEADBAND:
        out["exposure"] = round(ev * 20.0) / 20.0
    ev = out.get("exposure", 0.0)
    gain = 2.0 ** ev

    top = p99 * gain
    if top > 1.0:
        out["highlights"] = -int(min(100.0, 35.0 + 55.0 * math.log2(top)))
    bottom = max(p5 * gain, _FLOOR)
    # How buried the dark end is: encoded value of the 5th percentile.
    dark = float(encode(np.array([bottom]))[0])
    if dark < 0.12:
        out["shadows"] = int(min(85.0, (0.12 - dark) * 520.0))

    span = math.log2(max(p99 * gain, _FLOOR) / max(p1 * gain, _FLOOR))
    if span > 9.0:
        out["compress"] = int(min(70.0, (span - 9.0) * 14.0))
    return normalize(out)
