"""
The edit model for one photograph.

This module is the *description* of an edit: the controls, their ranges, how a
settings dictionary is cleaned and how it is written to and read from an XMP
sidecar. Performing an edit is :mod:`adjust.render`, which works in linear light
on a decode with headroom. The 8-bit tone table and Auto below are the original
implementation and are kept as the plain-picture fallback and for their tests;
nothing on the preview, thumbnail or export path uses them when a frame has an
edit.

Nothing here is a judgement about a photograph. It is the arithmetic of the
controls a photographer reaches for — exposure, white balance, and the tonal
ranges — written once so that the preview, the exported file, and the sidecar
all describe the same edit.

Two families of control, applied in that order:

  point corrections   exposure and white balance. These are per-channel curves,
                      the same shape the set-matching correction already builds,
                      so the two compose into one pass over the pixels instead
                      of two. Composing is also more accurate: two passes over
                      the pixels round twice, one does not.

  tone ranges         highlights, shadows, whites, blacks, contrast. Each is a
                      gain that applies only inside part of the tonal range, so
                      it is a function of a pixel's *luminance* and cannot be
                      folded into a per-channel curve. They become one
                      256-entry table indexed by luminance, applied as a single
                      multiplicative pass.

The order matters. Exposure comes first because the tone ranges are defined
relative to the exposure already chosen — which is what lets a highlight
recovery pull the same pixels down whether the frame was shot bright or dark.

What these controls deliberately do not do:

* They do not invent detail. Every tone control is a gain, so a pixel that is
  already pure black stays black when the shadows are lifted. Rebalancing a
  tonal range and recovering clipped detail are different claims, and only the
  first one is honest here.
* They do not decide what the photograph should look like. :func:`auto_tone`
  sets the black and white points from the frame's own histogram and stops
  there; it does not touch exposure, because a frame that is meant to be dark
  should stay dark.

Unlike a matching correction, a hand-dragged control is not clamped to a
whisper. The matching limits exist because that feature promises consistency,
not a re-edit; someone who asks for two stops more wants two stops more.
"""

import re
from typing import Optional

import numpy as np
from PIL import Image

from adjust.geometry import GEOMETRY_KEYS, normalize_geometry
from adjust.masks import normalize_masks, to_text as masks_to_text
from adjust.spots import SPOTS_KEY, normalize_spots, to_text as spots_to_text
from analysis.harmonize import apply_correction

# ── Point corrections ─────────────────────────────────────────────────
#
# Slider ranges. Exposure is in stops; temperature and tint are -100..100 like
# every other slider in this app, scaled to a gain here.

# Wide enough to rescue a frame that was shot three or four stops under. The
# float pipeline in adjust.render does the arithmetic in linear light with
# headroom above white, so a large exposure move is a real re-exposure and not
# a multiplication of values that were already clipped.
MAX_EXPOSURE_EV = 5.0

# A full temperature swing is a 25% gain on red and the same in the other
# direction on blue. That is a visible cast without being a colour filter: past
# this the sliders stop being an adjustment and start being a look, which is
# what the style presets are for.
MAX_WB_GAIN = 0.25

# Tint moves green the opposite way to magenta. It needs a smaller reach than
# temperature because green sits between the other two channels, so the same
# gain reads far stronger.
MAX_TINT_GAIN = 0.15

# Below these a control is doing nothing, and the difference is worth reporting
# as "no edit" rather than a curve that happens to be a straight line.
MIN_EV = 0.01
MIN_GAIN = 0.005

# ── Tone ranges ───────────────────────────────────────────────────────
#
# The tone sliders use Lightroom's -100..+100 scale. That is not imitation for
# its own sake: the sidecar writes the same numbers into crs:Highlights2012 and
# friends, so a value copied out of this app means the same thing in Lightroom.
# The gain each control applies at full deflection is what makes the two scales
# line up in practice.

POINT_SLIDERS = ("exposure", "temperature", "tint")
TONE_SLIDERS = ("contrast", "highlights", "shadows", "whites", "blacks")

# Controls that only the float pipeline (adjust.render) understands. They are
# the half of a develop edit that cannot be done honestly on an 8-bit picture:
# squeezing a wide scene into a narrow display range, local contrast, and the
# colour and look controls that sit on top of a finished exposure.
#
#   compress        0..100   squeeze the whole scene's range, keep local detail
#   clarity         -100..100  mid-tone local contrast
#   vibrance        -100..100  saturation that favours the flat colours
#   saturation      -100..100  all colours alike
#   fade            0..100   lift the blacks, ease the whites: the matte look
#   shadow_hue / shadow_sat, highlight_hue / highlight_sat
#                   split toning: a hue (0..360) and a strength (0..100) for
#                   each end of the range
#   style_fade      0..100   with a ``style`` (the id of a learned style, see
#                   adjust.styles), how far to fade it back towards the camera's
#                   own look: 0 is the whole style
#   flat            0..100   how far to leave the camera's own rendering behind
#                   (0 = the look learned from the embedded JPEG, 100 = the plain
#                   decode). Stored this way round so that "no edit" is the
#                   camera's look, which is where a RAW frame should start.
LOOK_SLIDERS = (
    "flat", "style_fade", "compress", "clarity", "vibrance", "saturation", "fade",
    "shadow_hue", "shadow_sat", "highlight_hue", "highlight_sat",
)
# The colour mixer: hue, saturation and luminance for each of eight bands of
# hue, -100..100 like Lightroom's HSL panel. The bands and their centres (in
# degrees) are Lightroom's too, so ``sat_blue -40`` means the same there.
HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HSL_CENTRES = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0)
HSL_SLIDERS = tuple(f"{kind}_{band}" for kind in ("hue", "sat", "lum") for band in HSL_BANDS)

# Black and white. ``mono`` (0 or 1) is the treatment: the picture is turned to
# grey after the tone and before the tone curve, so the curve shapes the grey as
# drawn and split toning tints it. ``bw_<band>`` (-100..100) is how light each
# band of colour turns in grey — the colour filters of film, Lightroom's B&W mix
# — and is kept when the treatment is switched off, as Lightroom keeps it.
BW_SLIDERS = ("mono",) + tuple(f"bw_{band}" for band in HSL_BANDS)

# Effects on the finished, framed picture. ``grain`` 0..100 is film grain and
# ``grain_size`` -100..100 its size (0 is a 35 mm film's grain, finer to the
# left); ``vignette`` -100..100 darkens (left) or lightens (right) the corners of
# the picture as cropped.
EFFECT_SLIDERS = ("grain", "grain_size", "vignette")

# Noise reduction (adjust.noise), 0..100 each: how much of the luminance noise
# and of the colour noise a learned denoiser takes out of a RAW frame. Both at 0
# is the frame as decoded, with the decode's own light colour smoothing.
NOISE_SLIDERS = ("nr_luma", "nr_color")

# Sharpening (adjust.sharpen): amount 0..150, and the radius, detail and edge
# masking that shape it, stored so that 0 is Lightroom's default for each.
SHARPEN_SLIDERS = ("sharpen", "sharpen_radius", "sharpen_detail", "sharpen_masking")

# Lens corrections (adjust.lens): ``lens_profile`` (0 or 1) corrects the
# distortion as the camera does for this frame, ``distortion`` (-100..100) by hand
# on top, positive straightening a barrel.
LENS_SLIDERS = ("lens_profile", "distortion")
# Transform (adjust.lens): the picture plane tilted about its horizontal axis
# (tf_vertical) and its vertical one (tf_horizontal), and its aspect, -100..100.
TRANSFORM_SLIDERS = ("tf_vertical", "tf_horizontal", "tf_aspect")

SETTINGS_SLIDERS = (POINT_SLIDERS + TONE_SLIDERS + LOOK_SLIDERS + HSL_SLIDERS
                    + BW_SLIDERS + EFFECT_SLIDERS + NOISE_SLIDERS + SHARPEN_SLIDERS
                    + LENS_SLIDERS + TRANSFORM_SLIDERS)

# The tone curve: points (x, y) on 0..255, input to output, for all three
# channels together and then for each one. A list rather than a number, so it is
# cleaned by :func:`normalize_curve` instead of clamped like a slider. Applied on
# the finished, display-encoded picture, as Lightroom's point curve is.
CURVE_KEYS = ("curve", "curve_r", "curve_g", "curve_b")
MAX_CURVE_POINTS = 16

# Local adjustments: a list of gradients, each with its own controls
# (adjust.masks). Like a curve, a value that is not a number.
MASKS_KEY = "masks"

# Every key an edit may carry, sliders, curves, gradients, spots and framing alike.
EDIT_KEYS = SETTINGS_SLIDERS + CURVE_KEYS + (MASKS_KEY, SPOTS_KEY) + GEOMETRY_KEYS
# What belongs to one frame and is not put on another with its edit (a paste,
# a preset, a burst): the crop and level, and the spots removed from it.
OWN_KEYS = GEOMETRY_KEYS + (SPOTS_KEY,) + TRANSFORM_SLIDERS

# (low, high) for the controls that are not the usual -100..100.
SLIDER_RANGES = {
    "flat": (0.0, 100.0),
    "style_fade": (0.0, 100.0),
    "compress": (0.0, 100.0),
    "fade": (0.0, 100.0),
    "shadow_sat": (0.0, 100.0),
    "highlight_sat": (0.0, 100.0),
    "shadow_hue": (0.0, 360.0),
    "highlight_hue": (0.0, 360.0),
    "mono": (0.0, 1.0),
    "grain": (0.0, 100.0),
    "nr_luma": (0.0, 100.0),
    "nr_color": (0.0, 100.0),
    "sharpen": (0.0, 150.0),
    "sharpen_radius": (-5.0, 20.0),
    "sharpen_detail": (-25.0, 75.0),
    "sharpen_masking": (0.0, 100.0),
    "lens_profile": (0.0, 1.0),
}
# A style id, as adjust.styles writes them.
_STYLE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")

# A setting that means nothing without another: a hue without its strength, a
# grain size without any grain. It is dropped when the other is at 0.
HUE_OF = {"shadow_hue": "shadow_sat", "highlight_hue": "highlight_sat", "grain_size": "grain",
          "sharpen_radius": "sharpen", "sharpen_detail": "sharpen", "sharpen_masking": "sharpen"}

MAX_TONE_VALUE = 100.0

# Gain at full deflection, per control. Shadows reach furthest because lifting
# a shadow is the correction people actually notice. Whites and blacks are
# anchors at the ends of the range and are kept tighter, so that they set a
# point rather than re-exposing the frame.
MAX_SHADOW_GAIN = 0.90
MAX_HIGHLIGHT_GAIN = 0.85
MAX_WHITE_GAIN = 0.50
MAX_BLACK_GAIN = 0.50

# Contrast is a power law through the mid-tone pivot, so it steepens the curve
# without moving the point the picture is built around. The exponent is the
# strength at full deflection: 0.3 is roughly a third of a stop either side of
# the pivot, which is as far as an automatic contrast change should reach.
MAX_CONTRAST_EXPONENT = 0.3
TONE_PIVOT = 0.5

# Where the tonal weights fall off, in normalised luma. Shadows and highlights
# overlap across the mid-tones because that is where a photograph's subject
# lives; whites and blacks are deliberately narrow, since they are meant to
# move the ends of the range and nothing near the middle.
_SHADOW_END = 0.55
_HIGHLIGHT_START = 0.45
_BLACK_END = 0.28
_WHITE_START = 0.72

# A composed gain is clamped to this, so a stack of extreme controls cannot
# turn a mid-tone into a blown highlight by arithmetic alone.
MAX_TONE_GAIN = 4.0

# Rec. 709 luma, matching analysis.image_stats so that the tonal range a
# control affects is the same tonal range the measurements describe.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# ── Auto tone ─────────────────────────────────────────────────────────
#
# Auto tone aims the 5th and 95th percentiles at these values rather than at 0
# and 255. A photograph is not a calibration target: a frame whose blacks
# genuinely sit at 6 is already set correctly, and moving it would be a change,
# not an improvement.

TARGET_BLACK = 6.0
TARGET_WHITE = 242.0

# How far off target a frame has to be before auto tone uses that control at
# all. Without a deadband auto tone would be nudging every frame in the
# library, and a nudge that is invisible is still a second pass over the pixels.
AUTO_DEADBAND = 12.0

# Ceilings on what an automatic decision may ask for — not on a hand-dragged
# slider, which is allowed to go as far as the person dragging it wants.
AUTO_MAX_HIGHLIGHTS = 70.0
AUTO_MAX_WHITES = 45.0
AUTO_MAX_SHADOWS = 60.0
AUTO_MAX_BLACKS = 45.0
AUTO_MAX_CONTRAST = 40.0

# How hard auto pushes contrast as a frame falls short of spanning the range. A
# frame occupying a fifth of it needs a real curve, not a nudge: a gentle power
# law on a flat frame moves every pixel by one level, which is a change nobody
# can see.
AUTO_CONTRAST_SLOPE = 110.0

# What counts as a change worth offering, measured on the probe: a clear step on
# the 0-255 scale, on at least this much of the frame.
AUTO_VISIBLE_STEP = 2.0
AUTO_VISIBLE_SHARE = 0.02

# Contrast is a curve anchored on the mid-tones, so it is only worth suggesting
# for a frame that has mid-tones to anchor it. A low-key frame whose whole
# content sits in the shadows would only be pushed further down.
CONTRAST_PIVOT_LOW = 64.0
CONTRAST_PIVOT_HIGH = 191.0

# Auto tone measures a small copy. Percentiles of a 256px frame are stable to
# well under a unit, and the alternative is a full-size sort on every click.
AUTO_MEASURE_SIZE = 256


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coerce(value, low: float, high: float) -> float:
    """Read one slider out of a stored blob, or 0.0 when it cannot be read.

    A control that cannot be read is treated as untouched rather than as an
    error: an edit set that a newer version of the app wrote should export the
    photo as it is, not fail the export.
    """
    try:
        return _clamp(float(value), low, high)
    except (TypeError, ValueError):
        return 0.0


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """0 before edge0, 1 after edge1, smooth between.

    Smooth matters. A linear ramp has a kink at both ends, and a kink in the
    gain curve shows up as a band of flat tone across the middle of a sky.
    """
    if edge1 <= edge0:
        return 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ── Point corrections ─────────────────────────────────────────────────

def adjustment(
    exposure: float = 0.0,
    temperature: int = 0,
    tint: int = 0,
) -> Optional[dict]:
    """Turn three sliders into a correction, or ``None`` when they are all at zero.

    ``exposure`` is in stops (an EV of +1 doubles the recorded values),
    ``temperature`` warms towards red and away from blue, ``tint`` moves
    towards magenta when positive and green when negative. Values outside the
    ranges are pulled back to them rather than rejected: a slider that has been
    dragged too far is still an intention.
    """
    try:
        ev = float(exposure)
        temperature = int(temperature)
        tint = int(tint)
    except (TypeError, ValueError):
        return None

    ev = max(-MAX_EXPOSURE_EV, min(MAX_EXPOSURE_EV, ev))
    temperature = max(-100, min(100, temperature))
    tint = max(-100, min(100, tint))

    warm = (temperature / 100.0) * MAX_WB_GAIN
    green = -(tint / 100.0) * MAX_TINT_GAIN

    change = {
        "ev": ev,
        "wb_r": 1.0 + warm,
        "wb_g": 1.0 + green,
        "wb_b": 1.0 - warm,
    }
    return None if is_identity(change) else change


def is_identity(change: Optional[dict]) -> bool:
    """True when a correction would leave every pixel exactly as it is."""
    if not change:
        return True
    if abs(float(change.get("ev", 0.0))) >= MIN_EV:
        return False
    return all(
        abs(float(change.get(key, 1.0)) - 1.0) < MIN_GAIN
        for key in ("wb_r", "wb_g", "wb_b")
    )


def combine(*changes: Optional[dict]) -> Optional[dict]:
    """Compose corrections into the one curve that does all of them.

    Every term is a multiplication — stops scale all three channels, white
    balance scales one — so the order they arrive in does not matter and the
    result is a single per-channel curve. Composing is also more accurate than
    applying each in turn: two passes over the pixels round twice, one does not.

    Returns ``None`` when there is nothing to apply, so a caller can skip the
    pass through the pixels entirely.
    """
    ev = 0.0
    gains = {"wb_r": 1.0, "wb_g": 1.0, "wb_b": 1.0}
    for change in changes:
        if not change:
            continue
        ev += float(change.get("ev", 0.0))
        for key in gains:
            gains[key] *= float(change.get(key, 1.0))

    combined = {"ev": ev, **gains}
    return None if is_identity(combined) else combined


def describe(change: Optional[dict]) -> str:
    """One line for the log: what is about to be done to this frame."""
    if not change:
        return "no adjustment"
    return (
        f"{change['ev']:+.2f} EV, "
        f"WB r{change['wb_r']:.3f} g{change['wb_g']:.3f} b{change['wb_b']:.3f}"
    )


def point_of(settings: Optional[dict]) -> Optional[dict]:
    """The point-correction half of an edit set, in the shape matching uses."""
    if not settings:
        return None
    return adjustment(
        exposure=settings.get("exposure", 0.0),
        temperature=settings.get("temperature", 0),
        tint=settings.get("tint", 0),
    )


# ── The settings dictionary ───────────────────────────────────────────

def normalize(settings: Optional[dict]) -> dict:
    """Clamp every control into range and drop the ones doing nothing.

    An empty dict means "no edit", and every caller treats that as a signal to
    skip the pass entirely rather than to run it with neutral values. Unknown
    keys are dropped, so an edit set written by a newer version of the app
    degrades to the controls this one understands instead of failing.
    """
    if not settings:
        return {}
    clean: dict = {}
    for key in SETTINGS_SLIDERS:
        if key not in settings:
            continue
        if key == "exposure":
            value = _coerce(settings[key], -MAX_EXPOSURE_EV, MAX_EXPOSURE_EV)
            if abs(value) < MIN_EV:
                continue
            clean[key] = round(value, 4)
        elif key in POINT_SLIDERS:
            value = int(_coerce(settings[key], -100.0, 100.0))
            if value == 0:
                continue
            clean[key] = value
        else:
            low, high = SLIDER_RANGES.get(key, (-MAX_TONE_VALUE, MAX_TONE_VALUE))
            value = int(round(_coerce(settings[key], low, high)))
            if value == 0:
                continue
            clean[key] = value
    for key in CURVE_KEYS:
        curve = normalize_curve(settings.get(key))
        if curve:
            clean[key] = curve
    masks = normalize_masks(settings.get(MASKS_KEY))
    if masks:
        clean[MASKS_KEY] = masks
    spots = normalize_spots(settings.get(SPOTS_KEY))
    if spots:
        clean[SPOTS_KEY] = spots
    # The style is the one control that is a name rather than a number.
    style = settings.get("style")
    if isinstance(style, str) and _STYLE_ID.match(style):
        clean["style"] = style
    else:
        clean.pop("style_fade", None)
    clean.update(normalize_geometry(settings))
    # A hue with no strength behind it is not an edit.
    for hue, strength in HUE_OF.items():
        if hue in clean and strength not in clean:
            del clean[hue]
    return clean


def normalize_curve(points) -> Optional[list]:
    """A tone curve as ``[[x, y], …]`` on 0..255, sorted, or ``None`` for no curve.

    Points may arrive as pairs or as ``{"x": …, "y": …}``. Each is rounded and
    clamped, a repeated x keeps its last y, and the curve is limited to
    :data:`MAX_CURVE_POINTS`. A curve with fewer than two points, or one whose
    every point sits on the diagonal, changes nothing and is dropped, so "no
    curve" has one spelling.
    """
    if not points or not isinstance(points, (list, tuple)):
        return None
    by_x: dict = {}
    for point in points:
        try:
            if isinstance(point, dict):
                x, y = point["x"], point["y"]
            else:
                x, y = point
            x = int(round(_clamp(float(x), 0.0, 255.0)))
            y = int(round(_clamp(float(y), 0.0, 255.0)))
        except (KeyError, TypeError, ValueError):
            continue
        by_x[x] = y
    curve = [[x, by_x[x]] for x in sorted(by_x)][:MAX_CURVE_POINTS]
    if len(curve) < 2 or all(x == y for x, y in curve):
        return None
    return curve


# ── Tone ranges ───────────────────────────────────────────────────────

def tone_table(settings: Optional[dict]) -> Optional[list[float]]:
    """Build the 256-entry luminance-indexed gain for an edit set.

    ``None`` when no tone control is set, which is the caller's signal to skip
    the pass. Building the table once and indexing it is both faster and more
    honest than evaluating the weights per pixel: the curve a slider describes
    is exactly the curve every pixel gets, and it can be read in a test.
    """
    clean = normalize(settings)
    if not any(key in clean for key in TONE_SLIDERS):
        return None

    contrast = clean.get("contrast", 0) / MAX_TONE_VALUE
    highlights = clean.get("highlights", 0) / MAX_TONE_VALUE
    shadows = clean.get("shadows", 0) / MAX_TONE_VALUE
    whites = clean.get("whites", 0) / MAX_TONE_VALUE
    blacks = clean.get("blacks", 0) / MAX_TONE_VALUE

    gamma = 2.0 ** (contrast * MAX_CONTRAST_EXPONENT)

    table: list[float] = []
    for value in range(256):
        luma = value / 255.0
        gain = 1.0
        if gamma != 1.0:
            # Power law through the pivot: the mid-tone stays where it is and
            # everything away from it moves. Clamping the base keeps a negative
            # contrast setting from dividing by zero at pure black.
            gain *= (max(luma, 1.0 / 255.0) / TONE_PIVOT) ** (gamma - 1.0)
        gain *= 1.0 + shadows * MAX_SHADOW_GAIN * (1.0 - _smoothstep(0.0, _SHADOW_END, luma))
        gain *= 1.0 + highlights * MAX_HIGHLIGHT_GAIN * _smoothstep(_HIGHLIGHT_START, 1.0, luma)
        gain *= 1.0 + blacks * MAX_BLACK_GAIN * (1.0 - _smoothstep(0.0, _BLACK_END, luma))
        gain *= 1.0 + whites * MAX_WHITE_GAIN * _smoothstep(_WHITE_START, 1.0, luma)
        table.append(round(_clamp(gain, 0.0, MAX_TONE_GAIN), 6))
    return table


def apply_tone(image: Image.Image, table: list[float]) -> Image.Image:
    """Apply a luminance-indexed gain to every pixel.

    The gain is looked up by the pixel's luma, so it moves brightness without
    touching the ratio between the channels — a shadow lifts, it does not turn
    blue. Working in float and rounding once at the end is the difference
    between one pass over the pixels and several.

    Rounded rather than truncated on the way back to whole numbers: a slight
    shadow lift puts a pixel at 5 into 5.4, and truncating that back to 5 would
    swallow the control exactly where it is supposed to be working. The point
    corrections round the same way, and the two passes have to agree.
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.size == 0:
        return image
    lum = arr @ _LUMA_WEIGHTS
    index = np.clip(np.rint(lum), 0, 255).astype(np.uint8)
    arr *= np.asarray(table, dtype=np.float32)[index][..., None]
    np.clip(np.rint(arr), 0.0, 255.0, out=arr)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def apply_develop(image: Image.Image, settings: Optional[dict]) -> Image.Image:
    """Apply an edit set to an image: point corrections, then tone ranges.

    Returns the image unchanged when the edit set is empty, so a caller can
    hand it whatever it has.
    """
    clean = normalize(settings)
    if not clean:
        return image
    change = point_of(clean)
    if change:
        image = apply_correction(image, change)
    table = tone_table(clean)
    if table:
        image = apply_tone(image, table)
    return image


# ── Auto tone ─────────────────────────────────────────────────────────

def _probe(image: Image.Image) -> Optional[Image.Image]:
    """A small copy to measure, and to try a suggestion on.

    Copied rather than resized in place: the frame handed in is usually the
    cached preview, which is shared and must not be written to.
    """
    probe = image.copy()
    probe.thumbnail((AUTO_MEASURE_SIZE, AUTO_MEASURE_SIZE), Image.BILINEAR)
    rgb = probe.convert("RGB")
    if rgb.width < 2 or rgb.height < 2:
        return None
    return rgb


def _measure_luma(probe: Image.Image) -> tuple:
    """Percentiles of a probe, in 0-255 luma."""
    flat = (np.asarray(probe, dtype=np.float32) @ _LUMA_WEIGHTS).reshape(-1)
    p05, p50, p95 = (float(v) for v in np.percentile(flat, [5.0, 50.0, 95.0]))
    return p05, p50, p95


def _visible(probe: Image.Image, settings: dict) -> bool:
    """Would this control change this frame enough to be worth offering?

    A control that moves a frame by a level or two on a handful of pixels is a
    number in the panel and nothing on the picture. Asking the frame rather than
    the slider is the difference between an automatic control and a plausible
    looking one: on a folder of night shots, every frame has its darkest pixels
    sitting on black, so a percentile rule alone would offer a shadow lift on
    all of them, and change none of them.
    """
    before = np.asarray(probe, dtype=np.float32)
    after = np.asarray(apply_develop(probe, settings).convert("RGB"), dtype=np.float32)
    moved = np.abs(after - before).max(axis=2)
    return bool((moved >= AUTO_VISIBLE_STEP).mean() >= AUTO_VISIBLE_SHARE)


def auto_tone(image: Image.Image) -> dict:
    """Set this frame's black and white points from its own histogram.

    This is the part of the edit that does not need a photographer. A frame
    whose 95th percentile is sitting on pure white has clipped highlights, and
    pulling them back is arithmetic. A frame whose blacks are milky wants the
    black point lowered. A frame that occupies only a narrow band of tone wants
    contrast.

    It measures the frame it is about to change, not the cached thumbnail
    statistics. That is deliberate: the stored measurements were taken from the
    camera's own rendering, which already has the camera's tone curve baked in,
    while a RAW export is decoded without it. Auto tone driven off the wrong
    histogram would over-correct every RAW file in the library.

    Exposure is left alone. Where a frame sits overall is a decision about the
    photograph; where its endpoints sit is a fact about the file.

    A control is only suggested when the frame has pixels it could act on, and
    then only when acting on them would show. A high-key frame with nothing
    below mid-grey does not need its blacks lowered, a dark frame has no whites
    to raise, and a frame whose shadows are already flat black gains nothing
    from being told its black point is 6 levels off. Offering those sliders
    anyway is how an automatic control ends up looking like a mistake.
    """
    probe = _probe(image)
    if probe is None:
        return {}
    p05, p50, p95 = _measure_luma(probe)

    candidates: dict = {}

    # The white end. Highlights only fire on a frame that is genuinely sitting
    # on white; whites only on a frame that has a bright end to move up.
    if p95 > TARGET_WHITE:
        candidates["highlights"] = -int(
            min(AUTO_MAX_HIGHLIGHTS, round((p95 - TARGET_WHITE) * 1.2))
        )
    elif p95 < TARGET_WHITE - AUTO_DEADBAND and p95 > _WHITE_START * 255.0:
        candidates["whites"] = int(
            min(AUTO_MAX_WHITES, round((TARGET_WHITE - p95) * 0.9))
        )

    # The black end. The deadband on the milky-blacks case is what keeps a
    # frame that is merely dark from being told its blacks are wrong.
    if p05 < TARGET_BLACK:
        candidates["shadows"] = int(
            min(AUTO_MAX_SHADOWS, round((TARGET_BLACK - p05) * 1.6))
        )
    elif p05 > TARGET_BLACK + AUTO_DEADBAND and p05 < _BLACK_END * 255.0:
        candidates["blacks"] = -int(
            min(AUTO_MAX_BLACKS, round((p05 - TARGET_BLACK) * 1.1))
        )

    # Contrast only when the frame occupies a narrow band of tone at all, and
    # only when its mid-tones sit where the curve is anchored. A frame that
    # already spans the range does not need it, and a low-key frame whose whole
    # content sits in the shadows would only be pushed further down by it. This
    # is how automatic tone controls end up making a picture worse.
    spread = (p95 - p05) / 255.0
    if spread < 0.55 and CONTRAST_PIVOT_LOW <= p50 <= CONTRAST_PIVOT_HIGH:
        candidates["contrast"] = int(
            min(AUTO_MAX_CONTRAST, round((0.55 - spread) * AUTO_CONTRAST_SLOPE))
        )

    return normalize(
        {
            key: value
            for key, value in candidates.items()
            if _visible(probe, {key: value})
        }
    )


# ── Describing and exporting an edit ──────────────────────────────────

def describe_settings(settings: Optional[dict]) -> str:
    """A one-line account of a whole edit set, for the log and for the user."""
    clean = normalize(settings)
    if not clean:
        return "no edit"
    parts: list[str] = []
    if "exposure" in clean:
        parts.append(f"{clean['exposure']:+.2f} EV")
    tone = describe_tone(clean)
    if tone != "no tone":
        parts.append(tone)
    change = point_of(clean)
    if change:
        parts.append(
            "WB r{:.3f} g{:.3f} b{:.3f}".format(
                change["wb_r"], change["wb_g"], change["wb_b"]
            )
        )
    looks = [
        f"{key} {clean[key]:+d}"
        for key in ("compress", "clarity", "vibrance", "saturation", "fade")
        if key in clean
    ]
    if "shadow_sat" in clean or "highlight_sat" in clean:
        looks.append("split tone")
    if any(key in clean for key in CURVE_KEYS):
        looks.append("tone curve")
    if any(key in clean for key in HSL_SLIDERS):
        looks.append("colour mixer")
    if clean.get("mono"):
        looks.append("black and white")
    looks.extend(f"{key} {clean[key]:+d}" for key in ("grain", "vignette") if key in clean)
    if "nr_luma" in clean or "nr_color" in clean:
        looks.append(f"noise reduction {clean.get('nr_luma', 0)}/{clean.get('nr_color', 0)}")
    if clean.get("lens_profile"):
        looks.append("lens profile")
    if "distortion" in clean:
        looks.append(f"distortion {clean['distortion']:+d}")
    turns = [f"{name} {clean[key]:+d}" for key, name in
             (("tf_vertical", "vertical"), ("tf_horizontal", "horizontal"), ("tf_aspect", "aspect")) if key in clean]
    if turns:
        looks.append("transform " + ", ".join(turns))
    if "sharpen" in clean:
        from adjust.sharpen import detail_of, radius_px

        looks.append(f"sharpening {clean['sharpen']} at {radius_px(clean.get('sharpen_radius', 0)):.1f} px, "
                     f"detail {detail_of(clean.get('sharpen_detail', 0)):g}"
                     + (f", masking {clean['sharpen_masking']}" if "sharpen_masking" in clean else ""))
    if "style" in clean:
        looks.append(f"style {clean['style']}"
                     + (f" faded {clean['style_fade']}" if "style_fade" in clean else ""))
    if MASKS_KEY in clean:
        brushes = sum(1 for m in clean[MASKS_KEY] if m["kind"] == "brush")
        for n, word in ((len(clean[MASKS_KEY]) - brushes, "gradient"), (brushes, "brush")):
            if n:
                looks.append(f"{n} {word}{('es' if word == 'brush' else 's') if n != 1 else ''}")
    if SPOTS_KEY in clean:
        n = len(clean[SPOTS_KEY])
        looks.append(f"{n} spot{'s' if n != 1 else ''} removed")
    parts.extend(looks)
    return ", ".join(parts) if parts else "no edit"


def describe_tone(settings: Optional[dict]) -> str:
    """The tone half of an edit set on one line, for the export log."""
    clean = normalize(settings)
    parts = [
        f"{key.capitalize()} {clean[key]:+d}"
        for key in ("highlights", "shadows", "whites", "blacks", "contrast")
        if key in clean
    ]
    return ", ".join(parts) if parts else "no tone"


_LOOK_CRS = {
    "clarity": "crs:Clarity2012",
    "vibrance": "crs:Vibrance",
    "saturation": "crs:Saturation",
    "compress": "crs:ToneCompress",
    "flat": "crs:FlatLook",
    "fade": "crs:Fade",
    "shadow_hue": "crs:SplitToningShadowHue",
    "shadow_sat": "crs:SplitToningShadowSaturation",
    "highlight_hue": "crs:SplitToningHighlightHue",
    "highlight_sat": "crs:SplitToningHighlightSaturation",
    "style_fade": "crs:PhotoolStyleFade",
}
# The colour mixer under Lightroom's own names (crs:HueAdjustmentRed, …): the
# same bands on the same -100..100 scale.
_LOOK_CRS.update({
    f"{kind}_{band}": f"crs:{name}Adjustment{band.capitalize()}"
    for kind, name in (("hue", "Hue"), ("sat", "Saturation"), ("lum", "Luminance"))
    for band in HSL_BANDS
})
# Black and white, grain and vignette under Lightroom's names where its scale is
# this app's: the B&W mix (crs:GrayMixerRed …), grain amount and the post-crop
# vignette. The treatment itself is crs:ConvertToGrayscale, a "True"/"False"
# attribute rather than a number, and is written and read on its own below.
_LOOK_CRS.update({f"bw_{band}": f"crs:GrayMixer{band.capitalize()}" for band in HSL_BANDS})
_LOOK_CRS.update({
    "grain": "crs:GrainAmount",
    "grain_size": "crs:PhotoolGrainSize",
    "vignette": "crs:PostCropVignetteAmount",
})
# Noise reduction under names of this app's own. Lightroom's LuminanceSmoothing and
# ColorNoiseReduction are not this denoiser, and Lightroom writes a colour amount
# into every sidecar it makes, which read as this would run the model on all of them.
_LOOK_CRS.update({"nr_luma": "crs:PhotoolNoiseLuminance", "nr_color": "crs:PhotoolNoiseColor"})
# Sharpening, likewise. Lightroom's crs:Sharpness and friends sit in most sidecars
# it writes (its default is 40): read as this app's, they would sharpen every such
# frame; written, they would wipe out the sharpening Lightroom set there.
_LOOK_CRS.update({
    "sharpen": "crs:PhotoolSharpenAmount",
    "sharpen_radius": "crs:PhotoolSharpenRadius",
    "sharpen_detail": "crs:PhotoolSharpenDetail",
    "sharpen_masking": "crs:PhotoolSharpenMasking",
})
# Lens corrections, likewise: Lightroom's crs:LensProfileEnable names a profile from
# its own database, and crs:LensManualDistortionAmount is on its own scale.
_LOOK_CRS.update({"lens_profile": "crs:PhotoolLensProfile", "distortion": "crs:PhotoolDistortion"})
# The transform, likewise: Lightroom's crs:PerspectiveVertical and friends are on its
# own scales, and it would read ours as a perspective it then crops its own way.
_LOOK_CRS.update({"tf_vertical": "crs:PhotoolTransformVertical",
                  "tf_horizontal": "crs:PhotoolTransformHorizontal",
                  "tf_aspect": "crs:PhotoolTransformAspect"})
GRAYSCALE_CRS = "crs:ConvertToGrayscale"
# The curves under names of this app's own, as "x,y x,y …". Lightroom keeps its
# point curve as an rdf:Seq rather than an attribute, which the sidecar writer
# does not produce, so these round-trip here and are not shown there.
_CURVE_CRS = {
    "curve": "crs:PhotoolToneCurve",
    "curve_r": "crs:PhotoolToneCurveRed",
    "curve_g": "crs:PhotoolToneCurveGreen",
    "curve_b": "crs:PhotoolToneCurveBlue",
}
# A learned style, by id: this app's own attribute (Lightroom has no such thing).
_STYLE_CRS = "crs:PhotoolStyle"
# The gradients, as compact JSON under a name of this app's own. Lightroom keeps
# its masks in nested rdf structures of its own process version; these are not
# written there, so they round-trip here and are not shown in Lightroom.
_MASKS_CRS = "crs:PhotoolMasks"
# The spots removed (adjust.spots), the same way.
_SPOTS_CRS = "crs:PhotoolSpots"
# Written without a sign: they are 0..100 / 0..360, never negative.
_UNSIGNED = {"flat", "style_fade", "compress", "fade", "shadow_hue", "shadow_sat", "highlight_hue", "highlight_sat",
             "grain", "nr_luma", "nr_color", "sharpen", "sharpen_masking", "lens_profile"}


def to_crs(settings: Optional[dict]) -> dict:
    """The edit set in the names an XMP sidecar uses.

    ``crs:Exposure2012`` is in stops and the tone controls are -100..+100, the
    same scales this app uses, so those values carry over exactly. White
    balance does not: Lightroom's ``crs:Temperature`` is an absolute colour
    temperature in kelvin, while this app's warmth slider is a relative channel
    shift. It is written as ``crs:TemperatureShift`` — a tag Lightroom will not
    read, but which round-trips through this app and does not pretend to be
    something it is not. ``crs:Tint2012`` is the one exception that is worth
    converting: Lightroom's tint is -150..+150 on the same axis, so it is a
    scale factor and not a guess.

    Tint is written twice on purpose. ``crs:Tint2012`` is for Lightroom, and its
    1.5 scale factor turns a -100..100 slider into a whole number the other way
    only sometimes: 7 becomes 10.5, which rounds to 10, which reads back as 6. A
    sidecar that is the durable record of an edit cannot lose a unit every time
    it is opened, so ``crs:TintShift`` carries this app's own value exactly and
    is what :func:`from_crs` prefers.
    """
    clean = normalize(settings)
    out: dict = {}
    if "exposure" in clean:
        out["crs:Exposure2012"] = f"{clean['exposure']:+.2f}"
    for key in ("contrast", "highlights", "shadows", "whites", "blacks"):
        if key in clean:
            out[f"crs:{key.capitalize()}2012"] = f"{clean[key]:+d}"
    # Names Lightroom reads, on scales that line up with its own. Compress and
    # fade have no counterpart there, so they get tags of this app's own.
    for key, name in _LOOK_CRS.items():
        if key in clean:
            out[name] = f"{clean[key]:+d}" if key not in _UNSIGNED else f"{clean[key]:d}"
    if clean.get("mono"):
        out[GRAYSCALE_CRS] = "True"
    if "temperature" in clean:
        out["crs:TemperatureShift"] = f"{clean['temperature']:+d}"
    if "tint" in clean:
        out["crs:Tint2012"] = f"{int(round(clean['tint'] * 1.5)):+d}"
        out["crs:TintShift"] = f"{clean['tint']:+d}"
    for key, name in _CURVE_CRS.items():
        if key in clean:
            out[name] = " ".join(f"{x},{y}" for x, y in clean[key])
    if "style" in clean:
        out[_STYLE_CRS] = clean["style"]
    if MASKS_KEY in clean:
        out[_MASKS_CRS] = masks_to_text(clean[MASKS_KEY])
    if SPOTS_KEY in clean:
        out[_SPOTS_CRS] = spots_to_text(clean[SPOTS_KEY])
    out.update(_geometry_to_crs(clean))
    return out


# Crop and level. This app's own attributes carry the exact numbers (the box is
# measured on the turned frame, which is not how Lightroom measures it). When
# there is no turn the same rectangle is also written under Lightroom's names,
# where the two coordinate systems agree, so the crop shows in Lightroom too.
_GEOMETRY_CRS = {
    "angle": "crs:PhotoolAngle",
    "crop_x": "crs:PhotoolCropX",
    "crop_y": "crs:PhotoolCropY",
    "crop_w": "crs:PhotoolCropW",
    "crop_h": "crs:PhotoolCropH",
}
LIGHTROOM_CROP_KEYS = (
    "crs:HasCrop", "crs:CropLeft", "crs:CropTop", "crs:CropRight",
    "crs:CropBottom", "crs:CropAngle",
)


def _geometry_to_crs(clean: dict) -> dict:
    out: dict = {}
    for key, name in _GEOMETRY_CRS.items():
        if key in clean:
            out[name] = f"{clean[key]:g}"
    if "crop_x" in clean and "angle" not in clean:
        x, y = clean["crop_x"], clean["crop_y"]
        out["crs:HasCrop"] = "True"
        out["crs:CropLeft"] = f"{x:.6f}"
        out["crs:CropTop"] = f"{y:.6f}"
        out["crs:CropRight"] = f"{x + clean['crop_w']:.6f}"
        out["crs:CropBottom"] = f"{y + clean['crop_h']:.6f}"
        out["crs:CropAngle"] = "0"
    return out


# How much of Lightroom's tint range one unit of this app's tint covers.
TINT_SCALE = 1.5


def from_crs(develop: Optional[dict]) -> dict:
    """The edit set an XMP sidecar describes, in this app's slider names.

    Only the attributes :func:`to_crs` writes are read, and the exact ones are
    preferred over the converted ones. That is the whole reason the sidecar can
    be the durable record: reading back what was read gives the same edit, so
    the file on disk and the sliders on screen cannot drift apart.

    Anything else in the crs: block is deliberately ignored. A sidecar touched
    by Lightroom carries ``crs:Temperature`` as kelvin and ``crs:Exposure2012``
    measured against its own process version; taking those for this app's
    relative warmth and stops would silently re-edit every frame the first time
    it was opened.
    """
    if not develop:
        return {}

    def number(key: str) -> Optional[float]:
        try:
            return float(develop[key])
        except (KeyError, TypeError, ValueError):
            return None

    raw: dict = {}

    exposure = number("crs:Exposure2012")
    if exposure is not None:
        raw["exposure"] = exposure

    for key in ("contrast", "highlights", "shadows", "whites", "blacks"):
        value = number(f"crs:{key.capitalize()}2012")
        if value is not None:
            raw[key] = value

    for key, name in _LOOK_CRS.items():
        value = number(name)
        if value is not None:
            raw[key] = value
    if str(develop.get(GRAYSCALE_CRS, "")).strip().lower() == "true":
        raw["mono"] = 1

    temperature = number("crs:TemperatureShift")
    if temperature is not None:
        raw["temperature"] = temperature

    # The exact value when this app wrote it, and the converted one only for a
    # sidecar from before TintShift existed.
    tint = number("crs:TintShift")
    if tint is None:
        converted = number("crs:Tint2012")
        tint = None if converted is None else converted / TINT_SCALE
    if tint is not None:
        raw["tint"] = tint

    for key, name in _CURVE_CRS.items():
        text = develop.get(name)
        if isinstance(text, str):
            raw[key] = [pair.split(",") for pair in text.split()]

    for key, name in _GEOMETRY_CRS.items():
        value = number(name)
        if value is not None:
            raw[key] = value

    style = develop.get(_STYLE_CRS)
    if isinstance(style, str) and style.strip():
        raw["style"] = style.strip()

    masks = develop.get(_MASKS_CRS)
    if isinstance(masks, str) and masks.strip():
        raw[MASKS_KEY] = masks
    spots = develop.get(_SPOTS_CRS)
    if isinstance(spots, str) and spots.strip():
        raw[SPOTS_KEY] = spots

    return normalize(raw)


# The sidecar attributes this app owns. Writing an edit replaces exactly these,
# so a crop, a curve or a lens profile another editor put in the same block is
# left where it was: the sidecar is shared, and a shared file that loses half
# its contents when a slider moves is worse than no sidecar at all.
CRS_KEYS = (
    "crs:Exposure2012",
    "crs:Contrast2012",
    "crs:Highlights2012",
    "crs:Shadows2012",
    "crs:Whites2012",
    "crs:Blacks2012",
    "crs:TemperatureShift",
    "crs:TintShift",
    "crs:Tint2012",
    *_LOOK_CRS.values(),
    GRAYSCALE_CRS,
    *_CURVE_CRS.values(),
    _STYLE_CRS,
    _MASKS_CRS,
    _SPOTS_CRS,
    *_GEOMETRY_CRS.values(),
)
