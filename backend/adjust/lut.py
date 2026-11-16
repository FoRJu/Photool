"""A look as a 3D LUT (.cube), for a video editor or another photo app.

A LUT gives one colour out for each colour in. So it holds what an edit does to a
pixel by its colour alone: exposure and the relative white balance, the tone
sliders, presence, fade and toning, the curves, the colour mixer and black and
white. Nothing that depends on the pixels round it (clarity, sharpening, noise
reduction, lens blur), on where it is in the frame (grain, the vignette, lens
corrections), or on the RAW file (a learned style, the camera's look, an
absolute white balance, which is worked out from one file's own white) can be in
it. Those are left out and named, never an error, unless nothing is left.

The colours are display-encoded Rec. 709, in and out: the picture as the app
shows it with no edit, and the picture with the look.

Exposure above zero meets the highlight shoulder (adjust.render._shoulder),
whose strength follows the frame's brightest value. The lattice holds a full
white, so the LUT is the edit as it is on a picture that has one; on a picture
with no white in it the app's own rendering rolls off a little less.
"""

from typing import Optional

import numpy as np

from adjust.halation import KEYS as HALATION_KEYS
from adjust.develop import BW_SLIDERS, CURVE_KEYS, HSL_SLIDERS, normalize, point_of
from adjust.render import (
    LUMA,
    _FLOOR,
    LinearFrame,
    _apply_curves,
    _apply_look,
    _apply_mixer,
    _light_and_tone,
    _to_mono,
    decode,
)

SIZES = (17, 33, 65)

SPLIT_KEYS = ("shadow_hue", "shadow_sat", "highlight_hue", "highlight_sat")
POINT_KEYS = frozenset(
    ("exposure", "temperature", "tint", "contrast", "highlights", "lights", "darks", "shadows",
     "whites", "blacks", "compress", "vibrance", "saturation", "fade")
    + SPLIT_KEYS + CURVE_KEYS + HSL_SLIDERS + BW_SLIDERS
)

# What a LUT cannot hold, by the name the panel gives it, in the panel's order.
LEFT_OUT = (
    ("White balance", ("wb_kelvin", "wb_tint")),
    ("Camera look", ("flat",)),
    ("Style", ("style", "style_fade")),
    ("Clarity", ("clarity",)),
    ("Noise reduction", ("nr_luma", "nr_color")),
    ("Sharpening", ("sharpen", "sharpen_radius", "sharpen_detail", "sharpen_masking")),
    ("Lens blur", ("lens_blur", "blur_focus", "blur_sharp", "blur_outline")),
    ("Lens corrections", ("lens_profile", "distortion")),
    # It depends on the pixels round each one, so no LUT can hold it.
    ("Halation & bloom", HALATION_KEYS),
    ("Vignette", ("vignette",)),
    ("Grain", ("grain", "grain_size")),
    # A look's masks find a subject, a sky or a distance on each photo: a place
    # in the picture, which a colour in, colour out table has none of.
    ("Masks", ("masks",)),
)
_LOOK_STEP_KEYS = ("vibrance", "saturation", "fade", "shadow_sat", "highlight_sat")


def split(settings: Optional[dict]) -> tuple[dict, list[str]]:
    """The edit's point keys, and the names of what is left out."""
    clean = normalize(settings)
    point = {k: v for k, v in clean.items() if k in POINT_KEYS}
    rest = set(clean) - set(point)
    left_out = []
    for name, keys in LEFT_OUT:
        if rest & set(keys):
            left_out.append(name)
            rest -= set(keys)
    if rest:
        # A look never carries framing, spots or gradients; an edit that does
        # has them named rather than dropped without a word.
        left_out.append("Framing and local edits")
    return point, left_out


def lattice(size: int) -> np.ndarray:
    """``size³`` display-encoded colours 0..1, red changing fastest, then green,
    then blue (the .cube order), as an image of shape ``(size * size, size, 3)``."""
    steps = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(steps, steps, steps, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(size * size, size, 3)


def per_pixel_frame(encoded: np.ndarray) -> LinearFrame:
    """A frame of ``encoded`` (display-encoded 0..1) whose tone is per pixel: the
    base is the pixel itself and there is no detail, so nothing depends on the
    pixels round it. No camera look."""
    frame = LinearFrame(decode(encoded.astype(np.float32)))
    log_luma = np.log2(np.maximum(frame.rgb @ LUMA, _FLOOR)).astype(np.float32)
    frame._layers = (log_luma, log_luma, np.zeros_like(log_luma))
    return frame


def colour_steps(frame: LinearFrame, point: dict) -> np.ndarray:
    """The colour steps of adjust.render.develop_frame, in its order, as float."""
    enc = _light_and_tone(frame, point, point_of(point))
    mono = bool(point.get("mono"))
    if mono:
        enc = _to_mono(enc, point)
    if any(k in point for k in CURVE_KEYS):
        enc = _apply_curves(enc, point)
    if not mono and any(k in point for k in HSL_SLIDERS):
        enc = _apply_mixer(enc, point)
    if any(k in point for k in _LOOK_STEP_KEYS):
        enc = _apply_look(enc, point)
    return np.clip(enc, 0.0, 1.0)


def cube_of(settings: Optional[dict], size: int = 33, title: str = "") -> tuple[str, list[str]]:
    """The look as the text of a .cube file, and the names of what it leaves out.
    ``ValueError`` when nothing of the edit can be carried."""
    point, left_out = split(settings)
    if not point:
        raise ValueError("Nothing in this look can be held by a LUT: "
                         + (", ".join(left_out) if left_out else "it changes nothing") + ".")
    out = colour_steps(per_pixel_frame(lattice(size)), point).reshape(-1, 3)
    name = (title or "Look").replace('"', "'").replace("\n", " ")
    lines = [
        "# Display-encoded Rec. 709 in and out.",
        "# Not in it: " + (", ".join(left_out) if left_out else "nothing left out") + ".",
        f'TITLE "{name}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in out.tolist())
    return "\n".join(lines) + "\n", left_out
