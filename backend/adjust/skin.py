"""
Skin, retouched: a mask's ``smooth``, ``even`` and ``texture`` (adjust.masks),
on the skin of the people it reaches.

Blemishes, blotches and shine sit at a middle scale: larger than pores, smaller
than the shape of a face. So the skin's brightness is split by scale, in stops,
and the middle band calmed while the finest (the pores, which make skin read as
skin) is kept:

  smooth   0..100   how much of the middle band goes: spots, uneven patches
                    (the darker fully, the brighter, its sheen, much less)
  texture  -100..100  the finest band: softer pores (left) or crisper (right)
  even     0..100   colour (and broad brightness) pulled toward the skin
                    round it: redness, a blotchy cheek

Every scale is set by the person's own face, as wide as the face detector found
it in these pixels, so a headshot and a face across a stage are treated alike.
Nothing is measured against a colour: each patch is compared with the skin of
the same face around it, so it works the same for any complexion.

It reaches skin only, where the mask reaches: the face and body skin the people
parsers find (adjust.parts), so eyes, brows, lips and hair are never touched,
and the averages are taken over skin alone, so no hair or background bleeds in.
Done on the linear frame after the spots, before the light and tone, and kept
on the frame for the masks that made it: moving another slider does not redo it.
"""

import json
from typing import Optional

import numpy as np

SKIN_SLIDERS = ("smooth", "even", "texture")
SKIN_AREAS = ("face_skin", "body_skin")

# How far each control goes at full deflection.
SMOOTH_MAX = 0.9          # share of the middle band's darker spots taken out
# Of its brighter spots, a share of that: a blemish or a blotch is darker or
# redder than the skin round it, and what is brighter is the skin's own sheen
# and glow, which taken out as well leaves a face flat, as under foundation.
SMOOTH_BRIGHT = 0.35
TEXTURE_MAX = 0.8         # share of the finest band added or taken away
EVEN_COLOUR_MAX = 0.75    # share of the way to the skin round about
EVEN_LIGHT_MAX = 0.15     # likewise for broad brightness: the face keeps its shape
# The scales, in face widths: the finest band is under this, the middle band
# between it and the next, and "even" looks this far round.
FINE_FACES = 1 / 160
MIDDLE_FACES = 1 / 12
EVEN_FACES = 1 / 8
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_EPS = 1e-5


def wanted(masks: Optional[list]) -> list:
    """The masks that retouch skin."""
    return [m for m in masks or [] if any(m.get(k) for k in SKIN_SLIDERS)]


def retouched(frame, masks: list, reach):
    """``frame`` with the skin its masks reach retouched, as a frame of its own
    (sharing the original's model mattes, which the retouch does not change).
    ``reach(frame, mask)`` is the mask's weight over these pixels, or ``None``.
    Kept on the frame for the masks that made it."""
    from adjust.render import LinearFrame

    masks = wanted(masks)
    key = json.dumps([{k: v for k, v in m.items() if k in SKIN_SLIDERS or k not in _OTHER_SLIDERS}
                      for m in masks], sort_keys=True)
    kept = getattr(frame, "_retouched", None)
    if kept is not None and kept[0] == key:
        return kept[1]
    rgb = retouch(frame, masks, reach)
    out = LinearFrame(rgb, look=frame.look, window=frame.window, scale=frame.scale, anchor_of=frame.anchor_of)
    out._mattes = frame._mattes
    frame._retouched = (key, out)
    return out


def retouch(frame, masks: list, reach) -> np.ndarray:
    """The frame's linear RGB with the skin retouched (a new array)."""
    from adjust.segment import face_boxes, sample

    rgb = frame.rgb
    h, w = rgb.shape[:2]
    whole = getattr(frame, "anchor_of", None) or frame
    left, top, right, bottom = frame.window
    sx, sy = w / max(right - left, 1e-6), h / max(bottom - top, 1e-6)
    out = rgb.copy()
    for m in masks:
        weight = reach(frame, m)
        if weight is None or not weight.any():
            continue
        person = m.get("person") if m["kind"] in ("people", "faces") else None
        skin = _skin(whole, person)
        if skin is None:
            continue
        here = weight * sample(skin, (h, w), frame.window)
        if not here.any():
            continue
        boxes = face_boxes(whole)
        if person:
            boxes = boxes[int(person) - 1:int(person)]
        # Each pixel of skin is retouched once, at the scale of the first face
        # whose person it is near (the people parsers read it from that face).
        done = np.zeros((h, w), bool)
        for fx, fy, fw, fh in boxes:
            face = fw * sx
            if face < 6:
                continue
            cx, ty = (fx + fw / 2 - left) * sx, (fy - top) * sy
            x0, x1 = int(max(0, cx - 2.8 * face)), int(min(w, cx + 2.8 * face))
            y0, y1 = int(max(0, ty - 1.2 * fh * sy)), int(min(h, ty + 9.0 * fh * sy))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            region = here[y0:y1, x0:x1] * ~done[y0:y1, x0:x1]
            if not region.any():
                continue
            out[y0:y1, x0:x1] = _region(out[y0:y1, x0:x1], region, face, m)
            done[y0:y1, x0:x1] |= region > 0
    return out


# A blur wider than this is made on a smaller copy and brought back up: it has
# no fine detail to lose, and a 200 px blur over a whole headshot took seconds.
_SMALL_SIGMA = 4.0


def _skin(whole, person):
    """The skin to retouch over the whole frame (adjust.parts.skin_matte), or
    ``None`` when the parsers are not here."""
    import logging

    from adjust import parts
    from faces.models import ModelsUnavailable

    try:
        return parts.skin_matte(whole, person)
    except ModelsUnavailable as exc:
        logging.getLogger(__name__).warning("[skin] cannot find the skin: %s", exc)
        return None


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    if sigma <= 2 * _SMALL_SIGMA:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    h, w = x.shape[:2]
    f = _SMALL_SIGMA / sigma
    small = cv2.resize(x, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), _SMALL_SIGMA, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _over_skin(x: np.ndarray, skin: np.ndarray, sigma: float) -> np.ndarray:
    """``x`` blurred over the skin alone (a normalised convolution): what the
    skin round each pixel is like, with no hair or background in it."""
    if x.ndim == 3:
        return _blur(x * skin[..., None], sigma) / (_blur(skin, sigma)[..., None] + _EPS)
    return _blur(x * skin, sigma) / (_blur(skin, sigma) + _EPS)


def _region(rgb: np.ndarray, weight: np.ndarray, face: float, mask: dict) -> np.ndarray:
    """One person's skin retouched over a crop of linear RGB."""
    smooth = mask.get("smooth", 0) / 100.0
    texture = mask.get("texture", 0) / 100.0
    even = mask.get("even", 0) / 100.0
    skin = weight.astype(np.float32)
    lum = np.maximum(rgb @ _LUMA, 0.0) + _EPS
    stops = np.log2(lum).astype(np.float32)
    colour = (rgb / lum[..., None]).astype(np.float32)

    fine_r = max(0.7, face * FINE_FACES)
    mid_r = max(1.5, face * MIDDLE_FACES)
    finer = _blur(stops, fine_r)
    broad = _over_skin(stops, skin, mid_r)
    fine = stops - finer
    middle = finer - broad
    dark = np.minimum(middle, 0.0)
    bright = middle - dark
    new = (broad + dark * (1.0 - SMOOTH_MAX * smooth) + bright * (1.0 - SMOOTH_MAX * SMOOTH_BRIGHT * smooth)
           + fine * (1.0 + TEXTURE_MAX * texture))
    if even:
        far = face * EVEN_FACES
        # Darker patches are lifted toward the skin round them; the brighter,
        # where the light falls and the skin glows, are left as they are.
        new = new + np.maximum(_over_skin(stops, skin, far) - broad, 0.0) * EVEN_LIGHT_MAX * even
        colour = colour + (_over_skin(colour, skin, far) - colour) * EVEN_COLOUR_MAX * even
    light = np.exp2(new)
    # Smoothing in stops averages in stops, which is darker than the light's own
    # average: each broad area of skin (a cheek, not a spot, whose being lifted
    # is the point) is put back to the brightness it had.
    keep_r = face * EVEN_FACES
    light = light * (_over_skin(lum, skin, keep_r) / (_over_skin(light, skin, keep_r) + _EPS))
    done = np.maximum(light[..., None] * colour - _EPS, 0.0)
    return (rgb + (done - rgb) * skin[..., None]).astype(np.float32)


# The other controls a mask carries, which do not change what the retouch does
# and so are not part of what it is kept by.
_OTHER_SLIDERS = ("exposure", "contrast", "highlights", "lights", "darks", "shadows", "temperature",
                  "saturation", "clarity")
