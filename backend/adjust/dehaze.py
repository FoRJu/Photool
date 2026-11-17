"""
Dehaze (the Presence section): haze taken out of a picture, or put in.

Haze is light scattered by the air between the camera and the scene. It adds the
colour of the sky (the *airlight*) to every part of the picture in proportion to
how much air lies in front of it, so a far hill is paler than a near one. Contrast
and clarity cannot undo that, because it grows with distance and not with tone: a
dark tree far away is as pale as a bright wall nearby.

The estimate is the dark channel prior (He, Sun and Tang, 2009): in a clear scene
almost every small patch has some pixel that is nearly black in at least one
colour (a shadow, a leaf, a saturated colour). Where even the darkest colour of a
patch is pale, the paleness is haze, and how pale it is says how much. The steps:

  1. On a small copy of the whole frame, in linear light: the darkest colour of
     each pixel, then the darkest of those over a small patch (the dark channel).
  2. The airlight: the mean colour of the haziest 0.1% of that copy.
  3. The haze: the dark channel of the copy divided by the airlight.
  4. Refined with a guided filter against the luminance: its coefficients are
     solved on the small copy and applied to the full-size picture, so the haze
     follows the picture's own edges (a branch against the sky) and not the
     patch's blocks.

Positive amounts take that haze out, up to 85% of the estimate at 100, by the
haze model ``I = J t + A (1 - t)`` read backwards. Negative amounts lay an even
haze of the airlight's colour over the picture.

The estimate belongs to the whole picture, so it is made once from the whole
frame (the frame a loupe's patch or a draft is anchored to) and kept there. A
preview, a draft, a loupe's patch and an export all sample the same coefficients
and the same airlight, and differ only in the full-size luminance the haze is
refined against, which is each one's own pixels.
"""

from typing import Optional

import numpy as np

# The copy the estimate is made on, by its longer side. The haze is a smooth field
# and the airlight a single colour, so a small copy loses nothing that matters.
WORKING_SIZE = 384
# The dark channel's patch, as a fraction of the picture's longer side: 15 px on a
# 600 px picture, as in the paper.
PATCH_FRACTION = 1.0 / 40.0
# The guided filter's radius as a multiple of the patch, and its edge threshold on
# display-encoded luminance (0..1).
GUIDE_RADIUS = 2.0
GUIDE_EPS = 1e-3
# The share of the copy, the haziest, whose mean colour is the airlight.
AIRLIGHT_SHARE = 0.001
# Dehaze 100 takes out this much of the haze the estimate finds. Taking out all of
# it leaves nothing of the distance, and divides the far sky's noise by nearly 0.
REMOVE = 0.85
# Dehaze -100 lays this much of the airlight over the picture, its colour kept and
# its brightness no more than white's, eased in by the square of the amount. Haze
# is added in linear light, where a little goes a long way in the shadows: in
# proportion, 0.6 made -30 a fog on a sunny street, and 0.15 still lifted its
# darkest twentieth from 16 to 103 of 255.
ADD = 0.3


def _root(frame):
    """The whole frame a frame is a part or a copy of."""
    return frame.anchor_of or frame


def _encoded_luma(rgb: np.ndarray) -> np.ndarray:
    from adjust.render import LUMA

    y = np.clip(rgb @ LUMA, 0.0, 1.0)
    return np.power(y, np.float32(1.0 / 2.2)).astype(np.float32)


def estimate(rgb: np.ndarray) -> dict:
    """The haze of a whole picture (linear, (H, W, 3)): the airlight and the guided
    filter's coefficients, on a small copy of it."""
    import cv2

    h, w = rgb.shape[:2]
    f = min(1.0, WORKING_SIZE / float(max(h, w)))
    size = (max(4, round(w * f)), max(4, round(h * f)))
    small = (cv2.resize(np.ascontiguousarray(rgb, dtype=np.float32), size, interpolation=cv2.INTER_AREA)
             if f < 1.0 else np.ascontiguousarray(rgb, dtype=np.float32))
    small = np.maximum(small, 0.0)
    sh, sw = small.shape[:2]
    patch = max(3, int(round(max(sh, sw) * PATCH_FRACTION)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))

    dark = cv2.erode(small.min(axis=2), kernel, borderType=cv2.BORDER_REFLECT)
    count = max(1, int(round(dark.size * AIRLIGHT_SHARE)))
    haziest = np.argpartition(dark.reshape(-1), dark.size - count)[-count:]
    airlight = small.reshape(-1, 3)[haziest].mean(axis=0)
    airlight = np.maximum(airlight, 1e-3).astype(np.float32)

    haze = cv2.erode((small / airlight).min(axis=2), kernel, borderType=cv2.BORDER_REFLECT)
    haze = np.clip(haze, 0.0, 1.0).astype(np.float32)

    # The guided filter (He, Sun and Tang, 2010), with the luminance as the guide.
    guide = _encoded_luma(small)
    r = max(2, int(round(patch * GUIDE_RADIUS)))

    def box(a):
        return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)

    mean_g, mean_p = box(guide), box(haze)
    cov = box(guide * haze) - mean_g * mean_p
    var = box(guide * guide) - mean_g * mean_g
    a = cov / (var + GUIDE_EPS)
    b = mean_p - a * mean_g
    return {"airlight": airlight, "a": box(a).astype(np.float32), "b": box(b).astype(np.float32)}


def _kept(frame) -> dict:
    root = _root(frame)
    if root._haze is None:
        root._haze = estimate(root.rgb)
    return root._haze


def _sampled(plane: np.ndarray, shape: tuple, window: tuple, of: tuple) -> np.ndarray:
    """``plane``, made over the part ``of`` of the picture, sampled at the pixel
    centres of a frame of ``shape`` that is the part ``window`` of it."""
    import cv2

    h, w = shape[:2]
    mh, mw = plane.shape[:2]
    ol, ot, or_, ob = of
    left, top, right, bottom = window
    fx = left + (np.arange(w, dtype=np.float64) + 0.5) / w * (right - left)
    fy = top + (np.arange(h, dtype=np.float64) + 0.5) / h * (bottom - top)
    xs = ((fx - ol) / max(or_ - ol, 1e-9) * mw - 0.5).astype(np.float32)
    ys = ((fy - ot) / max(ob - ot, 1e-9) * mh - 0.5).astype(np.float32)
    map_x = np.ascontiguousarray(np.broadcast_to(xs[None, :], (h, w)))
    map_y = np.ascontiguousarray(np.broadcast_to(ys[:, None], (h, w)))
    return cv2.remap(np.ascontiguousarray(plane), map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def haze_map(frame, rgb: np.ndarray, window: tuple) -> tuple[np.ndarray, np.ndarray]:
    """The haze over ``rgb`` (linear, the part ``window`` of the picture ``frame``
    belongs to), 0..1 per pixel, and the airlight."""
    kept = _kept(frame)
    of = tuple(_root(frame).window)
    a = _sampled(kept["a"], rgb.shape, window, of)
    b = _sampled(kept["b"], rgb.shape, window, of)
    haze = np.clip(a * _encoded_luma(rgb) + b, 0.0, 1.0)
    return haze.astype(np.float32), kept["airlight"]


def pixels(frame, rgb: np.ndarray, window: tuple, amount: float) -> np.ndarray:
    """``rgb`` (linear, the part ``window`` of the picture ``frame`` belongs to) with
    ``amount`` (-100..100) of dehaze: how a loupe's patch takes the whole frame's
    estimate."""
    amount = float(np.clip(amount, -100.0, 100.0)) / 100.0
    rgb = np.asarray(rgb, dtype=np.float32)
    if amount == 0.0:
        return rgb
    if amount < 0.0:
        airlight = _kept(frame)["airlight"]
        airlight = airlight / max(1.0, float(airlight.max()))
        f = np.float32(ADD * amount * amount)
        return (rgb * (1.0 - f) + airlight * f).astype(np.float32)
    haze, airlight = haze_map(frame, rgb, window)
    taken = (np.float32(REMOVE * amount) * haze)[..., None]
    # I = J t + A (1 - t), with 1 - t the haze taken out; t is never under 0.15.
    out = (rgb - airlight * taken) / (1.0 - taken)
    return np.maximum(out, 0.0).astype(np.float32)


def dehazed(frame, amount: float):
    """``frame`` with ``amount`` of dehaze, as a frame of its own (sharing its
    mattes); ``frame`` itself at 0. Kept on the frame for the amount."""
    from adjust.render import LinearFrame

    if not amount:
        return frame
    key = float(amount)
    if frame._dehazed is not None and frame._dehazed[0] == key:
        return frame._dehazed[1]
    rgb = pixels(frame, frame.rgb, tuple(frame.window), amount)
    whole = _root(frame)
    if tuple(frame.window) != (0.0, 0.0, 1.0, 1.0) and whole is not frame:
        # A patch is toned by the whole frame's base (adjust.render.region_frame),
        # so it takes the base of the whole frame dehazed.
        from adjust.render import region_frame

        out = region_frame(dehazed(whole, amount), rgb, tuple(frame.window))
        out.scale, out.lens, out.source_key = frame.scale, frame.lens, frame.source_key
    else:
        out = LinearFrame(rgb, look=frame.look, window=frame.window, scale=frame.scale,
                          anchor_of=whole, lens=frame.lens, wb=frame.wb, source_key=frame.source_key)
    out._mattes = frame._mattes
    frame._dehazed = (key, out)
    return out


def wanted(clean: Optional[dict]) -> bool:
    return bool(clean) and bool(clean.get("dehaze"))
