"""
Lens corrections: the lens's distortion, taken out as the camera takes it out.

A Canon camera corrects its JPEG for the lens's distortion (and its vignetting,
which adjust.look learns); a RAW decode is not corrected. The correction is
already measured: adjust.look registers the camera's JPEG onto the RAW with a
lens model to learn the camera's look. The same registration, kept, is the
camera's own distortion correction for that frame — for any lens and focal
length, without a lens database. Measured on an R6 II with the RF24-105 at 24 mm:
the corners move by 1.4% of the half-diagonal, about 85 px at full size.

Lateral chromatic aberration needs no control: the decode's highlight repair
smooths colour along edges (adjust.clip), and what is left of it on the same
frame measures 0.0 px between the channels.

The controls, 0 being no change:

  lens_profile   0 or 1   correct the distortion as the camera does for this frame
  distortion     -100..100  by hand, on top: positive straightens a barrel (lines
                 bowing out), negative a pincushion. The picture is enlarged just
                 enough that no corner comes up empty.

A model is a dict of normalised terms about the frame's centre, distances in
half-diagonals: ``ox, oy`` a shift, ``s`` a scale, ``a1, a2`` the radial terms,
so a point ``p`` of the RAW lands at ``o + s p + a1 p r² + a2 p r⁴`` in the
corrected picture. It does not depend on the size it is drawn at.

Transform (perspective), drawn through the same model after the lens, so the
loupe, masks, spots and exports follow it as they follow the lens:

  tf_vertical    -100..100  the picture plane tilted about its horizontal axis:
                 negative straightens verticals that lean in towards the top (a
                 building or a stage shot from below), positive the other way,
                 as Lightroom's Vertical does
  tf_horizontal  -100..100  the same about the vertical axis, for a wall shot
                 from one side
  tf_aspect      -100..100  taller (positive) or wider, for the squat look a
                 strong vertical correction leaves

They make ``H``, the 3x3 homography taking a point of the corrected picture to
the lens-corrected frame, and ``zoom``, how much the picture is enlarged so its
corners stay inside the frame. ``adjust.upright`` finds the tilt and the level
that make a frame's own vertical lines vertical.
"""

import math
from typing import Optional

import numpy as np

LENS_KEYS = ("lens_profile", "distortion")
TRANSFORM_KEYS = ("tf_vertical", "tf_horizontal", "tf_aspect")
# Every key that changes the frame's geometry before the crop.
FRAME_KEYS = LENS_KEYS + TRANSFORM_KEYS

# The tilt at ±100, in degrees, and the focal length the tilt is made with, in
# half-diagonals (about a 35 mm lens on full frame). A tilt is a picture-plane
# rotation seen through that lens; the upright search uses the same, so what it
# finds and what the sliders do agree.
MAX_TILT = 30.0
TILT_FOCAL = 1.25
# Aspect at ±100: this many times taller (or wider).
MAX_ASPECT = 1.5

# The hand control's reach at ±100, as the r² term: about the camera's own
# correction of a 24 mm zoom at its wide end.
MANUAL_REACH = 0.08
_ITERATIONS = 8


def from_registration(centre, radius: float, u) -> dict:
    """adjust.look's registration (centre, radius in pixels; ``u`` its terms) as
    a size-free model."""
    ox, oy, s, a1, a2 = (float(v) for v in u)
    return {"ox": ox / radius, "oy": oy / radius, "s": s, "a1": a1, "a2": a2}


IDENTITY = {"ox": 0.0, "oy": 0.0, "s": 1.0, "a1": 0.0, "a2": 0.0}


def model_for(settings: dict, profile: Optional[dict], aspect: float) -> Optional[dict]:
    """The model an edit asks for on a frame of width/height ``aspect``: the
    camera's own (``profile``) when ``lens_profile`` is set and there is one, with
    the hand control on top, then the transform. ``None`` when it changes nothing."""
    model = dict(profile) if settings.get("lens_profile") and profile else dict(IDENTITY)
    amount = float(settings.get("distortion", 0)) / 100.0
    if amount:
        # The hand's term is added, then the whole is enlarged so that the
        # picture's corners still come from inside the frame.
        model["a1"] += MANUAL_REACH * amount
        model["s"] = model["s"] * _fill(model, aspect)
    h = homography(settings)
    if h is not None:
        model["H"] = h
        model["zoom"] = _zoom(model, aspect)
    if model == IDENTITY:
        return None
    return model


def homography(settings: dict) -> Optional[list]:
    """The transform's 3x3 homography (a point of the corrected picture to the
    frame, both normalised about the centre), or ``None`` for no transform."""
    v = float(settings.get("tf_vertical", 0)) / 100.0
    hz = float(settings.get("tf_horizontal", 0)) / 100.0
    a = float(settings.get("tf_aspect", 0)) / 100.0
    if not (v or hz or a):
        return None
    f = TILT_FOCAL
    k = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    tx, ty = math.radians(MAX_TILT * v), math.radians(MAX_TILT * hz)
    rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    m = k @ rx @ ry @ np.linalg.inv(k)
    stretch = MAX_ASPECT ** a
    m = m @ np.diag([math.sqrt(stretch), 1.0 / math.sqrt(stretch), 1.0])
    # A tilt also moves the picture's middle up or down (or sideways); it is moved
    # back, so the picture stays where it was and is enlarged only for its corners.
    d = np.linalg.solve(m, np.array([0.0, 0.0, 1.0]))
    m = m @ np.array([[1.0, 0, d[0] / d[2]], [0, 1.0, d[1] / d[2]], [0, 0, 1.0]])
    return (m / m[2, 2]).round(9).tolist()


def _project(h, qx, qy):
    m = np.asarray(h, dtype=np.float64)
    w = m[2, 0] * qx + m[2, 1] * qy + m[2, 2]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return (m[0, 0] * qx + m[0, 1] * qy + m[0, 2]) / w, (m[1, 0] * qx + m[1, 1] * qy + m[1, 2]) / w


def _to_frame(model: dict, qx, qy):
    """A point of the finished picture to the lens-corrected frame: the zoom and
    the transform's homography."""
    if "H" not in model:
        return qx, qy
    zoom = model.get("zoom", 1.0)
    return _project(model["H"], qx / zoom, qy / zoom)


def _zoom(model: dict, aspect: float) -> float:
    """How much to enlarge a transformed picture so every point of its edge comes
    from inside the frame: found by halving, the edge's reach falling as it grows."""
    h = 1.0 / math.hypot(aspect, 1.0)
    w = aspect * h
    t = np.linspace(-1.0, 1.0, 41)
    ex = np.concatenate([t * w, t * w, np.full_like(t, -w), np.full_like(t, w)])
    ey = np.concatenate([np.full_like(t, -h), np.full_like(t, h), t * h, t * h])

    def fits(zoom):
        sx, sy = _project(model["H"], ex / zoom, ey / zoom)
        px, py = source_of(model, sx, sy)
        return bool((np.abs(px) <= w * 1.0005).all() and (np.abs(py) <= h * 1.0005).all())

    lo, hi = 1.0, 1.0
    while not fits(hi) and hi < 8.0:
        lo, hi = hi, hi * 1.5
    if fits(lo):
        return lo
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def _forward(model: dict, px: np.ndarray, py: np.ndarray):
    r2 = px * px + py * py
    k = model["s"] + model["a1"] * r2 + model["a2"] * r2 * r2
    return model["ox"] + k * px, model["oy"] + k * py


def _fill(model: dict, aspect: float) -> float:
    """How much larger to draw ``model``'s picture so that every point of the
    edge comes from the frame (no empty corner): the largest of what each point of
    the edge asks for."""
    h = 1.0 / math.hypot(aspect, 1.0)
    w = aspect * h
    t = np.linspace(-1.0, 1.0, 41)
    edge_x = np.concatenate([t * w, t * w, np.full_like(t, -w), np.full_like(t, w)])
    edge_y = np.concatenate([np.full_like(t, -h), np.full_like(t, h), t * h, t * h])
    sx, sy = source_of({**model, "s": model["s"]}, edge_x, edge_y)
    reach = np.maximum(np.abs(sx) / w, np.abs(sy) / h)
    return float(max(1.0, reach.max()))


def source_of(model: dict, qx: np.ndarray, qy: np.ndarray):
    """Where in the RAW each point ``(qx, qy)`` of the corrected picture comes
    from, normalised as the model is: the model inverted, by a few steps of
    fixed-point iteration (it is close to the identity, and each step gains
    about two digits)."""
    px = (qx - model["ox"]) / model["s"]
    py = (qy - model["oy"]) / model["s"]
    for _ in range(_ITERATIONS):
        r2 = px * px + py * py
        k = model["s"] + model["a1"] * r2 + model["a2"] * r2 * r2
        px = (qx - model["ox"]) / k
        py = (qy - model["oy"]) / k
    return px, py


def maps(model: dict, full_size: tuple, window: tuple, shape: tuple):
    """For each pixel of the ``shape`` (H, W) pixels being drawn — ``window``
    (left, top, right, bottom fractions) of the corrected whole frame of
    ``full_size`` (W, H) pixels — the RAW pixel it comes from, in whole-frame
    pixels: ``(map_x, map_y)`` float32."""
    fw, fh = full_size
    h, w = shape
    left, top, right, bottom = window
    cx, cy, radius = fw / 2.0, fh / 2.0, math.hypot(fw / 2.0, fh / 2.0)
    xs = (left + (np.arange(w, dtype=np.float64) + 0.5) / w * (right - left)) * fw
    ys = (top + (np.arange(h, dtype=np.float64) + 0.5) / h * (bottom - top)) * fh
    qx = np.broadcast_to(((xs - cx) / radius)[None, :], (h, w))
    qy = np.broadcast_to(((ys - cy) / radius)[:, None], (h, w))
    px, py = source_of(model, *_to_frame(model, qx, qy))
    return ((px * radius + cx - 0.5).astype(np.float32),
            (py * radius + cy - 0.5).astype(np.float32))


def warp(rgb: np.ndarray, model: dict) -> np.ndarray:
    """A whole frame corrected by ``model``, at its own size."""
    import cv2

    h, w = rgb.shape[:2]
    map_x, map_y = maps(model, (w, h), (0.0, 0.0, 1.0, 1.0), (h, w))
    return cv2.remap(np.ascontiguousarray(rgb, dtype=np.float32), map_x, map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp_region(fetch, model: dict, full_size: tuple, window: tuple, shape: tuple) -> np.ndarray:
    """Part of a corrected frame at full resolution (a loupe's patch), from the
    RAW pixels it needs: ``fetch(x0, y0, x1, y1)`` returns the uncorrected frame's
    pixels over that box of whole-frame pixels (clamped to the frame)."""
    import cv2

    fw, fh = full_size
    map_x, map_y = maps(model, full_size, window, shape)
    x0 = max(0, int(math.floor(float(map_x.min()))) - 2)
    y0 = max(0, int(math.floor(float(map_y.min()))) - 2)
    x1 = min(fw, int(math.ceil(float(map_x.max()))) + 3)
    y1 = min(fh, int(math.ceil(float(map_y.max()))) + 3)
    region = np.ascontiguousarray(fetch(x0, y0, x1, y1), dtype=np.float32)
    return cv2.remap(region, map_x - np.float32(x0), map_y - np.float32(y0),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
