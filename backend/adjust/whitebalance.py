"""White balance in Kelvin and tint, and the grey picker.

The develop panel's white balance is absolute: ``wb_kelvin`` (2000..50000) and
``wb_tint`` (-150..150) on Lightroom's scales, where a higher Kelvin warms the
picture and a positive tint turns it magenta. Absent, the frame is as shot.

All the colour maths of it lives here:

* Kelvin and tint to a chromaticity and back, by the DNG SDK's
  ``dng_temperature`` (Robertson's isotherms in CIE 1960 uv, the tint being the
  distance off the locus times -3000). That is what gives Lightroom's numbers.
* What a file knows of its own white balance (:func:`of_source`): for a RAW, the
  camera's matrix from XYZ to its own colours, its as-shot gains and the matrix
  the decode used to reach linear sRGB. A file without them is the *nominal*
  case, whose as shot is taken to be D65.
* The 3x3 matrix that re-balances the decoded, linear sRGB frame to a chosen
  white (:func:`matrix_for`), and the white that makes a picked patch neutral
  (:func:`white_point`).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

KELVIN_RANGE = (2000, 50000)
TINT_RANGE = (-150, 150)

# ── Kelvin, tint and xy (the DNG SDK's dng_temperature) ──────────────

# Robertson's table: reciprocal megakelvin, then u, v of the locus there, then
# the slope of the isotherm.
_TABLE = np.array([
    (0, 0.18006, 0.26352, -0.24341),
    (10, 0.18066, 0.26589, -0.25479),
    (20, 0.18133, 0.26846, -0.26876),
    (30, 0.18208, 0.27119, -0.28539),
    (40, 0.18293, 0.27407, -0.30470),
    (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156),
    (70, 0.18611, 0.28342, -0.37915),
    (80, 0.18740, 0.28668, -0.40955),
    (90, 0.18880, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888),
    (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471),
    (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512),
    (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637),
    (325, 0.24792, 0.34655, -2.4681),
    (350, 0.25591, 0.34951, -2.9641),
    (375, 0.26400, 0.35200, -3.5814),
    (400, 0.27218, 0.35407, -4.3633),
    (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262),
    (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324),
    (525, 0.31320, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325),
    (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45),
])
_TINT_SCALE = -3000.0

D65_XY = (0.31271, 0.32902)


def kelvin_tint_of(x: float, y: float) -> tuple[float, float]:
    """The colour temperature and tint of the chromaticity ``(x, y)``."""
    denom = 1.5 - x + 6.0 * y
    u, v = 2.0 * x / denom, 3.0 * y / denom
    last_dt = last_du = last_dv = 0.0
    for i in range(1, len(_TABLE)):
        r, tu, tv, slope = _TABLE[i]
        length = math.sqrt(1.0 + slope * slope)
        du, dv = 1.0 / length, slope / length
        dt = -(u - tu) * dv + (v - tv) * du
        if dt <= 0.0 or i == len(_TABLE) - 1:
            dt = -min(dt, 0.0)
            f = 0.0 if i == 1 else dt / (last_dt + dt)
            r0, u0, v0, _ = _TABLE[i - 1]
            kelvin = 1.0e6 / (r0 * f + r * (1.0 - f))
            uu = u - (u0 * f + tu * (1.0 - f))
            vv = v - (v0 * f + tv * (1.0 - f))
            du, dv = du * (1.0 - f) + last_du * f, dv * (1.0 - f) + last_dv * f
            length = math.sqrt(du * du + dv * dv)
            return kelvin, (uu * du / length + vv * dv / length) * _TINT_SCALE
        last_dt, last_du, last_dv = dt, du, dv
    raise AssertionError("unreachable")


def xy_of(kelvin: float, tint: float = 0.0) -> tuple[float, float]:
    """The chromaticity of colour temperature ``kelvin`` moved ``tint`` off the locus."""
    r = 1.0e6 / float(kelvin)
    offset = float(tint) / _TINT_SCALE
    last = len(_TABLE) - 2
    for i in range(last + 1):
        if r < _TABLE[i + 1][0] or i == last:
            r0, u0, v0, s0 = _TABLE[i]
            r1, u1, v1, s1 = _TABLE[i + 1]
            f = (r1 - r) / (r1 - r0)
            u = u0 * f + u1 * (1.0 - f)
            v = v0 * f + v1 * (1.0 - f)
            l0, l1 = math.sqrt(1.0 + s0 * s0), math.sqrt(1.0 + s1 * s1)
            uu = (1.0 / l0) * f + (1.0 / l1) * (1.0 - f)
            vv = (s0 / l0) * f + (s1 / l1) * (1.0 - f)
            length = math.sqrt(uu * uu + vv * vv)
            u += uu / length * offset
            v += vv / length * offset
            d = u - 4.0 * v + 2.0
            return 1.5 * u / d, v / d
    raise AssertionError("unreachable")


def xyz_of(xy) -> np.ndarray:
    """XYZ of a chromaticity at Y = 1."""
    x, y = float(xy[0]), float(xy[1])
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def xy_of_xyz(xyz) -> tuple[float, float]:
    xyz = np.asarray(xyz, dtype=np.float64)
    total = float(xyz.sum())
    return float(xyz[0] / total), float(xyz[1] / total)


# ── Linear sRGB and Bradford ──────────────────────────────────────────

RGB_TO_XYZ = np.array([
    (0.412453, 0.357580, 0.180423),
    (0.212671, 0.715160, 0.072169),
    (0.019334, 0.119193, 0.950227),
])
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)
_BRADFORD = np.array([
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
])


def bradford(source_xy, target_xy) -> np.ndarray:
    """XYZ to XYZ: what is white under ``source_xy`` made white under ``target_xy``."""
    s = _BRADFORD @ xyz_of(source_xy)
    t = _BRADFORD @ xyz_of(target_xy)
    return np.linalg.inv(_BRADFORD) @ np.diag(t / s) @ _BRADFORD


# The white sRGB's matrix maps (1, 1, 1) to: D65 as this module's matrices have it.
_NOMINAL_XY = xy_of_xyz(RGB_TO_XYZ @ np.ones(3))


# ── What a file knows ────────────────────────────────────────────────

@dataclass(frozen=True)
class WhiteBalance:
    """A frame's own white balance. ``raw`` False is the nominal case: no camera
    matrix to work in, the as shot taken to be D65 and the re-balance a Bradford
    adaptation in linear sRGB."""
    raw: bool
    as_shot: tuple                      # (kelvin, tint)
    cam_xyz: Optional[np.ndarray] = None        # XYZ to camera, 3x3
    neutral: Optional[np.ndarray] = None        # as-shot camera neutral, green = 1
    srgb_from_cam: Optional[np.ndarray] = None  # balanced camera to linear sRGB


NOMINAL = WhiteBalance(raw=False, as_shot=kelvin_tint_of(*_NOMINAL_XY))


def of_camera(cam_xyz, gains, srgb_from_cam) -> Optional[WhiteBalance]:
    """A RAW's white balance from its matrix, its as-shot gains and its decode's
    matrix to linear sRGB; ``None`` when they do not make one."""
    try:
        cam_xyz = np.asarray(cam_xyz, dtype=np.float64)[:3, :3]
        gains = np.asarray(gains, dtype=np.float64)[:3]
        srgb = np.asarray(srgb_from_cam, dtype=np.float64)
        if cam_xyz.shape != (3, 3) or srgb.shape != (3, 3) or not np.all(gains > 0):
            return None
        if not (np.all(np.isfinite(cam_xyz)) and np.all(np.isfinite(srgb))):
            return None
        if abs(np.linalg.det(cam_xyz)) < 1e-9 or abs(np.linalg.det(srgb)) < 1e-9:
            return None
        neutral = 1.0 / gains
        neutral = neutral / neutral[1]
        xyz = np.linalg.inv(cam_xyz) @ neutral
        if not np.all(np.isfinite(xyz)) or xyz.sum() <= 0 or xyz[1] <= 0:
            return None
        kelvin, tint = kelvin_tint_of(*xy_of_xyz(xyz))
        if not (math.isfinite(kelvin) and math.isfinite(tint)) or kelvin <= 0:
            return None
        return WhiteBalance(raw=True, as_shot=(kelvin, tint), cam_xyz=cam_xyz,
                            neutral=neutral, srgb_from_cam=srgb)
    except Exception:
        return None


_CACHE: "OrderedDict[tuple, Optional[WhiteBalance]]" = OrderedDict()
_CACHE_LIMIT = 512
_LOCK = threading.Lock()


def of_source(path) -> Optional[WhiteBalance]:
    """What ``path`` knows of its own white balance, from its metadata only, or
    ``None`` when it is not a RAW or lacks what is needed (the nominal case).
    Kept by path and modification time."""
    from export.engine import _RAW_EXTENSIONS, _srgb_from_camera

    path = Path(path)
    if path.suffix.lower() not in _RAW_EXTENSIONS:
        return None
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    found = None
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            srgb = _srgb_from_camera(raw)
            if srgb is not None:
                found = of_camera(np.asarray(raw.rgb_xyz_matrix)[:3],
                                  np.asarray(raw.camera_whitebalance)[:3], srgb)
    except Exception as exc:
        logger.debug("[wb] %s: %s", path.name, exc)
    with _LOCK:
        _CACHE[key] = found
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return found


def as_shot(wb: Optional[WhiteBalance]) -> tuple:
    """(kelvin, tint) the frame was shot at: the nominal one's when ``wb`` is None."""
    return (wb or NOMINAL).as_shot


# ── The re-balance ───────────────────────────────────────────────────

def is_as_shot(wb: Optional[WhiteBalance], kelvin: float, tint: float) -> bool:
    """Within a mired and a unit of tint of as shot: no change worth a pass."""
    k0, t0 = as_shot(wb)
    return abs(1.0e6 / float(kelvin) - 1.0e6 / k0) < 1.0 and abs(float(tint) - t0) < 1.0


def matrix_for(wb: Optional[WhiteBalance], kelvin: float, tint: float = 0.0) -> Optional[np.ndarray]:
    """The 3x3 matrix taking the decoded frame's linear sRGB to the same picture
    balanced for ``kelvin`` and ``tint``, applied as ``rgb @ W.T``; ``None`` when
    that is as shot."""
    if is_as_shot(wb, kelvin, tint):
        return None
    xy = xy_of(kelvin, tint)
    if wb is not None and wb.raw:
        n1 = wb.cam_xyz @ xyz_of(xy)
        n1 = n1 / n1[1]
        s = wb.srgb_from_cam
        out = s @ np.diag(wb.neutral / n1) @ np.linalg.inv(s)
    else:
        out = XYZ_TO_RGB @ bradford(xy, _NOMINAL_XY) @ RGB_TO_XYZ
    return out.astype(np.float32)


# ── The picker ───────────────────────────────────────────────────────

# A sample this bright on any channel may have clipped; one this dark is mostly noise.
CLIPPED = 0.98
TOO_DARK = 0.005


def white_point(wb: Optional[WhiteBalance], sample) -> tuple[int, int]:
    """(kelvin, tint) that make ``sample`` — linear sRGB of the frame as decoded,
    before any edit — neutral. Raises ValueError with a sentence for the person
    when it cannot."""
    p = np.asarray(sample, dtype=np.float64)[:3]
    if not np.all(np.isfinite(p)) or p.max() >= CLIPPED:
        raise ValueError("That spot is too bright: a channel is clipped. Pick a grey that is not blown out.")
    if float(RGB_TO_XYZ[1] @ p) < TOO_DARK or p.min() <= 0:
        raise ValueError("That spot is too dark to read its colour. Pick a lighter grey.")
    if wb is not None and wb.raw:
        camera = np.linalg.inv(wb.srgb_from_cam) @ p
        if np.any(camera <= 0):
            raise ValueError("That spot is too strongly coloured to be grey. Pick something neutral.")
        xyz = np.linalg.inv(wb.cam_xyz) @ (wb.neutral * camera)
    else:
        xyz = RGB_TO_XYZ @ p
    if xyz.sum() <= 0 or xyz[1] <= 0:
        raise ValueError("That spot is too strongly coloured to be grey. Pick something neutral.")
    kelvin, tint = kelvin_tint_of(*xy_of_xyz(xyz))
    low, high = KELVIN_RANGE
    if not (math.isfinite(kelvin) and low <= kelvin <= high and TINT_RANGE[0] <= tint <= TINT_RANGE[1]):
        raise ValueError("That spot is too strongly coloured to be grey. Pick something neutral.")
    return int(round(kelvin)), int(round(tint))


def sample(rgb: np.ndarray, u: float, v: float, radius: int = 3) -> np.ndarray:
    """The median of a small patch of ``rgb`` (H, W, 3) about (u, v), fractions."""
    h, w = rgb.shape[:2]
    x = int(round(min(max(u, 0.0), 1.0) * (w - 1)))
    y = int(round(min(max(v, 0.0), 1.0) * (h - 1)))
    patch = rgb[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].reshape(-1, 3)
    return np.median(patch, axis=0)


def of_frame(frame) -> Optional[WhiteBalance]:
    """A frame's white balance: its own, or that of the frame it was cut from."""
    for f in (frame, getattr(frame, "anchor_of", None)):
        get = getattr(f, "wb", None) if f is not None else None
        if get is not None:
            return get() if callable(get) else get
    return None


def matrix_of(frame, settings: dict) -> Optional[np.ndarray]:
    """The re-balance an edit asks of this frame, or ``None``."""
    if "wb_kelvin" not in settings:
        return None
    return matrix_for(of_frame(frame), settings["wb_kelvin"], settings.get("wb_tint", 0))
