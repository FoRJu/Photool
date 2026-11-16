"""
Film scans: TIFF and PSD files read at every bit they hold.

A scanner writes TIFF (or, through Photoshop, PSD), usually 16 bits a channel. A
negative holds the whole scene in a narrow band of the scan's values, which is then
stretched five or six stops apart: from 8 bits that is posterised. Pillow reads a
16-bit RGB TIFF as 8 bits and a 16-bit PSD not at all, so these files are read here
instead, into the scene-linear float the rest of the app works in.

**TIFF** is decoded by OpenCV (already required): 8, 16 or 32-bit float; grey, grey
with alpha, RGB, RGB with alpha or with an infrared dust channel as a fourth sample
(the extra sample is dropped, never multiplied in); strips or tiles; uncompressed,
LZW or deflate; either byte order; BigTIFF. Only the first page is read (a scanner's
thumbnail page after it is ignored). The Orientation tag is honoured. CMYK, Lab,
white-is-zero grey and what OpenCV refuses (palette, 1-bit, an odd compression) are
not read here: the caller falls back to Pillow at 8 bits, and failing that to the
placeholder. A 16-bit grey file with alpha is read by OpenCV only at 8 bits: it gets
the 8-bit road (``read_8bit``) and not the linear one.

**PSD and PSB**: the merged picture, as Photoshop writes it when the file is saved
with "Maximize compatibility". Grey or RGB, 8, 16 or 32 bits, raw or PackBits. A
layered file saved without compatibility has only a white page with a notice as its
merged picture, and says so in its version resource (1057): its picture is then
rebuilt from its layers (export.psd_layers), and if that cannot be done the file has
no picture, rather than the white page. CMYK, Lab, indexed, duotone and bitmap files
are not read (``None``).

**The tone of the values.** A scan is either gamma-encoded (most scanner software,
anything from Photoshop) or linear (a "raw" scan, gamma 1.0). With an ICC profile, a
grey ramp is pushed through it and the profile's gamma fitted: within 0.15 of 1.0 the
values are used as they are; near the display's 2.2 they are decoded with the app's
own display curve, as every JPEG is; otherwise (ProPhoto's 1.8) with the fitted
gamma. With no profile, a float file is linear and an integer file display-encoded.
A 32-bit PSD is linear float whatever its profile says (Photoshop's 32-bit mode).

**The colours.** A matrix profile's primaries (rXYZ, gXYZ, bXYZ) are honoured: a
ProPhoto or Adobe RGB scan is turned into the app's sRGB/Rec. 709 primaries, and
what lies outside them is clipped at zero. Without its primaries a scan was flat and
leaned red. A profile made of tables (no matrix) is read as if its primaries were
sRGB's.
"""

import hashlib
import io
import logging
import math
import struct
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

SCAN_EXTENSIONS = {".tif", ".tiff", ".psd", ".psb"}

# Raised when the picture a scan gives changes, so that thumbnails made from the old
# one are made again (thumbnails.store keys a scan's thumbnails by it). 2: layered
# PSDs rebuilt from their layers, profiles' primaries honoured.
READER_VERSION = 2

# A decode is reduced by OpenCV (or a PSD's planes shrunk) by the largest of these
# factors that still leaves twice what was asked for: the rule _load_raw follows for
# a half-size RAW. A medium-format scan is 60 to 120 megapixels; a preview must not
# build that as float32.
_REDUCTIONS = (8, 4, 2)
# A profile whose fitted gamma is this close to 1.0 is taken as linear, and one this
# close to the display's 2.2 is decoded with the app's display curve.
_LINEAR_SLACK = 0.15
_DISPLAY_GAMMA, _DISPLAY_SLACK = 2.2, 0.2
# Bradford-adapted XYZ (D50, as every ICC profile's matrix is) to linear sRGB.
_SRGB_FROM_D50 = np.array([[3.1338561, -1.6168667, -0.4906146],
                           [-0.9787684, 1.9161415, 0.0334540],
                           [0.0719453, -0.2289914, 1.4052427]])

_TONE_BY_PROFILE: dict = {}
_TONE_LOCK = threading.Lock()


def is_scan(path) -> bool:
    return Path(path).suffix.lower() in SCAN_EXTENSIONS


def _is_psd(path: Path) -> bool:
    return path.suffix.lower() in (".psd", ".psb")


# ── TIFF header (Pillow, no load) ────────────────────────────────────

_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
                    11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TIFF_INT_TYPES = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 16: "Q", 17: "q", 18: "Q"}


def _tiff_tags(path: Path) -> Optional[dict]:
    """The first page's tags, read from the file's own header: classic TIFF or
    BigTIFF, either byte order. Integer tags as tuples, the ICC profile as bytes.
    (Pillow cannot open a float RGB TIFF at all, so it is not asked.)"""
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:2] == b"II":
            e = "<"
        elif head[:2] == b"MM":
            e = ">"
        else:
            return None
        (magic,) = struct.unpack(e + "H", head[2:4])
        if magic == 42:
            (ifd,) = struct.unpack(e + "I", head[4:8])
            count_fmt, entry_fmt, inline = "H", "HHII", 4
        elif magic == 43:
            (ifd,) = struct.unpack(e + "Q", head[8:16])
            count_fmt, entry_fmt, inline = "Q", "HHQQ", 8
        else:
            return None
        f.seek(ifd)
        count_size = struct.calcsize(count_fmt)
        (n,) = struct.unpack(e + count_fmt, f.read(count_size))
        entry_size = struct.calcsize("=" + entry_fmt)
        table = f.read(n * entry_size)
        if len(table) < n * entry_size:
            return None
        tags = {}
        for i in range(n):
            entry = table[i * entry_size:(i + 1) * entry_size]
            tag, typ, count, value = struct.unpack(e + entry_fmt, entry)
            size = _TIFF_TYPE_SIZES.get(typ)
            if size is None:
                continue
            nbytes = size * count
            if tag == 34675:
                if nbytes > 1 << 24:
                    continue
                if nbytes <= inline:
                    data = entry[-inline:][:nbytes]
                else:
                    here = f.tell()
                    f.seek(value)
                    data = f.read(nbytes)
                    f.seek(here)
                tags[tag] = data
                continue
            if typ not in _TIFF_INT_TYPES or count > 16:
                continue
            fmt = e + _TIFF_INT_TYPES[typ] * count
            if nbytes <= inline:
                data = entry[-inline:][:nbytes]
            else:
                here = f.tell()
                f.seek(value)
                data = f.read(nbytes)
                f.seek(here)
            if len(data) == nbytes:
                tags[tag] = struct.unpack(fmt, data)
            if tag == 274 and nbytes <= inline:
                # Where the Orientation's value sits, to be set to 1 in the bytes
                # handed to OpenCV (see _read_tiff).
                start = ifd + count_size + i * entry_size + entry_size - inline
                tags["orientation_at"] = (start, e + _TIFF_INT_TYPES[typ])
        tags["byte_order"] = e
        return tags


def _tiff_info(path: Path) -> Optional[dict]:
    """Size, orientation, profile, photometric and sample format of a TIFF's first page."""
    try:
        tags = _tiff_tags(path)
        if not tags or 256 not in tags or 257 not in tags:
            return None
        return {
            "size": (int(tags[256][0]), int(tags[257][0])),
            "orientation": int(tags.get(274, (1,))[0]),
            "icc": tags.get(34675),
            "float": 3 in tags.get(339, (1,)),
            "samples": int(tags.get(277, (1,))[0]),
            "photometric": int(tags.get(262, (2,))[0]),
            "bits": int(max(tags.get(258, (8,)))),
            "orientation_at": tags.get("orientation_at"),
        }
    except Exception as e:
        logger.debug(f"No TIFF header for {path.name}: {e}")
        return None


def _turned(orientation: int) -> bool:
    return orientation in (5, 6, 7, 8)


def _orient(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Turn an (H, W[, C]) array as the Orientation tag says, like exif_transpose."""
    swap = (1, 0) + tuple(range(2, arr.ndim))
    if orientation == 2:
        return arr[:, ::-1]
    if orientation == 3:
        return arr[::-1, ::-1]
    if orientation == 4:
        return arr[::-1]
    if orientation == 5:
        return arr.transpose(swap)
    if orientation == 6:
        return np.rot90(arr, -1)
    if orientation == 7:
        return np.rot90(arr, 2).transpose(swap)
    if orientation == 8:
        return np.rot90(arr, 1)
    return arr


# ── PSD header ───────────────────────────────────────────────────────

def _psd_layout(path: Path) -> Optional[dict]:
    """The header, the ICC profile and where the merged picture starts."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if len(head) < 26:
                return None
            sig, version, _, channels, height, width, depth, mode = struct.unpack(">4sH6sHIIHH", head)
            if sig != b"8BPS" or version not in (1, 2):
                return None
            info = {"version": version, "channels": channels, "height": height,
                    "width": width, "depth": depth, "mode": mode, "icc": None,
                    "merged": True}
            (n,) = struct.unpack(">I", f.read(4))
            f.seek(n, 1)
            (n,) = struct.unpack(">I", f.read(4))
            end = f.tell() + n
            while f.tell() + 12 <= end:
                if f.read(4) != b"8BIM":
                    break
                (rid,) = struct.unpack(">H", f.read(2))
                (name_len,) = struct.unpack(">B", f.read(1))
                f.seek(name_len + ((name_len + 1) % 2), 1)
                (size,) = struct.unpack(">I", f.read(4))
                if rid == 1039:
                    info["icc"] = f.read(size)
                    f.seek(size % 2, 1)
                elif rid == 1057 and size >= 5:
                    # The version info: its fifth byte says whether the merged
                    # picture is the real one ("Maximize compatibility") or a white
                    # page with a notice.
                    version_info = f.read(size)
                    info["merged"] = bool(version_info[4])
                    f.seek(size % 2, 1)
                else:
                    f.seek(size + size % 2, 1)
            f.seek(end)
            info["layers"] = end
            if version == 1:
                (n,) = struct.unpack(">I", f.read(4))
            else:
                (n,) = struct.unpack(">Q", f.read(8))
            f.seek(n, 1)
            info["data"] = f.tell()
            return info
    except Exception as e:
        logger.debug(f"No PSD header for {path.name}: {e}")
        return None


def _psd_readable(info: Optional[dict]) -> bool:
    return bool(info) and info["mode"] in (1, 3) and info["depth"] in (8, 16, 32) \
        and info["channels"] >= (1 if info["mode"] == 1 else 3) \
        and info["width"] > 0 and info["height"] > 0


def _unpack_row(data: bytes, out: bytearray, start: int, length: int) -> None:
    """PackBits: one row into ``out[start:start + length]``."""
    i, pos, end, n = 0, start, start + length, len(data)
    while i < n and pos < end:
        h = data[i]
        i += 1
        if h < 128:
            run = h + 1
            out[pos:pos + run] = data[i:i + run]
            i += run
            pos += run
        elif h > 128:
            run = 257 - h
            out[pos:pos + run] = data[i:i + 1] * run
            i += 1
            pos += run
    if pos != end:
        raise ValueError("PackBits row of the wrong length")


def _psd_planes(path: Path, info: dict, wanted: int, target: Optional[tuple]):
    """The ``wanted`` first planes of the merged picture, native-endian, each shrunk
    to ``target`` (w, h) with INTER_AREA as soon as it is read, so that no more than
    one whole plane is held at a time."""
    import cv2

    w, h, depth, channels = info["width"], info["height"], info["depth"], info["channels"]
    dtype = {8: np.dtype("u1"), 16: np.dtype(">u2"), 32: np.dtype(">f4")}[depth]
    native = dtype.newbyteorder("=")
    row_bytes = w * dtype.itemsize
    planes = []

    def finish(plane):
        plane = plane.astype(native, copy=False)
        if target is not None:
            plane = cv2.resize(plane, target, interpolation=cv2.INTER_AREA)
        return plane

    with open(path, "rb") as f:
        f.seek(info["data"])
        (compression,) = struct.unpack(">H", f.read(2))
        if compression == 0:
            start = info["data"] + 2
            needed = start + channels * h * row_bytes
            if path.stat().st_size < needed:
                raise ValueError("truncated")
            raw = np.memmap(path, dtype=dtype, mode="r", offset=start, shape=(channels, h, w))
            for c in range(wanted):
                planes.append(finish(np.array(raw[c])))
            del raw
            return planes
        if compression != 1:
            return None
        count_type = ">u2" if info["version"] == 1 else ">u4"
        counts = np.frombuffer(f.read(channels * h * np.dtype(count_type).itemsize), dtype=count_type)
        if counts.size != channels * h:
            raise ValueError("truncated")
        counts = counts.reshape(channels, h).astype(np.int64)
        for c in range(wanted):
            data = f.read(int(counts[c].sum()))
            if len(data) != counts[c].sum():
                raise ValueError("truncated")
            out = bytearray(h * row_bytes)
            pos = 0
            for r in range(h):
                n = int(counts[c, r])
                _unpack_row(data[pos:pos + n], out, r * row_bytes, row_bytes)
                pos += n
            planes.append(finish(np.frombuffer(out, dtype=dtype).reshape(h, w)))
    return planes


# ── Size ─────────────────────────────────────────────────────────────

def size_of(path) -> Optional[tuple]:
    """(width, height) as shown (turned by the Orientation tag), from the header only."""
    path = Path(path)
    try:
        if _is_psd(path):
            info = _psd_layout(path)
            return (info["width"], info["height"]) if info else None
        info = _tiff_info(path)
        if info is None:
            return None
        w, h = info["size"]
        return (h, w) if _turned(info["orientation"]) else (w, h)
    except Exception:
        return None


def _reduction(long_side: int, max_resolution: int) -> int:
    if max_resolution <= 0 or not long_side:
        return 1
    for f in _REDUCTIONS:
        if long_side >= 2 * f * max_resolution:
            return f
    return 1


def _shrunk_size(w: int, h: int, max_resolution: int) -> Optional[tuple]:
    """(w, h) with the longer side at twice ``max_resolution``, or ``None`` when the
    picture is not more than that already."""
    if max_resolution <= 0 or max(w, h) <= 2 * max_resolution:
        return None
    ratio = 2 * max_resolution / max(w, h)
    return (max(1, round(w * ratio)), max(1, round(h * ratio)))


# ── The tone ─────────────────────────────────────────────────────────

def _profile_gamma(profile: bytes) -> Optional[float]:
    """The gamma a profile's grey ramp follows, fitted at 0.25, 0.5 and 0.75."""
    from PIL import ImageCms

    src = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    try:
        space = src.profile.xcolor_space.strip().upper()
    except Exception:
        space = "RGB"
    mode = "L" if space.startswith("GRAY") else "RGB"
    levels = np.array([64, 128, 191], dtype=np.uint8)
    ramp = Image.fromarray(levels.reshape(1, 3), "L")
    if mode == "RGB":
        ramp = ramp.convert("RGB")
    transform = ImageCms.buildTransform(src, ImageCms.createProfile("LAB"), mode, "LAB")
    lab = np.asarray(ImageCms.applyTransform(ramp, transform), dtype=np.float64)
    big_l = lab.reshape(-1, 3)[:, 0] * (100.0 / 255.0)
    y = np.where(big_l > 8.0, ((big_l + 16.0) / 116.0) ** 3, big_l / 903.3)
    v = levels / 255.0
    good = y > 0
    if not good.any():
        return None
    return float(np.mean(np.log(y[good]) / np.log(v[good])))


def _profile_matrix(profile: bytes) -> Optional[np.ndarray]:
    """The profile's primaries as a 3x3 from its linear values to linear sRGB, or
    ``None`` with no matrix or one that is sRGB's own."""
    if len(profile) < 132 or profile[16:20] != b"RGB ":
        return None
    (count,) = struct.unpack(">I", profile[128:132])
    tags = {}
    for i in range(min(count, 200)):
        sig, at, size = struct.unpack(">4sII", profile[132 + 12 * i:144 + 12 * i])
        tags[sig] = profile[at:at + size]
    columns = []
    for sig in (b"rXYZ", b"gXYZ", b"bXYZ"):
        data = tags.get(sig)
        if not data or len(data) < 20 or data[:4] != b"XYZ ":
            return None
        columns.append([v / 65536.0 for v in struct.unpack(">3i", data[8:20])])
    primaries = np.array(columns).T
    if abs(np.linalg.det(primaries)) < 1e-6:
        return None
    matrix = _SRGB_FROM_D50 @ primaries
    if np.abs(matrix - np.eye(3)).max() < 0.02:
        return None
    return matrix.astype(np.float32)


def _colour_of_profile(profile: bytes) -> dict:
    """``tone`` ("linear", "display" or ``None``), the ``gamma`` to decode a managed
    file with (see below), and the ``matrix`` to sRGB primaries; cached."""
    key = hashlib.sha1(profile).hexdigest()
    with _TONE_LOCK:
        if key in _TONE_BY_PROFILE:
            return _TONE_BY_PROFILE[key]
    try:
        gamma = _profile_gamma(profile)
    except Exception as e:
        logger.debug(f"Unreadable ICC profile: {e}")
        gamma = None
    tone = None if gamma is None else ("linear" if abs(gamma - 1.0) <= _LINEAR_SLACK else "display")
    try:
        matrix = _profile_matrix(profile)
    except Exception as e:
        logger.debug(f"No primaries in the ICC profile: {e}")
        matrix = None
    # A display-encoded file in other primaries, or with another gamma, is turned
    # into sRGB through true linear light (as Photoshop's own conversion would) and
    # then read as a JPEG is: "gamma" is set for it. Otherwise the app's curve.
    managed = tone == "display" and (matrix is not None or abs(gamma - _DISPLAY_GAMMA) > _DISPLAY_SLACK)
    colour = {"tone": tone, "gamma": gamma if managed else None, "matrix": matrix}
    with _TONE_LOCK:
        _TONE_BY_PROFILE[key] = colour
    return colour


def _tone_of_profile(profile: bytes) -> Optional[str]:
    return _colour_of_profile(profile)["tone"]


def _gamma_and_matrix(profile: Optional[bytes]) -> tuple:
    if not profile:
        return None, None
    colour = _colour_of_profile(profile)
    return colour["gamma"], colour["matrix"]


_AS_JPEG = None


def _as_jpeg_table() -> np.ndarray:
    """True linear light (0..1, 65536 steps) to the app's linear, as an sRGB JPEG of
    it would be read: encoded with sRGB's curve, decoded with the app's."""
    global _AS_JPEG
    if _AS_JPEG is None:
        from adjust.render import decode

        v = np.linspace(0.0, 1.0, 65536)
        srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)
        _AS_JPEG = decode(srgb.astype(np.float32))
    return _AS_JPEG


def _to_working(rgb: np.ndarray, matrix: Optional[np.ndarray], managed: bool) -> np.ndarray:
    """Linear light in the profile's primaries to sRGB's (``matrix``), clipped at
    zero; ``managed``, from true linear to the app's linear as well (the light of a
    JPEG of the same picture). In place, by strips."""
    if matrix is None and not managed:
        return rgb
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    if managed:
        from export.psd_layers import lut
    step = max(1, (1 << 22) // max(rgb.shape[1], 1))
    for top in range(0, rgb.shape[0], step):
        strip = rgb[top:top + step]
        if matrix is not None and strip.ndim == 3 and strip.shape[2] == 3:
            strip[...] = strip @ matrix.T
        np.maximum(strip, 0.0, out=strip)
        if managed:
            strip[...] = lut(strip, _as_jpeg_table())
    return rgb


def _tone(profile: Optional[bytes], is_float: bool) -> str:
    if profile:
        tone = _tone_of_profile(profile)
        if tone is not None:
            return tone
    return "linear" if is_float else "display"


def tone_of(path) -> Optional[str]:
    """``"linear"`` or ``"display"``: how the file's values are read; ``None`` for a
    file that is not a readable scan."""
    path = Path(path)
    if _is_psd(path):
        info = _psd_layout(path)
        if not _psd_readable(info):
            return None
        return "linear" if info["depth"] == 32 else _tone(info["icc"], False)
    info = _tiff_info(path)
    if info is None:
        return None
    return _tone(info["icc"], info["float"])


_DECODE_TABLES: dict = {}


def _decode_table(levels: int) -> np.ndarray:
    """The app's display decode for every integer level: a 16-bit scan is decoded by
    lookup, into the one float array it becomes, with no temporaries."""
    table = _DECODE_TABLES.get(levels)
    if table is None:
        from adjust.render import decode

        table = decode(np.arange(levels, dtype=np.float32) / np.float32(levels - 1))
        _DECODE_TABLES[levels] = table
    return table


def _to_float(arr: np.ndarray, tone: str, gamma: Optional[float] = None) -> Optional[np.ndarray]:
    """Integer or float samples to scene-linear float32, 1.0 = the file's white;
    display-encoded values with the app's curve, or with ``gamma`` when given."""
    if gamma is not None and tone == "display":
        if arr.dtype in (np.uint8, np.uint16):
            levels = 256 if arr.dtype == np.uint8 else 65536
            table = (np.arange(levels, dtype=np.float64) / (levels - 1)) ** gamma
            return table.astype(np.float32)[arr]
        out = np.clip(np.nan_to_num(arr.astype(np.float32)), 0.0, None)
        return np.ascontiguousarray(np.power(out, np.float32(gamma)), dtype=np.float32)
    if arr.dtype in (np.uint8, np.uint16):
        levels = 256 if arr.dtype == np.uint8 else 65536
        if tone == "display":
            return _decode_table(levels)[arr]
        return arr.astype(np.float32) / np.float32(levels - 1)
    if not np.issubdtype(arr.dtype, np.floating):
        return None
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(out, 0.0, out=out)
    if tone == "display":
        from adjust.render import decode, in_strips

        out = in_strips(decode, out)
    return np.ascontiguousarray(out, dtype=np.float32)


# ── Reading ──────────────────────────────────────────────────────────

def _read_tiff(path: Path, max_resolution: int, shallow: bool = False) -> Optional[np.ndarray]:
    import cv2

    info = _tiff_info(path)
    # Grey (black is zero) and RGB only. OpenCV would turn CMYK, Lab or a white-is-zero
    # grey into RGB on its own terms; those go to Pillow, as before.
    if info is not None and info["photometric"] not in (1, 2):
        return None
    grey = bool(info) and info["samples"] <= 2
    long_side = max(info["size"]) if info else 0
    f = _reduction(long_side, max_resolution)
    if f > 1:
        flag = {8: cv2.IMREAD_REDUCED_COLOR_8, 4: cv2.IMREAD_REDUCED_COLOR_4,
                2: cv2.IMREAD_REDUCED_COLOR_2}[f]
        if grey:
            flag = {8: cv2.IMREAD_REDUCED_GRAYSCALE_8, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
                    2: cv2.IMREAD_REDUCED_GRAYSCALE_2}[f]
        flag |= cv2.IMREAD_ANYDEPTH
    else:
        flag = cv2.IMREAD_UNCHANGED
    turned_in_file = bool(info and info["orientation_at"] and info["orientation"] != 1)
    if str(path).isascii() and not turned_in_file:
        # By path, so that the file is not held in memory beside its decode: on a
        # 100-megapixel uncompressed scan that is 600 MB less at the peak.
        arr = cv2.imread(str(path), flag)
    else:
        # By bytes: cv2.imread cannot open a path with non-ASCII characters on
        # Windows, and a turned file's bytes are changed first (below).
        buf = np.fromfile(str(path), dtype=np.uint8)
        if turned_in_file:
            # OpenCV 5 turns a TIFF by its Orientation in a full decode and not in
            # a reduced one, and IMREAD_IGNORE_ORIENTATION does not stop it. Told
            # the picture is upright, it never turns it, and it is turned here.
            at, fmt = info["orientation_at"]
            one = struct.pack(fmt, 1)
            buf[at:at + len(one)] = np.frombuffer(one, dtype=np.uint8)
        arr = cv2.imdecode(buf, flag)
        del buf
    if arr is None:
        return None
    if not shallow and arr.dtype == np.uint8 and info and info["bits"] > 8:
        # OpenCV read it, but at 8 bits (16-bit grey with alpha): the 8-bit road may
        # have it, this one may not.
        return None
    target = _shrunk_size(arr.shape[1], arr.shape[0], max_resolution)
    if target is not None:
        # Still on the integers, and before anything is copied.
        arr = cv2.resize(arr, target, interpolation=cv2.INTER_AREA)
    planes = 1 if arr.ndim == 2 else arr.shape[2]
    if planes in (1, 2):
        arr = arr if arr.ndim == 2 else arr[..., 0]
    elif planes >= 3:
        # BGR(A) to RGB; a fourth sample (alpha or infrared) is dropped, not multiplied in.
        arr = arr[..., 2::-1]
    tone = _tone(info["icc"], np.issubdtype(arr.dtype, np.floating)) if info else \
        ("linear" if np.issubdtype(arr.dtype, np.floating) else "display")
    gamma, matrix = _gamma_and_matrix(info["icc"] if info else None)
    out = _to_float(arr, tone, gamma)
    if out is None:
        return None
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
        matrix = None
    out = _to_working(out, matrix, gamma is not None and tone == "display")
    if info:
        out = _orient(out, info["orientation"])
    return np.ascontiguousarray(out)


def _read_psd(path: Path, max_resolution: int = 0) -> Optional[np.ndarray]:
    """The picture of a PSD or PSB as scene-linear (H, W, 3) float32, or ``None``: its
    merged picture, or, saved without "Maximize compatibility", its layers'."""
    info = _psd_layout(path)
    if not _psd_readable(info):
        return None
    wanted = 1 if info["mode"] == 1 else 3
    target = _shrunk_size(info["width"], info["height"], max_resolution)
    if info["merged"]:
        planes = _psd_planes(path, info, wanted, target)
    else:
        from export import psd_layers

        # Built at the size asked for, not twice it: every layer is shrunk with
        # INTER_AREA as it is read, so the margin left for a resize in linear light
        # buys nothing here, and on a 5600-pixel scan it meant blending, masking
        # and curving every layer at 4800 pixels for a 2400-pixel thumbnail.
        if max_resolution > 0 and max(info["width"], info["height"]) > max_resolution:
            ratio = max_resolution / max(info["width"], info["height"])
            target = (max(1, round(info["width"] * ratio)), max(1, round(info["height"] * ratio)))
        planes = psd_layers.composite(path, info, target)
        if planes is None:
            logger.warning(f"{path.name} was saved without Photoshop's \"Maximize "
                           f"compatibility\" and its layers could not be read: no picture")
            return None
    if planes is None:
        return None
    tone = "linear" if info["depth"] == 32 else _tone(info["icc"], False)
    gamma, matrix = _gamma_and_matrix(info["icc"])
    planes = [_to_float(p, tone, gamma) for p in planes]
    managed = gamma is not None and tone == "display"
    if wanted == 1:
        return _to_working(np.stack(planes * 3, axis=-1), None, managed)
    return _to_working(np.stack(planes, axis=-1), matrix, managed)


def _read(path: Path, max_resolution: int, shallow: bool) -> Optional[np.ndarray]:
    try:
        if _is_psd(path):
            return _read_psd(path, max_resolution)
        return _read_tiff(path, max_resolution, shallow)
    except Exception as e:
        logger.info(f"Scan reader could not read {path.name}: {e}")
        return None


def read_linear(path, max_resolution: int = 0) -> Optional[np.ndarray]:
    """(H, W, 3) float32, scene-linear as the rest of the app means it, 1.0 = the
    file's white; upright. With ``max_resolution`` the result's longer side is
    between it and about four times it (the caller finishes the resize in linear
    light). ``None`` for a file this cannot read at more than 8 bits, or at all;
    never raises."""
    return _read(Path(path), max_resolution, shallow=False)


def read_8bit(path, max_resolution: int = 0) -> Optional[Image.Image]:
    """The file as an 8-bit RGB picture, display-encoded and upright; ``None`` when
    neither this reader nor Pillow can read it."""
    path = Path(path)
    rgb = _read(path, max_resolution, shallow=True)
    if rgb is not None:
        try:
            from adjust.render import encode, in_strips, resize_linear

            if max_resolution > 0:
                rgb = resize_linear(rgb, max_resolution)
            enc = in_strips(encode, rgb)
            return Image.fromarray(np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8), "RGB")
        except Exception as e:
            logger.info(f"Scan reader could not encode {path.name}: {e}")
    if _is_psd(path):
        info = _psd_layout(path)
        if info is not None and not info["merged"]:
            # Pillow would read the white page that stands in for the picture.
            return None
    try:
        img = Image.open(path)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max_resolution > 0:
            img.thumbnail((max_resolution, max_resolution), Image.LANCZOS)
        return img
    except Exception as e:
        logger.info(f"Could not read {path.name}: {e}")
        return None
