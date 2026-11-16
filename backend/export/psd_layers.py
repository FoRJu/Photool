"""
A layered PSD's picture rebuilt from its layers, for a file saved without "Maximize
compatibility".

Such a file's merged picture is a white page: Photoshop writes the real one only with
that setting on, and says which it did in the version resource (1057). The picture
is then rebuilt here from the layers, as Photoshop would show it, in the file's own
encoded values (0..1), which the caller decodes as it decodes a merged picture.

What is applied:

- **Pixel layers** (a smart object's too, from the pixels it stores) in Normal,
  Multiply, Screen, Overlay, Soft Light, Hard Light, Darken, Lighten, Linear Dodge,
  Color Dodge, Color Burn and Difference, with their transparency, opacity, fill
  opacity and layer mask; clipped layers are clipped to the layer below.
- **Adjustment layers**: Levels, Curves, Invert, and Hue/Saturation (the master
  hue, saturation and lightness), each through its opacity and mask.
- **Groups**, pass-through or with their own blend mode, opacity and mask. Hidden
  layers and groups are left out.

What is not: other adjustments (Color Balance, Brightness/Contrast, Selective
Color…), fill layers, layer effects, Hue/Saturation's per-colour ranges and its
Colorize, other blend modes, a vector mask. Each such layer is skipped, and the
file's name and what was skipped are logged: the picture is then close to
Photoshop's but not the same.

With ``target`` every layer is shrunk to the canvas's scale as it is read, so a
preview of a 100 MB file holds a preview-sized canvas.
"""

import logging
import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Additional-info keys whose length is 8 bytes in a PSB.
_LONG_KEYS = {b"LMsk", b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32", b"Mtrn", b"Alph",
              b"FMsk", b"lnk2", b"FEid", b"FXid", b"PxSD", b"cinf"}
_ADJUSTMENTS = {b"levl", b"curv", b"nvrt", b"hue2"}
_UNSUPPORTED = {b"brit", b"blnc", b"selc", b"mixr", b"expA", b"vibA", b"phfl", b"thrs",
                b"post", b"grdm", b"clrL", b"blwh", b"hue ", b"SoCo", b"GdFl", b"PtFl"}


# ── Reading the records ──────────────────────────────────────────────

class _Layer:
    __slots__ = ("name", "rect", "channels", "blend", "opacity", "clipped", "visible",
                 "mask", "keys", "section")

    @property
    def fill(self) -> float:
        data = self.keys.get(b"iOpa")
        return data[0] / 255.0 if data else 1.0


def _u(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _layer_records(f, psb: bool, end: int) -> Optional[list]:
    """The layer records of a layer-info block, bottom layer first, each with where
    its channels' data lies in the file."""
    (count,) = _u(f, ">h")
    layers = []
    for _ in range(abs(count)):
        L = _Layer()
        L.rect = _u(f, ">4i")
        (n,) = _u(f, ">H")
        L.channels = [list(_u(f, ">hQ" if psb else ">hI")) for _ in range(n)]
        sig, L.blend, L.opacity, L.clipped, flags, _ = _u(f, ">4s4sBBBB")
        if sig != b"8BIM":
            return None
        L.visible = not flags & 2
        (extra,) = _u(f, ">I")
        extra_end = f.tell() + extra
        (mask_len,) = _u(f, ">I")
        L.mask = f.read(mask_len)
        (blend_len,) = _u(f, ">I")
        f.seek(blend_len, 1)
        (name_len,) = _u(f, ">B")
        L.name = f.read(name_len).decode("latin-1")
        f.seek((4 - (name_len + 1) % 4) % 4, 1)
        L.keys = {}
        while f.tell() + 12 <= extra_end:
            sig, key = _u(f, ">4s4s")
            if sig not in (b"8BIM", b"8B64"):
                break
            (size,) = _u(f, ">Q" if psb and key in _LONG_KEYS else ">I")
            L.keys[key] = f.read(size) if size < 1 << 20 else b""
            if key not in L.keys or len(L.keys[key]) != size:
                f.seek(f.tell() + size - len(L.keys[key]))
        if b"luni" in L.keys and len(L.keys[b"luni"]) >= 4:
            (chars,) = struct.unpack(">I", L.keys[b"luni"][:4])
            L.name = L.keys[b"luni"][4:4 + 2 * chars].decode("utf-16-be", "replace")
        section = L.keys.get(b"lsct")
        L.section = struct.unpack(">I", section[:4])[0] if section and len(section) >= 4 else 0
        if section and len(section) >= 12 and section[4:8] == b"8BIM":
            L.blend = section[8:12]
        f.seek(extra_end)
        layers.append(L)
    at = f.tell()
    for L in layers:
        for ch in L.channels:
            cid, length = ch
            ch[1] = (at, length)
            at += length
    if at > end:
        return None
    return layers


def read_records(path: Path, info: dict) -> Optional[list]:
    """The file's layers, bottom first; ``None`` when there are none."""
    psb = info["version"] == 2
    with open(path, "rb") as f:
        f.seek(info["layers"])
        (section,) = _u(f, ">Q" if psb else ">I")
        end = f.tell() + section
        (size,) = _u(f, ">Q" if psb else ">I")
        info_end = f.tell() + size
        if size:
            layers = _layer_records(f, psb, info_end)
            if layers:
                return layers
        # 16- and 32-bit files keep their layers in the additional info (Lr16, Lr32)
        # after the global mask, with an empty layer-info block before it.
        f.seek(info_end)
        (mask,) = _u(f, ">I")
        f.seek(mask, 1)
        while f.tell() + 12 <= end:
            sig, key = _u(f, ">4s4s")
            if sig not in (b"8BIM", b"8B64"):
                return None
            (size,) = _u(f, ">Q" if psb and key in _LONG_KEYS else ">I")
            here = f.tell()
            if key in (b"Lr16", b"Lr32", b"Layr"):
                return _layer_records(f, psb, here + size)
            f.seek(here + size)
    return None


# ── Channels ─────────────────────────────────────────────────────────

def _channel(f, where: tuple, h: int, w: int, depth: int, psb: bool) -> Optional[np.ndarray]:
    """One channel of a layer, ``h`` by ``w``, as the file holds it: native-endian
    uint8 or uint16, or float32 (see :func:`_unit_range`)."""
    at, length = where
    if length < 2 or h <= 0 or w <= 0:
        return None
    f.seek(at)
    (compression,) = _u(f, ">H")
    data = f.read(length - 2)
    dtype = {8: np.dtype("u1"), 16: np.dtype(">u2"), 32: np.dtype(">f4")}[depth]
    row = w * dtype.itemsize
    if compression == 0:
        arr = np.frombuffer(data, dtype=dtype, count=h * w).reshape(h, w)
    elif compression == 1:
        from export.scans import _unpack_row

        count = np.dtype(">u4" if psb else ">u2")
        counts = np.frombuffer(data, dtype=count, count=h).astype(np.int64)
        out = bytearray(h * row)
        pos = h * count.itemsize
        for r in range(h):
            n = int(counts[r])
            _unpack_row(data[pos:pos + n], out, r * row, row)
            pos += n
        arr = np.frombuffer(out, dtype=dtype).reshape(h, w)
    elif compression in (2, 3):
        raw = zlib.decompress(data)
        if compression == 2:
            arr = np.frombuffer(raw, dtype=dtype, count=h * w).reshape(h, w)
        elif depth == 32:
            # Each row's bytes are split into four planes (every float's first byte,
            # then its second…), and those bytes delta-coded along the row.
            b = np.frombuffer(raw, dtype=np.uint8, count=h * row).reshape(h, row)
            b = np.cumsum(b, axis=1, dtype=np.uint8)
            arr = b.reshape(h, 4, w).transpose(0, 2, 1).copy().view(">f4").reshape(h, w)
        else:
            # Each row delta-coded, sample by sample, wrapping as the integers do.
            arr = np.frombuffer(raw, dtype=dtype, count=h * w).reshape(h, w)
            return np.cumsum(arr, axis=1, dtype=dtype.newbyteorder("="))
    else:
        return None
    if depth == 32:
        return np.nan_to_num(arr.astype(np.float32))
    return arr.astype(dtype.newbyteorder("="), copy=False)


def _unit_range(plane: np.ndarray) -> np.ndarray:
    """Samples to float32 0..1 (a float plane is already in its own units)."""
    if plane.dtype == np.uint8:
        return plane.astype(np.float32) * np.float32(1 / 255)
    if plane.dtype == np.uint16:
        return plane.astype(np.float32) * np.float32(1 / 65535)
    return plane


# ── The canvas ───────────────────────────────────────────────────────

class _Canvas:
    """Colour planes and coverage at the canvas's scale."""

    def __init__(self, n: int, w: int, h: int, sx: float, sy: float, opaque=False):
        self.w, self.h, self.sx, self.sy = w, h, sx, sy
        self.c = [np.zeros((h, w), np.float32) for _ in range(n)]
        self.a = np.full((h, w), 1.0 if opaque else 0.0, np.float32)

    def box(self, rect) -> tuple:
        """A rect (top, left, bottom, right) in the file to (top, left, bottom, right)
        on the canvas, unclipped."""
        t, l, b, r = rect
        return (round(t * self.sy), round(l * self.sx), round(b * self.sy), round(r * self.sx))


def _fit(plane: np.ndarray, h: int, w: int) -> np.ndarray:
    if plane.shape == (h, w):
        return plane
    import cv2

    shrinking = h <= plane.shape[0] and w <= plane.shape[1]
    return cv2.resize(plane, (w, h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def _place(canvas: _Canvas, plane: Optional[np.ndarray], box: tuple, default: float) -> np.ndarray:
    """``plane`` (already at the box's size) put on a canvas-size plane at ``box``,
    the rest ``default``."""
    out = np.full((canvas.h, canvas.w), default, np.float32)
    if plane is None:
        return out
    t, l, b, r = box
    tt, ll, bb, rr = max(t, 0), max(l, 0), min(b, canvas.h), min(r, canvas.w)
    if bb > tt and rr > ll:
        out[tt:bb, ll:rr] = plane[tt - t:bb - t, ll - l:rr - l]
    return out


class _Reader:
    def __init__(self, f, info: dict, canvas_size: tuple, psd_name: str):
        self.f, self.info, self.name = f, info, psd_name
        self.depth, self.psb = info["depth"], info["version"] == 2
        self.colours = 1 if info["mode"] == 1 else 3
        w, h = canvas_size
        self.w, self.h = w, h
        self.sx, self.sy = w / info["width"], h / info["height"]
        self.skipped: list = []

    def new_canvas(self, opaque=False) -> _Canvas:
        return _Canvas(self.colours, self.w, self.h, self.sx, self.sy, opaque)

    def channel_at(self, canvas: _Canvas, L: _Layer, cid: int, rect: tuple):
        """Channel ``cid`` of ``L`` at the canvas's scale over its own box, or ``None``."""
        where = next((ch[1] for ch in L.channels if ch[0] == cid), None)
        t, l, b, r = rect
        if where is None or b <= t or r <= l:
            return None
        plane = _channel(self.f, where, b - t, r - l, self.depth, self.psb)
        if plane is None:
            return None
        ct, cl, cb, cr = canvas.box(rect)
        if cb <= ct or cr <= cl:
            return None
        # Shrunk while still the file's integers, and only then made float: a
        # preview of a 100-megapixel layer converts a tenth of the samples, and
        # moving memory, not arithmetic, is what limits several reads at once.
        return _unit_range(_fit(plane, cb - ct, cr - cl))

    def mask(self, canvas: _Canvas, L: _Layer):
        """The layer mask on a canvas-size plane, or ``None`` with no mask in use."""
        if len(L.mask) < 18:
            return None
        rect = struct.unpack(">4i", L.mask[:16])
        default, flags = L.mask[16], L.mask[17]
        if flags & 2:      # disabled
            return None
        plane = self.channel_at(canvas, L, -2, rect)
        m = _place(canvas, plane, canvas.box(rect), default / 255.0)
        if flags & 4:      # inverted when blending
            m = 1.0 - m
        if m.min() >= 1.0:
            return None
        return m


# ── Blend modes, on the encoded values ───────────────────────────────

def _soft_light(d, s):
    g = np.where(d <= 0.25, ((16 * d - 12) * d + 4) * d, np.sqrt(np.maximum(d, 0)))
    return np.where(s <= 0.5, d - (1 - 2 * s) * d * (1 - d), d + (2 * s - 1) * (g - d))


_BLENDS = {
    b"norm": lambda d, s: s,
    b"mul ": lambda d, s: d * s,
    b"scrn": lambda d, s: d + s - d * s,
    b"over": lambda d, s: np.where(d <= 0.5, 2 * d * s, 1 - 2 * (1 - d) * (1 - s)),
    b"hLit": lambda d, s: np.where(s <= 0.5, 2 * d * s, 1 - 2 * (1 - d) * (1 - s)),
    b"sLit": _soft_light,
    b"dark": np.minimum,
    b"lite": np.maximum,
    b"lddg": lambda d, s: np.minimum(d + s, 1.0),
    b"div ": lambda d, s: np.where(s >= 1, 1.0, np.minimum(d / np.maximum(1 - s, 1e-6), 1.0)),
    b"idiv": lambda d, s: np.where(s <= 0, 0.0, 1 - np.minimum((1 - d) / np.maximum(s, 1e-6), 1.0)),
    b"diff": lambda d, s: np.abs(d - s),
}


def _over(canvas: _Canvas, colours: list, cover: np.ndarray, box: tuple, blend) -> None:
    """Colour planes (at the box's size) with their coverage, blended onto the canvas."""
    t, l, b, r = box
    tt, ll, bb, rr = max(t, 0), max(l, 0), min(b, canvas.h), min(r, canvas.w)
    if bb <= tt or rr <= ll:
        return
    cut = (slice(tt - t, bb - t), slice(ll - l, rr - l))
    k = cover[cut]
    da = canvas.a[tt:bb, ll:rr]
    if da.min() >= 1.0:
        # On an opaque canvas (below a Background): a plain mix.
        for i, src in enumerate(colours):
            d = canvas.c[i][tt:bb, ll:rr]
            mixed = blend(d, np.maximum(src[cut], 0.0))
            d += (mixed - d) * k
        return
    out_a = k + da * (1 - k)
    safe = np.maximum(out_a, 1e-6)
    for i, src in enumerate(colours):
        d = canvas.c[i][tt:bb, ll:rr]
        s = np.clip(src[cut], 0, None) if src.dtype == np.float32 else src[cut]
        mixed = blend(d, s)
        # Where nothing is below, the layer shows as it is.
        mixed = s * (1 - da) + mixed * da
        canvas.c[i][tt:bb, ll:rr] = np.where(out_a > 0, (mixed * k + d * da * (1 - k)) / safe, d)
    canvas.a[tt:bb, ll:rr] = out_a


# ── Adjustments ──────────────────────────────────────────────────────

def _spline(points: list) -> np.ndarray:
    """A curve through ``points`` (input, output) on 0..255, natural cubic, as 256
    levels on 0..1."""
    pts = sorted(set(points))
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    grid = np.arange(256, dtype=np.float64)
    if len(xs) < 3:
        out = np.interp(grid, xs, ys)
    else:
        n = len(xs)
        hs = np.diff(xs)
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        A[0, 0] = A[-1, -1] = 1
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = hs[i - 1], 2 * (hs[i - 1] + hs[i]), hs[i]
            rhs[i] = 6 * ((ys[i + 1] - ys[i]) / hs[i] - (ys[i] - ys[i - 1]) / hs[i - 1])
        m = np.linalg.solve(A, rhs)
        j = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, n - 2)
        x0, x1, hj = xs[j], xs[j + 1], hs[j]
        a, b = (x1 - grid) / hj, (grid - x0) / hj
        out = a * ys[j] + b * ys[j + 1] + ((a ** 3 - a) * m[j] + (b ** 3 - b) * m[j + 1]) * hj * hj / 6
        out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return np.clip(out, 0, 255) / 255.0


def lut(plane: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``plane`` (0..1, clipped) through ``table``, a curve sampled evenly over 0..1,
    at 16 bits: a lookup, where np.interp over a 20-megapixel plane takes a second."""
    if len(table) != 65536:
        table = np.interp(np.linspace(0, 1, 65536), np.linspace(0, 1, len(table)), table)
    table = np.asarray(table, np.float32)
    index = np.empty(plane.shape, np.uint16)
    step = max(1, (1 << 22) // max(plane.shape[-1], 1))
    for top in range(0, plane.shape[0], step):
        part = np.clip(plane[top:top + step], 0.0, 1.0) * np.float32(65535) + np.float32(0.5)
        index[top:top + step] = part.astype(np.uint16)
    return table[index]


def _levels(data: bytes) -> Optional[list]:
    """Per plane (composite, red, green, blue) a function, or ``None`` for identity."""
    if len(data) < 2 + 4 * 10:
        return None
    funcs = []
    for i in range(4):
        lo, hi, out_lo, out_hi, gamma = struct.unpack(">5H", data[2 + 10 * i:12 + 10 * i])
        if (lo, hi, out_lo, out_hi, gamma) == (0, 255, 0, 255, 100):
            funcs.append(None)
            continue
        g = max(gamma, 1) / 100.0
        x = np.linspace(0.0, 1.0, 65536)
        v = np.clip((x * 255.0 - lo) / max(hi - lo, 1), 0, 1) ** (1.0 / g)
        table = (out_lo + v * (out_hi - out_lo)) / 255.0
        funcs.append(lambda plane, table=table: lut(plane, table))
    return funcs


def _curves(data: bytes) -> Optional[list]:
    if len(data) < 7:
        return None
    (bits,) = struct.unpack(">I", data[3:7])
    at, funcs = 7, [None] * 4
    for i in range(32):
        if not bits & (1 << i):
            continue
        (n,) = struct.unpack(">H", data[at:at + 2])
        pts = [struct.unpack(">2H", data[at + 2 + 4 * k:at + 6 + 4 * k]) for k in range(n)]
        at += 2 + 4 * n
        if i < 4:
            table = _spline([(inp, out) for out, inp in pts])
            funcs[i] = lambda x, table=table: lut(x, table)
    return funcs


def _per_channel(planes: list, funcs: list) -> list:
    """Each colour plane through its own function, then all through the composite
    one (Photoshop's order)."""
    out = list(planes)
    if len(out) == 3:
        for i in range(3):
            if funcs[i + 1]:
                out[i] = funcs[i + 1](out[i])
    if funcs[0]:
        out = [funcs[0](p) for p in out]
    return out


def _hue_saturation(planes: list, data: bytes, note) -> list:
    if len(data) < 16 or len(planes) != 3:
        return planes
    colorize = data[2]
    hue, sat, light = struct.unpack(">3h", data[10:16])
    if colorize:
        note("Hue/Saturation's Colorize")
        return planes
    if len(data) >= 100 and any(struct.unpack(">3h", data[16 + 14 * k + 8:16 + 14 * k + 14]) != (0, 0, 0)
                                for k in range(6)):
        note("Hue/Saturation's colour ranges")
    r, g, b = planes
    if hue:
        r, g, b = _rotate_hue(r, g, b, hue / 360.0)
    if sat:
        hi = np.maximum(np.maximum(r, g), b)
        lo = np.minimum(np.minimum(r, g), b)
        mid = (hi + lo) / 2
        s = sat / 100.0
        k = (1.0 / max(1 - s, 1e-3) - 1) if s > 0 else s
        r, g, b = (np.clip(c + (c - mid) * k, 0, 1) for c in (r, g, b))
    if light:
        v = light / 100.0
        r, g, b = ((c + (1 - c) * v) if v > 0 else c * (1 + v) for c in (r, g, b))
    return [np.asarray(c, np.float32) for c in (r, g, b)]


def _rotate_hue(r, g, b, turn):
    hi = np.maximum(np.maximum(r, g), b)
    lo = np.minimum(np.minimum(r, g), b)
    c = hi - lo
    safe = np.maximum(c, 1e-9)
    h = np.where(hi == r, ((g - b) / safe) % 6, np.where(hi == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = np.where(c > 0, (h / 6 + turn) % 1.0, 0)
    hp = h * 6
    x = c * (1 - np.abs(hp % 2 - 1))
    z = np.zeros_like(c)
    sector = np.floor(hp).astype(np.int8) % 6
    rr = np.choose(sector, [c, x, z, z, x, c])
    gg = np.choose(sector, [x, c, c, x, z, z])
    bb = np.choose(sector, [z, z, x, c, c, x])
    return rr + lo, gg + lo, bb + lo


# ── Compositing ──────────────────────────────────────────────────────

class _Group:
    __slots__ = ("layer", "children")

    def __init__(self, layer, children):
        self.layer, self.children = layer, children


def _tree(layers: list) -> list:
    """Bottom-first records to nested lists: a group's divider opens it, its own
    record (with its blend, opacity and mask) closes it."""
    stack = [[]]
    for L in layers:
        if L.section == 3:
            stack.append([])
        elif L.section in (1, 2):
            children = stack.pop() if len(stack) > 1 else []
            stack[-1].append(_Group(L, children))
        else:
            stack[-1].append(L)
    while len(stack) > 1:
        children = stack.pop()
        stack[-1].extend(children)
    return stack[0]


def _kind(L: _Layer) -> str:
    keys = set(L.keys)
    if keys & _ADJUSTMENTS:
        return "adjustment"
    if keys & _UNSUPPORTED:
        return "unsupported"
    t, l, b, r = L.rect
    if b > t and r > l and any(ch[0] >= 0 for ch in L.channels):
        return "pixels"
    return "empty"


def _composite(R: _Reader, nodes: list, canvas: _Canvas) -> None:
    base = None   # the coverage clipped layers are clipped to
    for i, node in enumerate(nodes):
        L = node.layer if isinstance(node, _Group) else node
        next_clipped = i + 1 < len(nodes) and not isinstance(nodes[i + 1], _Group) and nodes[i + 1].clipped
        if not L.visible:
            if not L.clipped:
                base = np.zeros((canvas.h, canvas.w), np.float32) if next_clipped else None
            continue
        weight = (L.opacity / 255.0) * L.fill
        mask = R.mask(canvas, L)
        if L.clipped and base is not None:
            mask = base if mask is None else mask * base

        if isinstance(node, _Group):
            _group(R, node, canvas, weight, mask)
            base = None
            continue
        kind = _kind(L)
        if kind == "adjustment":
            _adjust(R, L, canvas, weight, mask)
        elif kind == "pixels":
            cover = _pixels(R, L, canvas, weight, mask)
            if not L.clipped:
                base = cover if next_clipped else None
        elif kind == "unsupported":
            R.skipped.append(L.name)


def _blend_of(R: _Reader, L: _Layer):
    blend = _BLENDS.get(L.blend)
    if blend is None:
        R.skipped.append(f"{L.name} ({L.blend.decode('latin-1').strip()} blending, drawn as Normal)")
        blend = _BLENDS[b"norm"]
    return blend


def _pixels(R: _Reader, L: _Layer, canvas: _Canvas, weight: float, mask) -> Optional[np.ndarray]:
    box = canvas.box(L.rect)
    t, l, b, r = box
    if b <= t or r <= l:
        return None
    alpha = R.channel_at(canvas, L, -1, L.rect)
    cover = np.full((b - t, r - l), weight, np.float32) if alpha is None else alpha * np.float32(weight)
    if mask is not None:
        cover = cover * _crop(mask, box, canvas)
    colours = []
    for c in range(R.colours):
        plane = R.channel_at(canvas, L, c, L.rect)
        if plane is None:
            return None
        colours.append(plane)
    _over(canvas, colours, cover, box, _blend_of(R, L))
    return _place(canvas, alpha if alpha is not None else np.ones((b - t, r - l), np.float32), box, 0.0)


def _crop(plane: np.ndarray, box: tuple, canvas: _Canvas) -> np.ndarray:
    """A canvas-size plane cut to ``box`` (outside the canvas: 0)."""
    t, l, b, r = box
    out = np.zeros((b - t, r - l), np.float32)
    tt, ll, bb, rr = max(t, 0), max(l, 0), min(b, canvas.h), min(r, canvas.w)
    if bb > tt and rr > ll:
        out[tt - t:bb - t, ll - l:rr - l] = plane[tt:bb, ll:rr]
    return out


def _adjust(R: _Reader, L: _Layer, canvas: _Canvas, weight: float, mask) -> None:
    planes = canvas.c
    if b"levl" in L.keys:
        funcs = _levels(L.keys[b"levl"])
        new = _per_channel(planes, funcs) if funcs else planes
    elif b"curv" in L.keys:
        funcs = _curves(L.keys[b"curv"])
        new = _per_channel(planes, funcs) if funcs else planes
    elif b"nvrt" in L.keys:
        new = [1.0 - p for p in planes]
    else:
        new = _hue_saturation(planes, L.keys[b"hue2"],
                              lambda what: R.skipped.append(f"{L.name} ({what})"))
    k = None if mask is None else mask * np.float32(weight)
    for i, (old, adjusted) in enumerate(zip(planes, new)):
        if adjusted is old:
            continue
        if k is None and weight >= 1.0:
            canvas.c[i] = np.asarray(adjusted, np.float32)
        else:
            w = k if k is not None else np.float32(weight)
            canvas.c[i] = (old + (adjusted - old) * w).astype(np.float32)


def _group(R: _Reader, node: _Group, canvas: _Canvas, weight: float, mask) -> None:
    L = node.layer
    if L.blend == b"pass":
        if weight >= 1.0 and mask is None:
            _composite(R, node.children, canvas)
            return
        before = [p.copy() for p in canvas.c] + [canvas.a.copy()]
        _composite(R, node.children, canvas)
        k = np.float32(weight) if mask is None else mask * np.float32(weight)
        for i, old in enumerate(before[:-1]):
            canvas.c[i] = (old + (canvas.c[i] - old) * k).astype(np.float32)
        canvas.a = (before[-1] + (canvas.a - before[-1]) * k).astype(np.float32)
        return
    inner = R.new_canvas()
    _composite(R, node.children, inner)
    cover = inner.a * np.float32(weight)
    if mask is not None:
        cover = cover * mask
    _over(canvas, inner.c, cover, (0, 0, canvas.h, canvas.w), _blend_of(R, L))


def composite(path: Path, info: dict, target: Optional[tuple]) -> Optional[list]:
    """The picture as the layers make it: a list of float32 planes (one grey or three
    colour) on 0..1 in the file's encoded values, flattened on white; at ``target``
    (w, h) when given. ``None`` when the file has no layers this can read."""
    layers = read_records(path, info)
    if not layers:
        return None
    size = target or (info["width"], info["height"])
    with open(path, "rb") as f:
        R = _Reader(f, info, size, path.name)
        canvas = R.new_canvas()
        _composite(R, _tree(layers), canvas)
    if R.skipped:
        logger.info(f"{path.name}: rebuilt from its layers without " + ", ".join(R.skipped))
    a = canvas.a
    return [(p * a + (1.0 - a)).astype(np.float32) for p in canvas.c]
