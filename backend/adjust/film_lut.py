"""A film look from a .cube LUT, rendered on any RAW like a learned style.

Film-simulation LUTs such as abpy's (github.com/abpy/FujifilmCameraProfiles) are
made for a picture with *linear contrast*: the camera's colour matrix and white
balance as shot and no tone curve, sRGB- (or Display P3-) encoded, 0..1. All of
the film's character, tone curve included, is in the LUT.

That is nearly the decode every RAW is brought to before a look sees it
(adjust.styles: scene-linear light in Rec. 709/sRGB primaries, the camera's own
matrix and white balance). So the LUT is put on our decode as it is, through
the same two steps a learned style takes:

* **Anchored.** Cameras disagree about where middle grey lands in their raw
  numbers (a Canon R6 II's decode sat 2 to 2.5 stops under where these LUTs want
  it). One exposure factor puts the frame's middle brightness, through the LUT,
  where its own camera put it (adjust.styles.anchor).
* **A shoulder before the LUT.** A LUT's input stops at white and the decode has
  headroom above it; the highlight shoulder the app already uses
  (adjust.render._shoulder) folds that headroom in rather than clipping it.

What does not carry over exactly: Adobe's linear profile keeps Adobe Standard's
own hue and saturation twists, and this decode has the camera's matrix alone, so
strongly saturated colours may sit a little apart from the same LUT in Camera Raw.
"""

from typing import Optional

import numpy as np

SPACES = ("srgb", "p3")
MIN_SIZE, MAX_SIZE = 2, 129
MAX_BYTES = 12 * 1024 * 1024

# Linear Rec. 709 / sRGB to linear Display P3 (both D65), and back.
_TO_P3 = np.array([
    [0.8224621, 0.1775380, 0.0000000],
    [0.0331941, 0.9668058, 0.0000000],
    [0.0170827, 0.0723974, 0.9105199],
], dtype=np.float32)
_FROM_P3 = np.linalg.inv(_TO_P3).astype(np.float32)


class FilmLut:
    """A 3D LUT, ``table[b, g, r]`` (the .cube order, red fastest), on the given
    encoding: sRGB transfer, in sRGB or Display P3 primaries."""

    __slots__ = ("table", "space")

    def __init__(self, table: np.ndarray, space: str = "srgb"):
        self.table = np.ascontiguousarray(table, dtype=np.float32)
        self.space = space if space in SPACES else "srgb"

    @property
    def size(self) -> int:
        return int(self.table.shape[0])


def parse_cube(text: str) -> tuple[np.ndarray, str]:
    """The table of a .cube file, and its title. ``ValueError`` with a sentence
    when it is not a 3D LUT this can use."""
    size, title, rows = 0, "", []
    lo, hi = np.zeros(3, np.float32), np.ones(3, np.float32)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        word = line.split(None, 1)[0].upper()
        if word == "TITLE":
            title = line[5:].strip().strip('"')
        elif word == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif word == "LUT_1D_SIZE":
            raise ValueError("This is a 1D LUT (a curve per channel); a film look needs a 3D LUT.")
        elif word == "DOMAIN_MIN":
            lo = np.array([float(v) for v in line.split()[1:4]], np.float32)
        elif word == "DOMAIN_MAX":
            hi = np.array([float(v) for v in line.split()[1:4]], np.float32)
        elif word[0].isdigit() or word[0] in "-.":
            parts = line.split()
            if len(parts) != 3:
                raise ValueError("A line of colours does not have three numbers.")
            rows.append([float(v) for v in parts])
        # Other keywords (LUT_3D_INPUT_RANGE and the like) are ignored.
    if not (MIN_SIZE <= size <= MAX_SIZE):
        raise ValueError("This file does not say its LUT size (LUT_3D_SIZE).")
    if len(rows) != size ** 3:
        raise ValueError(f"A {size}-point LUT has {size ** 3} colours; this file has {len(rows)}.")
    if np.any(hi <= lo):
        raise ValueError("The LUT's domain is empty.")
    if np.any(lo != 0) or np.any(hi != 1):
        raise ValueError("Only LUTs on a 0..1 domain are supported.")
    table = np.asarray(rows, np.float32).reshape(size, size, size, 3)
    if not np.all(np.isfinite(table)):
        raise ValueError("The LUT holds numbers that are not finite.")
    return table, title


def guess_space(text: str, filename: str = "") -> str:
    """Display P3 when the file or its comments say so, else sRGB."""
    head = (filename + " " + "\n".join(text.splitlines()[:12])).lower().replace(" ", "")
    return "p3" if ("displayp3" in head or "_p3" in head or "p3." in head) else "srgb"


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    v = np.clip(linear, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055).astype(np.float32)


def _srgb_decode(encoded: np.ndarray) -> np.ndarray:
    v = np.clip(encoded, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4)).astype(np.float32)


def look_up(table: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Trilinear lookup of colours 0..1 (..., 3) in ``table[b, g, r]``."""
    n = table.shape[0]
    shape = colours.shape
    p = np.clip(colours.reshape(-1, 3), 0.0, 1.0) * np.float32(n - 1)
    i = np.minimum(p.astype(np.int32), n - 2)
    f = p - i
    r0, g0, b0 = i[:, 0], i[:, 1], i[:, 2]
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    out = np.zeros_like(p)
    for db, wb in ((0, 1.0 - fb), (1, fb)):
        for dg, wg in ((0, 1.0 - fg), (1, fg)):
            w = wb * wg
            for dr, wr in ((0, 1.0 - fr), (1, fr)):
                out += (w * wr) * table[b0 + db, g0 + dg, r0 + dr]
    return out.reshape(shape)


def apply_lut(rgb: np.ndarray, lut: FilmLut) -> np.ndarray:
    """Scene-linear Rec. 709 light (headroom already folded in) to the app's
    display encoding, through the LUT."""
    from adjust.render import encode

    linear = np.asarray(rgb, np.float32)
    if lut.space == "p3":
        linear = np.clip(linear @ _TO_P3.T, 0.0, None)
    shown = _srgb_decode(look_up(lut.table, _srgb_encode(linear)))
    if lut.space == "p3":
        shown = np.clip(shown @ _FROM_P3.T, 0.0, 1.0)
    # The LUT gives sRGB-encoded colour; the rest of the pipeline works on the
    # app's own display curve (adjust.render.encode).
    return encode(shown)


def render(rgb: np.ndarray, lut: FilmLut, factor: float) -> np.ndarray:
    """A frame's linear light through the LUT at its anchor ``factor``."""
    from adjust.render import _shoulder

    return apply_lut(_shoulder(np.asarray(rgb, np.float32) * np.float32(factor)), lut)


def from_text(text: str, filename: str = "", space: Optional[str] = None) -> tuple[FilmLut, str]:
    table, title = parse_cube(text)
    return FilmLut(table, space or guess_space(text, filename)), title
