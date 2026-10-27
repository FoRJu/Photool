"""
Measurements that make one photo comparable to its neighbours.

None of this is a judgement about a photograph. It is the arithmetic that lets
the app group near-identical frames, warn that one frame is softer than the
ones around it, and bring a set of frames to a common exposure and colour
before writing them out. Every number here is derived from the file and can be
recomputed, which is why it is stored as a cache and stamped with
``STATS_VERSION``: when a measurement changes shape, old rows are recomputed
instead of being compared against numbers produced by different code.

What each measurement is for:

  luminance percentiles   exposure matching, and the blown-highlight hint
  grey-world channel mean the colour the camera was treating as white, which is
                          what matching needs in order to make a set agree
  dHash                   "is this frame the same frame as that one"
  Laplacian variance      soft-frame hint, relative to its neighbours

All of them come from a small image on purpose. A 400px grid thumbnail is plenty
for a 64-bit perceptual hash and for exposure and colour statistics, and
sharpness only ever has to be *relative* to the other frames in the same folder,
which were measured the same way.
"""

import io
from typing import Optional

import numpy as np
from PIL import Image

# Bump when a measurement changes shape, so stored rows are recomputed rather
# than compared against numbers produced by different code.
STATS_VERSION = 1

# dHash compares adjacent columns of a 9x8 grayscale strip: 64 bits, one per
# comparison, which is what a near-identical frame preserves.
_DHASH_COLS = 9
_DHASH_ROWS = 8

# Rec. 709 luma coefficients, applied to 0-255 RGB.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# A pixel at or above this is blown; at or below, crushed.
_CLIP_HIGH = 250.0
_CLIP_LOW = 5.0

# Returned by hamming_distance for a hash it cannot read, so an unknown value
# never looks like a match.
_MAX_DISTANCE = 64


def compute_image_stats(image: Image.Image) -> dict:
    """Measure exposure, colour, sharpness, and a perceptual hash.

    Any mode is accepted; the image is converted to RGB first. The returned
    keys match the ``photo_stats`` columns, so a caller can hand this straight
    to :meth:`CatalogDB.upsert_photo_stats`.
    """
    rgb = image.convert("RGB")
    if rgb.width < 2 or rgb.height < 2:
        raise ValueError(f"image is too small to measure ({rgb.width}x{rgb.height})")

    arr = np.asarray(rgb, dtype=np.float32)
    lum = arr @ _LUMA_WEIGHTS
    flat = lum.reshape(-1)

    p05, p50, p95 = (float(v) for v in np.percentile(flat, [5.0, 50.0, 95.0]))

    # Grey-world: the scene's own average colour is the white the camera saw.
    # Stored as ratios against the green channel, which describes the cast
    # without pre-deciding what correction would remove it.
    channels = arr.reshape(-1, 3).mean(axis=0)
    neutral = float(channels[1]) or 1.0

    return {
        # The numbers carry the version of the code that produced them, so a
        # stored row can never be compared against one measured differently.
        "stats_version": STATS_VERSION,
        "width": int(rgb.width),
        "height": int(rgb.height),
        "mean_lum": float(flat.mean()),
        "lum_p05": p05,
        "lum_p50": p50,
        "lum_p95": p95,
        "clip_high": float((flat >= _CLIP_HIGH).mean()),
        "clip_low": float((flat <= _CLIP_LOW).mean()),
        "wb_r": round(float(channels[0]) / neutral, 6),
        "wb_g": 1.0,
        "wb_b": round(float(channels[2]) / neutral, 6),
        "sharpness": _laplacian_variance(lum),
        "dhash": _dhash(rgb.convert("L")),
    }


def stats_from_jpeg(jpeg: bytes) -> dict:
    """Decode JPEG bytes and measure them."""
    with Image.open(io.BytesIO(jpeg)) as img:
        return compute_image_stats(img)


def _laplacian_variance(lum: np.ndarray) -> float:
    """Variance of the discrete Laplacian — the usual focus measure.

    More edge energy means a sharper frame, so a frame that is soft *compared
    with its neighbours* scores lower. The absolute number says nothing across
    subjects — a landscape and a portrait of a face are not comparable — but
    frames inside one burst are, and that is the comparison this is used for.
    """
    if lum.shape[0] < 3 or lum.shape[1] < 3:
        return 0.0
    kernel = (
        lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
        - 4.0 * lum[1:-1, 1:-1]
    )
    return float(kernel.var())


def _dhash(gray: Image.Image) -> str:
    """64-bit difference hash as 16 hex digits.

    A difference hash encodes *which side of each pair is brighter* rather than
    brightness itself, so it survives the exposure and colour differences that
    separate frames of the same moment — which is exactly what grouping needs.
    """
    strip = gray.resize((_DHASH_COLS, _DHASH_ROWS), Image.LANCZOS)
    row = np.asarray(strip, dtype=np.float32)
    bits = (row[:, 1:] > row[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """How many bits differ between two dHashes.

    0 is the same framing; single digits are the same frame with a small change
    (a breath, a blink); anything around 10 or more is a different picture. An
    unreadable hash returns the maximum, so a missing value never groups with
    anything.
    """
    try:
        a = int(hash_a or "", 16)
        b = int(hash_b or "", 16)
    except ValueError:
        return _MAX_DISTANCE
    return bin(a ^ b).count("1")


def stats_are_current(row: Optional[dict], source_mtime: int, source_size: int) -> bool:
    """Whether a stored row still describes the file it was measured from."""
    if not row:
        return False
    if int(row.get("stats_version") or 0) != STATS_VERSION:
        return False
    return (
        int(row.get("source_mtime") or 0) == int(source_mtime)
        and int(row.get("source_size") or 0) == int(source_size)
    )
