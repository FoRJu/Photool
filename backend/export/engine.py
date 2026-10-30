"""
Export engine — loads a source photo (RAW or standard), applies its edit,
resizes, strips/embeds EXIF, and writes the final JPEG.
"""

import io
import logging
import math
import shutil
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# RAW extensions handled via rawpy
_RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2",
    ".dng", ".raf", ".pef", ".srw", ".3fr", ".iiq",
}

# The same extensions in the order to look for a RAW sibling, most likely first.
# A set has no stable order, and which sibling gets used should not depend on
# how the strings happened to hash.
_RAW_PREFERRED_ORDER = (
    ".cr3", ".cr2", ".nef", ".arw", ".orf", ".rw2",
    ".dng", ".raf", ".pef", ".srw", ".3fr", ".iiq",
)


def resolve_export_source(source: Path, prefer_raw: bool = True) -> Path:
    """Return the file to develop for a basked photo.

    A CR3+JPG pair is one frame with two renderings of it. Exporting whichever
    half happened to be basked means every frame keeps the camera's own auto
    exposure and white balance, which is exactly the shot-to-shot drift a school
    event is full of. Developing the RAW instead gives every frame the same kind
    of starting data, which is what makes matching them possible at all.

    Falls back to the basked file when there is no RAW sibling, or when the
    basked file is itself the RAW.
    """
    from config import is_video

    if not prefer_raw or is_video(source.suffix):
        return source
    if source.suffix.lower() in _RAW_EXTENSIONS:
        return source
    # Listed once rather than probed one extension at a time, so the sibling is
    # found under the name it really has on disk instead of a guessed case.
    try:
        siblings = {p.name.lower(): p for p in source.parent.iterdir()}
    except OSError:
        return source
    wanted = source.stem.lower()
    for ext in _RAW_PREFERRED_ORDER:
        candidate = siblings.get(wanted + ext)
        if candidate is not None:
            return candidate
    return source


def _embedded_camera_image(source: Path) -> Optional[Image.Image]:
    """The JPEG the camera embedded in a RAW file, upright, as a PIL image; or
    ``None``. The orientation lives in the RAW container rather than in the JPEG,
    so it is read from the file and applied here."""
    try:
        from thumbnails.extract import (
            _get_raw_orientation,
            _normalize_image,
            extract_embedded_jpeg,
        )

        data = extract_embedded_jpeg(source)
        if not data:
            return None
        img, _ = _normalize_image(data, _get_raw_orientation(source))
        return img.convert("RGB")
    except Exception as e:
        logger.debug(f"No embedded JPEG for {source.name}: {e}")
        return None


def camera_image(source: Path, max_resolution: int = 0) -> Optional[Image.Image]:
    """The camera's own rendering of a frame, resized, or ``None``.

    A JPEG is that rendering already. A RAW carries two: the sensor data and the
    JPEG the camera made from it — the picture the back of the camera showed, with
    its white balance, its Picture Style tone curve and saturation, its highlight
    and shadow handling, its lens correction and its noise reduction already in
    it. An export *developed* from the RAW can only ever approach that picture,
    however well the camera's look is learned (measured on one R6 II frame: a mean
    of 8.5 levels of 255 away, and the plain RAW decode 48 levels away and a stop
    dark). The JPEG is the picture, so it is what "no editing at all" has to
    resize.

    The full-size JPEG beside a RAW is taken first — the camera's whole rendering —
    then the one embedded in the file. A RAW with neither falls back to the plain
    decode, which is still the frame, resized. The resize is done here rather than
    by the caller so the export and the export preview show the same picture.
    """
    img = None
    if source.suffix.lower() in _RAW_EXTENSIONS:
        sibling = _sibling_jpeg_path(source)
        if sibling is not None:
            img = _load_image(sibling)
        if img is None:
            img = _embedded_camera_image(source)
        if img is None:
            logger.info(f"No camera JPEG for {source.name}: exporting the plain decode")
            img = _load_image(source)
    else:
        img = _load_image(source)
    if img is None:
        return None
    return _resize_image(img, max_resolution)


@dataclass
class ExportTemplate:
    """Export settings for one batch."""
    max_resolution: int = 2048
    quality: int = 85
    strip_exif: bool = True
    photographer: str = ""


@dataclass
class MatchSettings:
    """Match a set while exporting it: one target for the batch, applied per frame.

    ``applied`` is filled in by :func:`export_photo` as the batch runs, so the
    caller reports how many frames were actually changed rather than how many
    were offered the chance.
    """
    target: dict
    strength: float = 1.0
    applied: int = 0


def _load_image(source: Path, max_resolution: int = 0) -> Optional[Image.Image]:
    """Load an image from any supported format, returning an RGB PIL Image."""
    ext = source.suffix.lower()
    if ext in _RAW_EXTENSIONS:
        return _load_raw(source, max_resolution)
    try:
        img = Image.open(source)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Failed to load {source.name}: {e}")
        return None


def _load_raw(source: Path, max_resolution: int = 0) -> Optional[Image.Image]:
    """Decode a RAW file using rawpy, returning an RGB PIL Image.

    Uses half_size demosaic when the target resolution is much smaller
    than the sensor resolution — cuts decode time roughly in half.
    """
    try:
        import rawpy
        with rawpy.imread(str(source)) as raw:
            # If output is ≤ half the sensor size, use fast half-size decode
            use_half = False
            if max_resolution > 0:
                sensor_max = max(raw.sizes.width, raw.sizes.height)
                if max_resolution <= sensor_max // 2:
                    use_half = True
            rgb = raw.postprocess(
                use_camera_wb=True,
                # Auto-bright normalises every frame against its own histogram,
                # so two frames of the same moment come out different brightness.
                # Measured over eight frames of one real event it widened the
                # spread from 0.8 EV to 1.4 EV and lifted frames by an
                # unpredictable 0 to 0.9 EV. A fixed baseline is the only thing
                # matching can work from.
                no_auto_bright=True,
                half_size=use_half,
            )
        return Image.fromarray(rgb)
    except Exception as e:
        logger.error(f"rawpy decode failed for {source.name}: {e}")
        return None


def _resize_image(img: Image.Image, max_res: int) -> Image.Image:
    """Downscale preserving aspect ratio. No-op if already small enough."""
    if max_res <= 0:
        return img
    w, h = img.size
    if w <= max_res and h <= max_res:
        return img
    ratio = min(max_res / w, max_res / h)
    # Rounded, as adjust.render.resize_linear is: truncated, 6000 px at 2048 is 2047.
    new_w = max(1, min(max_res, round(w * ratio)))
    new_h = max(1, min(max_res, round(h * ratio)))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── Preview decoding ───────────────────────────────────────────────
#
# Adjusting a photo by hand means looking at it while the number changes. The
# only slow part of rendering a small preview is the decode — a RAW file has to
# be demosaiced, and a JPEG has to be read off a slow volume — while the
# adjustment itself is one pass through a lookup table. So the decoded, resized
# frame is kept in memory for a short while and every slider movement is applied
# to that copy instead of starting the decode again.
#
# Small on purpose: a handful of preview-sized frames, dropped when the file
# changes underneath them. The image is handed out as read-only — every step
# after this point returns a new image rather than writing into this one.
_PREVIEW_CACHE: "OrderedDict[tuple, Image.Image]" = OrderedDict()
_PREVIEW_CACHE_LIMIT = 4
_PREVIEW_LOCK = threading.Lock()


def preview_base(source: Path, size: int) -> Optional[Image.Image]:
    """Decode and resize a frame once, so it can be adjusted repeatedly."""
    try:
        stat = source.stat()
    except OSError:
        return None
    # The mtime and size are part of the key so an edit to the file on disk —
    # a sidecar written by another program, a re-import — is never served from
    # a stale frame.
    key = (str(source), int(size), stat.st_mtime_ns, stat.st_size)

    with _PREVIEW_LOCK:
        cached = _PREVIEW_CACHE.get(key)
        if cached is not None:
            _PREVIEW_CACHE.move_to_end(key)
            return cached

    img = _load_image(source, size)
    if img is None:
        return None
    img = _resize_image(img, size)

    with _PREVIEW_LOCK:
        _PREVIEW_CACHE[key] = img
        while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_LIMIT:
            _PREVIEW_CACHE.popitem(last=False)
    return img


def clear_preview_cache() -> None:
    """Drop every held preview frame."""
    with _PREVIEW_LOCK:
        _PREVIEW_CACHE.clear()
        _LINEAR_CACHE.clear()
        _LOOK_CACHE.clear()


# ── Linear-light decoding ──────────────────────────────────────────
#
# An edit that rescues exposure has to start from more than 8 bits of display
# data. A RAW file is decoded here to 16-bit linear with the rendering left
# flat — no auto-brightness, no camera tone curve — and with the headroom white
# balance normally throws away, so a channel that the camera's white balance
# pushed past white survives to be pulled back (a sunset's reds, a warm window).
# A JPEG has nothing above 255, but it goes through the same pipeline so one
# edit means one thing on both halves of a RAW+JPG pair.
#
# Measured against LibRaw on a synthetic DNG rather than assumed:
#
#  * Left to itself it rescales each frame so its brightest pixel reaches full
#    scale (``adjust_maximum_thr``), which would make the same exposure setting
#    mean something different on every frame. That is turned off.
#  * ``bright`` does not give headroom: white balance has already clipped by the
#    time it is applied.
#  * ``HighlightMode.Ignore`` does keep the boosted channels, by normalising the
#    white-balance gains to their maximum instead of their minimum. Everything is
#    then darker by max(gain)/min(gain), which is undone here from the camera's
#    own gains.
#  * Ignoring the clip means a pixel the *sensor* clipped comes out with its
#    boosted channels above its clipped one — magenta, for a daylight balance, and
#    pink on a flash-blown face or a sheet of music under a stage light.
#    adjust.clip finds where each channel really saturates (not where the file
#    says it should) and rebuilds the clipped channels from the ones that kept
#    their data and from the bright surface around them.
#  * That repair has to happen in the **camera's own primaries**, before the
#    colour matrix: the matrix mixes a clipped channel into the other two, after
#    which nothing is cleanly clipped and nothing can be repaired on its own. So
#    the frame is decoded with LibRaw's output in camera space, repaired, and only
#    then taken to linear sRGB with the same matrix LibRaw would have used
#    (computed here as LibRaw computes it; the two agree to 0.05%).
#  * Last, the colour (not the brightness) is smoothed over a few pixels, as every
#    camera's JPEG engine does. At high ISO this removes the chroma speckle the
#    camera's own picture never shows, and the thin coloured lines lateral
#    chromatic aberration draws along a bright edge; measured against the
#    camera's JPEGs it is the largest single step towards them at 100%.

# sRGB primaries to XYZ (D65), as LibRaw uses them to build its camera matrix.
_SRGB_TO_XYZ = (
    (0.412453, 0.357580, 0.180423),
    (0.212671, 0.715160, 0.072169),
    (0.019334, 0.119193, 0.950227),
)

# How far the colour is smoothed, in full-resolution pixels (a Gaussian sigma).
CHROMA_SMOOTHING = 3.0


def _srgb_from_camera(raw):
    """LibRaw's camera-to-linear-sRGB matrix for a three-colour sensor, or
    ``None`` when the file does not carry a usable one."""
    import numpy as np

    try:
        if int(raw.num_colors) != 3:
            return None
        cam_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3]
        cam_rgb = cam_xyz @ np.asarray(_SRGB_TO_XYZ)
        sums = cam_rgb.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(cam_rgb)) or np.any(np.abs(sums) < 1e-6):
            return None
        cam_rgb = cam_rgb / sums
        if abs(np.linalg.det(cam_rgb)) < 1e-6:
            return None
        return np.linalg.inv(cam_rgb).astype(np.float32)
    except Exception:
        return None


def has_opencv() -> bool:
    """Whether OpenCV is installed. It is only a faster blur: without it the same
    Gaussian is computed in NumPy (exactly at the half-size scale; within about
    0.0004 on a real frame's colour plane at full size)."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def _gaussian_kernel(sigma: float):
    import numpy as np

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32), radius


def _convolve_axis(a, kernel, radius: int, axis: int):
    import numpy as np

    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i, weight in enumerate(kernel):
        index = [slice(None)] * a.ndim
        index[axis] = slice(i, i + n)
        out += weight * padded[tuple(index)]
    return out


def _blur_plane(plane, sigma: float):
    """A Gaussian blur of one float plane: OpenCV's when it is installed, else the
    same separable Gaussian in NumPy. A wide blur is made on a block-reduced copy
    (so the kernel stays short) and enlarged again. The field is smooth: on a real
    frame the result is within 0.0004 of OpenCV's, and 0.002 on a plane busier than
    any frame's colour."""
    import numpy as np

    try:
        import cv2

        return cv2.GaussianBlur(np.ascontiguousarray(plane, dtype=np.float32), (0, 0), float(sigma))
    except Exception:
        pass
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    h, w = plane.shape
    factor = max(1, int(sigma // 1.5)) if sigma >= 3.0 else 1
    small = plane
    if factor > 1:
        hh, ww = (h // factor) * factor, (w // factor) * factor
        small = plane[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor).mean(axis=(1, 3))
    kernel, radius = _gaussian_kernel(sigma / factor)
    blurred = _convolve_axis(_convolve_axis(small, kernel, radius, 0), kernel, radius, 1)
    if factor > 1:
        blurred = np.asarray(
            Image.fromarray(np.ascontiguousarray(blurred), "F").resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )
    return blurred


def smooth_chroma(rgb, sigma: float):
    """Smooth a frame's colour over ``sigma`` pixels and leave its brightness.

    Weighted by brightness, so the colour of a dark, noisy pixel does not bleed
    into a bright one next to it.
    """
    import numpy as np

    if sigma < 0.5:
        return rgb
    from adjust.render import LUMA, in_strips, side_by_side

    luma = np.maximum(rgb @ LUMA, 1e-6)
    weight = in_strips(np.sqrt, luma)
    colour = in_strips(lambda px, y, w: px / y[..., None] * w[..., None], rgb, luma, weight)
    mass, *blurred = side_by_side(lambda plane: _blur_plane(plane, sigma),
                                  [weight, colour[..., 0], colour[..., 1], colour[..., 2]])
    mass = np.maximum(mass, 1e-6)
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = blurred[c] / mass * luma
    return out


def _decode_with_headroom(raw, common, gains, matrix):
    """The decode that keeps what white balance pushed past white, repaired where
    the sensor clipped, colour-smoothed, as linear sRGB.

    With the camera's matrix known the work is done in camera space and the matrix
    applied last. Without it (LibRaw does not expose one for every DNG) LibRaw
    converts to sRGB itself, and clipped pixels are rebuilt whole.
    """
    import numpy as np
    import rawpy

    from adjust.clip import reconstruct_highlights

    in_camera_space = matrix is not None
    options = dict(adjust_maximum_thr=0.0, highlight_mode=rawpy.HighlightMode.Ignore, **common)
    if in_camera_space:
        options["output_color"] = rawpy.ColorSpace.raw
    rgb16 = raw.postprocess(**options)
    rgb = rgb16.astype(np.float32) * (float(gains.max() / gains.min()) / 65535.0)
    # Each channel clips at its own multiple of the weakest one's gain: the level
    # a sensor that saturates at the file's white level would reach. The data
    # decides where it really does.
    rgb = reconstruct_highlights(rgb, gains / gains.min(), joint=not in_camera_space)
    sigma = CHROMA_SMOOTHING / (2.0 if common.get("half_size") else 1.0)
    rgb = smooth_chroma(rgb, sigma)
    if in_camera_space:
        rgb = np.clip(rgb @ matrix.T, 0.0, None)
    return rgb.astype(np.float32)


def _load_raw_linear(source: Path, max_resolution: int = 0, mosaic=None):
    """Decode a RAW file to scene-linear float32, or ``None`` on failure.

    Falls back to a plain clipped decode rather than failing: an older rawpy, a
    file with no usable white-balance gains or colour matrix, or one that rejects
    an option should still open, just without the headroom. ``mosaic`` replaces
    the sensor's data before the decode (the file's noise-reduced mosaic, see
    adjust.noise), so it is decoded exactly as the file's own would be.
    """
    import numpy as np

    try:
        import rawpy
    except ImportError:
        return None

    try:
        with rawpy.imread(str(source)) as raw:
            if mosaic is not None:
                raw.raw_image_visible[...] = mosaic
            use_half = False
            if max_resolution > 0:
                sensor_max = max(raw.sizes.width, raw.sizes.height)
                use_half = max_resolution <= sensor_max // 2

            common = dict(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
                gamma=(1, 1),
                half_size=use_half,
            )

            # The camera's gains, to undo the normalisation Ignore applies. A
            # frame without them is decoded the plain way.
            gains = None
            try:
                found = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
                if found.shape == (3,) and np.all(found > 0):
                    gains = found
            except Exception:
                gains = None
            matrix = _srgb_from_camera(raw)

            if gains is not None:
                try:
                    return _decode_with_headroom(raw, common, gains, matrix)
                except Exception as e:
                    logger.debug(f"rawpy headroom decode retry for {source.name}: {e}")

            for extra in (dict(adjust_maximum_thr=0.0), dict()):
                try:
                    rgb16 = raw.postprocess(**common, **extra)
                    return rgb16.astype(np.float32) / 65535.0
                except Exception as e:
                    logger.debug(f"rawpy linear decode retry for {source.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"rawpy linear decode failed for {source.name}: {e}")
        return None


def _load_linear(source: Path, max_resolution: int = 0):
    """Decode any supported file to scene-linear float32 (H, W, 3), or ``None``."""
    from adjust.render import linear_from_image

    if source.suffix.lower() in _RAW_EXTENSIONS:
        rgb = _load_raw_linear(source, max_resolution)
        if rgb is not None:
            return rgb
        # A RAW the linear path cannot read still gets the 8-bit decoder.
    img = _load_image(source, max_resolution)
    if img is None:
        return None
    return linear_from_image(img)


_LINEAR_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
# A preview-sized frame with its layers is ~100-160 MB: enough for the photograph
# being edited, its neighbours got ready in advance (see prefetch), and the
# small renders of the style gallery beside them.
_LINEAR_CACHE_LIMIT = 8
_LINEAR_BUSY: dict = {}


# ── The camera's look ──────────────────────────────────────────────
#
# A RAW decode is flat; the camera's own JPEG, embedded in the same file, is not
# (see adjust.look). The look is learned from the pair once per file and kept —
# it is a few kilobytes — so that the preview, the library thumbnails and the
# export all start from the same rendering. A file with no embedded JPEG, or one
# that does not match the decode, has no look and is rendered flat as before.
_LOOK_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_LOOK_CACHE_LIMIT = 512


def _sibling_jpeg_rgb(source: Path):
    """The camera's full-size JPEG that sits beside a RAW file, if there is one.

    Shooting RAW+JPEG writes the camera's complete rendering — its Picture Style or
    image control, in full resolution — as a second file with the same name. That is
    a better reference than the preview inside the RAW, which may be small and is
    not always made by the same engine, so it is tried first.
    """
    import numpy as np

    try:
        wanted = source.stem.lower()
        for sibling in source.parent.iterdir():
            if sibling.stem.lower() == wanted and sibling.suffix.lower() in (".jpg", ".jpeg"):
                img = Image.open(sibling)
                img.load()
                return np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        logger.debug(f"No sibling JPEG for {source.name}: {e}")
    return None


def _embedded_camera_rgb(source: Path):
    """The camera's own JPEG for a RAW file, upright, as 8-bit RGB; or ``None``."""
    import numpy as np

    img = _embedded_camera_image(source)
    return None if img is None else np.asarray(img, dtype=np.uint8)


# ── The camera's look, pooled over frames ──────────────────────────
#
# One frame is a poor witness to its camera's colour. A stage of black clothes and
# warm wood holds too little colour to pin a matrix down, and on such a frame a
# single-frame fit can wander far from what the camera does. The rendering belongs
# to the camera and its Picture Style, not to the frame, so the samples of every
# frame opened are pooled per camera and per Picture Style settings, and the look
# is fitted to the pool. Measured on four R6 II frames: a look fitted to the other
# three matched the fourth frame's JPEG better than the old single-frame fit
# matched its own frame.
#
# What is pooled is the camera's colour. Its tone is not: Auto Lighting
# Optimizer lifts a dark or high-contrast frame more than a bright, flat one and
# Highlight Tone Priority reshapes the top of the curve, so a frame with a JPEG
# keeps its own tone curve under the pool's colour matrix. On R6 II brackets
# shot with both on, a single pooled look was kept out of every high-contrast
# frame; the pooled colour with each frame's own tone fitted every frame better
# than its own single-frame fit did. A frame with no JPEG takes the pool's tone
# as well, as the best guess there is.
#
# A frame joins its pool only when the pooled colour describes it about as well
# as its own fit does; a frame shot with another style that the file's metadata
# did not tell apart (or that exiftool could not read) keeps its own look and
# stays out of the pool. The pools are kept on disk when a directory is set
# (``set_profile_dir``), so the look keeps improving across sessions.

_POOL_FRAMES = 32
# How much worse than its own fit a frame may find the pooled look and still be
# counted as the same rendering, in mean error over every channel (fraction of
# full scale), so that a difference in colour alone keeps a frame out.
_POOL_SLACK_RATIO = 1.25
# The pooled colour may draw the whole picture this much worse than the frame's own
# look (mean |difference| over every channel, fraction of full scale) and still join.
_POOL_PICTURE_RATIO = 1.05
_POOL_PICTURE_SLACK = 0.001
_POOL_SLACK = 0.004
_POOL_LOCK = threading.Lock()
_POOLS: dict = {}
_DECISIONS: dict = {}
_LENS_POOLS: dict = {}
_LENS_NOTES: dict = {}
# How many frames of the same kind of scene a frame's look was made beside (None: not known).
_SCENE_FRAMES: dict = {}
_LENS_KEYS: dict = {}
_PROFILE_DIR: Optional[Path] = None
_PROFILE_WRITABLE = False

# The settings that change a Canon rendering. Read through exiftool when it is
# installed; on other makes the tags are simply absent.
_STYLE_TAGS = ("Model", "PictureStyle", "Contrast", "Saturation", "ColorTone",
               "AutoLightingOptimizer", "HighlightTonePriority")
# The same for other makes, added to a file's style only when the file records them,
# so a Canon file's style (and the pool it is kept in) is the string it always was.
# Ricoh GR: Image Control (ImageEffects), toning, hue, highlight correction
# (DynamicRangeExpansion); Sony: Creative Style; Fujifilm: Film Simulation.
_STYLE_TAGS_OTHER = ("ImageEffects", "ToningEffect", "HueAdjust", "DynamicRangeExpansion",
                     "CreativeStyle", "FilmMode")


def set_profile_dir(path) -> tuple:
    """Keep pooled camera looks in ``path``; ``None`` keeps them in memory only.

    Returns ``(writable, error)``. A directory that refuses writes is still read
    (the looks already learned there are used) but nothing new is saved, and the
    reason is reported once here rather than on every frame.
    """
    global _PROFILE_DIR, _PROFILE_WRITABLE
    with _POOL_LOCK:
        _POOLS.clear()
        _DECISIONS.clear()
        _LENS_POOLS.clear()
        _LENS_NOTES.clear()
        _SCENE_FRAMES.clear()
        _LENS_KEYS.clear()
        _PROFILE_DIR = Path(path) if path else None
        _PROFILE_WRITABLE = False
        if _PROFILE_DIR is None:
            return False, ""
        from storage_health import probe_writable

        writable, error = probe_writable(_PROFILE_DIR)
        _PROFILE_WRITABLE = writable
        if not writable:
            logger.error(f"Camera profiles in {_PROFILE_DIR} are read only, nothing new "
                         f"will be saved: {error}")
        return writable, error


def pooling_decision(source: Path) -> Optional[dict]:
    """How the last look for ``source`` was chosen: its pool, whether it joined,
    and why (for the offline tool)."""
    decision = _DECISIONS.get(str(source))
    note = _LENS_NOTES.get(str(source))
    if decision is None and note is None:
        return None
    return {**(decision or {}), **({"lens": note} if note else {})}


# A look from fewer frames than this is still being learned.
_LOOK_SETTLED_FRAMES = 4
_OWN_LOOK_REASONS = ("pooled colour draws the picture worse", "coloured unlike the pool",
                     "pooled colour does not describe this frame")
# Mean |difference| of the whole picture from the camera's JPEG above which a look is
# called rough (about 4 on the OkLab scale of the offline tool; the frames of a
# settled pool are at 0.012 to 0.03).
_ROUGH_PICTURE_ERROR = 0.034


def look_status(source: Path, look) -> tuple:
    """How well established the look on a frame is, for the develop view to say so
    quietly: ``(state, frames, hint)``.

    state:
      ``none``      no look at all (nothing to learn from)
      ``learning``  from fewer than a handful of the camera's frames: rough
      ``own``       from this frame alone because the camera's other frames did not
                    describe it (another style, perhaps)
      ``alone``     from this frame alone because the camera cannot be told apart
                    from others, so frames share nothing
      ``ready``     from the camera's pool of ``frames``
    hint, when more frames would help (the view suggests learning from the folder):
      ``rough``     the look draws this picture noticeably unlike the camera's JPEG
      ``scene``     no other frame in the pool is of this kind of scene, so the tone
                    an exposure edit moves towards has nothing to learn from
      ``""``        nothing to suggest
    """
    if look is None:
        return "none", 0, ""
    decision = _DECISIONS.get(str(source)) or {}
    key = decision.get("pool")
    with _POOL_LOCK:
        pooled = len(_pool(key)) if key else 0
    if decision.get("reason") == "camera not identified":
        return "alone", 1, ""
    hint = ""
    if decision.get("picture_error", 0.0) > _ROUGH_PICTURE_ERROR:
        hint = "rough"
    elif _SCENE_FRAMES.get(str(source)) == 0:
        hint = "scene"
    if decision.get("reason") in _OWN_LOOK_REASONS:
        return "own", max(pooled, 1), hint
    frames = max(pooled, int(getattr(look, "frames", 1) or 1))
    return ("learning" if frames < _LOOK_SETTLED_FRAMES else "ready"), frames, hint


# ── Learning a folder from the app ───────────────────────────────────

_LEARN_JOBS: dict = {}


def look_health(source: Path, size: int = 1800) -> dict:
    """Whether the look ``source`` is drawn with has the shape of a broken one: a
    tone curve with a plateau (adjust.look.curve_evenness), or the photograph with
    no edit posterized against the camera's JPEG somewhere in its tones
    (adjust.look.tone_detail). Opens the photograph as the develop view does."""
    import numpy as np

    from adjust.look import BANDING_MAX, EVEN_MIN, align_camera, curve_evenness, tone_detail
    from adjust.render import develop_frame

    frame = linear_base(source, size)
    look = frame.look if frame is not None else None
    if look is None:
        return {"look": None}
    state, frames, _ = look_status(source, look)
    even = curve_evenness(look.curve)
    out = {"look": state, "pool_frames": frames,
           "even": None if even is None else round(even, 3), "even_min": EVEN_MIN,
           "banding_max": BANDING_MAX,
           "pooling": (pooling_decision(source) or {}).get("reason")}
    camera = _sibling_jpeg_rgb(source)
    if camera is None:
        camera = _embedded_camera_rgb(source)
    if camera is not None:
        shown = np.asarray(develop_frame(frame, {}), dtype=np.float32) / 255.0
        detail = tone_detail(shown, align_camera(frame.rgb, camera))
        if detail is not None:
            out.update(detail)
    out["off"] = bool(even is not None and even < EVEN_MIN)
    out["shows"] = bool(out["off"] and out.get("banding", 0.0) > BANDING_MAX)
    return out


def learn_status(folder: str) -> dict:
    return dict(_LEARN_JOBS.get(folder) or {"status": "idle", "done": 0, "total": 0})


def learn_folder(folder: str, files, size: int = 1000) -> dict:
    """Open every RAW of a folder the way the develop view does, so each teaches its
    camera's pool, on this thread (callers run it on a worker). The held previews
    are dropped at the end so every frame is drawn again from the finished pool.
    One run per folder at a time."""
    with _POOL_LOCK:
        job = _LEARN_JOBS.get(folder)
        if job and job.get("status") == "running":
            return dict(job)
        job = _LEARN_JOBS[folder] = {"status": "running", "done": 0, "total": len(files)}
    try:
        for path in files:
            try:
                linear_base(Path(path), size, relearn=True)
            except Exception as e:
                logger.warning(f"Learning from {Path(path).name} failed: {e}")
            job["done"] += 1
        clear_preview_cache()
        job["status"] = "done"
    except Exception as e:  # pragma: no cover - defensive
        job["status"] = "failed"
        job["error"] = str(e)
    return dict(job)


# ── Learning a style ─────────────────────────────────────────────────
#
# A style (adjust.styles) is a look learned from many frames and kept under a
# name, to be put on other cameras' files. Each frame teaches exactly as it does
# its camera's pool — its RAW against the JPEG beside it, or else the one inside
# it, the lens's vignetting corrected first — but the samples go to the style,
# not to any pool, so learning the look of your own edited exports (a RAW with
# an edited JPEG beside it) never disturbs what the app knows about the camera.

_STYLE_JOB: dict = {"status": "idle", "done": 0, "total": 0}
_STYLE_LOCK = threading.Lock()


def style_job() -> dict:
    return dict(_STYLE_JOB)


# How each rendering setting reads to a person; ``None`` shows the value alone
# (a camera model, a Picture Style, an Image Control are their own names).
_STYLE_LABELS = {
    "Model": None, "PictureStyle": None, "Contrast": "contrast", "Saturation": "saturation",
    "ColorTone": "colour tone", "AutoLightingOptimizer": "Auto Lighting Optimizer",
    "HighlightTonePriority": "Highlight Tone Priority", "ImageEffects": None,
    "ToningEffect": "toning", "HueAdjust": "hue", "DynamicRangeExpansion": "highlight correction",
    "CreativeStyle": None, "FilmMode": None,
}
_UNSET = ("", "-", "n/a")


def _settings(style: str) -> dict:
    """A style string (:func:`_style_of`) as {tag: value}, unset values left out."""
    names = _STYLE_TAGS + _STYLE_TAGS_OTHER
    return {t: v for t, v in zip(names, style.split("|") if style else []) if v not in _UNSET}


def _setting_text(tag: str, value: str) -> str:
    label = _STYLE_LABELS.get(tag)
    return f"{label} {value}" if label else value


def _describe_style(style: str) -> str:
    """The camera and its rendering settings, for a person to read: what is set,
    without the defaults ("Off", "Normal") that say nothing about the look."""
    return " · ".join(_setting_text(t, v) for t, v in _settings(style).items()
                      if v not in ("Off", "Normal", "Disable"))


def _difference(style: str, wanted: str) -> str:
    """How ``style`` differs from ``wanted``, in their settings, for a person."""
    mine, theirs = _settings(style), _settings(wanted)
    changed = [_setting_text(t, mine.get(t, "unset")) for t in _STYLE_TAGS + _STYLE_TAGS_OTHER
               if mine.get(t) != theirs.get(t)]
    return ", ".join(changed) or "another camera"


def _style_frames(files, like: Optional[Path]):
    """The RAWs among ``files`` a style is learned from, as ``(frames, like, keys)``.

    ``like`` keeps only the frames shot with that file's camera and rendering
    settings. A RAW+JPEG pair is shown by its JPEG, and a JPEG's key (no sensor
    matrix) never equals a RAW's: the pair stands for its RAW, and a JPEG with no
    RAW, or a file whose settings cannot be read, keeps every frame. ``like`` comes
    back as the RAW it was taken to be, or ``None`` when nothing was filtered."""
    files = [Path(f) for f in files if Path(f).suffix.lower() in _RAW_EXTENSIONS]
    if like is not None:
        like = resolve_export_source(Path(like))
        if like.suffix.lower() not in _RAW_EXTENSIONS:
            like = None
    if like is None:
        return files, None, {}
    keys = _camera_keys(files + [like])
    wanted = keys.get(like)
    if wanted is None:
        return files, None, keys
    return [f for f in files if keys.get(f) == wanted], like, keys


def style_preview(files, like: Optional[Path] = None) -> dict:
    """What learning a style from ``files`` like ``like`` would use, before it
    runs: how many frames match, the settings they share, and what the frames
    left out differ in."""
    from adjust import styles

    raws = [Path(f) for f in files if Path(f).suffix.lower() in _RAW_EXTENSIONS]
    frames, like, keys = _style_frames(raws, like)
    out = {"total": len(raws), "matching": len(frames), "filtered": like is not None,
           "describe": "", "others": [], "limit": styles.MAX_LEARN_FRAMES}
    if like is None:
        return out
    strings = _style_strings(raws + [like])
    wanted = strings.get(like, "")
    out["describe"] = _describe_style(wanted)
    groups: dict = {}
    for f in raws:
        if keys.get(f) != keys.get(like):
            group = groups.setdefault(keys.get(f), {"frames": 0, "style": strings.get(f, "")})
            group["frames"] += 1
    out["others"] = sorted(
        ({"frames": g["frames"], "differs": _difference(g["style"], wanted)} for g in groups.values()),
        key=lambda g: -g["frames"],
    )
    return out


def _style_sample(source: Path, size: int = 1000):
    """What one RAW teaches a style: its decode against the JPEG beside it, or
    else the one inside it, the lens's vignetting corrected first. ``None`` when
    neither matches the RAW."""
    from adjust.look import fit_with_lens

    rgb = _load_linear(source, size)
    if rgb is None:
        return None
    camera = _sibling_jpeg_rgb(source)
    found = fit_with_lens(rgb, camera) if camera is not None else (None, None)
    if found[0] is None:
        camera = _embedded_camera_rgb(source)
        found = fit_with_lens(rgb, camera) if camera is not None else (None, None)
    return found[0]


def learn_style(name: str, files, like: Optional[Path] = None, size: int = 1000) -> dict:
    """Learn a style from ``files`` (RAWs), on this thread; callers run it on a
    worker. ``like`` keeps only the frames shot with the same camera and the same
    rendering settings as that file, so a folder from two cameras, or one where
    the Picture Style changed half way, teaches one rendering and not an average
    of two. One run at a time; the answer is also kept in :func:`style_job`."""
    from adjust import styles

    with _STYLE_LOCK:
        if _STYLE_JOB.get("status") == "running":
            return dict(_STYLE_JOB)
        _STYLE_JOB.clear()
        _STYLE_JOB.update(status="running", kind="learn", name=name, done=0, total=0, used=0,
                          skipped=0, error="", style=None)
    try:
        files, like, keys = _style_frames(files, like)
        if not files:
            _STYLE_JOB.update(status="failed",
                              error="No RAW frame here was shot with this frame's camera and settings.")
            return dict(_STYLE_JOB)
        files = styles.spread(files)
        _STYLE_JOB["total"] = len(files)
        samples = []
        used = []
        for source in files:
            try:
                sample = _style_sample(source, size)
                if sample is not None:
                    samples.append(sample)
                    used.append(source)
                    _STYLE_JOB["used"] += 1
                else:
                    _STYLE_JOB["skipped"] += 1
            except Exception as e:
                logger.warning(f"Learning a style from {source.name} failed: {e}")
                _STYLE_JOB["skipped"] += 1
            _STYLE_JOB["done"] += 1
        if not samples:
            _STYLE_JOB.update(status="failed",
                              error="No frame here has a JPEG that matches its RAW to learn from.")
            return dict(_STYLE_JOB)
        described = like or files[0]
        settings = _style_strings([described]).get(described, "")
        style = styles.from_samples(name, samples, source=_describe_style(settings))
        if style is not None:
            # What frames from other folders must share to be added later
            # (improve_style): the camera, its settings, and where these came from.
            style.key = keys.get(described) or _camera_keys([described]).get(described) or ""
            style.settings = settings
            style.folders = sorted({str(Path(f).parent) for f in used})
        if style is None:
            _STYLE_JOB.update(status="failed",
                              error="These frames do not agree on one rendering; try fewer, or one camera setting.")
            return dict(_STYLE_JOB)
        styles.save(style, samples)
        _STYLE_JOB.update(status="done", style=style.summary())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Learning style {name!r} failed: {e}")
        _STYLE_JOB.update(status="failed", error=str(e))
    return dict(_STYLE_JOB)


def improve_style(style_id: str, folders, size: int = 1000) -> dict:
    """Try to make a style better with frames from other folders, on this
    thread; callers run it on a worker. ``folders`` are the folders to look in
    (the caller narrows them to the style's camera). Only frames shot with the
    style's camera and settings are read, a few per folder and at most
    MAX_ADDED_FRAMES in all; the result is kept only when styles.improve says it
    does no worse on the style's own frames, and the version before is kept to
    undo. Steps aside while someone is using the app (background.py)."""
    import random
    import time

    import background as gate
    from adjust import styles

    style = styles.load(style_id)
    with _STYLE_LOCK:
        if _STYLE_JOB.get("status") == "running":
            return dict(_STYLE_JOB)
        _STYLE_JOB.clear()
        _STYLE_JOB.update(status="running", kind="improve", name=style.name if style else style_id,
                          style_id=style_id, phase="looking", done=0, total=0, used=0,
                          skipped=0, folders=0, error="", style=None, report=None)
    try:
        if style is None:
            _STYLE_JOB.update(status="failed", error="No such style")
            return dict(_STYLE_JOB)
        if not style.key:
            _STYLE_JOB.update(status="failed", error=(
                "This style was learned before styles remembered their camera settings; "
                "learn it again to improve it from other folders."))
            return dict(_STYLE_JOB)
        own = styles.samples_of(style_id)
        room = min(styles.MAX_ADDED_FRAMES, styles.MAX_TOTAL_FRAMES - len(own))
        if room <= 0:
            _STYLE_JOB.update(status="failed", error="It already has as many frames as a style keeps.")
            return dict(_STYLE_JOB)
        skip = {str(Path(f)) for f in style.folders}
        folders = [Path(f) for f in folders if str(Path(f)) not in skip]
        random.Random(style_id).shuffle(folders)
        _STYLE_JOB["total"] = len(folders)
        # Three per folder spreads what is added over many shoots and lights, so
        # the style learns the rendering rather than one more evening's scenes.
        # Twice what can be added is read, for the screen to choose from.
        per_folder, wanted = 3, room * 2
        found, from_folders = [], []
        for folder in folders:
            if len(found) >= wanted:
                break
            if not gate.wait_turn("Improving a style"):
                break
            raws = raw_files_in(folder)
            if raws:
                # The settings of a sample of the folder first: reading every
                # file's would cost minutes on a big folder that has none.
                sample = styles.spread(raws, 12)
                keys = _camera_keys(sample)
                matching = [f for f in sample if keys.get(f) == style.key]
                if matching:
                    from_folders.append(str(folder))
                    found += [(f, str(folder)) for f in matching[:per_folder]]
            _STYLE_JOB["done"] += 1
        _STYLE_JOB["folders"] = len(from_folders)
        if not found:
            _STYLE_JOB.update(status="done", phase="", report={
                "offered": 0, "added": 0, "reason": "no other folder has frames shot with this camera and settings"})
            return dict(_STYLE_JOB)

        _STYLE_JOB.update(phase="reading", done=0, total=len(found))
        candidates, folder_of = [], {}
        for source, folder in found:
            gate.wait_turn("Improving a style")
            try:
                sample = _style_sample(source, size)
            except Exception as e:
                logger.warning(f"Reading {source.name} for a style failed: {e}")
                sample = None
            if sample is not None:
                candidates.append(sample)
                folder_of[id(sample)] = folder
                _STYLE_JOB["used"] += 1
            else:
                _STYLE_JOB["skipped"] += 1
            _STYLE_JOB["done"] += 1

        _STYLE_JOB.update(phase="checking")
        better, report, added = styles.improve(style, own, candidates)
        added_from = {folder_of[id(s)] for s in added}
        report["looked_in"] = len(from_folders)
        report["folders"] = len(added_from)
        if better is not None:
            report["when"] = time.time()
            better.improved = report
            better.folders = sorted(set(style.folders) | added_from)
            styles.keep_previous(style_id)
            styles.save(better, list(own) + added)
            # Previews drawn through the old version are stale now.
            clear_preview_cache()
            _STYLE_JOB.update(style=styles.load(style_id).summary())
        _STYLE_JOB.update(status="done", phase="", report=report)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Improving style {style_id!r} failed: {e}")
        _STYLE_JOB.update(status="failed", error=str(e))
    return dict(_STYLE_JOB)


def raw_files_in(folder: Path) -> list:
    try:
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _RAW_EXTENSIONS)
    except OSError:
        return []


def _sibling_jpeg_path(source: Path) -> Optional[Path]:
    try:
        wanted = source.stem.lower()
        for sibling in source.parent.iterdir():
            if sibling.stem.lower() == wanted and sibling.suffix.lower() in (".jpg", ".jpeg"):
                return sibling
    except OSError:
        pass
    return None


def _style_blocks(stdout: str) -> list:
    """exiftool's ``-s -f`` output as ``[(path or None, [values…])]``, one per file.
    A run over a single file prints no ``========`` header, hence the ``None``."""
    blocks, current = [], None
    for line in stdout.splitlines():
        if line.startswith("======== "):
            current = []
            blocks.append((Path(line[9:].strip()), current))
        else:
            if current is None:
                current = []
                blocks.append((None, current))
            current.append(line.split(":", 1)[-1].strip())
    return blocks


def _style_string(values: list, beside: Optional[list]) -> str:
    """One file's settings (and those of the JPEG beside it) as one string."""
    names = _STYLE_TAGS + _STYLE_TAGS_OTHER
    values = values[:len(names)]
    if not values:
        return ""
    base, other = values[:len(_STYLE_TAGS)], values[len(_STYLE_TAGS):]
    if beside is not None:
        beside = beside[len(_STYLE_TAGS):len(names)]
        other = [v if v not in ("", "-") else (beside[i] if i < len(beside) else v)
                 for i, v in enumerate(other)]
    if any(v not in ("", "-") for v in other):
        base = base + other
    return "|".join(base)


def _style_of(source: Path) -> str:
    """The rendering settings the file records, as one string; empty when they
    cannot be read. The camera's JPEG beside a RAW is asked too, for the settings
    of other makes (a Ricoh's Image Control may be written only into the JPEG)."""
    try:
        names = _STYLE_TAGS + _STYLE_TAGS_OTHER
        files = [source]
        sibling = _sibling_jpeg_path(source) if source.suffix.lower() in _RAW_EXTENSIONS else None
        if sibling is not None:
            files.append(sibling)
        out = subprocess.run(
            ["exiftool", "-s", "-f", *[f"-{t}" for t in names], *[str(f) for f in files]],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0:
            return ""
        blocks = [values for _, values in _style_blocks(out.stdout)]
        return _style_string(blocks[0] if blocks else [], blocks[1] if len(blocks) > 1 else None)
    except Exception:
        return ""


# What :func:`_style_strings` and :func:`_camera_keys` read, per file and the
# state it was read in, so a folder is read once and not again for the learning
# that follows its preview.
_STYLE_STRING_CACHE: dict = {}
_SIGNATURE_CACHE: dict = {}
_READ_CACHE_LIMIT = 50000
# Files per exiftool run: one process for many is what makes a folder quick (a
# process per file is a third of a second on Windows), and a list this long
# still goes through an argument file rather than the command line.
_STYLE_BATCH = 400


def _stamp(path: Optional[Path]) -> tuple:
    try:
        st = path.stat()
        return (str(path), st.st_mtime_ns, st.st_size)
    except (OSError, AttributeError):
        return (str(path), 0, 0)


def _remember(cache: dict, key, value) -> None:
    if len(cache) >= _READ_CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def _style_strings(files) -> dict:
    """:func:`_style_of` for many files at once: ``{path: style string}``.

    The same strings, read with one exiftool run per few hundred files instead of
    one per file, ``-fast`` (the settings sit at the start of the file; nothing is
    read past them), and each folder listed once to find the JPEG beside a RAW."""
    import tempfile

    files = list(dict.fromkeys(Path(f) for f in files))
    listing: dict = {}
    siblings = {}
    for f in files:
        if f.suffix.lower() not in _RAW_EXTENSIONS:
            continue
        if f.parent not in listing:
            try:
                listing[f.parent] = {p.stem.lower(): p for p in f.parent.iterdir()
                                     if p.suffix.lower() in (".jpg", ".jpeg")}
            except OSError:
                listing[f.parent] = {}
        siblings[f] = listing[f.parent].get(f.stem.lower())
    stamps = {f: (_stamp(f), _stamp(siblings.get(f)) if siblings.get(f) else None) for f in files}
    found = {f: _STYLE_STRING_CACHE[stamps[f]] for f in files if stamps[f] in _STYLE_STRING_CACHE}
    todo = [f for f in files if f not in found]
    names = _STYLE_TAGS + _STYLE_TAGS_OTHER
    for start in range(0, len(todo), _STYLE_BATCH):
        chunk = todo[start:start + _STYLE_BATCH]
        paths = []
        for f in chunk:
            paths.append(f)
            if siblings.get(f):
                paths.append(siblings[f])
        read: dict = {}
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8",
                                         delete=False) as arg:
            arg.write("\n".join(str(p) for p in paths))
        try:
            out = subprocess.run(
                ["exiftool", "-fast", "-charset", "filename=utf8", "-s", "-f",
                 *[f"-{t}" for t in names], "-@", arg.name],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60 + 2 * len(paths),
            )
            for path, values in _style_blocks(out.stdout):
                read[path if path is not None else paths[0]] = values
        except Exception as e:
            logger.debug(f"Reading rendering settings failed: {e}")
        finally:
            Path(arg.name).unlink(missing_ok=True)
        for f in chunk:
            values = read.get(f)
            style = "" if values is None else _style_string(
                values, read.get(siblings[f]) if siblings.get(f) else None)
            found[f] = style
            if values is not None:
                _remember(_STYLE_STRING_CACHE, stamps[f], style)
    return found


def _matrix_signature(source: Path) -> str:
    """The sensor's colour matrix as text, which tells one camera from another;
    empty when the file has none (a JPEG) or cannot be opened."""
    import numpy as np

    try:
        import rawpy

        with rawpy.imread(str(source)) as raw:
            matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3]
            if np.any(matrix != 0):
                return ",".join(f"{v:.4f}" for v in matrix.ravel())
    except Exception:
        pass
    return ""


def _key_of(signature: str, style: str) -> Optional[str]:
    import hashlib

    if not signature and not style.strip("|-"):
        return None
    return hashlib.sha1(f"{signature}#{style}".encode()).hexdigest()[:20]


def _camera_key(source: Path) -> Optional[str]:
    """Which pool a RAW file's samples belong to, or ``None`` if it cannot be
    told apart from other cameras' files."""
    return _key_of(_matrix_signature(source), _style_of(source))


def _camera_keys(files) -> dict:
    """:func:`_camera_key` for many files at once: ``{path: key}``.

    The matrix belongs to the camera model, and opening a RAW for it reads the
    whole file (half a second each, from a slow disk), so it is read from one
    file per model and file type and shared; a file that names no model is
    opened itself."""
    files = list(dict.fromkeys(Path(f) for f in files))
    strings = _style_strings(files)
    by_model: dict = {}
    keys = {}
    for f in files:
        style = strings.get(f, "")
        model = style.split("|", 1)[0].strip("-")
        group = (model, f.suffix.lower()) if model else None
        signature = by_model.get(group) if group else None
        if signature is None:
            stamp = _stamp(f)
            signature = _SIGNATURE_CACHE.get(stamp)
            if signature is None:
                signature = _matrix_signature(f)
                _remember(_SIGNATURE_CACHE, stamp, signature)
            if group:
                by_model[group] = signature
        keys[f] = _key_of(signature, style)
    return keys


def _pool(key: str) -> "OrderedDict[str, object]":
    """The frames pooled under ``key``, loaded from disk on first use."""
    pool = _POOLS.get(key)
    if pool is not None:
        return pool
    pool = OrderedDict()
    if _PROFILE_DIR is not None:
        path = _PROFILE_DIR / f"{key}.npz"
        try:
            import numpy as np

            from adjust.look import FrameSamples

            with np.load(path, allow_pickle=False) as data:
                bounds = np.concatenate([[0], np.cumsum(data["counts"])])
                for i, frame_id in enumerate(data["frames"]):
                    a, b = int(bounds[i]), int(bounds[i + 1])
                    scene = None
                    if "scene" in data.files and np.isfinite(data["scene"][i]).all():
                        scene = data["scene"][i]
                    pool[str(frame_id)] = FrameSamples(linear=data["linear"][a:b],
                                                       camera=data["camera"][a:b], scene=scene)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Camera profile {path.name} unreadable, starting afresh: {e}")
    _POOLS[key] = pool
    return pool


def _save_pool(key: str, pool) -> None:
    if _PROFILE_DIR is None or not _PROFILE_WRITABLE or not pool:
        return
    import numpy as np

    from adjust.look import SCENE_TERMS

    samples = list(pool.values())
    path = _PROFILE_DIR / f"{key}.npz"
    try:
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            frames=np.array(list(pool.keys())),
            counts=np.array([len(s.linear) for s in samples]),
            linear=np.concatenate([s.linear for s in samples]),
            camera=np.concatenate([s.camera for s in samples]),
            scene=np.stack([s.scene if s.scene is not None else np.full(SCENE_TERMS, np.nan, np.float32)
                            for s in samples]),
        )
        tmp.replace(path)
    except Exception as e:
        logger.warning(f"Camera profile {path.name} not saved: {e}")


def _lens_key(source: Path) -> Optional[str]:
    """Which lens setting a file was shot at: camera, lens, focal length and
    aperture, as one key; ``None`` when exiftool cannot say.

    A picture style that adds its own effects to the JPEG is part of it: a Ricoh GR's
    Retro darkens the corners, so its correction is not that of its Standard, and
    pooling them gave Retro frames the other's brightening (+0.85 stops at the corner
    against a camera that darkened them). Canon files record none of those settings
    (see ``_STYLE_TAGS_OTHER``) and keep the key they had."""
    import hashlib
    import json

    try:
        stat = source.stat()
    except OSError:
        return None
    cache_key = (str(source), stat.st_mtime_ns)
    if cache_key in _LENS_KEYS:
        return _LENS_KEYS[cache_key]
    key = None
    try:
        out = subprocess.run(
            ["exiftool", "-j", "-n", "-Model", "-LensModel", "-FocalLength", "-FNumber", str(source)],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        tags = json.loads(out)[0]
        if tags.get("FocalLength"):
            text = "|".join(str(tags.get(t)) for t in ("Model", "LensModel")) + \
                f"|{round(float(tags['FocalLength']))}|{round(float(tags.get('FNumber') or 0), 1)}"
            effects = _style_of(source).split("|")[len(_STYLE_TAGS):]
            if effects:
                text += "|" + "|".join(effects)
            key = hashlib.sha1(text.encode()).hexdigest()[:16]
    except Exception:
        key = None
    _LENS_KEYS[cache_key] = key
    return key


def _lens_pool(key: str) -> dict:
    """The vignetting corrections learned for a lens setting: ``{frame: (a, b)}``."""
    pool = _LENS_POOLS.get(key)
    if pool is not None:
        return pool
    pool = {}
    if _PROFILE_DIR is not None:
        import json

        try:
            pool = {k: (float(v[0]), float(v[1]))
                    for k, v in json.loads((_PROFILE_DIR / f"lens_{key}.json").read_text()).items()}
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Lens profile lens_{key}.json unreadable, starting afresh: {e}")
    _LENS_POOLS[key] = pool
    return pool


def _save_lens_pool(key: str, pool: dict) -> None:
    if _PROFILE_DIR is None or not _PROFILE_WRITABLE or not pool:
        return
    import json

    path = _PROFILE_DIR / f"lens_{key}.json"
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({k: list(v) for k, v in pool.items()}))
        tmp.replace(path)
    except Exception as e:
        logger.warning(f"Lens profile {path.name} not saved: {e}")


_LENS_POOL_FRAMES = 64
# A pooled correction this close to the frame's own (stops at the corner) changes nothing.
_LENS_SAME = 0.05


def _with_pooled_lens(source: Path, rgb, camera, samples, own):
    """Replace a frame's own vignetting search by the camera's correction for its
    lens setting, once enough frames of that setting have been seen.

    The frame's own result still joins the pool when the search found something.
    Returns ``(samples, look)``, either unchanged or refitted with the pooled
    correction, and notes which in the pooling decision.
    """
    from adjust.look import fit_with_lens, pool_vignettes

    frame_id = str(source)
    key = _lens_key(source)
    if key is None:
        _LENS_NOTES[frame_id] = "own (lens setting unknown)"
        return samples, own
    searched = tuple(own.vignette)
    with _POOL_LOCK:
        pool = _lens_pool(key)
        others = [v for k, v in pool.items() if k != frame_id]
        if any(searched):
            pool.pop(frame_id, None)
            pool[frame_id] = searched
            while len(pool) > _LENS_POOL_FRAMES:
                pool.pop(next(iter(pool)))
            _save_lens_pool(key, pool)
    pooled = pool_vignettes(others)
    if pooled is None:
        _LENS_NOTES[frame_id] = "own" if any(searched) else "none found"
        return samples, own
    if abs(sum(pooled) - sum(searched)) < _LENS_SAME:
        _LENS_NOTES[frame_id] = f"own, agrees with the pool ({len(others)} frames)"
        return samples, own
    refit_samples, refit = fit_with_lens(rgb, camera, vignette=pooled)
    if refit is None:
        _LENS_NOTES[frame_id] = "own (pooled correction did not fit)"
        return samples, own
    _LENS_NOTES[frame_id] = f"pooled from {len(others)} frames (found {sum(searched):+.2f}, using {sum(pooled):+.2f})"
    return refit_samples, refit


def _reference_samples(source: Path, rgb):
    """What this frame teaches about its camera, from the first of its JPEGs
    that is the same picture: (samples, its own look), or (None, None). The
    look carries the camera's vignetting correction (adjust.look.fit_with_lens),
    pooled over the frames of the same lens setting once there are enough, and
    the samples are taken from the corrected frame."""
    from adjust.look import fit_with_lens

    for name, reference in (("sibling JPEG", _sibling_jpeg_rgb),
                            ("embedded JPEG", _embedded_camera_rgb)):
        camera = reference(source)
        if camera is None:
            continue
        samples, own = fit_with_lens(rgb, camera)
        if own is not None:
            samples, own = _with_pooled_lens(source, rgb, camera, samples, own)
            logger.info(f"Camera look for {source.name} from its {name}")
            return samples, own
        logger.info(f"Camera look not used for {source.name}: its {name} does not match")
    return None, None


def _pooled_look(source: Path, samples, own):
    """The look to use for this frame, given what it teaches on its own, and the
    pool updated with it when it belongs there."""
    from adjust.look import (fit_look, look_error, nearest_scenes, probe_error, scene_curve,
                             trustworthy, with_colour_of)

    key = _camera_key(source)
    frame_id = str(source)

    def decided(look, joined, reason, **errors):
        _DECISIONS[frame_id] = {"pool": key, "joined": joined, "reason": reason,
                                **{k: round(v, 4) for k, v in errors.items()}}
        if look is not None and samples is not None:
            picture = probe_error(look, samples)
            if math.isfinite(picture):
                _DECISIONS[frame_id]["picture_error"] = round(picture, 4)
        if look is not None and own is not None and look is not own:
            look.vignette = own.vignette      # the lens is this frame's, not the pool's
        return look

    if key is None:
        return decided(own, False, "camera not identified")
    with _POOL_LOCK:
        pool = _pool(key)
        others = [s for k, s in pool.items() if k != frame_id]
        if samples is None:
            # No JPEG to learn from: the camera's look from its other frames is
            # still a far better start than a flat decode.
            if len(others) >= 2:
                pooled = fit_look(others)
                if pooled is not None:
                    return decided(pooled, False, "no camera JPEG; look from the pool")
            return decided(own, False, "no camera JPEG")
        if not others:
            pool[frame_id] = samples
            _save_pool(key, pool)
            return decided(own, True, "first frame of its pool")
        # The pool decides the colour; this frame's own JPEG decides its tone,
        # which the camera varies from frame to frame (see adjust.look.with_colour_of).
        everyone = fit_look(others + [samples])
        # Colour is the camera's; the tone an exposure edit moves towards is the
        # scene's: from the pool's frames most like this one, not all of them.
        like = nearest_scenes(others, samples)
        pooled = with_colour_of(everyone, samples, like) if everyone is not None else None
        if own is not None and like is not None:
            own.pool_curve = scene_curve(own.matrix, like + [samples]) if like else None
        _SCENE_FRAMES[frame_id] = None if like is None else len(like)
        if not trustworthy(pooled, [samples]):
            return decided(own, False, "pooled colour does not describe this frame")
        # What a look is finally judged on is the picture, not the flat patches it was
        # fitted to: the pool's colour can fit a bright frame's patches and still draw
        # the frame wrongly (7533 of an R6 II bracket: 20% worse on the patches, 37%
        # worse on the picture, and the old patch-only test let it through).
        picture_own, picture_pooled = probe_error(own, samples), probe_error(pooled, samples)
        if picture_pooled > picture_own * _POOL_PICTURE_RATIO + _POOL_PICTURE_SLACK:
            logger.info(f"{source.name}: the pool's colour draws it worse than its own look "
                        f"({picture_pooled:.4f} against {picture_own:.4f}); using its own")
            return decided(own, False, "pooled colour draws the picture worse",
                           own_error=picture_own, pooled_error=picture_pooled)
        mine = look_error(own, [samples], colour=True)
        theirs = look_error(pooled, [samples], colour=True)
        if theirs > mine * _POOL_SLACK_RATIO + _POOL_SLACK:
            logger.info(f"{source.name} is coloured unlike its camera's other frames; "
                        f"using its own look")
            return decided(own, False, "coloured unlike the pool", own_error=mine, pooled_error=theirs)
        pool.pop(frame_id, None)
        pool[frame_id] = samples
        while len(pool) > _POOL_FRAMES:
            pool.popitem(last=False)
        _save_pool(key, pool)
        return decided(pooled, True, "joined", own_error=mine, pooled_error=theirs)


# ── The camera's look, kept per photograph ─────────────────────────
#
# Learning a frame's look is most of the time it takes to open: on an R6 II frame
# at ISO 25600, 6.7 s of an 8.6 s open, against 0.1 s for LibRaw's decode. Held in
# memory only, it was learned again for every photograph after every restart. So a
# settled look is kept on disk beside the pools, one small file per photograph,
# tied to the file's size and modification time, with the pooling decision that
# made it (the develop view's note on the look reads that).
#
# A look made while its pool was still learning is not kept, and one made from a
# pool that has since doubled is learned again, so a photograph keeps up with its
# camera's pool as it fills, a few times at most, instead of being frozen at the
# first answer.
#
# Version 2: looks are learned from a decode of at least FIT_SIZE (see
# _learning_copy). Version 1 learned from whatever size first opened the file, and
# a grid thumbnail's or a 512 px decode's look was kept for every size after.
_LOOK_FILE_VERSION = 2


def _look_path(source: Path) -> Optional[Path]:
    if _PROFILE_DIR is None:
        return None
    import hashlib
    import os

    digest = hashlib.sha1(os.path.normcase(str(source)).encode("utf-8")).hexdigest()[:24]
    return _PROFILE_DIR / "looks" / f"{digest}.npz"


def _store_look(source: Path, stat, look) -> None:
    """Keep a settled look for ``source`` on disk (see above)."""
    path = _look_path(source)
    if path is None or not _PROFILE_WRITABLE or look is None:
        return
    import json

    import numpy as np

    state, frames, _ = look_status(source, look)
    if state == "learning":
        return
    frame_id = str(source)
    meta = {"source": frame_id, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size,
            "version": _LOOK_FILE_VERSION, "pool_frames": frames,
            "decision": _DECISIONS.get(frame_id)}
    if frame_id in _SCENE_FRAMES:
        meta["scene_frames"] = _SCENE_FRAMES[frame_id]
    if frame_id in _LENS_NOTES:
        meta["lens_note"] = _LENS_NOTES[frame_id]
    arrays = dict(meta=np.array(json.dumps(meta)), curve=look.curve, matrix=look.matrix,
                  numbers=np.array([look.white_stops, look.error, look.frames], dtype=np.float64),
                  vignette=np.array(look.vignette, dtype=np.float64))
    if look.pool_curve is not None:
        arrays["pool_curve"] = look.pool_curve
    if look.hue is not None:
        arrays["hue"] = look.hue
    try:
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception as e:
        logger.warning(f"Camera look for {source.name} not saved: {e}")


def read_kept_look(path: Path) -> tuple:
    """``(meta, look)`` from a kept look's file, whether or not it still stands for
    its photograph. Raises when the file is missing or unreadable."""
    import json

    import numpy as np

    from adjust.look import CameraLook

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        white_stops, error, frames = (float(v) for v in data["numbers"])
        look = CameraLook(curve=data["curve"], matrix=data["matrix"], white_stops=white_stops,
                          error=error, frames=int(frames),
                          pool_curve=data["pool_curve"] if "pool_curve" in data.files else None,
                          vignette=tuple(float(v) for v in data["vignette"]),
                          hue=data["hue"] if "hue" in data.files else None)
    return meta, look


def _stored_look(source: Path, stat):
    """The look kept on disk for ``source`` when it still stands, else ``None``.

    Restores the pooling decision with it, so the look is reported as it was made.
    """
    path = _look_path(source)
    if path is None:
        return None
    try:
        meta, look = read_kept_look(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Kept camera look {path.name} unreadable, learning it again: {e}")
        return None
    if (meta.get("version") != _LOOK_FILE_VERSION or meta.get("source") != str(source)
            or meta.get("mtime_ns") != stat.st_mtime_ns or meta.get("size") != stat.st_size):
        return None
    decision = meta.get("decision") or {}
    key = decision.get("pool")
    if key:
        with _POOL_LOCK:
            pooled = len(_pool(key))
        if pooled >= 2 * max(1, int(meta.get("pool_frames") or 1)):
            return None
    frame_id = str(source)
    if meta.get("decision") is not None:
        _DECISIONS[frame_id] = meta["decision"]
    if "scene_frames" in meta:
        _SCENE_FRAMES[frame_id] = meta["scene_frames"]
    if "lens_note" in meta:
        _LENS_NOTES[frame_id] = meta["lens_note"]
    return look


def _learning_copy(source: Path, rgb):
    """What the look of ``source`` is learned from: ``rgb``, or, when that is smaller
    than the look is fitted at, the file decoded again at that size.

    The look is learned once per file and kept for every size, so it must not
    depend on which size opened the file first. A grid thumbnail or a 512 px
    decode is below adjust.look.FIT_SIZE, and a fit at that size goes wrong on
    exactly the frames that need it most: on a stage frame of an R6 II at high
    ISO, a 512 px fit turned the camera's smooth curve into a plateau across a
    stop of skin tones and a cliff above it (faces flat and posterized), where
    every decode from 768 px up fitted the same smooth curve.
    """
    from adjust.look import FIT_SIZE
    from adjust.render import resize_linear

    if max(rgb.shape[:2]) >= FIT_SIZE:
        return rgb
    larger = _load_linear(source, FIT_SIZE)
    if larger is None or max(larger.shape[:2]) <= max(rgb.shape[:2]):
        return rgb
    return resize_linear(larger, FIT_SIZE)


def camera_look_for(source: Path, rgb, relearn: bool = False):
    """The camera's look for a RAW file, learned from ``rgb`` (its linear decode).

    The reference is the full-size JPEG beside the file when there is one, else the
    JPEG embedded in it; a reference that does not match the decode (a different
    crop, say) is passed over for the next. What the frame teaches is pooled with
    the camera's other frames (see above). ``None`` for anything that is not a RAW
    file, or that has neither a matching reference nor a pool to fall back on. The
    answer, including "none", is kept per file and modification time, in memory,
    and a settled look on disk as well. ``relearn`` passes over the one on disk, so
    the frame teaches its pool again (learning a folder).
    """
    if source.suffix.lower() not in _RAW_EXTENSIONS:
        return None
    try:
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), stat.st_mtime_ns, stat.st_size)
    with _PREVIEW_LOCK:
        if key in _LOOK_CACHE:
            _LOOK_CACHE.move_to_end(key)
            return _LOOK_CACHE[key]

    look = None if relearn else _stored_look(source, stat)
    if look is None:
        try:
            samples, own = _reference_samples(source, _learning_copy(source, rgb))
            look = _pooled_look(source, samples, own)
        except Exception as e:
            logger.warning(f"Camera look failed for {source.name}: {e}")
        _store_look(source, stat, look)

    with _PREVIEW_LOCK:
        _LOOK_CACHE[key] = look
        while len(_LOOK_CACHE) > _LOOK_CACHE_LIMIT:
            _LOOK_CACHE.popitem(last=False)
    return look


def _with_camera_look(source: Path, frame, relearn: bool = False) -> None:
    """Give ``frame`` its camera's look, and the camera's correction of the lens
    vignetting with it (applied to the linear frame, as the camera applies it)."""
    from adjust.look import correct_lens

    frame.look = camera_look_for(source, frame.rgb, relearn)
    frame.rgb = correct_lens(frame.rgb, frame.look)
    frame._layers = None


# ── Noise reduction ─────────────────────────────────────────────────
#
# A RAW frame carries a way to its noise-reduced twin: the same file decoded the
# same way from its denoised mosaic (adjust.noise), at the same size and with the
# same lens correction, so the two line up pixel for pixel and the sliders only
# blend them (adjust.render.noise_reduced). It is made the first time an edit
# asks for it and kept with the frame.

def _denoised_decode(source: Path, size: int, wait: bool):
    """The file decoded from its noise-reduced mosaic, as :func:`_load_raw_linear`
    decodes it at ``size``; ``None`` when there is none (yet)."""
    from adjust import noise

    mosaic = noise.mosaic_for(source, wait=wait)
    if mosaic is None:
        return None
    return _load_raw_linear(source, size, mosaic)


def _with_denoiser(source: Path, frame, size: int) -> None:
    """Give ``frame`` (a decode of ``source`` at ``size`` with its camera's look)
    the way to its noise-reduced twin."""
    if source.suffix.lower() not in _RAW_EXTENSIONS:
        return
    kept = []

    def denoised(wait: bool = True):
        if not kept:
            from adjust.look import correct_lens
            from adjust.render import resize_linear

            rgb = _denoised_decode(source, size, wait)
            if rgb is not None:
                rgb = correct_lens(resize_linear(rgb, size), frame.look)
                if rgb.shape == frame.rgb.shape:
                    kept.append(rgb)
        return kept[0] if kept else None

    frame.denoiser = denoised


_SOURCE_SIZES: "OrderedDict[tuple, int]" = OrderedDict()


def _source_long_side(source: Path) -> Optional[int]:
    """The whole photograph's longer side in pixels, as a full-size decode makes
    it, without decoding it: from the RAW's header or the image's. Kept by file."""
    try:
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), stat.st_mtime_ns)
    with _PREVIEW_LOCK:
        if key in _SOURCE_SIZES:
            return _SOURCE_SIZES[key]
    side = None
    try:
        if source.suffix.lower() in _RAW_EXTENSIONS:
            import rawpy

            with rawpy.imread(str(source)) as raw:
                side = max(raw.sizes.width, raw.sizes.height)
        else:
            with Image.open(source) as img:
                side = max(img.size)
    except Exception:
        side = None
    with _PREVIEW_LOCK:
        _SOURCE_SIZES[key] = side
        while len(_SOURCE_SIZES) > 1024:
            _SOURCE_SIZES.popitem(last=False)
    return side


def frame_of(source: Path, rgb):
    """A ``LinearFrame`` of ``rgb``, a decode of ``source`` at whatever size, that
    knows how much smaller than the whole photograph it is (for sharpening) and
    how the camera corrects its lens (for lens_profile, found when first asked)."""
    from adjust.render import LinearFrame

    frame = LinearFrame(rgb)
    side = _source_long_side(source)
    if side:
        frame.scale = min(1.0, max(rgb.shape[:2]) / side)
    frame.lens = lambda: lens_profile(source, frame.rgb)
    return frame


# The camera's distortion correction, by file: (path, mtime) -> model or None.
_LENS_PROFILES: "OrderedDict[tuple, object]" = OrderedDict()
_LENS_PROFILE_LIMIT = 256
# The size the camera's JPEG is registered onto the RAW at: its terms agree with
# those found at 768 px, and the shift and the corners are measured twice as finely.
LENS_REGISTER_SIZE = 1536


def lens_profile(source: Path, rgb) -> Optional[dict]:
    """How the camera corrected this frame's lens distortion (adjust.lens), from
    its own JPEG registered onto ``rgb`` (any decode of it), or ``None``: a file
    that is not a RAW is the camera's corrected picture already, and a RAW with no
    JPEG that matches it has nothing to learn from."""
    if source.suffix.lower() not in _RAW_EXTENSIONS:
        return None
    try:
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), stat.st_mtime_ns)
    with _PREVIEW_LOCK:
        if key in _LENS_PROFILES:
            return _LENS_PROFILES[key]
    model = None
    try:
        import numpy as np

        from adjust.lens import from_registration
        from adjust.look import _register, _same_shape
        from adjust.render import LUMA, encode, resize_linear

        camera = _sibling_jpeg_rgb(source)
        if camera is None:
            camera = _embedded_camera_rgb(source)
        if camera is not None and _same_shape(rgb, camera):
            small = resize_linear(np.ascontiguousarray(rgb, dtype=np.float32), LENS_REGISTER_SIZE)
            h, w = small.shape[:2]
            cam = np.asarray(Image.fromarray(np.asarray(camera, dtype=np.uint8), "RGB")
                             .resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
            found = _register(encode(np.clip(small @ LUMA, 0.0, 1.0)), cam @ LUMA)
            if found is not None:
                model = from_registration(*found)
    except Exception as e:
        logger.warning(f"Lens profile for {source.name} not found: {e}")
    with _PREVIEW_LOCK:
        _LENS_PROFILES[key] = model
        while len(_LENS_PROFILES) > _LENS_PROFILE_LIMIT:
            _LENS_PROFILES.popitem(last=False)
    return model


def linear_base(source: Path, size: int, relearn: bool = False):
    """Decode, resize and keep a frame as a ``LinearFrame``, for repeated edits.

    The same idea as :func:`preview_base`, one level up: the decode and the
    base/detail split are the slow parts of an edit, and neither depends on the
    sliders, so they are done once per frame and every slider movement after
    that starts from the kept copy. ``relearn``: see :func:`camera_look_for`.
    """
    from adjust.render import resize_linear

    try:
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), int(size), stat.st_mtime_ns, stat.st_size)

    with _PREVIEW_LOCK:
        cached = _LINEAR_CACHE.get(key)
        if cached is not None:
            _LINEAR_CACHE.move_to_end(key)
            return cached
        busy = _LINEAR_BUSY.setdefault(key, threading.Lock())

    # One decode per frame at a time: a frame asked for while it is being got
    # ready in the background (see prefetch) waits for that decode instead of
    # starting a second one.
    with busy:
        with _PREVIEW_LOCK:
            cached = _LINEAR_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            rgb = _load_linear(source, size)
            if rgb is None:
                return None
            frame = frame_of(source, resize_linear(rgb, size))
            _with_camera_look(source, frame, relearn)
            _with_denoiser(source, frame, size)

            with _PREVIEW_LOCK:
                _LINEAR_CACHE[key] = frame
                while len(_LINEAR_CACHE) > _LINEAR_CACHE_LIMIT:
                    _LINEAR_CACHE.popitem(last=False)
            return frame
        finally:
            with _PREVIEW_LOCK:
                _LINEAR_BUSY.pop(key, None)


_FULL_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_FULL_LOCK = threading.Lock()


def _full_linear(source: Path):
    """The whole frame at sensor resolution, kept for the loupe, or ``None``.

    Held as half-precision floats and only the last frame: a 24 MP decode is
    ~300 MB as float32, and the loupe is asked about one photograph at a time.
    Half precision keeps ~3 significant digits, which is far more than the 8-bit
    picture it ends up in.
    """
    import numpy as np

    try:
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), stat.st_mtime_ns, stat.st_size)
    with _FULL_LOCK:
        cached = _FULL_CACHE.get(key)
        if cached is not None:
            return cached
        rgb = _load_linear(source, 0)
        if rgb is None:
            return None
        packed = rgb.astype(np.float16)
        _FULL_CACHE.clear()
        _FULL_CACHE[key] = packed
        return packed


_FULL_DENOISED: "OrderedDict[tuple, object]" = OrderedDict()


def _full_denoised(source: Path):
    """:func:`_full_linear` of the file's noise-reduced mosaic, kept the same way;
    ``None`` when the file has none."""
    import numpy as np

    try:
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), stat.st_mtime_ns, stat.st_size)
    with _FULL_LOCK:
        cached = _FULL_DENOISED.get(key)
        if cached is not None:
            return cached
        rgb = _denoised_decode(source, 0, wait=True)
        if rgb is None:
            return None
        packed = rgb.astype(np.float16)
        _FULL_DENOISED.clear()
        _FULL_DENOISED[key] = packed
        return packed


_WARMING: set = set()


def warm_denoised(source: Path) -> None:
    """Decode the file's noise-reduced twin at full resolution in the background,
    once its mosaic is made, so the loupe does not wait ~10 s for it."""
    try:
        stat = source.stat()
    except OSError:
        return
    key = (str(source), stat.st_mtime_ns, stat.st_size)
    with _PREVIEW_LOCK:
        if key in _FULL_DENOISED or key in _WARMING:
            return
        _WARMING.add(key)

    def run():
        try:
            _full_denoised(source)
        finally:
            with _PREVIEW_LOCK:
                _WARMING.discard(key)

    threading.Thread(target=run, name="loupe-denoised", daemon=True).start()


def upright_for(source: Path, settings: Optional[dict], mode: str = "vertical",
                size: int = 1600) -> Optional[dict]:
    """What straightens ``source``'s lines (adjust.upright), found on the frame as
    the develop view shows it with the edit's lens correction (not its transform,
    which is what is being found). ``None`` if it cannot be read."""
    from adjust.develop import normalize
    from adjust.render import lens_corrected
    from adjust.upright import upright

    clean = normalize(settings)
    frame = linear_base(source, size)
    if frame is None:
        return None
    frame = lens_corrected(frame, clean.get("lens_profile", 0), clean.get("distortion", 0))
    return upright(frame.shown(), mode)


def loupe_image(
    source: Path, settings: dict, u: float, v: float, side: int = 480, preview_size: int = 1600
) -> Optional[Image.Image]:
    """A 1:1 patch of the developed frame around a point of the preview.

    ``(u, v)`` are fractions of the picture as the develop view shows it —
    levelled and cropped — so they are walked back to the frame first. The patch
    is cut from the full-resolution decode and toned with the preview's own base
    layer (see :func:`adjust.render.region_frame`), so it is the same picture,
    magnified, rather than a second opinion about it. ``None`` if it cannot be read.
    """
    from adjust.develop import normalize
    from adjust.geometry import GEOMETRY_KEYS, source_point
    from adjust.render import develop_frame, region_frame

    clean = normalize(settings)
    small = linear_base(source, preview_size)
    if small is None:
        return None
    # Noise reduction comes first, as in develop_frame: it is blended into the
    # full-resolution pixels here, before the spots are healed from them.
    luma, color = clean.pop("nr_luma", 0), clean.pop("nr_color", 0)
    # Then the lens: the preview's frame is corrected as develop_frame corrects
    # it, and the patch is cut from the corrected picture below.
    from adjust.lens import TRANSFORM_KEYS

    lens_on, distortion = clean.pop("lens_profile", 0), clean.pop("distortion", 0)
    transform = {k: clean.pop(k) for k in TRANSFORM_KEYS if k in clean}
    lens_model = None
    if lens_on or distortion or transform:
        from adjust.lens import model_for
        from adjust.render import lens_corrected

        sw0, sh0 = small.size
        camera = small.lens() if lens_on and small.lens is not None else None
        lens_model = model_for({"lens_profile": lens_on, "distortion": distortion, **transform}, camera, sw0 / sh0)
        small = lens_corrected(small, lens_on, distortion, transform)
    spots = clean.pop("spots", None)
    if spots:
        from adjust.render import healed

        # The patch is toned with the preview's base, so that is healed too.
        small = healed(small, spots)
    sw, sh = small.size
    su, sv = source_point(clean, u, v, sw, sh)

    full = _full_linear(source)
    if full is None:
        return None
    fh, fw = full.shape[:2]
    side = max(64, min(int(side), 1024, fw, fh))
    left = max(0, min(fw - side, int(round(su * fw - side / 2))))
    top = max(0, min(fh - side, int(round(sv * fh - side / 2))))
    reduced = _full_denoised(source) if (luma or color) else None
    if reduced is not None and reduced.shape != full.shape:
        reduced = None

    def pixels(index):
        rgb = full[index].astype("float32")
        if reduced is not None:
            from adjust.noise import blend

            rgb = blend(rgb, reduced[index].astype("float32"), luma, color)
        return rgb

    import numpy as np

    from adjust.look import correct_lens

    def raw(x0, y0, x1, y1):
        """The decode over a box of its pixels (clamped to the frame), with the
        camera's vignetting correction, which belongs to the RAW's geometry."""
        xs = np.clip(np.arange(x0, x1), 0, fw - 1)
        ys = np.clip(np.arange(y0, y1), 0, fh - 1)
        return correct_lens(pixels(np.ix_(ys, xs)), small.look, (x0 / fw, y0 / fh, x1 / fw, y1 / fh))

    def corrected(x0, y0, w, h):
        """The picture as developed over a box of its pixels: the decode, or with
        its lens corrected the part of the corrected picture there."""
        if lens_model is None:
            return raw(x0, y0, x0 + w, y0 + h)
        from adjust.lens import warp_region

        return warp_region(raw, lens_model, (fw, fh), (x0 / fw, y0 / fh, (x0 + w) / fw, (y0 + h) / fh), (h, w))

    patch = corrected(left, top, side, side)
    if spots:
        from adjust.spots import apply_spots

        def fetch(x0, y0, size):
            # A spot's source is read from the full decode, wherever it lies.
            return corrected(x0, y0, size, size)

        patch = apply_spots(patch, spots, (left / fw, top / fh, (left + side) / fw, (top + side) / fh), fetch)
    frame = region_frame(
        small, patch, (left / fw, top / fh, (left + side) / fw, (top + side) / fh)
    )
    # The framing was spent finding the point; the patch itself is not re-cropped.
    tone = {k: val for k, val in clean.items() if k not in GEOMETRY_KEYS}
    return develop_frame(frame, tone)


# ── Getting the next frame ready ───────────────────────────────────
#
# Stepping through a shoot in the develop view opens one photograph after
# another, and each paid its whole decode on arrival. While one is being looked
# at, the ones either side of it are decoded on a background thread, and the
# one on screen at full resolution for the loupe, so stepping or zooming finds
# them ready. Only the latest request matters: a new one replaces whatever was
# still waiting, and the frame being decoded is finished (it is kept either way).
_PREFETCH_LOCK = threading.Lock()
_PREFETCH_WAITING: list = []
_PREFETCH_THREAD: Optional[threading.Thread] = None


def prefetch(sources, size: int, full: Optional[Path] = None) -> None:
    """Decode ``sources`` at ``size``, then ``full`` at full resolution, in the
    background, so the develop view finds them in the caches."""
    global _PREFETCH_THREAD
    jobs = [("preview", Path(s), int(size)) for s in sources]
    if full is not None:
        jobs.append(("full", Path(full), 0))
    with _PREFETCH_LOCK:
        _PREFETCH_WAITING[:] = jobs
        if _PREFETCH_THREAD is None or not _PREFETCH_THREAD.is_alive():
            _PREFETCH_THREAD = threading.Thread(target=_prefetch_worker, name="develop-prefetch",
                                                daemon=True)
            _PREFETCH_THREAD.start()


def _prefetch_worker() -> None:
    global _PREFETCH_THREAD
    while True:
        with _PREFETCH_LOCK:
            if not _PREFETCH_WAITING:
                _PREFETCH_THREAD = None
                return
            kind, source, size = _PREFETCH_WAITING.pop(0)
        try:
            if kind == "full":
                _full_linear(source)
            else:
                linear_base(source, size)
        except Exception as e:
            logger.debug(f"Prefetch of {source.name} failed: {e}")


def free_path(folder: Path, stem: str, ext: str) -> Path:
    """``folder/stem.ext``, or ``stem-2.ext`` and so on when that name is taken.

    An export never writes over a file. Two cameras both shoot IMG_0001, and on
    Windows ``IMG_0001.jpg`` *is* the camera's ``IMG_0001.JPG``: overwriting by
    name is how an export would destroy an original, which nothing else in this
    app can do.
    """
    path = folder / (stem + ext)
    n = 2
    while path.exists():
        path = folder / f"{stem}-{n}{ext}"
        n += 1
    return path


def _write_exif(
    output_path: Path,
    photographer: str,
    strip: bool,
    exif_source: Optional[Path] = None,
) -> None:
    """Use exiftool to strip/embed EXIF on the written JPEG.

    ``exif_source`` is the file the dates came from, which is not the output
    whenever the pixels came from a RAW: rawpy writes pixels and nothing else,
    so without a copy step a RAW-sourced export loses DateTimeOriginal and the
    delivered files stop carrying the day they were taken.
    """
    if not strip and not photographer:
        return
    args = ["exiftool", "-overwrite_original"]
    if strip:
        # Strip all but keep the dates. If the output has no dates to keep,
        # they are copied from the file the photo was shot into.
        keep_from = "@"
        if exif_source and exif_source != output_path and exif_source.is_file():
            keep_from = str(exif_source)
        args += ["-all=", "-tagsFromFile", keep_from,
                 "-DateTimeOriginal", "-CreateDate"]
    if photographer:
        args += [f"-Artist={photographer}", f"-Copyright={photographer}"]
    args.append(str(output_path))
    try:
        subprocess.run(args, capture_output=True, timeout=30, check=False)
    except FileNotFoundError:
        logger.warning("exiftool not found — skipping EXIF processing")
    except Exception as e:
        logger.warning(f"exiftool EXIF processing failed: {e}")


def _develop_linear(
    source: Path,
    template: "ExportTemplate",
    develop: dict,
    match: Optional["MatchSettings"],
) -> Optional[Image.Image]:
    """Load, resize and edit a frame in linear light; ``None`` if it cannot.

    ``None`` sends the caller to the 8-bit path, which can still write the file
    without the edit's rescue controls — better than failing an export.
    """
    counted = match.applied if match else 0
    try:
        from adjust.develop import describe_settings
        from adjust.render import develop_frame, resize_linear

        rgb = _load_linear(source, template.max_resolution)
        if rgb is None:
            return None
        frame = frame_of(source, resize_linear(rgb, template.max_resolution))
        _with_camera_look(source, frame)
        _with_denoiser(source, frame, template.max_resolution)

        matched = None
        if match and match.target:
            try:
                from analysis.harmonize import correction
                from analysis.image_stats import compute_image_stats

                matched = correction(
                    compute_image_stats(frame.encoded()), match.target, match.strength
                )
                if matched:
                    match.applied += 1
            except Exception as e:
                logger.warning(f"Matching skipped for {source.name}: {e}")

        out = develop_frame(frame, develop, matched)
        logger.info(f"Developed {source.name}: {describe_settings(develop)}")
        return out
    except Exception as e:
        logger.warning(f"Develop skipped for {source.name}: {e}")
        if match:
            match.applied = counted  # the 8-bit path counts it again
        return None


def export_photo(
    source: Path,
    output_dir: Path,
    template: ExportTemplate,
    exif_source: Optional[Path] = None,
    match: Optional[MatchSettings] = None,
    develop: Optional[dict] = None,
    suffix: str = "",
    resize_only: bool = False,
) -> Path:
    """
    Export a single photo: load → resize → correct (match + edit set) → save → EXIF.

    Noise reduction and sharpening are part of the edit set (see :mod:`adjust.noise`
    and :mod:`adjust.sharpen`), not batch settings.

    ``exif_source`` names the file the dates are copied from. It defaults to
    ``source``, which is correct for both paths: a JPEG keeps its own EXIF, and
    a RAW never had any in the decoded pixels to begin with.

    ``match`` nudges this frame towards the brightness and white balance of the
    set it belongs to. It is optional because a set of one has nothing to match.

    ``develop`` is this frame's own edit set (see :mod:`adjust.develop`). It is
    performed by :mod:`adjust.render` in linear light on a decode with headroom,
    and the match, if there is one, is composed into the same exposure and white
    balance step rather than applied as a separate 8-bit pass. A frame with no
    edit takes the plain 8-bit path, so an unedited export is byte-for-byte what
    it was.

    ``suffix`` goes between the name and the extension: a virtual copy of a frame
    is written beside the frame itself rather than over it.

    ``resize_only`` writes the photograph as it stands, resized: a frame that has
    been developed is written as it was developed (:func:`_develop_linear`, the
    edit on top of the camera's learned look), and one that has not is the camera's
    own picture (:func:`camera_image`). Neither is matched to the set. This is
    how the export panel writes every photo. Use it instead
    of passing empty settings: an empty edit on a RAW is the flat decode, which is
    a different picture again.

    Returns the output path on success.
    Raises RuntimeError on failure.
    """
    from adjust.develop import normalize
    from config import is_video

    if is_video(source.suffix):
        # A video is exported as it is: copied, with its own dates, which are
        # what Google Photos sorts by.
        output_path = free_path(output_dir, source.stem + suffix, source.suffix.lower())
        shutil.copy2(source, output_path)
        logger.info(f"Exported {source.name} as it is (a video)")
        return output_path

    develop = normalize(develop)
    img = None

    if resize_only:
        # "None": the photograph as it stands, at the size asked for. An edit is
        # not batch processing — it belongs to the photograph — so a frame that
        # has been developed is written as it was developed, and a frame that has
        # not is the camera's own picture, resized.
        if develop:
            img = _develop_linear(source, template, develop, None)
        if img is None:
            img = camera_image(source, template.max_resolution)
            if img is not None and develop:
                # The edit could not be rendered in linear light; the framing can
                # still be, as it is on the ordinary path.
                from adjust.geometry import apply_geometry

                img = apply_geometry(img, develop)
        if img is None:
            raise RuntimeError(f"Could not load {source.name}")
        output_path = free_path(output_dir, source.stem + suffix, ".jpg")
        img.save(output_path, format="JPEG", quality=template.quality)
        _write_exif(
            output_path,
            template.photographer,
            template.strip_exif,
            exif_source or source,
        )
        logger.info(f"Exported {source.name} as it stands, resized")
        return output_path

    if develop or source.suffix.lower() in _RAW_EXTENSIONS:
        # A hand edit is made in linear light, from a decode with headroom: that
        # is what lets exposure, highlights and shadows rescue a frame instead
        # of re-scaling one that has already been clipped. Resized in linear
        # light too, before the edit, so it works on the ~2 MP that is written.
        #
        # A RAW with *no* edit takes the same road. What the plain decoder renders
        # from the sensor data is not the camera's picture either — it carries none
        # of the learned in-camera look, so it came out a stop dark and coloured
        # unlike the JPEG beside it. Edit or no edit, the frame is given the
        # camera's own rendering first, and matching goes on top of that.
        img = _develop_linear(source, template, develop, match)

    if img is None:
        img = _load_image(source, template.max_resolution)
        if img is None:
            raise RuntimeError(f"Could not load {source.name}")

        # Resize first: matching measures the pixels about to be written.
        img = _resize_image(img, template.max_resolution)

        # The linear path could not take this frame, but the framing still can.
        if develop:
            from adjust.geometry import apply_geometry

            img = apply_geometry(img, develop)

        # Match against the set, measured on the pixels about to be written so it
        # corrects what the decoder actually produced rather than what was
        # expected of it.
        if match and match.target:
            try:
                from analysis.harmonize import apply_correction, correction
                from analysis.image_stats import compute_image_stats

                matched = correction(
                    compute_image_stats(img), match.target, match.strength
                )
                if matched:
                    match.applied += 1
                    img = apply_correction(img, matched)
            except Exception as e:
                logger.warning(f"Matching skipped for {source.name}: {e}")

    # Save JPEG
    output_path = free_path(output_dir, source.stem + suffix, ".jpg")
    img.save(output_path, format="JPEG", quality=template.quality)

    # EXIF post-processing
    _write_exif(
        output_path,
        template.photographer,
        template.strip_exif,
        exif_source or source,
    )

    return output_path
