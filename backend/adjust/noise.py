"""
Noise reduction: a learned denoiser run on the sensor's own data, and the two
sliders that blend its result in.

What the app did before was blur the colour of every RAW by a fixed amount and
leave the luminance noise alone. At ISO 25600 that leaves the colour blotchy and
the grain heavy, and nothing could be done about either. Measured against that,
RawRefinery's TreeNet denoiser (github.com/rymuelle/RawForge, MIT; trained on the
Raw Natural Image Noise Dataset) keeps a shirt's weave, hair and a fence's mesh
while taking the noise out.

How it is used here:

* **On the mosaic, before the decode.** The model is given the sensor's data as
  RawForge gives it — black level off, the camera's colours taken to linear
  Rec.2020 by the file's own matrix, Malvar-demosaiced — and what it returns is
  taken back to the camera's colours and sampled back into a mosaic at the
  file's own levels. That mosaic is then decoded exactly as the original is, so
  the highlight repair, the colour matrix, the camera's look and the lens
  correction all happen as they always have. Where the sensor came near to
  clipping the original data is kept, so the highlight repair still finds the
  clip where it is; and so it is where the colour given to the model went past
  1 (a bright warm colour), so the bright pixels the repair borrows its colour
  from keep theirs.
* **Once per photograph.** The model takes ~10 s on a GPU for 24 MP, so its
  mosaic is kept, in memory and on disk, and every slider move after that is a
  blend of two decodes that are both already there.
* **At one strength.** The model is told the photograph's ISO, which is how hard
  it works. Above ISO 12800 it smooths into flat patches (an R6 II frame at
  25600, measured), so it is never told more than that; the sliders then take
  back as much of the noise as is wanted.

The sliders, 0..100 each:

* ``nr_luma``: how much of the luminance noise is taken out. Below 100 the
  rest is put back, which reads as fine grain rather than as noise.
* ``nr_color``: how much of the colour noise is taken out. At 0 the colour is
  what it was before this existed (the decode's own colour smoothing).
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

NOISE_KEYS = ("nr_luma", "nr_color")

MODEL = ModelSpec(
    name="ShadowWeightedL1_24_deep_500.onnx",     # RawForge's TreeNetDenoiseHeavy
    url="https://github.com/rymuelle/RawForge/releases/download/onnx_v1.0.0/"
        "ShadowWeightedL1_24_deep_500.onnx",
    sha256="0074e2f7bec359c613305811c2c4e7123f738010fb3a98ce67044c48ecfc0129",
    size=21863084,
)
# Bump when what the mosaic is made from changes, so kept mosaics are not reused.
_MOSAIC_VERSION = 2

# The ISO the model is told at most (see above), and what it is told when the
# file does not say.
MAX_CONDITIONING_ISO = 12800
_UNKNOWN_ISO = 3200
# The model was trained on ISO / 6400.
_ISO_SCALE = 6400.0

# Tiles the frame is run in. The model takes any size; larger tiles waste less
# on the overlaps, and a batch keeps the GPU busy.
_TILE = 512
_OVERLAP = 64
_BATCH = 4

# Where the sensor came this near to its white level (a fraction of the range
# above black), the original data is blended back in, wholly by _KEEP_FULL.
_KEEP_FROM = 0.85
_KEEP_FULL = 0.95

# The model is given its colours in 0..1, and the file's matrix can take a
# bright warm colour well past 1: on a Canon CR2 red is multiplied by about 3, so
# a red at a third of the sensor's range is already over. Cut off there, the
# model hands back a red that is too low (-20 % at half the range, -35 % near the
# clip, IMG_6937), and those bright unclipped pixels are what the highlight repair
# takes its colour from, so a rescued highlight came back green. Where the input
# had to be cut, from _CUT_FROM wholly by _CUT_FULL, the original data is kept,
# and a little round it (the model reads its neighbours), softly.
_CUT_FROM = 0.9
_CUT_FULL = 1.0
_CUT_REACH = 4        # pixels
_CUT_SOFTEN = 3.0     # pixels, the blur on the edge of what is kept

# XYZ (D65) to linear Rec.2020, as RawForge's RawHandler has it.
_XYZ_TO_REC2020 = np.array([
    [1.71666343, -0.35567332, -0.25336809],
    [-0.66667384, 1.61645574, 0.0157683],
    [0.01764248, -0.04277698, 0.94224328],
])

# Kept mosaics on disk: one is ~50 MB for 24 MP, so the oldest are let go past this.
_DISK_LIMIT = 10 * 1024 ** 3
_MEMORY_FRAMES = 3

_models_dir: Optional[str] = None
_cache_dir: Optional[Path] = None
_session = None
_session_failed: Optional[str] = None
_lock = threading.Lock()          # the session and the GPU: one frame at a time
_state_lock = threading.Lock()
_memory: "OrderedDict[str, np.ndarray]" = OrderedDict()
_working: dict = {}               # key -> threading.Event, set when done
_failed: dict = {}                # key -> reason


def configure(models_dir: Optional[str], cache_dir: Optional[str]) -> None:
    global _models_dir, _cache_dir, _session, _session_failed
    with _lock:
        _models_dir = models_dir
        _cache_dir = Path(cache_dir) if cache_dir else None
        _session = None
        _session_failed = None
    with _state_lock:
        _memory.clear()
        _failed.clear()


def wanted(settings: Optional[dict]) -> bool:
    """True when an edit asks for any noise reduction."""
    return bool(settings) and any(settings.get(k) for k in NOISE_KEYS)


def blend(rgb: np.ndarray, denoised: np.ndarray, luma: float, color: float) -> np.ndarray:
    """``rgb`` with ``luma`` % of its luminance noise and ``color`` % of its colour
    noise taken away, the noise being what the denoiser took out.

    In linear light, the luminance part of a difference is the grey that carries
    its luma, and the colour part is what is left.
    """
    from adjust.render import LUMA

    diff = rgb - denoised
    grey = (diff @ LUMA)[..., None]
    keep_luma = 1.0 - max(0.0, min(100.0, float(luma))) / 100.0
    keep_color = 1.0 - max(0.0, min(100.0, float(color))) / 100.0
    out = denoised + keep_luma * grey + keep_color * (diff - grey)
    return out.astype(np.float32, copy=False)


# ── The model ───────────────────────────────────────────────────────

def _model_file() -> Path:
    if not _models_dir:
        raise ModelsUnavailable("no models folder is configured")
    target = Path(_models_dir) / MODEL.name
    if not verify(target, MODEL):
        _download(MODEL, target)
    return target


def _get_session():
    """The model, loaded once, on the GPU when onnxruntime has one (DirectML on
    Windows, Core ML on a Mac, CUDA elsewhere), else on the CPU. Called with
    ``_lock`` held."""
    global _session, _session_failed
    if _session is not None:
        return _session
    if _session_failed:
        raise RuntimeError(_session_failed)
    try:
        import onnxruntime as ort
    except ImportError:
        _session_failed = "onnxruntime is not installed (pip install onnxruntime-directml)"
        raise RuntimeError(_session_failed)
    path = _model_file()
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider",
                             "CPUExecutionProvider")
                 if p in available]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    _session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    logger.info(f"Noise reduction model loaded on {_session.get_providers()[0]}")
    return _session


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _models_dir is not None


# ── The sensor's data, as the model wants it ────────────────────────

def _malvar(cfa: np.ndarray) -> np.ndarray:
    """Malvar-He-Cutler demosaicing of an RGGB mosaic, (H, W) -> (H, W, 3),
    as colour_demosaicing does it (mirrored borders)."""
    import cv2

    k = lambda rows: np.array(rows, dtype=np.float32) / 8.0  # noqa: E731
    g_at_rb = k([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])
    rb_at_g_row = k([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]])
    rb_at_g_col = rb_at_g_row.T.copy()
    rb_at_br = k([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]])

    def conv(kernel):
        return cv2.filter2D(cfa, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    h, w = cfa.shape
    red_row = (np.arange(h) % 2 == 0)[:, None]
    red_col = (np.arange(w) % 2 == 0)[None, :]
    blue_row, blue_col = ~red_row, ~red_col
    r_site, b_site = red_row & red_col, blue_row & blue_col
    g_site = ~(r_site | b_site)

    g = np.where(g_site, cfa, conv(g_at_rb))
    along_row, along_col, across = conv(rb_at_g_row), conv(rb_at_g_col), conv(rb_at_br)
    r = np.where(r_site, cfa, 0.0)
    b = np.where(b_site, cfa, 0.0)
    r = np.where(red_row & blue_col, along_row, r)
    r = np.where(blue_row & red_col, along_col, r)
    b = np.where(blue_row & red_col, along_row, b)
    b = np.where(red_row & blue_col, along_col, b)
    r = np.where(b_site, across, r)
    b = np.where(r_site, across, b)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _rggb_offset(raw) -> Optional[tuple]:
    """(dy, dx) that puts a red site at (0, 0) of the visible mosaic, or ``None``
    for a sensor that is not a Bayer RGGB one in some phase."""
    try:
        if raw.color_desc.decode() != "RGBG" or raw.raw_pattern.shape != (2, 2):
            return None
        pattern = "".join("RGBG"[i] for i in raw.raw_pattern.flatten())
    except Exception:
        return None
    return {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}.get(pattern)


def _camera_to_rec2020(raw) -> np.ndarray:
    """The file's camera colours to linear Rec.2020, from its colour matrix.

    The model was trained on Rec.2020, so a file whose matrix LibRaw does not
    expose (some DNGs) is refused rather than given in the camera's own colours."""
    rgb_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3]
    if not np.all(np.isfinite(rgb_xyz)) or abs(np.linalg.det(rgb_xyz)) < 1e-9:
        raise ValueError("the file has no colour matrix to give the model its colours")
    return _XYZ_TO_REC2020 @ np.linalg.inv(rgb_xyz)


def cut_off(rgb: np.ndarray) -> np.ndarray:
    """0..1 per pixel of an (H, W, 3) model input not yet held to 0..1: how far
    the original data is to be kept because the input had to be cut off at 1
    there, or beside such a place (see _CUT_FROM)."""
    import cv2

    over = np.clip((rgb.max(axis=2) - _CUT_FROM) / (_CUT_FULL - _CUT_FROM), 0.0, 1.0).astype(np.float32)
    if not over.any():
        return over
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _CUT_REACH + 1,) * 2)
    over = cv2.dilate(over, reach)
    over = cv2.GaussianBlur(over, (0, 0), _CUT_SOFTEN)
    # The blur would let the middle of a kept place slip below whole.
    return np.clip(over * 2.0, 0.0, 1.0)


def _run_model(rgb: np.ndarray, conditioning: float) -> np.ndarray:
    """The model over a whole (H, W, 3) frame, in overlapping tiles blended
    together; (H, W, 3) float32 back. Called with ``_lock`` held."""
    session = _get_session()
    h0, w0 = rgb.shape[:2]
    if h0 < 16 or w0 < 16:
        rgb = np.pad(rgb, ((0, max(0, 16 - h0)), (0, max(0, 16 - w0)), (0, 0)), mode="edge")
    h, w = rgb.shape[:2]
    tile = min(_TILE, h - h % 16, w - w % 16)
    overlap = min(_OVERLAP, tile // 4)
    step = tile - overlap

    def starts(n):
        out = list(range(0, max(1, n - tile + 1), step))
        if out[-1] != n - tile:
            out.append(n - tile)
        return out

    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    ramp = np.minimum(ramp / max(overlap, 1), 1.0)
    weight = np.outer(ramp, ramp)
    out = np.zeros((h, w, 3), np.float32)
    total = np.zeros((h, w), np.float32)
    places = [(y, x) for y in starts(h) for x in starts(w)]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float16)
    for i in range(0, len(places), _BATCH):
        group = places[i:i + _BATCH]
        batch = np.stack([chw[:, y:y + tile, x:x + tile] for y, x in group])
        cond = np.full((len(group), 1), conditioning, np.float16)
        result = session.run(["output"], {"input": batch, "cond": cond})[0].astype(np.float32)
        for (y, x), piece in zip(group, result):
            out[y:y + tile, x:x + tile] += piece.transpose(1, 2, 0) * weight[..., None]
            total[y:y + tile, x:x + tile] += weight
    return (out / np.maximum(total, 1e-6)[..., None])[:h0, :w0]


def _iso_of(source: Path) -> float:
    try:
        from thumbnails.extract import get_exif_summary

        return float(str(get_exif_summary(source).get("iso") or "").split()[0])
    except Exception:
        return 0.0


def _compute(source: Path) -> np.ndarray:
    """The photograph's mosaic with its noise taken out, at the file's own levels
    and shape (``raw_image_visible``)."""
    import rawpy

    with rawpy.imread(str(source)) as raw:
        offset = _rggb_offset(raw)
        if offset is None:
            raise ValueError("not a Bayer sensor")
        visible = raw.raw_image_visible.copy()
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)
        to_rec2020 = _camera_to_rec2020(raw)

    dy, dx = offset
    h, w = visible.shape
    h2, w2 = (h - dy) - (h - dy) % 2, (w - dx) - (w - dx) % 2
    mosaic = visible[dy:dy + h2, dx:dx + w2].astype(np.float32)

    # Per site: 0 red, 1 green beside red, 3 green beside blue, 2 blue (LibRaw's order).
    site = np.empty((2, 2), np.int64)
    site[:] = [[0, 1], [3, 2]]
    sites = np.tile(site, (h2 // 2, w2 // 2))
    level = (mosaic - black[sites]) / (white - black[sites])

    # The colour transform at quad resolution, as RawHandler does it: each 2x2
    # cell is one camera colour (the greens shared) taken to Rec.2020.
    quad = level.reshape(h2 // 2, 2, w2 // 2, 2).transpose(1, 3, 0, 2).reshape(4, h2 // 2, w2 // 2)
    t = to_rec2020
    to_quad = np.array([
        [t[0, 0], t[0, 1] / 2, t[0, 1] / 2, t[0, 2]],
        [t[1, 0], t[1, 1], 0.0, t[1, 2]],
        [t[1, 0], 0.0, t[1, 1], t[1, 2]],
        [t[2, 0], t[2, 1] / 2, t[2, 1] / 2, t[2, 2]],
    ], dtype=np.float32)
    quad = (to_quad @ quad.reshape(4, -1)).reshape(4, h2 // 2, w2 // 2)
    bayer = quad.reshape(2, 2, h2 // 2, w2 // 2).transpose(2, 0, 3, 1).reshape(h2, w2)
    rgb = _malvar(bayer)
    cut = cut_off(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)

    iso = _iso_of(source) or _UNKNOWN_ISO
    conditioning = min(iso, MAX_CONDITIONING_ISO) / _ISO_SCALE
    with _lock:
        clean = _run_model(rgb, conditioning)

    # Back to the camera's colours, and to one value per site at the file's levels.
    camera = clean @ np.linalg.inv(to_rec2020).T.astype(np.float32)
    channel = np.array([[0, 1], [1, 2]])
    picked = np.take_along_axis(camera, np.tile(channel, (h2 // 2, w2 // 2))[..., None], axis=2)[..., 0]
    denoised = picked * (white - black[sites]) + black[sites]
    keep = np.clip((level - _KEEP_FROM) / (_KEEP_FULL - _KEEP_FROM), 0.0, 1.0)
    keep = np.maximum(keep, cut)
    denoised = denoised * (1.0 - keep) + mosaic * keep

    out = visible.copy()
    out[dy:dy + h2, dx:dx + w2] = np.clip(np.rint(denoised), 0, np.iinfo(visible.dtype).max).astype(visible.dtype)
    return out


# ── Kept mosaics ────────────────────────────────────────────────────

def _key(source: Path) -> Optional[str]:
    try:
        stat = source.stat()
    except OSError:
        return None
    text = f"{os.path.normcase(str(source))}|{stat.st_mtime_ns}|{stat.st_size}|{MODEL.sha256[:12]}|" \
           f"{MAX_CONDITIONING_ISO}|{_MOSAIC_VERSION}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:32]


def _disk_path(key: str) -> Optional[Path]:
    return _cache_dir / f"{key}.npy" if _cache_dir is not None else None


def _remember(key: str, mosaic: np.ndarray) -> None:
    with _state_lock:
        _memory[key] = mosaic
        _memory.move_to_end(key)
        while len(_memory) > _MEMORY_FRAMES:
            _memory.popitem(last=False)


def _from_disk(key: str) -> Optional[np.ndarray]:
    path = _disk_path(key)
    if path is None:
        return None
    try:
        mosaic = np.load(path, allow_pickle=False)
        os.utime(path)           # the oldest-used go first when the space is needed
        return mosaic
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Kept noise reduction {path.name} unreadable: {e}")
        return None


def _to_disk(key: str, mosaic: np.ndarray) -> None:
    path = _disk_path(key)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, mosaic, allow_pickle=False)
        tmp.replace(path)
        kept = sorted(path.parent.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        used = 0
        for p in kept:
            used += p.stat().st_size
            if used > _DISK_LIMIT:
                p.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Noise reduction for a frame not kept: {e}")


def status(source: Path) -> str:
    """``ready``, ``working``, ``failed`` or ``none`` for a photograph."""
    key = _key(source)
    if key is None:
        return "failed"
    with _state_lock:
        if key in _memory:
            return "ready"
        if key in _working:
            return "working"
        if key in _failed:
            return "failed"
    path = _disk_path(key)
    return "ready" if path is not None and path.exists() else "none"


def failure(source: Path) -> str:
    key = _key(source)
    with _state_lock:
        return _failed.get(key, "") if key else ""


def _make(source: Path, key: str, done: threading.Event) -> None:
    try:
        mosaic = _from_disk(key)
        if mosaic is None:
            logger.info(f"Reducing noise in {source.name}")
            mosaic = _compute(source)
            _to_disk(key, mosaic)
        _remember(key, mosaic)
    except Exception as e:
        logger.warning(f"Noise reduction failed for {source.name}: {e}")
        with _state_lock:
            _failed[key] = str(e)
    finally:
        with _state_lock:
            _working.pop(key, None)
        done.set()


def mosaic_for(source: Path, wait: bool = True) -> Optional[np.ndarray]:
    """The photograph's denoised mosaic, or ``None``.

    Kept ones come back at once. Otherwise it is made: on this thread when
    ``wait`` (an export), or on a background one with ``None`` returned meanwhile
    (the develop view, which asks again and shows the frame without it until then).
    A photograph that failed once is not tried again until restart.
    """
    key = _key(source)
    if key is None or not available():
        return None
    with _state_lock:
        mosaic = _memory.get(key)
        if mosaic is not None:
            _memory.move_to_end(key)
            return mosaic
        if key in _failed:
            return None
        done = _working.get(key)
        mine = done is None
        if mine:
            done = _working[key] = threading.Event()
    if mine:
        if wait:
            _make(source, key, done)
        else:
            # A kept one is quick to read, but not quick enough to hold a slider.
            threading.Thread(target=_make, args=(source, key, done), daemon=True,
                             name="noise-reduction").start()
    if not wait:
        return None
    done.wait()
    with _state_lock:
        return _memory.get(key)
