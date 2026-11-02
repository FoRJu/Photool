"""
How far away each part of a picture is, and what can be done with that.

Depth Anything V2 Small (Apache-2.0; its larger versions are for non-commercial use only)
reads depth from the picture alone, so it works on any photograph from any camera. What it
gives is relative, not metres: this frame's nearest and farthest are 100 and 0, and two
frames' numbers are not comparable. It reads about 1000 px of the frame in about half a
second on a graphics card, and the map is kept on the frame.

Two things use it:

  * a ``depth`` mask (adjust.masks): any local adjustment, reaching a band of distance.
  * lens blur (:func:`blurred`): what a wider aperture would have done.

Lens blur is done in linear light, so a bright point spreads into a bright disc rather than
a grey smudge, and in layers from far to near. A disc blurs each layer of distance by as
much as it is out of focus, and each is laid over those behind it, its blurred edge letting
them show through. So a sharp subject does not bleed into the blurred background behind it,
and something blurred in front spreads over what is behind. Only at the end is the picture
evened out by how much of each pixel was covered, which fills what a blurred layer left
thin beside a nearer one. Doing that layer by layer would stretch each one to the full width
of its disc and stop it dead there, drawing a hard line. What a frame cannot show, it cannot
blur: fine hair against a busy background follows the map's edge, which is soft by a few
pixels.

Dual-pixel data, which some Canon bodies can record, could give a second reading of depth
later. Nothing here assumes where the map came from.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from adjust import gpu
from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

# Pinned to one revision: a file changed under the same name would fail its hash.
MODEL = ModelSpec(
    name="depth_anything_v2_small.onnx",
    url="https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/"
        "4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx",
    sha256="afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
    size=99060839,
)
# The model's input size must be a multiple of this.
_PATCH = 14
# How much of the frame the model reads (its longer side). Its own default of 518 is smoother;
# this keeps a railing and the edge of a statue.
READ_SIZE = 1036
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# A wide lens wide open, at the far end of the slider: the disc's radius as a share of the
# longer side, for what is as far from focus as anything can be.
MAX_BLUR = 0.035
# Layers of distance. Each costs about the same; fewer shows as steps along a floor.
LAYERS = 12
# While a slider is held: half the time, and the full picture follows when it is let go.
DRAFT_LAYERS = 6
# How much on each side of focus stays sharp when not said, on the 0..1 scale.
DEFAULT_SHARP = 0.04
# Below this radius (pixels) a layer is left sharp.
_MIN_RADIUS = 0.7
# A layer whose disc is larger than this (pixels) is blurred at reduced size: its result is
# smooth, so nothing is lost, and the cost of a disc grows with its area.
_WORKING_RADIUS = 6

_models_dir: Optional[str] = None
_session = None
_failed: Optional[str] = None
# One download at a time: the panel asks for the model as a render comes to need it.
_fetching = threading.Lock()


def configure(models_dir: Optional[str]) -> None:
    global _models_dir, _failed
    if models_dir != _models_dir:
        gpu.release_all()
        _models_dir = models_dir
        _failed = None


def _forget_session() -> None:
    global _session
    _session = None


gpu.register(_forget_session)


def ensure(download: bool = True) -> Path:
    """The model file, fetched if it is missing. Raises :class:`ModelsUnavailable`."""
    global _failed
    if not _models_dir:
        raise ModelsUnavailable("no models folder is configured")
    target = Path(_models_dir) / MODEL.name
    if verify(target, MODEL):
        return target
    if not download:
        raise ModelsUnavailable(f"{MODEL.name} is not in {_models_dir} yet")
    with _fetching:
        if verify(target, MODEL):
            return target
        if _failed:
            raise ModelsUnavailable(_failed)
        try:
            _download(MODEL, target)
        except ModelsUnavailable as exc:
            _failed = str(exc)
            raise
    return target


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    try:
        ensure(download=False)
        return True
    except ModelsUnavailable:
        return False


def _get_session():
    """The model, opened when wanted. Called with ``gpu.LOCK`` held, and not to be kept
    beyond it (see adjust.gpu)."""
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError:
        raise ModelsUnavailable("onnxruntime is not installed (pip install onnxruntime-directml)")
    path = ensure()
    have = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in have] + ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    logger.info("[depth] model loaded on %s", _session.get_providers()[0])
    return _session


def _read(picture: np.ndarray) -> np.ndarray:
    """The model's depth for an 8-bit RGB picture, at the picture's size: 0 far, 1 near."""
    import cv2

    h, w = picture.shape[:2]
    k = READ_SIZE / max(h, w)
    nw, nh = max(_PATCH, round(w * k / _PATCH) * _PATCH), max(_PATCH, round(h * k / _PATCH) * _PATCH)
    x = cv2.resize(picture, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    with gpu.LOCK:
        session = _get_session()
        out = session.run(None, {session.get_inputs()[0].name: x})[0]
        del session
    out = np.squeeze(out).astype(np.float32)
    # The nearest and farthest of this frame, past a stray pixel or two.
    low, high = np.percentile(out, [1.0, 99.0])
    out = np.clip((out - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def whole_map(frame) -> np.ndarray:
    """How near each part of the whole frame is, 0 far to 1 near, as float32 about the size
    of a matte. Its edges are brought to the picture's. Kept on the frame."""
    from adjust.segment import MATTE_SIZE, _guided, _small_picture

    whole = getattr(frame, "anchor_of", None) or frame
    kept = whole._mattes.get("depth")
    if kept is not None:
        return kept
    picture = _small_picture(whole, MATTE_SIZE)
    rough = _read(picture)
    grey = (picture.astype(np.float32) / 255.0).mean(axis=2)
    fine = np.clip(_guided(grey, rough, 6, 1e-4), 0.0, 1.0).astype(np.float32)
    whole._mattes["depth"] = fine
    return fine


def _step(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def band(depth: np.ndarray, near: float, far: float, soft: float) -> np.ndarray:
    """How much of a band of distance each part of ``depth`` is in, 0..1. ``near`` and
    ``far`` are 0..100 (100 nearest in this frame), and ``soft`` (0..100) how far it fades
    out beyond them. The ends of the range are not faded: a band up to 100 takes in the
    nearest thing there is."""
    high, low = max(near, far) / 100.0, min(near, far) / 100.0
    fade = max(soft, 0.5) / 100.0 * 0.5
    out = _step(low - fade, low, depth) if low > 0.0 else np.ones_like(depth)
    if high < 1.0:
        out = out * (1.0 - _step(high, high + fade, depth))
    return out.astype(np.float32)


def depth_at(frame, u: float, v: float) -> int:
    """The distance at a point of the whole frame, on the sliders' scale: 1 far to 100 near
    (0 means no choice has been made), from a small area round it, so one stray pixel
    doesn't decide where the focus goes."""
    depth = whole_map(frame)
    h, w = depth.shape
    x, y = int(np.clip(u, 0.0, 1.0) * (w - 1)), int(np.clip(v, 0.0, 1.0) * (h - 1))
    r = max(2, max(h, w) // 200)
    return int(np.clip(round(float(np.median(depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1])) * 100.0), 1, 100))


def auto_focus(depth: np.ndarray) -> float:
    """Where to focus when not told: on what is near, in the middle of the frame, which is
    what most pictures are of. The nearest few percent are left out, since they are as often
    a foreground edge as the subject."""
    h, w = depth.shape
    middle = depth[h // 5:h - h // 5, w // 5:w - w // 5]
    return float(np.percentile(middle, 85.0))


def _disc(radius: float) -> np.ndarray:
    """A lens's blur: a disc of ``radius`` pixels, its edge antialiased, summing to one."""
    size = int(np.ceil(radius)) + 1
    ys, xs = np.mgrid[-size:size + 1, -size:size + 1]
    kernel = np.clip(radius + 0.5 - np.hypot(xs, ys), 0.0, 1.0).astype(np.float32)
    return kernel / kernel.sum()


def _layer(rgb: np.ndarray, mine: np.ndarray, radius: float) -> tuple:
    """One layer through a disc of ``radius`` pixels: its light, and how much of each pixel
    it now covers. Worked at a reduced size when the disc is large, and brought back at the
    end."""
    import cv2

    h, w = rgb.shape[:2]
    shrink = max(1.0, radius / _WORKING_RADIUS)
    disc = _disc(radius / shrink)
    light = cv2.multiply(rgb, cv2.merge((mine, mine, mine)))
    if shrink > 1.0:
        small = (max(1, round(w / shrink)), max(1, round(h / shrink)))
        light = cv2.resize(light, small, interpolation=cv2.INTER_AREA)
        mine = cv2.resize(mine, small, interpolation=cv2.INTER_AREA)
    light = cv2.filter2D(light, -1, disc, borderType=cv2.BORDER_REFLECT)
    cover = cv2.filter2D(mine, -1, disc, borderType=cv2.BORDER_REFLECT)
    if shrink > 1.0:
        light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
        cover = cv2.resize(cover, (w, h), interpolation=cv2.INTER_LINEAR)
    return light, np.clip(cover, 0.0, 1.0)


def lens_blur(rgb: np.ndarray, depth: np.ndarray, focus: float, sharp: float, radius: float,
              layers: int = LAYERS) -> np.ndarray:
    """``rgb`` (linear, H, W, 3) as if shot wider open. ``depth`` is 0 far to 1 near, the same
    size; ``focus`` the distance in focus and ``sharp`` how much stays sharp each side of it
    (both 0..1); ``radius`` the disc's pixels for what is furthest from focus.

    Blur goes with distance on this scale because the map is, near enough, inverse distance,
    and a lens's blur is proportional to the difference of inverse distances."""
    import cv2

    centres = np.linspace(0.0, 1.0, layers)
    position = np.clip(depth, 0.0, 1.0) * (layers - 1.0)
    out = np.zeros_like(rgb, dtype=np.float32)
    covered = np.zeros(rgb.shape[:2], np.float32)
    for i, centre in enumerate(centres):
        # This layer's share of each pixel, far to near; shares add up to one.
        mine = np.maximum(1.0 - np.abs(position - i), 0.0).astype(np.float32)
        if not mine.any():
            continue
        r = radius * float(np.clip((abs(centre - focus) - sharp) / max(1.0 - sharp, 1e-3), 0.0, 1.0))
        if r < _MIN_RADIUS:
            light, cover = cv2.multiply(rgb, cv2.merge((mine, mine, mine))), mine
        else:
            light, cover = _layer(rgb, mine, r)
        through = 1.0 - cover
        out = light + out * through[..., None]
        covered = cover + covered * through
    # What is left thin is evened out, here and not layer by layer (see the top of this file).
    return (out / np.maximum(covered, 1e-3)[..., None]).astype(np.float32)


def reach(amount: float, longest: int) -> int:
    """How many pixels a blur can carry a point at most, on a frame whose longer side is
    ``longest``: what a part of a frame must be cut with, to be blurred like the whole."""
    return int(np.ceil(float(amount) / 100.0 * MAX_BLUR * longest)) + 1


def blurred_pixels(frame, rgb: np.ndarray, window: tuple, amount: float, focus: float = 0,
                   sharp: float = 0, layers: int = LAYERS) -> np.ndarray:
    """``rgb``, which is the part ``window`` of ``frame``'s picture at any resolution, with
    lens blur (see :func:`blurred`). ``rgb`` itself when the model can't be had."""
    from adjust.segment import sample

    try:
        whole = whole_map(frame)
    except ModelsUnavailable as exc:
        logger.warning("[depth] cannot read depth: %s", exc)
        return rgb
    h, w = rgb.shape[:2]
    left, top, right, bottom = window
    depth = np.ascontiguousarray(sample(whole, (h, w), window), dtype=np.float32)
    # In the whole frame's pixels, so a preview and a full-size export agree.
    longest = max(w / max(right - left, 1e-9), h / max(bottom - top, 1e-9))
    at = float(focus) / 100.0 if focus else auto_focus(whole)
    around = float(sharp) / 200.0 if sharp else DEFAULT_SHARP
    return lens_blur(rgb, depth, at, around, float(amount) / 100.0 * MAX_BLUR * longest, layers)


def blurred(frame, amount: float, focus: float = 0, sharp: float = 0):
    """``frame`` with lens blur, as a frame of its own: ``amount`` 0..100, ``focus`` 1..100
    (0 chooses, see :func:`auto_focus`), ``sharp`` 0..100 (how deep a range stays sharp).
    ``frame`` itself when the model can't be had. Kept on the frame for these values."""
    from adjust.render import LinearFrame

    key = (float(amount), float(focus), float(sharp))
    if frame._blurred is not None and frame._blurred[0] == key:
        return frame._blurred[1]
    layers = DRAFT_LAYERS if frame._is_draft else LAYERS
    rgb = blurred_pixels(frame, frame.rgb, frame.window, amount, focus, sharp, layers)
    if rgb is frame.rgb:
        return frame
    out = LinearFrame(rgb, look=frame.look, window=frame.window, scale=frame.scale,
                      anchor_of=frame.anchor_of or frame, lens=frame.lens)
    frame._blurred = (key, out)
    return out
