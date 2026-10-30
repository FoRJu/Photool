"""
Masks of one thing in the picture, chosen by pointing at it: a box dragged
round it, clicks on it (and Alt-clicks on what is not it), as Segment Anything
takes them.

  sam2.1  SAM 2.1 Hiera base+ (Meta), Apache-2.0, 330 MB. The default.
  sam3    SAM 3's tracker: the same prompts, a larger image encoder that keeps
          small and thin things (a drummer across a stage, a trombone's slide)
          better. Under Meta's SAM License; 960 MB, fetched only when chosen.

A model reads a *view* of the frame (all of it, or the part the develop view is
zoomed into) at about 1024 px, and answers each prompt with three candidate
selections at a quarter of that, each with the model's own score: one click is
often several things (her hair, or her), so the person steps through them.
Reading the view is the slow part (0.2 s for sam2.1 on a mid-range graphics
card, 0.4 s for sam3); it is kept, and each prompt then costs about 10 ms.

Unlike the subject and sky masks (adjust.segment), the selection is not worked
out again at render time. The same click on the same view decoded at another
size can choose another candidate altogether, so a mask chosen in the develop
view must be exactly the mask exported: it is kept as a picture, a grey PNG of
the whole frame in ``mattes_dir`` named by its own hash, and the mask stores
that name with the prompts that made it.

A mask's selection is the union of its *parts*, each a prompt on its own view:
the drummer clicked at 4x, then his cymbal at 2x. Each part's answer is brought
to the picture's edges with a guided filter over its own view, so a part chosen
zoomed in keeps its fine edge.
"""

import hashlib
import io
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

_SAM21 = ("https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX/resolve/"
          "bab18593f44e652f04cf18b60b3690f60e8996b0/onnx/")
_SAM3 = ("https://huggingface.co/onnx-community/sam3-tracker-ONNX/resolve/"
         "429305c8a5b3de597243d919a07e4e6bdcd00ef7/onnx/")


def _spec(folder: str, base: str, name: str, sha256: str, size: int) -> ModelSpec:
    # Kept in a folder of their own, under their own names: a graph finds its
    # weights (the ``.onnx_data`` beside it) by the name it was saved with.
    return ModelSpec(name=f"{folder}/{name}", url=base + name, sha256=sha256, size=size)


MODELS = {
    "sam2.1": {
        "label": "SAM 2.1",
        "input": 1024,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "encoder": "sam2.1-base-plus/vision_encoder.onnx",
        "decoder": "sam2.1-base-plus/prompt_encoder_mask_decoder.onnx",
        "files": (
            _spec("sam2.1-base-plus", _SAM21, "vision_encoder.onnx",
                  "6fd992cf796e6ad5a29f3b92a127f2ce33f7f320d63c8b3f02bf3775f5f692f4", 702836),
            _spec("sam2.1-base-plus", _SAM21, "vision_encoder.onnx_data",
                  "f8a5f1b14d02eb7ed3b3576ad6774b8ebcfef6cc20674d5f316d6144c60a08a5", 305746496),
            _spec("sam2.1-base-plus", _SAM21, "prompt_encoder_mask_decoder.onnx",
                  "f39eeec20243ed1c8f2cd013812e77813d937ddbc800fa4bc703761adc7e63cd", 213114),
            _spec("sam2.1-base-plus", _SAM21, "prompt_encoder_mask_decoder.onnx_data",
                  "445cd3f72a218815db10e336f4f1c46a6eb2713a0160a85af5365134607f32a7", 20958208),
        ),
    },
    "sam3": {
        "label": "SAM 3",
        "input": 1008,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "encoder": "sam3-tracker/vision_encoder_fp16.onnx",
        "decoder": "sam3-tracker/prompt_encoder_mask_decoder.onnx",
        "files": (
            _spec("sam3-tracker", _SAM3, "vision_encoder_fp16.onnx",
                  "f8c52be6de99124bb17f25792054406abc482ce3502166f948d5c5849cd14d02", 1263005),
            _spec("sam3-tracker", _SAM3, "vision_encoder_fp16.onnx_data",
                  "4b021a4d3068c9e9153f2f2fcc9e6d280156228671c00ab6c2ddfa737da7f151", 934729728),
            _spec("sam3-tracker", _SAM3, "prompt_encoder_mask_decoder.onnx",
                  "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8", 213114),
            _spec("sam3-tracker", _SAM3, "prompt_encoder_mask_decoder.onnx_data",
                  "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae", 22072320),
        ),
    },
}
DEFAULT_MODEL = "sam2.1"
# The size of the picture an object is chosen on and kept at (its longer side):
# the develop view's largest preview, so the decode is usually one already held.
# A view zoomed in further than the model's own size reaches is read from a
# larger decode instead: the drummer across a stage, read off the preview, is a
# few hundred pixels stretched to 1024, and the model takes his hand for him.
PICTURE_SIZE = 2560
ZOOMED_PICTURE_SIZE = 4096
# Views kept read, by frame, view and model: zooming back out, or a second part
# on the same view, does not read it again. Each is about 17 MB.
_EMBED_LIMIT = 6
_MATTE_LIMIT = 8

_models_dir: Optional[str] = None
_mattes_dir: Optional[str] = None
_sessions: dict = {}
_failed: dict = {}
_lock = threading.Lock()
# DirectML runs one model call at a time per session: two at once (a click made
# while the last one is being answered) crash the process. Every model call
# takes this lock.
_run_lock = threading.Lock()
_embeds: "OrderedDict[tuple, list]" = OrderedDict()
_mattes: "OrderedDict[str, np.ndarray]" = OrderedDict()
_cache_lock = threading.Lock()


def configure(models_dir: str, mattes_dir: Optional[str] = None) -> None:
    global _models_dir, _mattes_dir
    if models_dir != _models_dir:
        _models_dir = models_dir
        _sessions.clear()
        _failed.clear()
    if mattes_dir != _mattes_dir:
        _mattes_dir = mattes_dir
        with _cache_lock:
            _mattes.clear()


def model_name(model) -> str:
    return model if model in MODELS else DEFAULT_MODEL


def ensure(model: str, download: bool = True) -> Path:
    """The folder holding ``model``'s files, each fetched if missing and checked
    against its hash. Raises :class:`ModelsUnavailable` with a reason to show."""
    model = model_name(model)
    if not _models_dir:
        raise ModelsUnavailable("no models folder is configured")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ModelsUnavailable("onnxruntime is not installed (pip install onnxruntime-directml)")
    for spec in MODELS[model]["files"]:
        target = Path(_models_dir) / spec.name
        if verify(target, spec):
            continue
        if not download:
            raise ModelsUnavailable(f"{spec.name} is not in {_models_dir} yet")
        if model in _failed:
            raise ModelsUnavailable(_failed[model])
        try:
            _download(spec, target)
        except ModelsUnavailable as exc:
            _failed[model] = str(exc)
            raise
    return Path(_models_dir)


def available(model: str) -> bool:
    """Whether ``model`` can choose without a download."""
    try:
        ensure(model, download=False)
        return True
    except ModelsUnavailable:
        return False


def download_size(model: str) -> int:
    return sum(spec.size for spec in MODELS[model_name(model)]["files"])


def _session_pair(model: str):
    model = model_name(model)
    with _lock:
        pair = _sessions.get(model)
        if pair is None:
            import onnxruntime as ort

            folder = ensure(model)
            have = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                     "CoreMLExecutionProvider") if p in have] + ["CPUExecutionProvider"]
            options = ort.SessionOptions()
            options.log_severity_level = 3
            # As DirectML asks: no memory pattern, one node at a time.
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            spec = MODELS[model]
            pair = tuple(ort.InferenceSession(str(folder / spec[part]), sess_options=options,
                                              providers=providers)
                         for part in ("encoder", "decoder"))
            logger.info("[objects] %s loaded on %s", spec["label"], pair[0].get_providers()[0])
            _sessions[model] = pair
    return pair


# ── The picture ─────────────────────────────────────────────────────────


def picture_size(parts: list) -> int:
    """The decode a mask's parts are chosen on: the larger one when any part's
    view, read off the usual picture, would hold fewer pixels than the model
    reads."""
    for part in parts:
        left, top, right, bottom = part["view"]
        if max(right - left, bottom - top) * PICTURE_SIZE < 1024:
            return ZOOMED_PICTURE_SIZE
    return PICTURE_SIZE


def picture(frame) -> np.ndarray:
    """The frame (decoded at :func:`picture_size`) as a viewer sees it, as 8-bit
    RGB: what the model reads and what the selection's edges follow. Made once
    per frame."""
    kept = frame._mattes.get("_object_rgb8")
    if kept is None:
        from adjust.segment import _small_picture

        kept = frame._mattes["_object_rgb8"] = _small_picture(frame, ZOOMED_PICTURE_SIZE)
    return kept


def _view_box(view, h: int, w: int) -> tuple:
    """A view (left, top, right, bottom fractions) as whole pixels of the picture,
    at least a few pixels each way."""
    left, top, right, bottom = view
    x0 = int(np.clip(np.floor(left * w), 0, w - 1))
    y0 = int(np.clip(np.floor(top * h), 0, h - 1))
    x1 = int(np.clip(np.ceil(right * w), x0 + 8, w))
    y1 = int(np.clip(np.ceil(bottom * h), y0 + 8, h))
    return max(0, min(x0, x1 - 8)), max(0, min(y0, y1 - 8)), x1, y1


def _embedding(rgb8: np.ndarray, view: tuple, model: str, key) -> list:
    """The model's reading of one view of the picture, kept by ``key``."""
    model = model_name(model)
    full_key = (key, tuple(round(v, 5) for v in view), model)
    if key is not None:
        with _cache_lock:
            kept = _embeds.get(full_key)
            if kept is not None:
                _embeds.move_to_end(full_key)
                return kept
    with _run_lock:
        if key is not None:
            # A second click on the same view, sent while the first was reading it.
            with _cache_lock:
                kept = _embeds.get(full_key)
            if kept is not None:
                return kept
        return _read_view(rgb8, view, model, key, full_key)


def _read_view(rgb8: np.ndarray, view: tuple, model: str, key, full_key) -> list:
    """Run the image encoder over one view. Called with ``_run_lock`` held."""
    import cv2

    spec = MODELS[model]
    h, w = rgb8.shape[:2]
    x0, y0, x1, y1 = _view_box(view, h, w)
    crop = rgb8[y0:y1, x0:x1]
    n = spec["input"]
    # The model stretches the view to its square, as it was trained to.
    small = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA if max(crop.shape[:2]) > n
                       else cv2.INTER_CUBIC)
    x = (small.astype(np.float32) / 255.0 - np.array(spec["mean"], np.float32)) / np.array(spec["std"], np.float32)
    encoder, _ = _session_pair(model)
    feats = encoder.run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})
    if key is not None:
        with _cache_lock:
            _embeds[full_key] = feats
            while len(_embeds) > _EMBED_LIMIT:
                _embeds.popitem(last=False)
    return feats


def predict(rgb8: np.ndarray, part: dict, model: str, key=None) -> tuple:
    """One part's three candidates over its view: ``(logits, scores)``, logits
    (3, h, w) at the model's mask size, scores the model's guess of each one's
    quality."""
    model = model_name(model)
    spec = MODELS[model]
    view = part["view"]
    feats = _embedding(rgb8, view, model, key)
    n = spec["input"]
    left, top, right, bottom = view
    sx, sy = n / max(right - left, 1e-6), n / max(bottom - top, 1e-6)
    pts = part.get("points") or []
    points = np.array([[[[(x - left) * sx, (y - top) * sy] for x, y, _ in pts]]],
                      np.float32).reshape(1, 1, len(pts), 2)
    labels = np.array([[[int(lab) for *_, lab in pts]]], np.int64).reshape(1, 1, len(pts))
    box = part.get("box")
    boxes = (np.array([[[(box[0] - left) * sx, (box[1] - top) * sy,
                         (box[2] - left) * sx, (box[3] - top) * sy]]], np.float32)
             if box else np.zeros((1, 0, 4), np.float32))
    _, decoder = _session_pair(model)
    feeds = {f"image_embeddings.{i}": f for i, f in enumerate(feats)}
    with _run_lock:
        scores, masks, _ = decoder.run(None, {**feeds, "input_points": points, "input_labels": labels,
                                              "input_boxes": boxes})
    return masks[0, 0].astype(np.float32), scores[0, 0].astype(np.float32)


def chosen(scores: np.ndarray, pick) -> int:
    """The candidate a part takes: the one picked, or the model's best."""
    if pick is not None and 0 <= int(pick) < len(scores):
        return int(pick)
    return int(np.argmax(scores))


def _part_matte(rgb8: np.ndarray, grey: np.ndarray, part: dict, model: str, key) -> np.ndarray:
    """One part's selection over the whole picture, 0..1, brought to the edges of
    its own view."""
    import cv2

    from adjust.segment import _guided

    h, w = rgb8.shape[:2]
    logits, scores = predict(rgb8, part, model, key)
    x0, y0, x1, y1 = _view_box(part["view"], h, w)
    vw, vh = x1 - x0, y1 - y0
    rough = cv2.resize(logits[chosen(scores, part.get("pick"))], (vw, vh), interpolation=cv2.INTER_LINEAR)
    rough = 1.0 / (1.0 + np.exp(-np.clip(rough, -30.0, 30.0)))
    # Followed to the picture's own edges, on the view's scale: a part chosen at
    # 4x keeps a finer edge than one chosen on the whole frame.
    radius = max(1, round(max(vw, vh) / 384))
    fine = np.clip(_guided(grey[y0:y1, x0:x1], rough.astype(np.float32), radius, 1e-3), 0.0, 1.0)
    t = np.clip((fine - 0.25) / 0.5, 0.0, 1.0)
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = t * t * (3.0 - 2.0 * t)
    return out


def selection(rgb8: np.ndarray, parts: list, model: str, key=None) -> np.ndarray:
    """The whole picture's selection for a mask's parts, 0..1, float32."""
    import cv2

    grey = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out = np.zeros(rgb8.shape[:2], np.float32)
    for part in parts:
        np.maximum(out, _part_matte(rgb8, grey, part, model, key), out=out)
    return out


# ── Kept selections ─────────────────────────────────────────────────────


def _matte_path(matte_id: str) -> Optional[Path]:
    if not _mattes_dir or not matte_id:
        return None
    return Path(_mattes_dir) / matte_id[:2] / f"{matte_id}.png"


def save(matte: np.ndarray) -> str:
    """Keep a selection; its name is its own hash, so saving it twice is once."""
    from PIL import Image

    if not _mattes_dir:
        raise ModelsUnavailable("no folder is configured for chosen selections")
    buf = io.BytesIO()
    Image.fromarray((np.clip(matte, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L").save(
        buf, format="PNG", optimize=True)
    data = buf.getvalue()
    matte_id = hashlib.sha256(data).hexdigest()[:32]
    path = _matte_path(matte_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(path)
    return matte_id


def load(matte_id: str) -> Optional[np.ndarray]:
    """A kept selection, 0..1 float32, or ``None`` when it is not here (an edit
    from another computer whose catalog folder did not come with it)."""
    if not matte_id:
        return None
    with _cache_lock:
        kept = _mattes.get(matte_id)
        if kept is not None:
            _mattes.move_to_end(matte_id)
            return kept
    path = _matte_path(matte_id)
    if path is None or not path.exists():
        return None
    from PIL import Image

    try:
        with Image.open(path) as im:
            matte = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    except OSError as exc:
        logger.warning("[objects] cannot read %s: %s", path, exc)
        return None
    with _cache_lock:
        _mattes[matte_id] = matte
        while len(_mattes) > _MATTE_LIMIT:
            _mattes.popitem(last=False)
    return matte


def png(matte: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.clip(matte, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L").save(buf, format="PNG")
    return buf.getvalue()
