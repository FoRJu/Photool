"""
Masks of one thing in the picture, chosen by pointing at it: a box dragged
round it, clicks on it (and Alt-clicks on what is not it), as Segment Anything
takes them. With SAM 3, also by naming it: "hair", "sky", "dog".

  sam2.1  SAM 2.1 Hiera base+ (Meta), Apache-2.0, 330 MB. The default.
  sam3    SAM 3 (Meta), under Meta's SAM License; 1.8 GB, fetched only when
          chosen. The same prompts, read by a larger image encoder that keeps
          small and thin things (a drummer across a stage, a trombone's slide)
          better, and words: every instance of what is named, each with its own
          selection. One reading of the view serves both (the graphs are our
          own export of Meta's weights, made to run on DirectML).

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

from analysis import clip
from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

_SAM21 = ("https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX/resolve/"
          "bab18593f44e652f04cf18b60b3690f60e8996b0/onnx/")
# Our export: vision (one reading for clicks and words), text and detector
# graphs from Meta's SAM 3 weights, and onnx-community's prompt decoder.
_SAM3 = ("https://huggingface.co/Photool/sam3-ONNX/resolve/"
         "main/")


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
        "license": "https://github.com/facebookresearch/sam3/blob/main/LICENSE",
        "input": 1008,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "encoder": "sam3/vision_fp16.onnx",
        "decoder": "sam3/prompt_encoder_mask_decoder.onnx",
        "text": "sam3/text_fp16.onnx",
        "detector": "sam3/detector.onnx",
        "files": (
            _spec("sam3", _SAM3, "vision_fp16.onnx",
                  "e47a46c3facb14675eaafb2841c592897f84d09e80b6e805d4ec3f0eb1084220", 5780933),
            _spec("sam3", _SAM3, "vision_fp16.onnx.data",
                  "e7fea269dbebea013537810b1a05affa677b3f5053ef6a93c131726e1ce0608e", 921428992),
            _spec("sam3", _SAM3, "prompt_encoder_mask_decoder.onnx",
                  "4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8", 213114),
            _spec("sam3", _SAM3, "prompt_encoder_mask_decoder.onnx_data",
                  "2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae", 22072320),
            _spec("sam3", _SAM3, "text_fp16.onnx",
                  "bc4c4cc5976e2eba35afb2f7f599005dc26fba149d3accfacbb103273f30bb2d", 2261534),
            _spec("sam3", _SAM3, "text_fp16.onnx.data",
                  "57b5e29bd7010dd8888549f49de2524c2437cefc7b9d40841171f194ac651ed0", 706401280),
            _spec("sam3", _SAM3, "detector.onnx",
                  "f90162580a07b018fd2af7db3960b24435d0802b073078b3f8e55e23c1fa9979", 2690377),
            _spec("sam3", _SAM3, "detector.onnx.data",
                  "9d34bac595f6feffb3f01116aebf01c343e87e711a4852557b440af0d5d0dae7", 101799296),
            # Meta's terms, which come with the files.
            _spec("sam3", _SAM3, "LICENSE",
                  "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be", 7353),
            # Its words are CLIP's: the same vocabulary, the files smart search uses.
            clip.MODELS[clip.VOCAB],
            clip.MODELS[clip.MERGES],
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
# on the same view, does not read it again. Each is about 17 MB; SAM 3's reading
# for words is 56 MB more, kept only for the latest two.
_EMBED_LIMIT = 6
_FEATURES_LIMIT = 2
_MATTE_LIMIT = 8
# Words: what counts as found (the model's score for an instance times its
# score for the phrase being in the picture at all), and how many at most.
FIND_THRESHOLD = 0.5
MAX_FOUND = 20
_TEXT_TOKENS = 32
_TEXT_LIMIT = 32
# The graphics card is let go of when nothing has been chosen for a while: SAM 3
# holds 6.5 GB of an 8 GB card while it works (its working memory is kept
# between runs), and loading it again takes a few seconds.
IDLE_RELEASE = 180.0

_models_dir: Optional[str] = None
_mattes_dir: Optional[str] = None
_sessions: dict = {}
_failed: dict = {}
_idle: Optional[threading.Timer] = None
_lock = threading.Lock()
# DirectML runs one model call at a time per session: two at once (a click made
# while the last one is being answered) crash the process. Every model call
# takes this lock.
_run_lock = threading.Lock()
_embeds: "OrderedDict[tuple, dict]" = OrderedDict()
_texts: "OrderedDict[tuple, tuple]" = OrderedDict()
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


def finds(model) -> bool:
    """Whether ``model`` can find things by name."""
    return "detector" in MODELS[model_name(model)]


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


def release() -> None:
    """Let go of every model loaded, and the graphics memory it holds."""
    with _run_lock, _lock:
        if _sessions:
            logger.info("[objects] models released")
        _sessions.clear()


def _keep_awake() -> None:
    """Hold the models for :data:`IDLE_RELEASE` seconds more."""
    global _idle
    if _idle is not None:
        _idle.cancel()
    _idle = threading.Timer(IDLE_RELEASE, release)
    _idle.daemon = True
    _idle.start()


def _sessions_of(model: str, parts: tuple) -> tuple:
    model = model_name(model)
    with _lock:
        _keep_awake()
        missing = [part for part in parts if (model, part) not in _sessions]
        if missing:
            # One model at a time: the other's memory is wanted for this one.
            for other in [k for k in _sessions if k[0] != model]:
                del _sessions[other]
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
            for part in missing:
                session = ort.InferenceSession(str(folder / spec[part]), sess_options=options,
                                               providers=providers)
                logger.info("[objects] %s %s loaded on %s", spec["label"], part, session.get_providers()[0])
                _sessions[(model, part)] = session
        return tuple(_sessions[(model, part)] for part in parts)


def _session_pair(model: str):
    """The image encoder and the prompt decoder."""
    return _sessions_of(model, ("encoder", "decoder"))


def _find_pair(model: str):
    """The text encoder and the detector, loaded the first time words are used."""
    if not finds(model):
        raise ModelsUnavailable(f"{MODELS[model_name(model)]['label']} cannot find things by name: choose SAM 3")
    return _sessions_of(model, ("text", "detector"))


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


def _embedding(rgb8: np.ndarray, view: tuple, model: str, key, need: str = "image_embeddings.0") -> dict:
    """The model's reading of one view of the picture, by output name, kept by
    ``key``. ``need`` is an output the caller reads: one dropped from an older
    reading (the features for words) has the view read again."""
    model = model_name(model)
    full_key = (key, tuple(round(v, 5) for v in view), model)
    if key is not None:
        with _cache_lock:
            kept = _embeds.get(full_key)
            if kept is not None and need in kept:
                _embeds.move_to_end(full_key)
                return kept
    with _run_lock:
        if key is not None:
            # A second click on the same view, sent while the first was reading it.
            with _cache_lock:
                kept = _embeds.get(full_key)
            if kept is not None and need in kept:
                return kept
        return _read_view(rgb8, view, model, key, full_key)


def _read_view(rgb8: np.ndarray, view: tuple, model: str, key, full_key) -> dict:
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
    outs = encoder.run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})
    feats = {o.name: v for o, v in zip(encoder.get_outputs(), outs)}
    # The detector's features are large: kept at half precision, and only for
    # the latest few views.
    feats.update({k: v.astype(np.float16) for k, v in feats.items() if k.startswith("features_")})
    if key is not None:
        with _cache_lock:
            _embeds[full_key] = feats
            while len(_embeds) > _EMBED_LIMIT:
                _embeds.popitem(last=False)
            for old in list(_embeds)[:-_FEATURES_LIMIT]:
                _embeds[old] = {k: v for k, v in _embeds[old].items() if not k.startswith("features_")}
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
    feeds = {f"image_embeddings.{i}": feats[f"image_embeddings.{i}"] for i in range(3)}
    with _run_lock:
        scores, masks, _ = decoder.run(None, {**feeds, "input_points": points, "input_labels": labels,
                                              "input_boxes": boxes})
    return masks[0, 0].astype(np.float32), scores[0, 0].astype(np.float32)


def _tokens(phrase: str) -> tuple:
    """SAM 3's input for a phrase: CLIP's tokens, at most 32, padded with the end
    mark, and which of them are the phrase."""
    ids = clip.tokenize(phrase)
    if len(ids) > _TEXT_TOKENS:
        ids = ids[:_TEXT_TOKENS - 1] + ids[-1:]
    mask = [1] * len(ids) + [0] * (_TEXT_TOKENS - len(ids))
    ids = ids + [ids[-1]] * (_TEXT_TOKENS - len(ids))
    return np.array([ids], np.int64), np.array([mask], np.int64)


def _text_features(phrase: str, model: str) -> tuple:
    """A phrase's encoding, kept: the same word on the next photo is not read again."""
    key = (model, phrase)
    with _cache_lock:
        kept = _texts.get(key)
        if kept is not None:
            _texts.move_to_end(key)
            return kept
    ids, mask = _tokens(phrase)
    text, _ = _find_pair(model)
    with _run_lock:
        features = text.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    with _cache_lock:
        _texts[key] = (features, mask)
        while len(_texts) > _TEXT_LIMIT:
            _texts.popitem(last=False)
    return features, mask


def find(rgb8: np.ndarray, part: dict, model: str, key=None) -> tuple:
    """Every instance of what a part names on its view: ``(logits, scores)``,
    logits (n, h, w) at the detector's mask size, best first, scores 0..1.
    Nothing found is ``n == 0``."""
    model = model_name(model)
    _, detector = _find_pair(model)
    features, mask = _text_features(part["text"], model)
    feats = _embedding(rgb8, part["view"], model, key, need="features_72")
    feeds = {k: feats[k].astype(np.float32) for k in ("features_288", "features_144", "features_72")}
    with _run_lock:
        outs = detector.run(None, {**feeds, "text_features": features, "text_mask": mask})
    got = {o.name: v for o, v in zip(detector.get_outputs(), outs)}
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -30.0, 30.0)))  # noqa: E731
    scores = sig(got["pred_logits"][0].reshape(-1)) * sig(got["presence_logits"].reshape(-1)[0])
    masks = got["pred_masks"][0]
    order = [int(i) for i in np.argsort(-scores)
             if scores[i] > FIND_THRESHOLD and (masks[i] > 0).any()][:MAX_FOUND]
    return masks[order].astype(np.float32), scores[order].astype(np.float32)


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
    x0, y0, x1, y1 = _view_box(part["view"], h, w)
    vw, vh = x1 - x0, y1 - y0
    if part.get("text"):
        # The instances found that the part keeps (all of them, unless it says).
        logits, _ = find(rgb8, part, model, key)
        keep = [i for i in part.get("keep", range(len(logits))) if i < len(logits)]
        if not keep:
            return np.zeros((h, w), np.float32)
        logit = logits[keep].max(axis=0)
    else:
        logits, scores = predict(rgb8, part, model, key)
        logit = logits[chosen(scores, part.get("pick"))]
    rough = cv2.resize(logit, (vw, vh), interpolation=cv2.INTER_LINEAR)
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
