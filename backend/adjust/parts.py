"""
Parts of people: their face skin, eyes, brows, lips, teeth, hair, body skin and
clothes, for a people mask that names ``areas`` (adjust.masks).

Two SegFormer models, each reading close crops of one person at a time, so a
face across a stage is read as closely as a portrait:

  face   jonathandinu/face-parsing (SegFormer-B5 on CelebAMask-HQ): a crop
         round each face, skin, nose, eyes, brows, ears, lips, the mouth
         between them, hair, neck and the collar's cloth. 340 MB.
  body   mattmdjaga/segformer_b2_clothes (SegFormer-B2 on ATR): a crop round
         each body below its face, hair, face, arms, legs and the clothes,
         hat and shoes. 110 MB.

Both are NVIDIA's SegFormer, released for non-commercial use, trained on
research datasets; fetched into ``models_dir`` the first time a part is chosen.

A person is a face the face detector found (adjust.segment.face_boxes), so the
parts are those of the people whose faces are seen, everyone or one of them.
Each crop's answer goes back where it came from on the whole frame, the parts
of everyone taken together. Large parts (skin, hair, clothes) are brought to
the picture's edges with a guided filter as the other chosen masks are; the
small ones (eyes, lips) keep the model's own outline, which a filter the size
of a hair's edge would blur away. Worked out on a 2048 px picture, once per
decoded frame, and kept on it.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

FOLDER = "people-parts"
MODELS = {
    "face": ModelSpec(
        name=f"{FOLDER}/face_parsing.onnx",
        url="https://huggingface.co/jonathandinu/face-parsing/resolve/"
            "758b82e15a0178c9db39c1ff666a8b56e3a550c8/onnx/model.onnx",
        sha256="6d4e67af60ff78184745ebf74cc15163c0adc27d45cdeba31e3a03d1096fb8c3",
        size=340316611,
    ),
    "body": ModelSpec(
        name=f"{FOLDER}/human_parsing.onnx",
        url="https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/"
            "584abc1e1d260e23c0fc627c5217a09b2b461046/onnx/model.onnx",
        sha256="a93a8dac171b5c1fcc53632a8bfc180bfd9759ea69a3e207451bb07f76add54f",
        size=110039290,
    ),
}
DOWNLOAD_MB = round(sum(s.size for s in MODELS.values()) / 1e6)

# The models' classes, as they number them.
FACE_CLASSES = ("background", "skin", "nose", "glasses", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear",
                "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "earring", "necklace", "neck", "cloth")
BODY_CLASSES = ("background", "hat", "hair", "sunglasses", "upper", "skirt", "pants", "dress", "belt",
                "l_shoe", "r_shoe", "face", "l_leg", "r_leg", "l_arm", "r_arm", "bag", "scarf")

# Each area a mask can name: the classes of each model it is made of, and
# whether it is large (brought to the picture's edges) or small (kept as the
# model drew it).
AREAS = {
    "face_skin": ({"face": ("skin", "nose", "l_ear", "r_ear")}, True),
    "eyes": ({"face": ("l_eye", "r_eye")}, False),
    "brows": ({"face": ("l_brow", "r_brow")}, False),
    "lips": ({"face": ("u_lip", "l_lip")}, False),
    "teeth": ({"face": ("mouth",)}, False),
    "hair": ({"face": ("hair",), "body": ("hair",)}, True),
    "body_skin": ({"face": ("neck",), "body": ("l_arm", "r_arm", "l_leg", "r_leg")}, True),
    "clothes": ({"face": ("cloth", "hat"),
                 "body": ("hat", "upper", "skirt", "pants", "dress", "belt", "scarf", "l_shoe", "r_shoe")}, True),
}

INPUT = 512
PICTURE_SIZE = 2048
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_models_dir: Optional[str] = None
_sessions: dict = {}
_failed: dict = {}
_lock = threading.Lock()
# One model call at a time: DirectML crashes the process when one session
# runs on two threads (see adjust.objects).
_run_lock = threading.Lock()


def configure(models_dir: str) -> None:
    global _models_dir
    if models_dir != _models_dir:
        _models_dir = models_dir
        _sessions.clear()
        _failed.clear()


def ensure(download: bool = True) -> Path:
    """The folder holding both models, each fetched if missing and checked
    against its hash. Raises :class:`ModelsUnavailable` with a reason to show."""
    if not _models_dir:
        raise ModelsUnavailable("no models folder is configured")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ModelsUnavailable("onnxruntime is not installed")
    for key, spec in MODELS.items():
        target = Path(_models_dir) / spec.name
        if verify(target, spec):
            continue
        if not download:
            raise ModelsUnavailable(f"{spec.name} is not in {_models_dir} yet")
        if key in _failed:
            raise ModelsUnavailable(_failed[key])
        try:
            _download(spec, target)
        except ModelsUnavailable as exc:
            _failed[key] = str(exc)
            raise
    return Path(_models_dir)


def available() -> bool:
    try:
        ensure(download=False)
        return True
    except ModelsUnavailable:
        return False


def _session(key: str):
    with _lock:
        session = _sessions.get(key)
        if session is None:
            import onnxruntime as ort

            folder = ensure()
            have = ort.get_available_providers()
            # Not Core ML: it splits a graph like this and copies between the
            # parts, slower than the CPU alone (see adjust.objects).
            providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
                         if p in have] + ["CPUExecutionProvider"]
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(folder / MODELS[key].name), sess_options=options,
                                           providers=providers)
            logger.info("[parts] %s parser loaded on %s", key, session.get_providers()[0])
            _sessions[key] = session
    return session


def _parse(key: str, crop: np.ndarray) -> np.ndarray:
    """Each class's probability over ``crop`` (C, h, w), at the crop's size."""
    import cv2

    x = cv2.resize(crop, (INPUT, INPUT), interpolation=cv2.INTER_AREA if min(crop.shape[:2]) > INPUT
                   else cv2.INTER_CUBIC)
    x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    session = _session(key)
    with _run_lock:
        logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    e = np.exp(logits - logits.max(axis=0))
    probs = e / e.sum(axis=0)
    h, w = crop.shape[:2]
    return np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs]).astype(np.float32)


def _face_crop(box: tuple, h: int, w: int) -> tuple:
    """A square round a face box with room for the hair: CelebAMask-HQ's faces
    fill about half their picture."""
    x, y, bw, bh = box
    side = 2.2 * max(bw * w, bh * h)
    cx, cy = (x + bw / 2) * w, (y + bh * 0.4) * h
    return _clip(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, h, w)


def _body_crop(box: tuple, h: int, w: int) -> tuple:
    """A person's body below their face: arms out, down to the feet."""
    x, y, bw, bh = box
    fw, fh = bw * w, bh * h
    cx = (x + bw / 2) * w
    return _clip(cx - 2.6 * fw, y * h - 0.8 * fh, cx + 2.6 * fw, y * h + 8.5 * fh, h, w)


def _clip(x0, y0, x1, y1, h, w) -> tuple:
    return (int(max(0, np.floor(x0))), int(max(0, np.floor(y0))),
            int(min(w, np.ceil(x1))), int(min(h, np.ceil(y1))))


def _picture(frame) -> np.ndarray:
    kept = frame._mattes.get("_parts_rgb8")
    if kept is None:
        from adjust.segment import _small_picture

        kept = frame._mattes["_parts_rgb8"] = _small_picture(frame, PICTURE_SIZE)
    return kept


def _read_people(frame) -> list:
    """Each person's parsing, kept on the frame: for each face, the two crops
    and each model's probabilities over them."""
    kept = frame._mattes.get("_parts_read")
    if kept is not None:
        return kept
    from adjust.segment import face_boxes

    rgb8 = _picture(frame)
    h, w = rgb8.shape[:2]
    people = []
    for box in face_boxes(frame):
        person = {}
        for key, crop_of in (("face", _face_crop), ("body", _body_crop)):
            x0, y0, x1, y1 = crop_of(box, h, w)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            person[key] = ((x0, y0, x1, y1), _parse(key, rgb8[y0:y1, x0:x1]))
        people.append(person)
    frame._mattes["_parts_read"] = people
    return people


def outline(read: dict, h: int, w: int) -> np.ndarray:
    """One person as the parsers see them, 0..1 over the 2048 px picture:
    everything of theirs (hair, face, skin, clothes, a bag) where either
    model reads it, from that person's crops."""
    out = np.zeros((h, w), np.float32)
    for (x0, y0, x1, y1), probs in read.values():
        np.maximum(out[y0:y1, x0:x1], 1.0 - probs[0], out=out[y0:y1, x0:x1])
    return out


# The share of a face box the face parser must read as face (skin, nose, eyes,
# brows, lips, mouth) for the face detector's box to be a face.
FACE_SHARE = 0.25
_FACE_PARTS = ("skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip")


def is_a_face(read: dict, box: tuple, h: int, w: int) -> bool:
    """Whether the face parser agrees that the face detector's ``box`` holds a
    face: a second opinion that drops a hand or a pattern taken for one. True
    when the face parser did not read it."""
    if "face" not in read:
        return True
    (x0, y0, x1, y1), probs = read["face"]
    bx, by, bw, bh = box
    fx0, fy0 = max(int(bx * w), x0), max(int(by * h), y0)
    fx1, fy1 = min(int((bx + bw) * w), x1), min(int((by + bh) * h), y1)
    if fx1 <= fx0 or fy1 <= fy0:
        return True
    face = probs[[FACE_CLASSES.index(c) for c in _FACE_PARTS]].sum(axis=0)
    return float(face[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].mean()) >= FACE_SHARE


def people_read(frame) -> list:
    """Each person's parsing (see _read_people): for adjust.segment."""
    return _read_people(frame)


# Classes that cover skin as often as not: the glasses (the skin seen through
# the lenses) and the lips' outer edge (the parser's lips run onto the skin
# round the mouth). A pixel of them counts as skin when it looks like this
# person's own skin. Only the edge of the lips: pale lips look like the skin
# too, and are not to be evened toward it.
_LIPS = ("u_lip", "l_lip")
# How far from the face's own skin a pixel may be, in its own spread.
_LIKE_SPREAD = 2.5


def skin_matte(frame, person: Optional[int] = None) -> np.ndarray:
    """The skin of everyone, or of one person, for retouching: their face and
    body skin, and what the parsers call glasses or lips but which looks like
    that face's own skin (behind a lens, round the mouth). Compared with the
    same face, not with a colour, so it is the same for any complexion. At the
    2048 px picture's size; kept on the frame."""
    key = ("_skin", int(person or 0))
    kept = frame._mattes.get(key)
    if kept is not None:
        return kept
    base = matte(frame, ("face_skin", "body_skin"), person)
    rgb8 = _picture(frame)
    people = _read_people(frame)
    if person:
        people = people[int(person) - 1:int(person)]
    out = base.copy()
    glasses = FACE_CLASSES.index("glasses")
    lip_classes = [FACE_CLASSES.index(c) for c in _LIPS]
    for read in people:
        if "face" not in read:
            continue
        (x0, y0, x1, y1), probs = read["face"]
        crop = rgb8[y0:y1, x0:x1].astype(np.float32) / 255.0
        sure = base[y0:y1, x0:x1] > 0.8
        if sure.sum() < 50:
            continue
        feats = _skin_features(crop)
        mean = feats[sure].mean(axis=0)
        spread = np.maximum(feats[sure].std(axis=0), (0.04, 0.01, 0.01))
        far = (((feats - mean) / (_LIKE_SPREAD * spread)) ** 2).sum(axis=-1)
        like = _firm(np.exp(-0.5 * far), 0.3, 0.7)
        lips = np.minimum(probs[lip_classes].sum(axis=0), 1.0)
        # The lips' own body: what is lips a little way in from their edge.
        import cv2

        reach = max(1, round((x1 - x0) / 2.2 / 45))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        body = cv2.erode((lips > 0.5).astype(np.uint8), kernel).astype(np.float32)
        body = cv2.GaussianBlur(body, (0, 0), max(0.5, reach / 2))
        maybe = np.maximum(probs[glasses], lips * (1.0 - body))
        np.maximum(out[y0:y1, x0:x1], maybe * like, out=out[y0:y1, x0:x1])
    frame._mattes[key] = out
    return out


def _skin_features(rgb: np.ndarray) -> np.ndarray:
    """Per pixel: its brightness in stops, and its colour as shares of the
    whole (red and blue), which do not change with the light on it."""
    total = rgb.sum(axis=-1) + 1e-4
    return np.stack([np.log2(total / 3.0 + 1e-4), rgb[..., 0] / total, rgb[..., 2] / total], axis=-1)


def matte(frame, areas, person: Optional[int] = None) -> np.ndarray:
    """How much of the named ``areas`` each part of the whole frame is, 0..1,
    float32 at the 2048 px picture's size: everyone's, or with ``person``
    (1 is the leftmost face) one person's. Kept on the frame."""
    import cv2

    from adjust.segment import _guided

    areas = tuple(sorted(a for a in set(areas) if a in AREAS))
    key = ("_parts", areas, int(person or 0))
    kept = frame._mattes.get(key)
    if kept is not None:
        return kept
    rgb8 = _picture(frame)
    h, w = rgb8.shape[:2]
    people = _read_people(frame)
    if person:
        people = people[int(person) - 1:int(person)]
    large = np.zeros((h, w), np.float32)
    small = np.zeros((h, w), np.float32)
    face_px = 0.0
    for read in people:
        for model, ((x0, y0, x1, y1), probs) in read.items():
            if model == "face":
                face_px = max(face_px, (x1 - x0) / 2.2)
            names = FACE_CLASSES if model == "face" else BODY_CLASSES
            for area in areas:
                classes, is_large = AREAS[area]
                wanted = [names.index(c) for c in classes.get(model, ())]
                if not wanted:
                    continue
                p = probs[wanted].sum(axis=0)
                target = large if is_large else small
                np.maximum(target[y0:y1, x0:x1], p, out=target[y0:y1, x0:x1])
    out = np.zeros((h, w), np.float32)
    if large.any():
        grey = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # The filter's reach follows the faces: a hair's edge in a portrait, a
        # few pixels for a band across a stage.
        radius = int(np.clip(round(face_px / 40), 1, 12))
        large = np.clip(_guided(grey, large, radius, 1e-4), 0.0, 1.0)
        out = np.maximum(out, _firm(large, 0.25, 0.75))
    if small.any():
        out = np.maximum(out, _firm(small, 0.3, 0.7))
    frame._mattes[key] = out
    return out


def _firm(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)
