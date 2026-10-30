"""
Masks the picture chooses: its subject, its sky, its people and their faces.

A ``subject`` or ``sky`` mask (adjust.masks) reaches where a model finds one,
worked out from the frame's own pixels, so the edit stores only the kind — not
a picture of the selection — and a frame developed again later is selected
again the same way.

  subject  U²-Net-p, a salient-object model: what a viewer would call the
           subject — the runners in front, not the crowd behind; a dish on a
           table. Apache-2.0, 4.6 MB.
  sky      SegFormer-B0 trained on ADE20K, its "sky" class. NVIDIA's weights
           are released for non-commercial use, which a family album is.
           15 MB.
  people   the same model's "person" class: everyone in the frame, whole.
  faces    the faces in it, found by the face detector the People tab uses
           (YuNet) and drawn as a soft oval each, brought to the picture's edges
           and kept inside the people: light on the faces of a group, softer
           skin with negative clarity.

``people`` and ``faces`` take a ``person``: the one person, counted from the
left by where their face is, instead of everyone. A face found is that person's;
for the people, each part of the people selection goes to the face it is under
or nearest to, which is how one of a row of band members is chosen.

Both are small and see the picture small (320 and 512 px). The rough selection
is brought to the frame's edges with a guided filter against the frame itself
at about 1024 px and firmed up, which is the selection kept: a hair line or a roof edge
follows the picture, not the model's grid. It is computed once per decoded
frame and kept on it; a loupe's patch reads its part of the whole frame's.

The models are fetched once into ``models_dir`` and checked against a hash, as
the face models are (faces.models).
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

KINDS = ("subject", "sky", "people", "faces")

SUBJECT_MODEL = "u2netp.onnx"
SKY_MODEL = "segformer_b0_ade.onnx"
MODELS = {
    "subject": ModelSpec(
        name=SUBJECT_MODEL,
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        size=4574861,
    ),
    # Pinned to one revision of the repository: a file that changed under the
    # same name would fail its hash on every new install.
    "sky": ModelSpec(
        name=SKY_MODEL,
        url="https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/"
            "d3e5499fa8701ff0453ca940a8dfeae39b2f1504/onnx/model.onnx",
        sha256="3e5c18a4be395f16646438d54c42377ddc202edfa33d5eced0c9506de75c44c2",
        size=15335446,
    ),
}
# ADE20K's classes, as the model numbers them: wall, building, sky, floor, tree,
# ceiling, road, bed, windowpane, grass, cabinet, sidewalk, person, …
_ADE_CLASS = {"sky": 2, "people": 12}
_INPUT = {"subject": 320, "sky": 512, "people": 512}
# The model a kind runs: people are read from the same answer as the sky.
_MODEL_OF = {"subject": "subject", "sky": "sky", "people": "sky"}
# The size the selection is refined and kept at (its longer side).
MATTE_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_models_dir: Optional[str] = None
_nets: dict = {}
_lock = threading.Lock()
# A fetch that failed is not tried again on every render until restart: a
# computer without a network would wait a minute for each preview.
_failed: dict = {}


def configure(models_dir: str) -> None:
    global _models_dir
    if models_dir != _models_dir:
        _models_dir = models_dir
        _nets.clear()
        _failed.clear()


def ensure(kind: str, download: bool = True) -> Path:
    """The model file for ``kind``, fetched if it is missing. Raises
    :class:`ModelsUnavailable` with a reason worth showing."""
    if kind not in KINDS:
        raise ValueError(f"no model selects {kind!r}")
    if not _models_dir:
        raise ModelsUnavailable("no models folder is configured")
    if kind == "faces":
        # The faces are found by the face detector and kept inside the people.
        ensure("people", download)
        from faces.models import DETECTOR_MODEL
        from faces.models import MODELS as FACE_MODELS

        spec = FACE_MODELS[DETECTOR_MODEL]
    else:
        kind = _MODEL_OF[kind]
        spec = MODELS[kind]
    target = Path(_models_dir) / spec.name
    if verify(target, spec):
        return target
    if not download:
        raise ModelsUnavailable(f"{spec.name} is not in {_models_dir} yet")
    if kind in _failed:
        raise ModelsUnavailable(_failed[kind])
    try:
        _download(spec, target)
    except ModelsUnavailable as exc:
        _failed[kind] = str(exc)
        raise
    return target


def available(kind: str) -> bool:
    """Whether ``kind`` can be selected without a download."""
    try:
        ensure(kind, download=False)
        return True
    except (ModelsUnavailable, ValueError):
        return False


def _net(kind: str):
    import cv2

    kind = _MODEL_OF[kind]
    net = _nets.get(kind)
    if net is None:
        net = cv2.dnn.readNet(str(ensure(kind)))
        _nets[kind] = net
    return net


def _small_picture(frame, size: int = MATTE_SIZE) -> np.ndarray:
    """The frame as a viewer sees it, at most ``size`` on its longer side (the
    matte's): 8-bit RGB through its own camera look, lifted when it is dark so a
    night sky is still seen."""
    import cv2

    from adjust.render import LinearFrame

    rgb = frame.rgb
    h, w = rgb.shape[:2]
    scale = min(1.0, size / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    bright = float(np.percentile(rgb.max(axis=2), 99))
    gain = float(np.clip(0.9 / max(bright, 1e-4), 1.0, 8.0))
    small = LinearFrame(np.ascontiguousarray(rgb * np.float32(gain)), look=frame.look)
    return np.asarray(small.encoded().convert("RGB"))


def _blob(rgb8: np.ndarray, size: int) -> np.ndarray:
    import cv2

    x = cv2.resize(rgb8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].copy()


def _rough(kind: str, rgb8: np.ndarray, kept: Optional[dict] = None) -> np.ndarray:
    """The model's own selection, 0..1, at its own small size. ``kept`` holds
    SegFormer's answer for this picture, so the sky and the people are one run."""
    if kind in _ADE_CLASS:
        probs = kept.get("_ade") if kept is not None else None
        if probs is None:
            with _lock:      # an OpenCV net is not to be run from two threads at once
                net = _net(kind)
                net.setInput(_blob(rgb8, _INPUT[kind]))
                logits = net.forward()[0]
            e = np.exp(logits - logits.max(axis=0))
            probs = (e / e.sum(axis=0)).astype(np.float32)
            if kept is not None:
                kept["_ade"] = probs
        return probs[_ADE_CLASS[kind]]
    with _lock:
        net = _net(kind)
        net.setInput(_blob(rgb8, _INPUT[kind]))
        out = net.forward(net.getUnconnectedOutLayersNames())
    # U²-Net answers at several depths; the first is the finished one.
    d = out[0].reshape(out[0].shape[-2:]).astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    return (d - lo) / max(hi - lo, 1e-6)


def _guided(guide: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He, Sun & Tang's guided filter, grey guide: ``p`` given the edges of
    ``guide``."""
    import cv2

    box = lambda x: cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1))  # noqa: E731
    mean_i, mean_p = box(guide), box(p)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * p) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def matte(frame, kind: str, person: Optional[int] = None, areas=None) -> np.ndarray:
    """How much of ``kind`` each part of the whole frame is, 0..1, as a float32
    picture about MATTE_SIZE on its longer side: everyone's, or with ``person``
    (1 is the leftmost face) one person's. People naming ``areas`` are those
    parts of them (adjust.parts), at that module's own size. Kept on the frame."""
    if kind == "people" and areas:
        from adjust import parts

        return parts.matte(frame, areas, person)
    key = (kind, int(person)) if person else kind
    kept = frame._mattes.get(key)
    if kept is not None:
        return kept
    if kind == "faces":
        fine = _faces_matte(frame, person)
    elif person and kind == "people":
        fine = _one_person(frame, int(person))
    else:
        fine = _model_matte(frame, kind)
    frame._mattes[key] = fine
    return fine


def _picture(frame) -> np.ndarray:
    """The frame at the matte's size as 8-bit RGB, made once per frame."""
    rgb8 = frame._mattes.get("_rgb8")
    if rgb8 is None:
        rgb8 = frame._mattes["_rgb8"] = _small_picture(frame)
    return rgb8


def _refined(rgb8: np.ndarray, rough: np.ndarray) -> np.ndarray:
    """A rough selection (at the picture's size) brought to its edges and firmed."""
    import cv2

    h, w = rgb8.shape[:2]
    grey = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    radius = max(2, round(max(h, w) / 128))
    fine = np.clip(_guided(grey, rough, radius, 1e-4), 0.0, 1.0)
    # Firmed up: a model's 0.8 is sky, and its 0.2 is not. A soft selection
    # darkens a sky unevenly and leaves a lighter ring round a head against it.
    t = np.clip((fine - 0.2) / 0.6, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _model_matte(frame, kind: str) -> np.ndarray:
    import cv2

    rgb8 = _picture(frame)
    h, w = rgb8.shape[:2]
    rough = cv2.resize(_rough(kind, rgb8, frame._mattes), (w, h), interpolation=cv2.INTER_LINEAR)
    return _refined(rgb8, rough)


# Smaller faces than this (pixels across, in the matte's picture) are a crowd in
# the distance, not someone to light; so is a face less than this share as wide
# as the largest in the frame (the people behind a family at a graduation), while
# a row of band members, all alike, are all kept.
MIN_FACE_PX = 10
MIN_FACE_SHARE = 0.4
FACE_SCORE = 0.6


def face_boxes(frame) -> list:
    """The faces in the whole frame, left to right, each ``(x, y, w, h)`` as
    fractions of it. Found once per frame, on the matte's picture."""
    import cv2

    whole = getattr(frame, "anchor_of", None) or frame
    kept = whole._mattes.get("_faces")
    if kept is not None:
        return kept
    from faces.detect import NMS_THRESHOLD, TOP_K

    rgb8 = _picture(whole)
    h, w = rgb8.shape[:2]
    with _lock:
        detector = _nets.get("faces")
        if detector is None:
            detector = cv2.FaceDetectorYN.create(str(ensure("faces")), "", (w, h),
                                                 FACE_SCORE, NMS_THRESHOLD, TOP_K)
            _nets["faces"] = detector
        detector.setInputSize((w, h))
        _, found = detector.detect(cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    boxes = []
    for row in found if found is not None else ():
        fx, fy, fw, fh = (float(v) for v in row[:4])
        if min(fw, fh) >= MIN_FACE_PX:
            boxes.append((fx / w, fy / h, fw / w, fh / h))
    if boxes:
        largest = max(b[2] for b in boxes)
        boxes = [b for b in boxes if b[2] >= MIN_FACE_SHARE * largest]
    boxes.sort(key=lambda b: b[0] + b[2] / 2)
    whole._mattes["_faces"] = boxes
    return boxes


def _oval(box: tuple, h: int, w: int) -> np.ndarray:
    """A soft oval over one face box. The detector's box runs from the brows to
    the chin, so the oval is taller and sits a little higher, over the forehead."""
    x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh * 0.42) * h
    rx, ry = bw * w * 0.58, bh * h * 0.68
    xs = (np.arange(w, dtype=np.float32) + 0.5 - cx) / rx
    ys = (np.arange(h, dtype=np.float32) + 0.5 - cy) / ry
    r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    t = np.clip((1.0 - r) / 0.35, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _faces_matte(frame, person: Optional[int]) -> np.ndarray:
    rgb8 = _picture(frame)
    h, w = rgb8.shape[:2]
    boxes = face_boxes(frame)
    if person:
        boxes = boxes[int(person) - 1:int(person)]
    rough = np.zeros((h, w), dtype=np.float32)
    for box in boxes:
        np.maximum(rough, _oval(box, h, w), out=rough)
    if not boxes:
        return rough
    # Kept inside the people, so an oval wider than a face does not light the
    # wall beside it. A face the people model missed (side-on, in costume)
    # keeps its oval as it is.
    people = matte(frame, "people")
    missed = np.zeros((h, w), dtype=np.float32)
    for box in boxes:
        x, y, bw, bh = box
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = max(x0 + 1, int((x + bw) * w)), max(y0 + 1, int((y + bh) * h))
        if float(people[y0:y1, x0:x1].mean()) < 0.3:
            np.maximum(missed, (_oval(box, h, w) > 0).astype(np.float32), out=missed)
    inside = np.maximum(people, missed)
    return (_refined(rgb8, rough) * inside).astype(np.float32)


# How far a person reaches from their face, in face widths across (and, scaled
# as below, about ten face heights down): a body with its arms out.
PERSON_REACH = 2.0


def _one_person(frame, person: int) -> np.ndarray:
    """One person of the people selection: each part of it goes to the face it is
    under or nearest to (across more than down, since a body hangs below its
    face), and this person's parts are kept. A part too far from every face is
    someone whose face was not found (a back, the crowd), and is nobody's."""
    import cv2

    people = matte(frame, "people")
    boxes = face_boxes(frame)
    if not 1 <= person <= len(boxes):
        return np.zeros_like(people)
    h, w = people.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + 0.5
    ys = np.arange(h, dtype=np.float32)[:, None] + 0.5
    best = np.full((h, w), PERSON_REACH ** 2, dtype=np.float32)
    owner = np.zeros((h, w), dtype=np.int32)
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        cx, cy, fw, fh = (x + bw / 2) * w, (y + bh / 2) * h, bw * w, bh * h
        dx = (xs - cx) / fw
        dy = (ys - cy) / fh
        # Above a face is the top of its head, then someone else's space; below
        # it is the body, several face heights down.
        dy = np.where(dy < 0, dy * 1.5, dy / 5.0)
        cost = dx * dx + dy * dy
        nearer = cost < best
        best = np.where(nearer, cost, best)
        owner = np.where(nearer, i, owner)
    mine = (owner == person).astype(np.float32)
    # The seam between two people is softened over a few pixels, so a lifted
    # face does not end on a hard line down a shoulder shared with a friend.
    soft = cv2.GaussianBlur(mine, (0, 0), max(1.0, max(h, w) / 512))
    return (people * soft).astype(np.float32)


def sample(full: np.ndarray, shape: tuple, window: tuple) -> np.ndarray:
    """A whole-frame matte at the pixels being rendered: ``shape`` (H, W) over
    ``window`` (left, top, right, bottom fractions of the whole frame)."""
    import cv2

    h, w = shape
    mh, mw = full.shape
    left, top, right, bottom = window
    xs = ((left + (np.arange(w, dtype=np.float32) + 0.5) / w * (right - left)) * mw - 0.5)
    ys = ((top + (np.arange(h, dtype=np.float32) + 0.5) / h * (bottom - top)) * mh - 0.5)
    map_x = np.broadcast_to(xs[None, :], (h, w)).astype(np.float32)
    map_y = np.broadcast_to(ys[:, None], (h, w)).astype(np.float32)
    return cv2.remap(full, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def whole_matte(frame, kind: str, person: Optional[int] = None, areas=None) -> Optional[np.ndarray]:
    """The whole frame's matte for ``frame`` (a loupe patch reads its anchor's),
    or ``None`` when the model cannot be had."""
    whole = getattr(frame, "anchor_of", None) or frame
    try:
        return matte(whole, kind, person, areas)
    except ModelsUnavailable as exc:
        logger.warning("[masks] cannot select %s: %s", kind, exc)
        return None
