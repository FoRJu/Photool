"""
Find faces in a frame and describe each one as a vector.

Two models, both run on this machine through OpenCV:

  * **YuNet** finds a face and its five landmarks — the corners of the eyes, the
    tip of the nose, the corners of the mouth. The landmarks are what make the
    second step possible: recognition wants the face upright, scaled and centred
    the same way every time, and a bounding box alone cannot say where the eyes
    are.
  * A **recogniser** takes that aligned 112px crop and returns a vector. Two
    faces of the same person point the same way; two different people do not.
    ArcFace (512 numbers, the default) separates people far better than SFace
    (128 numbers, the original): on labelled portraits the gap between "least
    alike pair of one person" and "most alike pair of two people" was 0.50
    against 0.35, and two strangers sit around 0 rather than 0.1. That gap is
    what decides whether two siblings end up in one group.

Each face is described twice, as it is and mirrored, and the two vectors are
added. A face is close to symmetric and the network is not, so the mirror
cancels some of the network's quirks for free (one batch of two, not two runs).

Every face also gets a **quality** from 0 to 1: how big it is, how frontal (from
the landmarks), how sharp (the aligned crop's Laplacian) and how sure the
detector was. The grouping lets good faces define a person and lets poor ones
only join a person who is already clear, which is what stops a group of
blurred profiles forming around nobody.

``cv2`` is imported inside the calls rather than at the top of the module, for
the same reason the export pipeline imports ``torch`` late: the catalog and the
API should start on a machine that has never had OpenCV installed, and say so
only when someone asks for faces.

The detector carries its input size as state, so instances are per-thread rather
than shared. Two folders can then be scanned at once without either one resizing
the other's input.
"""

import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from faces.models import (
    DEFAULT_RECOGNIZER,
    DETECTOR_MODEL,
    RECOGNIZER_MODEL,
    ModelsUnavailable,
    ensure_models,
    recognizer_file,
    recognizer_name,
)

logger = logging.getLogger(__name__)

# Detection confidence. Below this YuNet starts returning patterns in foliage.
DETECT_SCORE = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Faces whose width in the analysed frame is under this are recorded but left
# out of clustering — see ``config.face_min_px`` for why.
MIN_GROUP_PX = 40.0

# Where ArcFace (and SFace) expect the five landmarks in a 112x112 crop: the eye
# on the left of the picture, the other eye, the nose, the mouth corners.
ALIGN_SIZE = 112
ALIGN_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Quality: a face this wide in the analysed frame has as much detail as the
# 112px crop can hold; below it the crop is an upscale.
QUALITY_FULL_PX = 100.0
# How far the nose sits off the line between the eyes, in eye distances, at
# which a face counts as a full profile.
QUALITY_PROFILE_OFFSET = 0.7

_local = threading.local()
_models_dir: Optional[str] = None
_recognizer: str = DEFAULT_RECOGNIZER


@dataclass
class Face:
    """One detected face, in coordinates that survive a resize.

    Only faces at or above the caller's size floor are ever built, so there is no
    "found but not kept" state here: what comes out of :func:`find_faces` is what
    the scan stores, and the scan stores only what it could group.
    """

    x: float          # left edge, as a fraction of the frame's width
    y: float          # top edge, as a fraction of the frame's height
    w: float          # width, as a fraction of the frame's width
    h: float
    score: float      # detector confidence
    px: float         # width in the analysed frame, for judging the embedding
    embedding: np.ndarray  # float32, unit length (512 values for ArcFace)
    quality: float = 1.0   # 0..1, see quality()


class FacesUnavailable(RuntimeError):
    """OpenCV or the models are missing, with a reason worth showing."""


def configure(models_dir: str, recognizer: str = DEFAULT_RECOGNIZER) -> None:
    """Point the module at a models directory and a recogniser. Cheap; called on
    every use."""
    global _models_dir, _recognizer
    recognizer = recognizer_name(recognizer)
    if _models_dir != models_dir or _recognizer != recognizer:
        _models_dir = models_dir
        _recognizer = recognizer
        _local.__dict__.clear()


def model_id(recognizer: str = None) -> str:
    """What the stored vectors were made with. Vectors from two models live in
    different spaces and are never compared; a frame described by another model
    is looked at again."""
    return recognizer_name(recognizer or _recognizer)


def engine_present() -> bool:
    """Whether OpenCV is installed, without importing it.

    This is what decides whether fetching the models would help at all: a
    missing package is a fact the user has to change, a missing model is one the
    next scan will fetch.
    """
    import importlib.util

    return importlib.util.find_spec("cv2") is not None


def available(models_dir: str, download: bool = False,
              recognizer: str = DEFAULT_RECOGNIZER) -> tuple[bool, str]:
    """Whether faces can be found at all, and why not when they cannot.

    Never raises: the answer is for a status endpoint or a panel, and a missing
    optional dependency is not an error to hand a user as a stack trace.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False, "OpenCV is not installed (pip install opencv-python-headless)"
    try:
        ensure_models(models_dir, download=download, recognizer=recognizer)
    except ModelsUnavailable as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not prepare the face models: {exc}"
    return True, ""


class _SFace:
    """OpenCV's own recogniser: it aligns and describes in one object."""

    def __init__(self, path):
        import cv2

        self._net = cv2.FaceRecognizerSF.create(str(path), "")

    def describe(self, aligned: np.ndarray) -> np.ndarray:
        import cv2

        a = self._net.feature(aligned).reshape(-1)
        b = self._net.feature(cv2.flip(aligned, 1)).reshape(-1)
        return _unit(a) + _unit(b)


class _ArcFace:
    """An InsightFace ArcFace network run with OpenCV's DNN module.

    The network wants RGB scaled to -1..1; it was trained without any
    normalisation layer of its own (checked on both files), so the scaling is
    done here. The plain and mirrored crops go through as one batch of two.
    """

    def __init__(self, path):
        import cv2

        self._net = cv2.dnn.readNetFromONNX(str(path))

    def describe(self, aligned: np.ndarray) -> np.ndarray:
        import cv2

        blob = cv2.dnn.blobFromImages(
            [aligned, cv2.flip(aligned, 1)], 1.0 / 127.5, (ALIGN_SIZE, ALIGN_SIZE),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        self._net.setInput(blob)
        out = self._net.forward()
        return _unit(out[0]) + _unit(out[1])


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if np.isfinite(n) and n > 0 else np.zeros_like(v)


def _engines():
    """The per-thread detector and recogniser, created on first use."""
    import cv2

    # One frame is one small inference and OpenCV cannot spread it across
    # cores: measured on this library, a serial pass over 1024px renderings
    # held 0.95 of a core with the default pool of 20. Leaving that pool on
    # buys nothing and fights the scan's own threads, which is where the
    # parallelism has to come from (faces.service.FACE_WORKERS).
    cv2.setNumThreads(1)

    engine = getattr(_local, "engine", None)
    if engine is not None:
        return engine

    if not _models_dir:
        raise FacesUnavailable("no models directory has been configured")
    try:
        paths = ensure_models(_models_dir, recognizer=_recognizer)
    except ModelsUnavailable as exc:
        raise FacesUnavailable(str(exc)) from exc

    detector = cv2.FaceDetectorYN.create(
        str(paths[DETECTOR_MODEL]), "", (320, 320), DETECT_SCORE, NMS_THRESHOLD, TOP_K
    )
    name = recognizer_file(_recognizer)
    try:
        recognizer = _SFace(paths[name]) if name == RECOGNIZER_MODEL else _ArcFace(paths[name])
    except cv2.error as exc:
        raise FacesUnavailable(f"OpenCV could not load {name}: {exc}") from exc
    engine = (detector, recognizer)
    _local.engine = engine
    return engine


def align(bgr: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    """The 112px crop recognition wants: the face turned upright and scaled so
    the five landmarks land on the template. A similarity transform (rotation,
    one scale, shift) so the face is never stretched."""
    import cv2

    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(points, ALIGN_TEMPLATE, method=cv2.LMEDS)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    return cv2.warpAffine(bgr, matrix, (ALIGN_SIZE, ALIGN_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def pose(landmarks: Optional[np.ndarray]) -> float:
    """How frontal a face is, 0.15..1: how far the nose sits off the line between
    the eyes (1 when it is on the bisector, lower as the head turns)."""
    if landmarks is None:
        return 1.0
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    eyes = pts[1] - pts[0]
    eye_dist = float(np.hypot(*eyes))
    if eye_dist <= 1e-3:
        return 1.0
    mid = (pts[0] + pts[1]) / 2.0
    # Distance of the nose from the eye line's perpendicular bisector.
    along = float(np.dot(pts[2] - mid, eyes / eye_dist))
    offset = abs(along) / eye_dist
    return min(1.0, max(0.15, 1.0 - offset / QUALITY_PROFILE_OFFSET))


def quality(px: float, score: float, landmarks: Optional[np.ndarray],
            aligned: Optional[np.ndarray]) -> float:
    """How much one face's vector can be trusted, 0..1.

    The product of four plain measurements, each floored so that no single one
    zeroes a face out on its own:

      * size — the face's width against what the 112px crop can hold;
      * pose — how far the nose sits off the line between the eyes (a profile
        is the case the vector is least like the person's frontal ones);
      * sharpness — the variance of the Laplacian over the aligned crop, on a
        log scale (motion blur and missed focus are common at an event);
      * the detector's own confidence.
    """
    size_f = min(1.0, max(0.25, px / QUALITY_FULL_PX))
    pose_f = pose(landmarks)

    sharp_f = 1.0
    if aligned is not None:
        import cv2

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_f = min(1.0, max(0.2, (math.log10(variance + 1.0) - 1.0) / 1.4))

    det_f = min(1.0, max(0.2, (float(score) - 0.5) / 0.4))
    return round(size_f * pose_f * sharp_f * det_f, 4)


def find_faces(image: Image.Image, min_px: float = MIN_GROUP_PX) -> list[Face]:
    """Every face worth keeping in one frame, with an embedding for each.

    ``min_px`` is applied *before* the face is described, not after. Detection is
    cheap and describing a face is not, and on some photographs YuNet returns
    hundreds of specks — a frame in one real library produced 221 of them, none
    wider than 28 pixels, which at one recognition pass each is three seconds
    spent on nothing. It is also the honest place for the limit: a face a few
    dozen pixels across describes noise, and a vector of noise in a group's
    average is what merges two people.

    Raises :class:`FacesUnavailable` when the models cannot be loaded; the scan
    turns that into the reason shown in the panel rather than failing per photo.
    """
    import cv2

    detector, recognizer = _engines()

    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        return []
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    detector.setInputSize((width, height))
    _, detected = detector.detect(bgr)
    if detected is None or len(detected) == 0:
        return []

    faces: list[Face] = []
    for row in detected:
        score = float(row[14])
        fx, fy, fw, fh = (float(v) for v in row[:4])
        if fw < min_px or fh < min_px:
            continue
        landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        try:
            aligned = align(bgr, landmarks)
            if aligned is None:
                continue
            vector = recognizer.describe(aligned)
        except cv2.error as exc:  # a box the aligner cannot use
            logger.debug("could not describe a face at %.0f,%.0f: %s", fx, fy, exc)
            continue
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            continue
        faces.append(
            Face(
                x=max(0.0, fx / width),
                y=max(0.0, fy / height),
                w=min(1.0, fw / width),
                h=min(1.0, fh / height),
                score=score,
                px=fw,
                embedding=(vector / norm).astype(np.float32),
                quality=quality(fw, score, landmarks, aligned),
            )
        )
    # Largest first: the caller takes the first as the frame's main subject, and
    # a cluster's picture is its most prominent member.
    faces.sort(key=lambda f: -f.px)
    return faces
