"""
Where the ONNX models live, and how they get there.

The models are binary that is fetched once and checked against a known hash, so
they are not in the repository. Everything here is standard library on purpose:
a download that has to happen before the feature works is a poor place to add a
dependency.

Two families of model:

* **YuNet** (the detector) and **SFace** come from the OpenCV model zoo, which
  stores them in git-lfs — the plain GitHub URL serves a 130-byte pointer, not
  the model, which is why the URLs below go to the media endpoint.
* **ArcFace** recognisers come from InsightFace's own release packs (``buffalo_l``
  holds the ResNet-50, ``buffalo_sc`` the MobileFaceNet). Each pack is a zip of
  several models; only the recogniser is taken out of it, checked, and kept. The
  InsightFace weights are released for non-commercial use, which a family album
  is; SFace (Apache-2.0) stays available as ``face_model: sface``.
"""

import hashlib
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 256
_TIMEOUT = 60


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    size: int
    # When set, ``url`` is a zip and the model is this member of it. The member's
    # size and hash are what ``size`` and ``sha256`` describe.
    member: str = ""


DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
ARCFACE_MODEL = "w600k_r50.onnx"
ARCFACE_LITE_MODEL = "w600k_mbf.onnx"

_INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7"

# Which file each ``face_model`` setting describes faces with.
RECOGNIZERS = {
    "arcface": ARCFACE_MODEL,
    "arcface_lite": ARCFACE_LITE_MODEL,
    "sface": RECOGNIZER_MODEL,
}
DEFAULT_RECOGNIZER = "arcface"


def recognizer_file(recognizer: str) -> str:
    """The model file for a ``face_model`` setting; an unknown one is the default."""
    return RECOGNIZERS.get((recognizer or "").strip().lower(), RECOGNIZERS[DEFAULT_RECOGNIZER])


def recognizer_name(recognizer: str) -> str:
    """The canonical setting name (``arcface`` when the setting is not one we know)."""
    name = (recognizer or "").strip().lower()
    return name if name in RECOGNIZERS else DEFAULT_RECOGNIZER

_ZOO = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
)

MODELS = {
    DETECTOR_MODEL: ModelSpec(
        name=DETECTOR_MODEL,
        url=f"{_ZOO}/face_detection_yunet/{DETECTOR_MODEL}",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size=232589,
    ),
    RECOGNIZER_MODEL: ModelSpec(
        name=RECOGNIZER_MODEL,
        url=f"{_ZOO}/face_recognition_sface/{RECOGNIZER_MODEL}",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size=38696353,
    ),
    # ArcFace ResNet-50 trained on WebFace600K, 512 numbers per face. On a set of
    # labelled portraits the gap between the least similar pair of one person
    # and the most similar pair of two people was 0.50 against SFace's 0.35.
    ARCFACE_MODEL: ModelSpec(
        name=ARCFACE_MODEL,
        url=f"{_INSIGHTFACE}/buffalo_l.zip",
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        size=174383860,
        member="w600k_r50.onnx",
    ),
    # The same training on a MobileFaceNet: a 15 MB download instead of 290 MB and
    # a tenth of the arithmetic, at a gap of 0.42 on the same set.
    ARCFACE_LITE_MODEL: ModelSpec(
        name=ARCFACE_LITE_MODEL,
        url=f"{_INSIGHTFACE}/buffalo_sc.zip",
        sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
        size=13616099,
        member="w600k_mbf.onnx",
    ),
}


class ModelsUnavailable(RuntimeError):
    """The models are not on disk and could not be fetched."""


def model_path(models_dir: str | Path, name: str) -> Path:
    return Path(models_dir) / name


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(path: Path, spec: ModelSpec) -> bool:
    """True when the file is the model it claims to be.

    A truncated download is worse than none: it fails at load time with a
    message about the network rather than about the file, and the retry that
    would fix it never happens. The hash is checked, not just the size.
    """
    try:
        if path.stat().st_size != spec.size:
            return False
        return _digest(path) == spec.sha256
    except OSError:
        return False


def _download(spec: ModelSpec, target: Path) -> None:
    """Fetch one model, then move it into place in one step.

    Written to a ``.part`` file first so an interrupted download never leaves
    something that looks like a model behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if spec.member:
        partial = target.with_suffix(target.suffix + ".zip.part")
    logger.info("[faces] downloading %s (%.0f MB)", spec.name, spec.size / 1e6)
    request = urllib.request.Request(spec.url, headers={"User-Agent": "canon-photo-manager"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            with open(partial, "wb") as handle:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    handle.write(block)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise ModelsUnavailable(
            f"could not download {spec.name}: {exc}. Fetch {spec.url} by hand"
            f"{f' and take {spec.member} out of it' if spec.member else ''}, put it in "
            f"{target.parent} and try again."
        ) from exc

    if spec.member:
        # A release pack: take the one model out and let the zip go.
        import zipfile

        unpacked = target.with_suffix(target.suffix + ".part")
        try:
            with zipfile.ZipFile(partial) as archive:
                member = next(
                    (n for n in archive.namelist() if n.split("/")[-1] == spec.member), None
                )
                if member is None:
                    raise ModelsUnavailable(f"{spec.url} does not contain {spec.member}")
                with archive.open(member) as source, open(unpacked, "wb") as handle:
                    while True:
                        block = source.read(_CHUNK)
                        if not block:
                            break
                        handle.write(block)
        except zipfile.BadZipFile as exc:
            unpacked.unlink(missing_ok=True)
            raise ModelsUnavailable(f"{spec.url} was not a readable zip: {exc}") from exc
        finally:
            partial.unlink(missing_ok=True)
        partial = unpacked

    if not verify(partial, spec):
        partial.unlink(missing_ok=True)
        raise ModelsUnavailable(
            f"{spec.name} did not match its checksum after downloading. Delete "
            f"anything with that name in {target.parent} and try again."
        )
    partial.replace(target)


def required(recognizer: str = DEFAULT_RECOGNIZER) -> list[str]:
    """The model files a scan with this recogniser needs: the detector first."""
    return [DETECTOR_MODEL, recognizer_file(recognizer)]


def ensure_models(
    models_dir: str | Path,
    download: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    recognizer: str = DEFAULT_RECOGNIZER,
) -> dict[str, Path]:
    """Return the path of each model a scan needs, fetching any that is missing.

    Only the detector and the one recogniser ``recognizer`` names are fetched, so
    choosing the small model never downloads the large one.

    Raises :class:`ModelsUnavailable` when a model is absent and cannot be
    fetched, with the reason rather than a stack trace from OpenCV.
    """
    paths: dict[str, Path] = {}
    for name in required(recognizer):
        spec = MODELS[name]
        target = model_path(models_dir, name)
        if not verify(target, spec):
            if target.exists():
                logger.warning("[faces] %s is not the expected file; fetching it again", name)
            if not download:
                raise ModelsUnavailable(f"{name} is missing from {models_dir}")
            if progress:
                progress(name)
            _download(spec, target)
        paths[name] = target
    return paths
