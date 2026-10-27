"""
Person clustering: finding faces, describing them, and grouping them by who.

Four parts, each useful on its own:

  * :mod:`faces.models` — the two ONNX models, fetched and checksummed.
  * :mod:`faces.detect` — OpenCV finds a face and turns it into 128 numbers.
  * :mod:`faces.clustering` — pure arithmetic that decides who is who.
  * :mod:`faces.service` — the passes over a folder, and the crops to show.

Nothing here is imported by the catalog or the export path. A machine without
OpenCV installed runs the rest of the application exactly as before, and only
asks for faces when someone asks for faces.
"""

from faces.clustering import centroid, group, unit
from faces.detect import (
    MIN_GROUP_PX,
    Face,
    FacesUnavailable,
    available,
    configure,
    find_faces,
)
from faces.models import (
    DETECTOR_MODEL,
    MODELS,
    RECOGNIZER_MODEL,
    ModelSpec,
    ModelsUnavailable,
    ensure_models,
    model_path,
    verify,
)
from faces.service import crop, rebuild_people, scan_folder

__all__ = [
    "DETECTOR_MODEL",
    "Face",
    "FacesUnavailable",
    "MIN_GROUP_PX",
    "MODELS",
    "ModelSpec",
    "ModelsUnavailable",
    "RECOGNIZER_MODEL",
    "available",
    "centroid",
    "configure",
    "crop",
    "ensure_models",
    "find_faces",
    "group",
    "model_path",
    "rebuild_people",
    "scan_folder",
    "unit",
    "verify",
]
