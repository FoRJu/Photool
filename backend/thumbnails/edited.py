"""
Thumbnails that show the edit.

The grid and the cull view draw the picture the camera embedded in the file, and
an edit is made on the sensor data, so without this a frame that had been
pulled two stops brighter sat in the library looking exactly as it was shot.
An edited frame is rendered through the same pipeline the develop view and the
export use, at the thumbnail's size, and kept under its own key — beside the
plain thumbnail, never over it — so removing the edit puts the camera's
rendering straight back.

Only edited frames pay for this. A RAW decode is the slow part, so it is done
at half size when the target allows, a couple at a time, and then it is cached
like any other thumbnail.
"""

import hashlib
import io
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

from adjust.develop import normalize
from thumbnails.store import ThumbnailSize

logger = logging.getLogger(__name__)

# Bumped when the pipeline changes what an edit looks like, so thumbnails drawn
# by the old one are not served for the new one.
RENDER_VERSION = 2

_QUALITY = {ThumbnailSize.GRID: 80, ThumbnailSize.PREVIEW: 90}

# A browser asks for every visible thumbnail at once. Decoding RAW files is
# heavy, so only a few run together and the rest wait their turn.
_SLOTS = threading.BoundedSemaphore(2)


def edit_variant(settings: Optional[dict]) -> str:
    """The cache variant for an edit, or ``""`` when the frame is not edited.

    Derived from the settings themselves, so two frames with the same edit share
    nothing by accident (their sources differ) but one frame whose edit changes
    always gets a new key.
    """
    clean = normalize(settings)
    if not clean:
        return ""
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return f"e{RENDER_VERSION}-{hashlib.sha1(blob.encode()).hexdigest()[:16]}"


def render_edited(source: Path, settings: dict, size: ThumbnailSize) -> Optional[bytes]:
    """The frame with its edit applied, as JPEG bytes at a thumbnail size."""
    from adjust.render import develop_frame, resize_linear
    from export.engine import _load_linear, _with_camera_look, _with_denoiser, frame_of

    target = size.value
    with _SLOTS:
        rgb = _load_linear(source, target)
        if rgb is None:
            return None
        frame = frame_of(source, resize_linear(rgb, target))
        _with_camera_look(source, frame)
        # The large view shows the noise reduction; at grid size it cannot be
        # seen, and is not worth a run of the model.
        if size == ThumbnailSize.PREVIEW:
            _with_denoiser(source, frame, target)
        image: Image.Image = develop_frame(frame, settings)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=_QUALITY.get(size, 85))
    return buf.getvalue()


def get_edited_thumbnail(
    source: Path, settings: dict, cache, size: ThumbnailSize
) -> Optional[bytes]:
    """Cached edited thumbnail, rendered and stored on a miss.

    ``None`` when the frame is not edited or cannot be rendered; the caller then
    serves the camera's own thumbnail, which is a better answer than an error.
    """
    variant = edit_variant(settings)
    if not variant or size not in _QUALITY:
        return None
    cached = cache.get(source, size, variant)
    if cached:
        return cached
    try:
        data = render_edited(source, normalize(settings), size)
    except Exception as exc:
        logger.warning("Edited thumbnail failed for %s: %s", source.name, exc)
        return None
    if data:
        cache.put(source, size, data, variant)
    return data
