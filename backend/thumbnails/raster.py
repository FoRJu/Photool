"""
Pictures that are not a camera's: film and slide scans, Photoshop documents,
screenshots and the like (TIFF, PSD, PNG, BMP).

They are shown, rated and developed like any photograph. What makes them their
own case is how they are stored, not what they show:

- A scanner writes 16 bits a channel (VueScan's "48-bit RGB" TIFF). Pillow reads
  that as 8-bit RGB from version 11; an older Pillow cannot, and OpenCV reads it
  instead.
- A Photoshop document is read for its composite, the flattened picture
  Photoshop saves beside the layers ("maximize compatibility", the default).
- A scan may be CMYK or greyscale, and a PNG may carry transparency, which is
  laid on white as a print would be.

``open_raster`` gives every one of them as an upright 8-bit RGB PIL image.
"""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

from config import RASTER_EXTENSIONS  # noqa: E402  (one list, kept in config)


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.split()[-1])
        return white
    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        import numpy as np

        a = np.asarray(img, dtype=np.uint32)
        top = 65535 if a.max() > 255 else 255
        return Image.fromarray((a * 255 // top).astype("uint8"), "L").convert("RGB")
    if img.mode == "F":
        import numpy as np

        a = np.clip(np.asarray(img, dtype=np.float32), 0, 1)
        return Image.fromarray((a * 255).astype("uint8"), "L").convert("RGB")
    return img if img.mode == "RGB" else img.convert("RGB")


def _with_opencv(source: Path) -> Optional[Image.Image]:
    """What Pillow could not read, by OpenCV (16-bit TIFF on an older Pillow)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    data = np.fromfile(str(source), dtype=np.uint8)      # a path with any characters
    a = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, "L").convert("RGB")
    if a.shape[2] == 4:
        rgba = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        return _to_rgb(Image.fromarray(rgba, "RGBA"))
    return Image.fromarray(cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2RGB), "RGB")


def open_raster(source: Path) -> Optional[Image.Image]:
    """The picture in a TIFF, PSD, PNG or BMP (or any file Pillow reads) as an
    upright 8-bit RGB image; ``None`` when nothing can read it."""
    source = Path(source)
    try:
        with Image.open(source) as img:
            img.load()
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:  # a damaged orientation tag is not worth losing the picture
                pass
            return _to_rgb(img)
    except Exception as e:
        logger.info(f"Pillow could not read {source.name} ({e}); trying OpenCV")
    try:
        img = _with_opencv(source)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to read {source.name}: {e}")
        return None
    if img is None:
        logger.error(f"Failed to read {source.name}")
    return img
