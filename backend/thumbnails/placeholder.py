"""
The tile shown for a file with no picture to give: an empty recording (a
camera that stopped a clip as it started leaves one of about a kilobyte), or a
damaged file. Drawn on request, dark, with what is wrong in a line of text, so
the grid says why instead of showing a broken picture.
"""

import io
from functools import lru_cache

from .store import ThumbnailSize

_SIDES = {ThumbnailSize.GRID: (400, 300), ThumbnailSize.PREVIEW: (1200, 900), ThumbnailSize.FULL: (1200, 900)}


@lru_cache(maxsize=16)
def no_picture_jpeg(text: str, size: ThumbnailSize = ThumbnailSize.GRID) -> bytes:
    """A dark 4:3 tile with ``text`` across its middle, as JPEG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = _SIDES.get(size, (400, 300))
    im = Image.new("RGB", (w, h), (28, 28, 30))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(14, w // 18))
    except TypeError:                       # an older Pillow: one size only
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text(((w - tw) / 2, (h - th) / 2), text, fill=(150, 150, 155), font=font)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
