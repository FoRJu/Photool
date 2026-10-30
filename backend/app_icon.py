"""
Photool's icon, drawn when a phone or an iPad asks for it: added to a home
screen, the page opens as an app of its own, and the device wants a picture to
show for it. Drawn rather than kept as a file, so no picture is in the
repository: an amber ring round an amber dot, a lens, on near black.
"""

import io
from functools import lru_cache

# The sizes the page's manifest and the Apple touch icon name.
SIZES = {"apple-touch-icon.png": 180, "icon-192.png": 192, "icon-512.png": 512}


@lru_cache(maxsize=len(SIZES))
def png(size: int) -> bytes:
    """The icon at ``size`` pixels square, as PNG bytes."""
    from PIL import Image, ImageDraw

    scale = 4                      # drawn large and shrunk, for smooth edges
    n = size * scale
    im = Image.new("RGB", (n, n), (10, 10, 10))
    draw = ImageDraw.Draw(im)
    c = n / 2
    ring = n * 0.30
    draw.ellipse([c - ring, c - ring, c + ring, c + ring], outline=(245, 158, 11), width=int(n * 0.075))
    dot = n * 0.11
    draw.ellipse([c - dot, c - dot, c + dot, c + dot], fill=(245, 158, 11))
    buf = io.BytesIO()
    im.resize((size, size), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
