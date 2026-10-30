"""The icon a phone or an iPad shows for Photool on its home screen: drawn on
request, so the repository holds no picture, at the sizes the page names."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from test_develop_routes import client  # noqa: E402,F401  (a fixture)


def test_each_icon_is_a_square_png_of_its_size(client):
    from PIL import Image

    for name, size in (("apple-touch-icon.png", 180), ("icon-192.png", 192), ("icon-512.png", 512)):
        res = client.get(f"/{name}")
        assert res.status_code == 200 and res.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(res.content)) as im:
            assert im.size == (size, size)
