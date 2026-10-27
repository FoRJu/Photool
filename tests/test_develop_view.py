"""
What the develop view reads back: the loupe's patch, lossless and at the size
asked (2:1 asks for half), and the whole photograph's size beside the preview,
for the pixel readout.
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from test_develop_routes import PHOTO, client  # noqa: E402,F401  (a fixture)


def test_the_loupe_is_lossless(client):
    res = client.post("/api/export/develop/loupe",
                      json={"filepath": PHOTO, "u": 0.5, "v": 0.5, "size": 128, "preview_size": 256})
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(res.content))
    assert img.format == "PNG" and img.size == (128, 128)


def test_two_to_one_asks_for_half_the_patch(client):
    res = client.post("/api/export/develop/loupe",
                      json={"filepath": PHOTO, "u": 0.5, "v": 0.5, "size": 64, "preview_size": 256})
    assert Image.open(io.BytesIO(res.content)).size == (64, 64)


def test_the_preview_says_how_big_the_photograph_is(client):
    res = client.post("/api/export/preview",
                      json={"filepath": PHOTO, "size": 128, "linear": True, "adjustment": {}})
    assert res.status_code == 200
    # The test photograph is 400 x 300; the preview is a quarter of that.
    assert res.headers["X-Photo-Size"] == "400x300"
    assert max(Image.open(io.BytesIO(res.content)).size) == 128


def test_sharpening_shows_on_a_small_picture():
    """At a quarter of full size the radius is under a pixel of the picture; the
    finest blur used still lifts fine detail visibly."""
    from adjust.sharpen import MIN_SIGMA, sharpen

    assert MIN_SIGMA >= 0.7
    rng = np.random.default_rng(4)
    enc = (0.4 + 0.1 * rng.standard_normal((64, 64, 1))).repeat(3, axis=2).astype(np.float32)
    detail = lambda a: float(np.abs(np.diff(a[..., 1], axis=1)).mean())
    assert detail(sharpen(enc, {"sharpen": 40}, scale=0.27)) > 1.15 * detail(enc)
