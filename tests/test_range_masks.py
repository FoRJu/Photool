"""
Range masks: a mask narrowed to a band of brightness or to a colour of the
picture as it was before the edit.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.masks import colour_of, normalize_range, range_weight  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402
from test_develop_routes import FOLDER, PHOTO, client  # noqa: E402,F401  (a fixture)


def _strips():
    """Display-encoded: a dark, a mid and a bright grey, a red and a blue."""
    enc = np.zeros((10, 50, 3), dtype=np.float32)
    enc[:, 0:10] = 0.1
    enc[:, 10:20] = 0.5
    enc[:, 20:30] = 0.9
    enc[:, 30:40] = (0.8, 0.2, 0.15)
    enc[:, 40:50] = (0.15, 0.25, 0.8)
    return enc


class TestTheRange:
    def test_a_range_that_narrows_nothing_is_none(self):
        assert normalize_range({"kind": "luminance", "lo": 0, "hi": 100}) is None
        assert normalize_range({"kind": "color"}) is None
        assert normalize_range({"kind": "nonsense"}) is None

    def test_it_is_cleaned(self):
        assert normalize_range({"kind": "luminance", "lo": 80, "hi": 20}) == \
            {"kind": "luminance", "lo": 20, "hi": 80, "smooth": 30}
        assert normalize_range({"kind": "color", "cb": 2, "cr": -0.1234567, "lum": 300, "width": 50}) == \
            {"kind": "color", "cb": 0.5, "cr": -0.1235, "lum": 100, "width": 50}

    def test_it_travels_with_its_mask_through_the_sidecar(self):
        edit = {"masks": [{"kind": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.5, "exposure": -0.5,
                           "range": {"kind": "luminance", "lo": 60, "hi": 100, "smooth": 20}}]}
        clean = normalize(edit)
        assert clean["masks"][0]["range"] == {"kind": "luminance", "lo": 60, "hi": 100, "smooth": 20}
        assert from_crs(to_crs(clean)) == clean


class TestTheWeight:
    def test_brightness_takes_the_band_only(self):
        w = range_weight(_strips(), {"kind": "luminance", "lo": 70, "hi": 100, "smooth": 10})
        assert w[5, 25] == 1.0 and w[5, 5] == 0.0 and w[5, 15] == 0.0

    def test_shadows_only(self):
        w = range_weight(_strips(), {"kind": "luminance", "lo": 0, "hi": 30, "smooth": 10})
        assert w[5, 5] == 1.0 and w[5, 25] == 0.0

    def test_it_fades_rather_than_cuts(self):
        enc = np.linspace(0, 1, 101, dtype=np.float32)[None, :, None].repeat(3, axis=2)
        w = range_weight(enc, {"kind": "luminance", "lo": 50, "hi": 100, "smooth": 40})
        assert 0.0 < w[0, 40] < 1.0 and w[0, 25] == 0.0 and w[0, 55] == 1.0

    def test_a_colour_takes_that_colour_and_not_another(self):
        enc = _strips()
        y, cb, cr = colour_of(enc[5:6, 35:36])
        red = {"kind": "color", "cb": float(cb[0, 0]), "cr": float(cr[0, 0]), "lum": float(y[0, 0]) * 100, "width": 40}
        w = range_weight(enc, red)
        assert w[5, 35] == 1.0
        assert w[5, 45] == 0.0 and w[5, 15] == 0.0 and w[5, 25] == 0.0

    def test_the_same_colour_in_shade_is_still_taken(self):
        enc = np.zeros((2, 2, 3), dtype=np.float32)
        enc[0, 0] = (0.8, 0.2, 0.15)
        enc[0, 1] = (0.6, 0.12, 0.08)           # the same red, darker
        y, cb, cr = colour_of(enc[0:1, 0:1])
        w = range_weight(enc, {"kind": "color", "cb": float(cb[0, 0]), "cr": float(cr[0, 0]),
                                "lum": float(y[0, 0]) * 100, "width": 40})
        assert w[0, 1] > 0.5

    def test_wider_takes_more(self):
        enc = np.random.default_rng(2).random((40, 40, 3)).astype(np.float32)
        y, cb, cr = colour_of(enc[:1, :1])
        base = {"kind": "color", "cb": float(cb[0, 0]), "cr": float(cr[0, 0]), "lum": float(y[0, 0]) * 100}
        narrow = range_weight(enc, {**base, "width": 10}).sum()
        wide = range_weight(enc, {**base, "width": 90}).sum()
        assert wide > narrow


class TestInTheDevelop:
    def test_a_masks_exposure_reaches_only_its_range(self):
        rgb = np.zeros((20, 40, 3), dtype=np.float32)
        rgb[:, :20] = 0.02                   # dark
        rgb[:, 20:] = 0.5                    # bright
        frame = LinearFrame(rgb)
        plain = np.asarray(frame.encoded(), dtype=np.float32)
        whole = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 3, "ry": 3, "feather": 0, "exposure": -1.0}
        out = np.asarray(develop_frame(frame, {"masks": [
            {**whole, "range": {"kind": "luminance", "lo": 50, "hi": 100, "smooth": 10}}]}), dtype=np.float32)
        assert out[10, 30].mean() < plain[10, 30].mean() - 20       # the bright half, darkened
        assert abs(out[10, 5].mean() - plain[10, 5].mean()) < 2     # the dark half, untouched

    def test_the_range_is_judged_before_the_edit(self):
        """Lifting the global exposure does not move what a range reaches."""
        rgb = np.zeros((10, 20, 3), dtype=np.float32)
        rgb[:, :10] = 0.02
        rgb[:, 10:] = 0.5
        frame = LinearFrame(rgb)
        mask = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 3, "ry": 3, "feather": 0, "saturation": -100,
                "range": {"kind": "luminance", "lo": 0, "hi": 30, "smooth": 5}}
        develop_frame(frame, {"exposure": 3.0, "masks": [mask]})
        weights = list(frame._mask_weights.values())
        assert weights and weights[0][5, 2] == 1.0 and weights[0][5, 15] == 0.0


class TestThroughTheApi:
    def test_a_ranges_picture(self, client):
        res = client.post("/api/export/develop/range-image", json={
            "filepath": PHOTO, "range": {"kind": "luminance", "lo": 70, "hi": 100}, "size": 256})
        assert res.status_code == 200 and res.headers["content-type"] == "image/png"
        from io import BytesIO

        from PIL import Image

        w = np.asarray(Image.open(BytesIO(res.content)))
        # The test photo is dark with a bright strip along its top.
        assert w[:5].mean() > 200 and w[-5:].mean() < 20

    def test_a_range_that_narrows_nothing_is_refused(self, client):
        res = client.post("/api/export/develop/range-image", json={
            "filepath": PHOTO, "range": {"kind": "luminance", "lo": 0, "hi": 100}})
        assert res.status_code == 400

    def test_a_colour_is_sampled(self, client):
        bright = client.post("/api/export/develop/sample", json={"filepath": PHOTO, "x": 0.5, "y": 0.05, "size": 256}).json()
        dark = client.post("/api/export/develop/sample", json={"filepath": PHOTO, "x": 0.5, "y": 0.9, "size": 256}).json()
        assert bright["lum"] > 80 and dark["lum"] < 30
        assert abs(bright["cb"]) < 0.05 and abs(bright["cr"]) < 0.05     # grey
        assert len(bright["rgb"]) == 3
