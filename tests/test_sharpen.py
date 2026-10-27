"""
Sharpening: its controls in the edit model and the sidecar, what it does to an
edge, and that a smaller picture is sharpened by the photograph's own radius.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import CRS_KEYS, describe_settings, from_crs, normalize, to_crs
from adjust.render import LinearFrame, develop_frame
from adjust.sharpen import detail_of, radius_px, sharpen


def _edge(h=48, w=48, low=0.3, high=0.6):
    """A vertical edge, grey on each side, display-encoded."""
    enc = np.full((h, w, 3), low, dtype=np.float32)
    enc[:, w // 2:] = high
    return enc


def _contrast_across(enc, w):
    row = enc[enc.shape[0] // 2, :, 1]
    return float(row[w // 2] - row[w // 2 - 1])


class TestTheControls:
    def test_shaping_controls_mean_nothing_without_an_amount(self):
        assert normalize({"sharpen_radius": 5, "sharpen_detail": 10, "sharpen_masking": 40}) == {}

    def test_they_are_kept_with_an_amount_and_clamped_to_lightrooms_ranges(self):
        clean = normalize({"sharpen": 400, "sharpen_radius": 99, "sharpen_detail": -99, "sharpen_masking": 50})
        assert clean == {"sharpen": 150, "sharpen_radius": 20, "sharpen_detail": -25, "sharpen_masking": 50}
        assert radius_px(clean["sharpen_radius"]) == 3.0
        assert detail_of(clean["sharpen_detail"]) == 0

    def test_zero_is_lightrooms_default(self):
        assert radius_px(0) == 1.0
        assert detail_of(0) == 25

    def test_the_sidecar_round_trips_under_this_apps_own_names(self):
        edit = {"sharpen": 60, "sharpen_radius": -3, "sharpen_detail": 15, "sharpen_masking": 40}
        crs = to_crs(edit)
        assert crs["crs:PhotoolSharpenAmount"] == "60"
        assert from_crs(crs) == edit
        # Lightroom's own sharpening is neither written nor read: most of its
        # sidecars carry crs:Sharpness 40.
        assert "crs:Sharpness" not in crs and "crs:Sharpness" not in CRS_KEYS
        assert from_crs({"crs:Sharpness": "40", "crs:SharpenRadius": "+1.0"}) == {}

    def test_it_is_described(self):
        assert "sharpening 60 at 0.7 px, detail 40, masking 40" in describe_settings(
            {"sharpen": 60, "sharpen_radius": -3, "sharpen_detail": 15, "sharpen_masking": 40})


class TestWhatItDoes:
    def test_no_amount_is_no_change(self):
        enc = _edge()
        assert sharpen(enc, {}) is enc

    def test_an_edge_gets_more_contrast(self):
        enc = _edge()
        out = sharpen(enc, {"sharpen": 80})
        assert _contrast_across(out, 48) > _contrast_across(enc, 48)
        # Darker on the dark side and lighter on the light side, as sharpening is.
        mid = enc.shape[0] // 2
        assert out[mid, 23, 1] < enc[mid, 23, 1] and out[mid, 24, 1] > enc[mid, 24, 1]

    def test_flat_areas_are_left_alone(self):
        enc = np.full((32, 32, 3), 0.4, dtype=np.float32)
        assert np.allclose(sharpen(enc, {"sharpen": 150}), enc, atol=1e-6)

    def test_brightness_only_so_a_grey_edge_stays_grey(self):
        out = sharpen(_edge(), {"sharpen": 150, "sharpen_detail": 75})
        assert np.allclose(out[..., 0], out[..., 1], atol=1e-6)
        assert np.allclose(out[..., 1], out[..., 2], atol=1e-6)

    def test_more_amount_is_more(self):
        enc = _edge()
        weak = _contrast_across(sharpen(enc, {"sharpen": 30}), 48)
        strong = _contrast_across(sharpen(enc, {"sharpen": 120}), 48)
        assert strong > weak

    def test_low_detail_holds_a_strong_edge_back(self):
        enc = _edge(low=0.1, high=0.9)
        held = sharpen(enc, {"sharpen": 100, "sharpen_detail": -25})
        full = sharpen(enc, {"sharpen": 100, "sharpen_detail": 75})
        # The halo beside a strong edge is what Detail limits.
        assert abs(held[24, 21, 1] - enc[24, 21, 1]) < abs(full[24, 21, 1] - enc[24, 21, 1])

    def test_masking_spares_fine_texture_and_keeps_edges(self):
        rng = np.random.default_rng(1)
        enc = _edge(low=0.2, high=0.8)
        enc[:, :16] += rng.normal(0, 0.01, (48, 16, 1)).astype(np.float32)   # skin-like texture
        open_ = sharpen(enc, {"sharpen": 100})
        masked = sharpen(enc, {"sharpen": 100, "sharpen_masking": 100})
        texture = lambda out: float(np.abs(out[:, 2:14] - enc[:, 2:14]).mean())
        assert texture(masked) < 0.3 * texture(open_)
        assert _contrast_across(masked, 48) > _contrast_across(enc, 48)

    def test_the_output_stays_in_range(self):
        out = sharpen(_edge(low=0.0, high=1.0), {"sharpen": 150, "sharpen_detail": 75})
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestInThePipeline:
    def _frame(self, scale=1.0):
        rgb = np.full((64, 96, 3), 0.05, dtype=np.float32)
        rgb[:, 48:] = 0.4
        return LinearFrame(rgb, scale=scale)

    def test_develop_frame_sharpens(self):
        frame = self._frame()
        plain = np.asarray(develop_frame(frame, {"exposure": 0.1}), dtype=np.float32)
        sharp = np.asarray(develop_frame(frame, {"exposure": 0.1, "sharpen": 100}), dtype=np.float32)
        assert sharp[32, 48, 1] - sharp[32, 47, 1] > plain[32, 48, 1] - plain[32, 47, 1]

    def test_a_sharpen_only_edit_is_an_edit(self):
        frame = self._frame()
        assert develop_frame(frame, {"sharpen": 100}).tobytes() != frame.encoded().tobytes()

    def test_a_smaller_picture_keeps_the_radius_from_reaching_wider(self):
        # The same 3 px radius on a quarter-size frame is under a pixel of it:
        # the change reaches fewer of its pixels from the edge.
        reach = lambda img: int((np.abs(np.asarray(img, dtype=np.float32)[32, :, 1]
                                        - np.asarray(self._frame().encoded(), dtype=np.float32)[32, :, 1]) > 1).sum())
        full = reach(develop_frame(self._frame(1.0), {"sharpen": 150, "sharpen_radius": 20}))
        small = reach(develop_frame(self._frame(0.25), {"sharpen": 150, "sharpen_radius": 20}))
        assert small < full

    def test_derived_frames_keep_the_scale(self):
        from adjust.render import healed

        frame = self._frame(0.3)
        spot = {"x": 0.2, "y": 0.5, "r": 0.02, "sx": 0.1, "sy": 0.5}
        assert healed(frame, [spot]).scale == 0.3


def test_a_frame_knows_how_much_smaller_than_the_photograph_it_is(tmp_path):
    from export.engine import frame_of

    path = tmp_path / "big.jpg"
    Image.new("RGB", (800, 600), "gray").save(path)
    frame = frame_of(path, np.zeros((150, 200, 3), dtype=np.float32))
    assert frame.scale == 0.25

