"""Depth: the band a depth mask reaches, lens blur, and how both are stored."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from adjust import depth  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.masks import normalize_mask, weight  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402

MODELS = Path(os.environ.get("PHOTOOL_TEST_MODELS", Path(__file__).resolve().parent.parent / "models"))


def steps(w=400, h=300):
    """Stripes to show blur, the left half far away and the right half near."""
    rgb = np.full((h, w, 3), 0.1, np.float32)
    rgb[:, ::10] = 0.6
    rgb[:, 1::10] = 0.6
    d = np.zeros((h, w), np.float32)
    d[:, w // 2:] = 1.0
    return rgb, d


def detail(rgb):
    return float(np.abs(np.diff(rgb[..., 1], axis=1)).mean())


def framed(rgb, d):
    """A frame whose depth is already known, so no model is needed."""
    frame = LinearFrame(rgb.copy())
    frame._mattes["depth"] = d
    return frame


class TestBand:
    def test_inside_is_whole_and_far_outside_is_nothing(self):
        d = np.array([[0.0, 0.3, 0.5, 0.7, 1.0]], np.float32)
        out = depth.band(d, 80, 40, 0)
        assert out[0, 0] == 0.0 and out[0, 4] == 0.0
        assert out[0, 2] == 1.0 and out[0, 3] == 1.0

    def test_the_ends_of_the_scale_are_not_faded(self):
        d = np.array([[0.0, 1.0]], np.float32)
        assert depth.band(d, 100, 0, 50).tolist() == [[1.0, 1.0]]

    def test_soft_fades_beyond_the_band(self):
        d = np.array([[0.35]], np.float32)
        assert depth.band(d, 80, 40, 0)[0, 0] == 0.0
        assert 0.0 < depth.band(d, 80, 40, 40)[0, 0] < 1.0

    def test_either_way_round(self):
        d = np.linspace(0, 1, 11, dtype=np.float32)[None]
        assert np.array_equal(depth.band(d, 30, 70, 10), depth.band(d, 70, 30, 10))


class TestMask:
    def test_normalized(self):
        m = normalize_mask({"kind": "depth", "near": 300, "far": -4, "soft": 15.4, "exposure": 1, "invert": 1})
        assert (m["near"], m["far"], m["soft"], m["invert"]) == (100, 0, 15, True)

    def test_reaches_the_band_through_the_pipeline(self):
        rgb, d = steps()
        frame = framed(rgb, d)
        out = develop_frame(frame, {"masks": [{"kind": "depth", "near": 100, "far": 60, "soft": 0, "exposure": 2}]})
        out = np.asarray(out, np.float32)
        before = np.asarray(develop_frame(frame, {}), np.float32)
        assert out[:, 300:].mean() > before[:, 300:].mean() + 20
        assert abs(out[:, :100].mean() - before[:, :100].mean()) < 1

    def test_weight_inverts(self):
        matte = np.zeros((10, 10), np.float32)
        out = weight({"kind": "depth", "invert": True}, (10, 10), matte=matte)
        assert out.min() == 1.0


class TestSettings:
    def test_focus_and_range_need_the_amount(self):
        assert normalize({"blur_focus": 40, "blur_sharp": 10}) == {}
        assert normalize({"lens_blur": 30, "blur_focus": 40}) == {"lens_blur": 30, "blur_focus": 40}

    def test_clamped(self):
        assert normalize({"lens_blur": 400})["lens_blur"] == 100

    def test_survives_the_sidecar(self):
        edit = {"lens_blur": 35, "blur_focus": 70, "blur_sharp": 12}
        assert from_crs(to_crs(edit)) == edit


class TestBlur:
    def test_what_is_in_focus_stays_exactly_sharp(self):
        rgb, d = steps()
        out = depth.lens_blur(rgb, d, 1.0, 0.04, 20.0)
        assert np.allclose(out[:, 300:], rgb[:, 300:], atol=1e-4)

    def test_what_is_out_of_focus_loses_its_detail(self):
        rgb, d = steps()
        out = depth.lens_blur(rgb, d, 1.0, 0.04, 20.0)
        assert detail(out[:, :150]) < detail(rgb[:, :150]) * 0.1

    def test_light_is_kept(self):
        rgb, d = steps()
        out = depth.lens_blur(rgb, np.zeros_like(d), 1.0, 0.04, 20.0)
        assert out.mean() == pytest.approx(rgb.mean(), rel=0.02)

    def test_a_sharp_subject_does_not_bleed_into_the_blur_behind_it(self):
        """A bright near object, dark behind it: the halo a plain blur would leave."""
        h, w = 200, 300
        rgb = np.full((h, w, 3), 0.02, np.float32)
        d = np.zeros((h, w), np.float32)
        rgb[:, 150:] = 2.0
        d[:, 150:] = 1.0
        out = depth.lens_blur(rgb, d, 1.0, 0.04, 30.0)
        assert out[:, 140:148].max() < 0.1

    def test_a_blurred_foreground_spreads_over_what_is_behind(self):
        h, w = 200, 300
        rgb = np.full((h, w, 3), 0.02, np.float32)
        d = np.zeros((h, w), np.float32)
        rgb[:, 150:] = 1.0
        d[:, 150:] = 1.0
        out = depth.lens_blur(rgb, d, 0.0, 0.04, 30.0)
        assert out[:, 140:148].min() > 0.1

    def test_a_bright_point_becomes_a_disc(self):
        rgb = np.full((201, 201, 3), 0.01, np.float32)
        rgb[100, 100] = 400.0
        out = depth.lens_blur(rgb, np.zeros((201, 201), np.float32), 1.0, 0.04, 20.0)[..., 0]
        # About even across the disc, and nothing outside it: not a Gaussian's hump.
        assert out[100, 100] == pytest.approx(out[100, 112], rel=0.25)
        assert out[100, 130] < 0.02

    def test_a_far_layer_fades_out_instead_of_stopping_dead(self):
        """Bright sky through a gap in nearer leaves, all out of focus: its light has to thin
        out into the leaves, not stay as bright as the sky and then stop in a hard line."""
        d = np.full((201, 201), 0.3, np.float32)
        rgb = np.full((201, 201, 3), 0.02, np.float32)
        d[95:106, 95:106] = 0.0
        rgb[95:106, 95:106] = 1.0
        out = depth.lens_blur(rgb, d, 1.0, 0.04, 20.0)[100, :, 0]
        assert out.max() < 0.5
        assert np.abs(np.diff(out)).max() < 0.5 * (out.max() - out.min())

    def test_a_draft_uses_fewer_layers_and_keeps_the_picture(self):
        rgb = np.random.default_rng(1).random((120, 160, 3)).astype(np.float32)
        d = np.tile(np.linspace(0.0, 1.0, 160, dtype=np.float32), (120, 1))
        full = depth.lens_blur(rgb, d, 1.0, 0.04, 8.0)
        draft = depth.lens_blur(rgb, d, 1.0, 0.04, 8.0, depth.DRAFT_LAYERS)
        assert depth.DRAFT_LAYERS < depth.LAYERS
        assert np.abs(full - draft).mean() < 0.03

    def test_reduced_working_matches_a_direct_disc(self):
        """Large discs are worked small: the result should be the same picture."""
        rng = np.random.default_rng(3)
        rgb = rng.random((240, 320, 3), np.float32)
        d = np.zeros((240, 320), np.float32)
        out = depth.lens_blur(rgb, d, 1.0, 0.04, 18.0)
        import cv2

        exact = cv2.filter2D(rgb, -1, depth._disc(18.0 * (1.0 - 0.0)), borderType=cv2.BORDER_REFLECT)
        assert np.abs(out - exact)[30:-30, 30:-30].mean() < 0.03

    def test_same_blur_at_any_preview_size(self):
        """The radius is a share of the frame, so a small preview matches a large one."""
        import cv2

        rgb, d = steps(800, 600)
        large = framed(rgb, d)
        small = framed(cv2.resize(rgb, (400, 300), interpolation=cv2.INTER_AREA), d)
        a = depth.blurred(large, 60, 100).rgb
        b = depth.blurred(small, 60, 100).rgb
        a = cv2.resize(a, (400, 300), interpolation=cv2.INTER_AREA)
        assert np.abs(a - b)[:, 20:180].mean() < 0.02

    def test_kept_for_the_same_values(self):
        rgb, d = steps()
        frame = framed(rgb, d)
        assert depth.blurred(frame, 50, 100) is depth.blurred(frame, 50, 100)
        assert depth.blurred(frame, 50, 100) is not depth.blurred(frame, 51, 100)

    def test_through_the_pipeline(self):
        rgb, d = steps()
        frame = framed(rgb, d)
        out = np.asarray(develop_frame(frame, {"lens_blur": 60, "blur_focus": 100}), np.float32)
        before = np.asarray(develop_frame(frame, {}), np.float32)
        assert np.abs(np.diff(out[:, :150, 1], axis=1)).mean() < np.abs(np.diff(before[:, :150, 1], axis=1)).mean() * 0.3

    def test_without_the_model_the_picture_is_left_alone(self, monkeypatch):
        rgb, _ = steps()
        frame = LinearFrame(rgb.copy())
        monkeypatch.setattr(depth, "_models_dir", None)
        monkeypatch.setattr(depth, "_session", None)
        assert depth.blurred(frame, 50) is frame


class TestFocus:
    def test_at_a_point(self):
        d = np.zeros((100, 100), np.float32)
        d[:, 50:] = 0.8
        frame = framed(np.zeros((100, 100, 3), np.float32), d)
        assert depth.depth_at(frame, 0.8, 0.5) == 80
        # Never 0, which would mean automatic.
        assert depth.depth_at(frame, 0.1, 0.5) == 1

    def test_automatic_goes_to_what_is_near_in_the_middle(self):
        d = np.zeros((100, 100), np.float32)
        d[30:70, 30:70] = 0.7
        d[:5, :] = 1.0
        assert depth.auto_focus(d) == pytest.approx(0.7)


@pytest.mark.skipif(not (MODELS / depth.MODEL.name).exists(), reason="the depth model is not here")
class TestModel:
    def test_near_is_nearer_than_far(self):
        """A ball on a floor seen from above the horizon: the bottom of a picture of a
        receding plane is nearer than its top."""
        depth.configure(str(MODELS))
        h, w = 400, 600
        ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        noise = np.random.default_rng(5).random((h, w), np.float32)
        grey = (0.15 + 0.5 * ys) * (0.7 + 0.3 * noise)
        frame = LinearFrame(np.stack([grey, grey, grey], axis=-1))
        d = depth.whole_map(frame)
        assert d.dtype == np.float32
        assert d.min() >= 0.0 and d.max() <= 1.0
        assert d is depth.whole_map(frame)
