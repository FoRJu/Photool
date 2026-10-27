"""
Noise reduction: the sliders' blend, the mosaic round trip the model is run
inside, the develop pipeline's use of it, and keeping what was made.

The model itself is not run: an identity stands in for it, so what is tested is
everything around it — that the frame comes back to the file's own mosaic, and
that the sliders, the spots and the views use the result as they should.
"""

import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import noise
from adjust.develop import from_crs, normalize, to_crs
from adjust.render import LUMA, LinearFrame, develop_frame, noise_reduced
from test_develop_routes import FOLDER, PHOTO, client  # noqa: F401  (a fixture)
from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng


def _noisy(seed=0, shape=(40, 60)):
    rng = np.random.default_rng(seed)
    clean = np.full(shape + (3,), 0.2, np.float32)
    clean[:, shape[1] // 2:] = (0.05, 0.3, 0.1)
    return clean, clean + rng.normal(0, 0.03, clean.shape).astype(np.float32)


class TestBlend:
    def test_none_taken_out_is_the_frame_and_all_is_the_denoised_one(self):
        clean, noisy = _noisy()
        assert np.allclose(noise.blend(noisy, clean, 0, 0), noisy, atol=1e-6)
        assert np.allclose(noise.blend(noisy, clean, 100, 100), clean, atol=1e-6)

    def test_luminance_alone_leaves_the_colour_noise(self):
        clean, noisy = _noisy()
        out = noise.blend(noisy, clean, 100, 0)
        # The grey of the noise is gone; what is left has no luma.
        assert np.allclose((out - clean) @ LUMA, 0.0, atol=1e-5)
        assert np.abs(out - clean).max() > 0.01

    def test_colour_alone_leaves_a_grey_grain(self):
        clean, noisy = _noisy()
        out = noise.blend(noisy, clean, 0, 100)
        left = out - clean
        assert np.allclose(left[..., 0], left[..., 1], atol=1e-5)
        assert np.allclose(left[..., 1], left[..., 2], atol=1e-5)
        assert np.allclose(left @ LUMA, (noisy - clean) @ LUMA, atol=1e-5)

    def test_amounts_outside_the_range_are_held_to_it(self):
        clean, noisy = _noisy()
        assert np.allclose(noise.blend(noisy, clean, 250, -5), noise.blend(noisy, clean, 100, 0))


class TestSettings:
    def test_the_sliders_are_part_of_an_edit_and_held_to_0_100(self):
        assert normalize({"nr_luma": 70, "nr_color": 140}) == {"nr_luma": 70, "nr_color": 100}
        assert normalize({"nr_luma": -3}) == {}

    def test_they_survive_the_sidecar_under_names_of_their_own(self):
        crs = to_crs({"nr_luma": 60, "nr_color": 100})
        assert crs["crs:PhotoolNoiseLuminance"] == "60"
        assert "crs:LuminanceSmoothing" not in crs and "crs:ColorNoiseReduction" not in crs
        assert from_crs(crs) == {"nr_luma": 60, "nr_color": 100}

    def test_lightrooms_own_noise_settings_do_not_turn_it_on(self):
        assert from_crs({"crs:ColorNoiseReduction": "25", "crs:LuminanceSmoothing": "10"}) == {}

    def test_wanted(self):
        assert not noise.wanted({}) and not noise.wanted({"exposure": 1})
        assert noise.wanted({"nr_color": 50})


class TestThroughTheDevelopPipeline:
    def _frame(self):
        clean, noisy = _noisy(shape=(80, 120))
        frame = LinearFrame(noisy)
        asked = []

        def denoiser(wait=True):
            asked.append(wait)
            return clean

        frame.denoiser = denoiser
        return frame, clean, asked

    def test_the_sliders_draw_the_denoised_frame(self):
        frame, clean, _ = self._frame()
        out = np.asarray(develop_frame(frame, {"nr_luma": 100, "nr_color": 100}), np.float32)
        want = np.asarray(LinearFrame(clean).encoded(), np.float32)
        assert np.abs(out - want).mean() < 1.0

    def test_without_a_denoised_twin_the_frame_is_drawn_as_it_is(self):
        frame, _, _ = self._frame()
        frame.denoiser = lambda wait=True: None
        a = np.asarray(develop_frame(frame, {"nr_luma": 100, "exposure": 0.5}), np.float32)
        b = np.asarray(develop_frame(frame, {"exposure": 0.5}), np.float32)
        assert np.abs(a - b).max() < 1.0

    def test_the_preview_does_not_wait_for_it(self):
        frame, _, asked = self._frame()
        develop_frame(frame, {"nr_luma": 50}, noise_wait=False)
        assert asked == [False]

    def test_the_blend_is_kept_for_its_values(self):
        frame, _, asked = self._frame()
        first = noise_reduced(frame, 50, 100)
        assert noise_reduced(frame, 50, 100) is first and len(asked) == 1
        assert noise_reduced(frame, 60, 100) is not first

    def test_it_comes_before_the_spots(self):
        """A spot is healed from the clean picture: blending afterwards would put
        the unhealed spot back from the denoised twin."""
        frame, clean, _ = self._frame()
        clean[30:40, 10:20] = 0.9            # a speck in both versions
        frame.rgb[30:40, 10:20] = 0.9
        spot = {"x": 15 / 120, "y": 35 / 80, "r": 0.06, "sx": 0.8, "sy": 0.5}
        out = np.asarray(develop_frame(frame, {"nr_luma": 100, "nr_color": 100, "spots": [spot]}), np.float32)
        assert out[33:37, 13:17].mean() < 200


class TestTheMosaicRoundTrip:
    def test_malvar_keeps_a_flat_colour_flat(self):
        cfa = np.empty((32, 32), np.float32)
        cfa[0::2, 0::2], cfa[0::2, 1::2], cfa[1::2, 0::2], cfa[1::2, 1::2] = 0.6, 0.4, 0.4, 0.2
        rgb = noise._malvar(cfa)
        assert np.allclose(rgb[4:-4, 4:-4], (0.6, 0.4, 0.2), atol=1e-5)

    @pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
    def test_an_identity_model_gives_back_the_files_own_mosaic(self, tmp_path, monkeypatch):
        import rawpy

        dng = _write_dng(tmp_path / "IMG_0001.dng", [(v, v, v) for v in np.geomspace(0.01, 0.8, 12)])

        class Identity:
            def run(self, names, feeds):
                return [feeds["input"]]

        monkeypatch.setattr(noise, "_get_session", lambda: Identity())
        monkeypatch.setattr(noise, "_iso_of", lambda source: 6400.0)
        # The synthetic DNG carries no matrix LibRaw exposes; any proper one will do.
        monkeypatch.setattr(noise, "_camera_to_rec2020", lambda raw: np.array(
            [[1.2, -0.15, -0.05], [-0.1, 1.15, -0.05], [0.0, -0.1, 1.1]]))
        out = noise._compute(dng)
        with rawpy.imread(str(dng)) as raw:
            original = raw.raw_image_visible.astype(np.float32)
            white = float(raw.white_level)
        assert out.shape == original.shape and out.dtype == np.uint16
        # Flat patches come back to the same level; edges between patches are
        # where demosaicing and the colour transform do not commute.
        inner = (slice(8, -8), slice(8, -8))
        assert np.median(np.abs(out[inner] - original[inner])) / white < 0.002


    def test_where_the_models_input_is_cut_off_the_original_is_kept(self):
        rgb = np.full((64, 64, 3), 0.3, np.float32)
        rgb[20:40, 20:40, 0] = 2.5          # a bright warm colour, red far past 1
        cut = noise.cut_off(rgb)
        assert cut[25:35, 25:35].min() == 1.0
        assert cut[22, 22] == 1.0           # and a little round it
        assert cut[:5, :5].max() == 0.0 and cut[60:, 60:].max() == 0.0
        assert not noise.cut_off(np.full((8, 8, 3), 0.5, np.float32)).any()

    @pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
    def test_a_bright_warm_colour_comes_back_as_the_sensor_had_it(self, tmp_path, monkeypatch):
        # IMG_6937: a matrix that triples red takes a lit riser past 1; cut off
        # there, the model gave back too little red, and the highlight repair,
        # borrowing its colour from those pixels, turned the riser green.
        import rawpy

        dng = _write_dng(tmp_path / "IMG_0002.dng", [(0.05, 0.05, 0.05)] * 6 + [(0.6, 0.6, 0.6)] * 6)

        class Darken:
            """A model that gets everything wrong, so whatever it touched shows."""
            def run(self, names, feeds):
                return [feeds["input"] * np.float16(0.5)]

        monkeypatch.setattr(noise, "_get_session", lambda: Darken())
        monkeypatch.setattr(noise, "_iso_of", lambda source: 3200.0)
        monkeypatch.setattr(noise, "_camera_to_rec2020", lambda raw: np.array(
            [[3.0, -0.4, 0.1], [0.25, 1.1, -0.35], [0.15, -0.2, 1.5]]))
        out = noise._compute(dng).astype(np.float32)
        with rawpy.imread(str(dng)) as raw:
            original = raw.raw_image_visible.astype(np.float32)
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
        level = (original - black) / (white - black)
        bright, dark = level > 0.4, level < 0.1
        assert np.array_equal(out[bright], original[bright])
        # The dark half went through the model.
        assert np.median(out[dark]) < np.median(original[dark])


@pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
def test_a_file_without_a_colour_matrix_is_refused_with_a_reason(tmp_path, monkeypatch):
    dng = _write_dng(tmp_path / "IMG_0003.dng", [(0.2, 0.2, 0.2)] * 4)
    monkeypatch.setattr(noise, "_get_session", lambda: (_ for _ in ()).throw(AssertionError("ran")))
    with pytest.raises(ValueError, match="colour matrix"):
        noise._compute(dng)


class TestKeptMosaics:
    @pytest.fixture
    def configured(self, tmp_path, monkeypatch):
        monkeypatch.setattr(noise, "available", lambda: True)
        noise.configure(str(tmp_path / "models"), str(tmp_path / "denoised"))
        yield tmp_path
        noise.configure(None, None)

    def _source(self, tmp_path):
        path = tmp_path / "IMG_0002.CR3"
        path.write_bytes(b"raw")
        return path

    def test_the_view_gets_none_at_first_then_the_mosaic(self, configured, monkeypatch):
        source = self._source(configured)
        gate = threading.Event()
        made = np.arange(12, dtype=np.uint16).reshape(3, 4)

        def compute(path):
            gate.wait(10)
            return made

        monkeypatch.setattr(noise, "_compute", compute)
        assert noise.mosaic_for(source, wait=False) is None
        assert noise.status(source) == "working"
        gate.set()
        for _ in range(100):
            if noise.status(source) == "ready":
                break
            time.sleep(0.05)
        assert np.array_equal(noise.mosaic_for(source, wait=False), made)

    def test_it_is_kept_on_disk_across_a_restart(self, configured, monkeypatch):
        source = self._source(configured)
        made = np.ones((4, 4), np.uint16) * 7
        monkeypatch.setattr(noise, "_compute", lambda path: made)
        noise.mosaic_for(source, wait=True)
        noise.configure(str(configured / "models"), str(configured / "denoised"))
        monkeypatch.setattr(noise, "_compute", lambda path: (_ for _ in ()).throw(AssertionError("made again")))
        assert noise.status(source) == "ready"
        assert np.array_equal(noise.mosaic_for(source, wait=True), made)

    def test_a_failure_is_reported_and_not_retried(self, configured, monkeypatch):
        source = self._source(configured)
        calls = []

        def broken(path):
            calls.append(path)
            raise ValueError("not a Bayer sensor")

        monkeypatch.setattr(noise, "_compute", broken)
        assert noise.mosaic_for(source, wait=True) is None
        assert noise.mosaic_for(source, wait=True) is None
        assert noise.status(source) == "failed" and "Bayer" in noise.failure(source)
        assert len(calls) == 1

    def test_a_changed_file_is_made_again(self, configured, monkeypatch):
        source = self._source(configured)
        monkeypatch.setattr(noise, "_compute", lambda path: np.zeros((2, 2), np.uint16))
        noise.mosaic_for(source, wait=True)
        source.write_bytes(b"a different raw")
        assert noise.status(source) == "none"


class TestThePreviewSaysWhereItStands:
    def _preview(self, client, adjustment):
        return client.post("/api/export/preview", json={
            "filepath": PHOTO, "linear": True, "size": 256, "adjustment": adjustment})

    def test_off_when_not_asked_for(self, client):
        res = self._preview(client, {"exposure": 0.3})
        assert res.status_code == 200 and res.headers["X-Noise-Reduction"] == "off"

    def test_a_jpeg_is_told_it_works_on_raws_only(self, client):
        res = self._preview(client, {"nr_luma": 60})
        assert res.status_code == 200 and res.headers["X-Noise-Reduction"] == "raw-only"
