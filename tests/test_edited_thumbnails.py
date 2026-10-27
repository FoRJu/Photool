"""
Tests for what an edit does outside the develop view: the RAW decode that gives
it headroom to work with, and the library thumbnails that show it.
"""

import io
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from export import engine
from export.engine import export_photo
from thumbnails.edited import edit_variant, get_edited_thumbnail
from thumbnails.extract import ThumbnailCache, get_thumbnail
from thumbnails.store import ThumbnailSize


def _grey_jpeg(path: Path, value: int, size: int = 600) -> Path:
    Image.new("RGB", (size, size), (value, value, value)).save(path, format="JPEG", quality=95)
    return path


def _mean(data: bytes) -> float:
    return float(np.asarray(Image.open(io.BytesIO(data)).convert("L")).mean())


# ── The cache key ────────────────────────────────────────────────────

class TestVariantKey:
    def test_a_plain_thumbnail_keeps_the_key_it_always_had(self, tmp_path):
        from thumbnails.store import ThumbnailStore

        source = _grey_jpeg(tmp_path / "a.jpg", 100)
        assert ThumbnailStore.cache_key(source) == ThumbnailStore.cache_key(source, "")
        assert ThumbnailStore.cache_key(source) != ThumbnailStore.cache_key(source, "e1-abc")

    def test_the_variant_depends_on_the_edit_and_nothing_else(self):
        a = edit_variant({"exposure": 1.0, "shadows": 20})
        assert a == edit_variant({"shadows": 20, "exposure": 1.0})
        assert a != edit_variant({"exposure": 1.0, "shadows": 21})

    def test_no_edit_is_no_variant(self):
        assert edit_variant({}) == ""
        assert edit_variant({"exposure": 0, "highlights": 0}) == ""
        assert edit_variant(None) == ""


# ── Edited thumbnails ────────────────────────────────────────────────

class TestEditedThumbnails:
    def _cache(self, tmp_path):
        return ThumbnailCache(str(tmp_path / "cache"))

    def test_an_edited_frame_is_drawn_with_its_edit(self, tmp_path):
        source = _grey_jpeg(tmp_path / "a.jpg", 100)
        cache = self._cache(tmp_path)
        plain = get_thumbnail(source, cache, ThumbnailSize.GRID)
        edited = get_edited_thumbnail(source, {"exposure": 1.5}, cache, ThumbnailSize.GRID)
        assert _mean(edited) > _mean(plain) + 30

    def test_the_edit_never_overwrites_the_plain_thumbnail(self, tmp_path):
        """Take the edit off and the camera's rendering is straight back."""
        source = _grey_jpeg(tmp_path / "a.jpg", 100)
        cache = self._cache(tmp_path)
        before = get_thumbnail(source, cache, ThumbnailSize.GRID)
        get_edited_thumbnail(source, {"exposure": 1.5}, cache, ThumbnailSize.GRID)
        assert get_thumbnail(source, cache, ThumbnailSize.GRID) == before

    def test_a_changed_edit_is_a_new_thumbnail_not_a_stale_one(self, tmp_path):
        source = _grey_jpeg(tmp_path / "a.jpg", 100)
        cache = self._cache(tmp_path)
        up = get_edited_thumbnail(source, {"exposure": 1.0}, cache, ThumbnailSize.GRID)
        more = get_edited_thumbnail(source, {"exposure": 2.0}, cache, ThumbnailSize.GRID)
        assert _mean(more) > _mean(up) + 10

    def test_a_second_request_is_served_from_the_cache(self, tmp_path, monkeypatch):
        source = _grey_jpeg(tmp_path / "a.jpg", 100)
        cache = self._cache(tmp_path)
        first = get_edited_thumbnail(source, {"exposure": 1.0}, cache, ThumbnailSize.GRID)

        def boom(*a, **k):
            raise AssertionError("rendered again")

        monkeypatch.setattr("thumbnails.edited.render_edited", boom)
        assert get_edited_thumbnail(source, {"exposure": 1.0}, cache, ThumbnailSize.GRID) == first

    def test_an_unedited_frame_is_not_this_modules_business(self, tmp_path):
        source = _grey_jpeg(tmp_path / "a.jpg", 100)
        assert get_edited_thumbnail(source, {}, self._cache(tmp_path), ThumbnailSize.GRID) is None

    def test_the_preview_size_is_edited_too(self, tmp_path):
        source = _grey_jpeg(tmp_path / "a.jpg", 100, size=1200)
        cache = self._cache(tmp_path)
        out = get_edited_thumbnail(source, {"shadows": 100}, cache, ThumbnailSize.PREVIEW)
        assert out is not None and max(Image.open(io.BytesIO(out)).size) <= 2000

    def test_an_unreadable_file_falls_back_rather_than_raising(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not a picture")
        assert get_edited_thumbnail(bad, {"exposure": 1.0}, self._cache(tmp_path),
                                    ThumbnailSize.GRID) is None


# ── The RAW decode ───────────────────────────────────────────────────

class _FakeRaw:
    """Stands in for a rawpy.RawPy: records how it was asked to decode."""

    def __init__(self, reject=(), value=16384, gains=(2.0, 1.0, 2.0, 0.0)):
        self.sizes = types.SimpleNamespace(width=6000, height=4000)
        self.camera_whitebalance = list(gains)
        self.calls = []
        self.reject = reject
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **kwargs):
        self.calls.append(kwargs)
        for name in self.reject:
            if name in kwargs:
                raise TypeError(f"unexpected {name}")
        return np.full((4, 6, 3), self.value, dtype=np.uint16)


def _install_fake(monkeypatch, raw):
    module = types.ModuleType("rawpy")
    module.imread = lambda path: raw
    module.HighlightMode = types.SimpleNamespace(Clip=0, Ignore=1, Blend=2)
    monkeypatch.setitem(sys.modules, "rawpy", module)


class TestRawLinearDecode:
    def test_it_asks_for_flat_16_bit_linear_that_keeps_the_boosted_channels(self, monkeypatch, tmp_path):
        raw = _FakeRaw()
        _install_fake(monkeypatch, raw)
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        call = raw.calls[0]
        assert call["output_bps"] == 16
        assert call["gamma"] == (1, 1)
        assert call["no_auto_bright"] is True
        assert call["use_camera_wb"] is True
        assert call["highlight_mode"] == 1            # Ignore: do not clip at white balance
        # Left on, LibRaw rescales each frame so its brightest pixel is full scale,
        # and the same exposure setting would then mean something different on
        # every frame.
        assert call["adjust_maximum_thr"] == 0.0
        assert out.dtype == np.float32

    def test_the_normalisation_ignore_applies_is_undone(self, monkeypatch, tmp_path):
        """Ignore divides everything by max(gain)/min(gain). A quarter of full
        scale on the wire, from a camera whose red gain is 2, is half of white."""
        _install_fake(monkeypatch, _FakeRaw(value=16384, gains=(2.0, 1.0, 2.0, 0.0)))
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert float(out.mean()) == pytest.approx(0.5, abs=0.01)

    def test_a_gain_of_one_everywhere_changes_nothing(self, monkeypatch, tmp_path):
        _install_fake(monkeypatch, _FakeRaw(value=16384, gains=(1.0, 1.0, 1.0, 0.0)))
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert float(out.mean()) == pytest.approx(0.25, abs=0.01)

    def test_a_decoder_that_rejects_an_option_still_opens_the_file(self, monkeypatch, tmp_path):
        raw = _FakeRaw(reject=("highlight_mode",), value=16384)
        _install_fake(monkeypatch, raw)
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert out is not None
        assert "highlight_mode" not in raw.calls[-1]
        assert float(out.mean()) == pytest.approx(0.25, abs=0.01)    # no gain scale to undo

    def test_a_file_with_no_white_balance_gains_gets_the_plain_decode(self, monkeypatch, tmp_path):
        raw = _FakeRaw(gains=(0.0, 0.0, 0.0, 0.0), value=16384)
        _install_fake(monkeypatch, raw)
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert "highlight_mode" not in raw.calls[0]
        assert float(out.mean()) == pytest.approx(0.25, abs=0.01)

    def test_a_small_target_uses_the_fast_half_size_decode(self, monkeypatch, tmp_path):
        raw = _FakeRaw()
        _install_fake(monkeypatch, raw)
        engine._load_raw_linear(tmp_path / "a.CR3", 400)
        assert raw.calls[0]["half_size"] is True
        engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert raw.calls[1]["half_size"] is False

    def test_an_unreadable_raw_is_none_not_an_exception(self, monkeypatch, tmp_path):
        module = types.ModuleType("rawpy")

        def boom(path):
            raise RuntimeError("corrupt")

        module.imread = boom
        module.HighlightMode = types.SimpleNamespace(Clip=0, Ignore=1, Blend=2)
        monkeypatch.setitem(sys.modules, "rawpy", module)
        assert engine._load_raw_linear(tmp_path / "a.CR3", 0) is None

    def test_a_jpeg_goes_through_the_same_pipeline(self, tmp_path):
        source = _grey_jpeg(tmp_path / "a.jpg", 128, size=64)
        rgb = engine._load_linear(source, 0)
        assert rgb.shape == (64, 64, 3)
        assert float(rgb.max()) <= 1.0 + 1e-3        # nothing above 255 to keep


class _FakeCameraRaw(_FakeRaw):
    """A three-colour sensor whose file carries a colour matrix, decoded in its
    own camera space: ``image`` is what LibRaw hands back there (white-balanced
    linear light, before the Ignore normalisation)."""

    # Camera to linear sRGB: rows sum to one, so a neutral stays neutral, and the
    # channels are mixed, as every real camera's are.
    RGB_CAM = np.array([[1.6, -0.4, -0.2], [-0.2, 1.5, -0.3], [0.0, -0.5, 1.5]])

    def __init__(self, image, gains=(2.0, 1.0, 2.0, 0.0)):
        super().__init__(gains=gains)
        self.image = np.asarray(image, dtype=np.float64)
        self.num_colors = 3
        srgb_to_xyz = np.array(engine._SRGB_TO_XYZ)
        self.rgb_xyz_matrix = np.vstack([np.linalg.inv(self.RGB_CAM) @ np.linalg.inv(srgb_to_xyz),
                                         np.zeros((1, 3))])

    def postprocess(self, **kwargs):
        self.calls.append(kwargs)
        g = np.asarray(self.camera_whitebalance[:3])
        wire = self.image / (g.max() / g.min())
        return np.clip(np.rint(wire * 65535.0), 0, 65535).astype(np.uint16)


def _install_camera_fake(monkeypatch, raw):
    _install_fake(monkeypatch, raw)
    sys.modules["rawpy"].ColorSpace = types.SimpleNamespace(raw=0, sRGB=1)


class TestDecodingInCameraSpace:
    """With the camera's matrix known, the repair is made before it."""

    def test_it_asks_for_the_cameras_own_primaries_and_applies_the_matrix_itself(self, monkeypatch, tmp_path):
        # Real data has noise; a perfectly flat frame would look like a clipped one.
        noise = 1.0 + 0.004 * np.random.default_rng(1).standard_normal((40, 60, 3))
        image = np.full((40, 60, 3), 0.2) * np.array([1.0, 0.6, 0.3]) * noise
        raw = _FakeCameraRaw(image)
        _install_camera_fake(monkeypatch, raw)
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert raw.calls[0]["output_color"] == 0
        want = np.clip(image.mean(axis=(0, 1)) @ _FakeCameraRaw.RGB_CAM.T, 0, None)
        assert out.mean(axis=(0, 1)) == pytest.approx(want, abs=0.003)

    def test_a_clipped_green_is_rebuilt_before_the_matrix_mixes_it_in(self, monkeypatch, tmp_path):
        """A white sheet lit past green's level: red and blue (after white balance)
        went on to 1.6 and green stopped at 1.0. Repaired in camera space it is
        white; mixed by the matrix first, it would be magenta for good."""
        noise = 1.0 + 0.004 * np.random.default_rng(2).standard_normal((80, 120, 3))
        image = np.full((80, 120, 3), 0.7) * noise           # bright neutral surround
        image[25:55, 35:85] = (1.6, 1.0, 1.6)
        _install_camera_fake(monkeypatch, _FakeCameraRaw(image))
        out = engine._load_raw_linear(tmp_path / "a.CR3", 0)
        r, g, b = out[40, 60]
        assert r == pytest.approx(g, rel=0.05) and b == pytest.approx(g, rel=0.05)
        assert g > 1.2

    def test_without_a_matrix_libraw_converts_and_the_pixel_is_rebuilt_whole(self, monkeypatch, tmp_path):
        raw = _FakeRaw()
        _install_fake(monkeypatch, raw)
        engine._load_raw_linear(tmp_path / "a.CR3", 0)
        assert "output_color" not in raw.calls[0]


class TestColourSmoothing:
    def test_colour_noise_goes_and_brightness_stays(self):
        rng = np.random.default_rng(4)
        luma = 0.2 + 0.6 * np.linspace(0, 1, 120)[None, :] * np.ones((80, 1))
        tint = 1.0 + 0.15 * rng.standard_normal((80, 120, 3))
        rgb = (luma[..., None] * tint).astype(np.float32)
        from adjust.render import LUMA

        out = engine.smooth_chroma(rgb, 3.0)
        chroma = lambda x: x / (x @ LUMA)[..., None]
        assert chroma(out).std(axis=(0, 1)).max() < 0.4 * chroma(rgb).std(axis=(0, 1)).max()
        assert np.allclose(out @ LUMA, rgb @ LUMA, rtol=1e-4)

    def test_no_smoothing_is_no_change(self):
        rgb = np.random.default_rng(0).random((10, 10, 3)).astype(np.float32)
        assert engine.smooth_chroma(rgb, 0.0) is rgb


# ── The RAW decode, against the real LibRaw ──────────────────────────
#
# The tests above pin what the loader asks for. These pin what it gets back,
# from real (synthetic) DNG files: Bayer data with known linear values, a known
# white level and known white-balance gains, decoded by the actual library.

try:
    import rawpy as _rawpy_real  # noqa: F401
    import tifffile
    _HAVE_REAL_RAW = True
except ImportError:                                # the rest of the file still runs
    tifffile = None
    _HAVE_REAL_RAW = False

_WHITE = 16383
_XYZ_TO_SRGB = [3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570]


def _rational(values, den=10000):
    out = []
    for v in values:
        out += [int(round(v * den)), den]
    return tuple(out)


def _write_dng(path: Path, patches, neutral=(1.0, 1.0, 1.0), size=64, noise=0.0):
    """A Bayer RGGB DNG: one flat patch per entry, each (R, G, B) as a fraction
    of the sensor's white level, side by side. ``noise`` is the relative standard
    deviation of the sensor noise: real data always has some, and a perfectly flat
    patch is indistinguishable from a clipped one."""
    raw = np.zeros((size, size * len(patches)))
    for i, (r, g, b) in enumerate(patches):
        x0 = i * size
        raw[0:size:2, x0:x0 + size:2] = r
        raw[0:size:2, x0 + 1:x0 + size:2] = g
        raw[1:size:2, x0:x0 + size:2] = g
        raw[1:size:2, x0 + 1:x0 + size:2] = b
    if noise:
        raw = raw * (1.0 + noise * np.random.default_rng(7).standard_normal(raw.shape))
    extra = [
        (50706, "B", 4, (1, 4, 0, 0), True),           # DNGVersion
        (50708, "s", 0, "Synthetic", True),            # UniqueCameraModel
        (33421, "H", 2, (2, 2), True),                 # CFARepeatPatternDim
        (33422, "B", 4, (0, 1, 1, 2), True),           # CFAPattern: RGGB
        (50714, "H", 1, (0,), True),                   # BlackLevel
        (50717, "H", 1, (_WHITE,), True),              # WhiteLevel
        (50721, "i", 18, _rational(_XYZ_TO_SRGB), True),   # ColorMatrix1
        (50728, "2I", 3, _rational(neutral), True),    # AsShotNeutral
        (50778, "H", 1, (21,), True),                  # CalibrationIlluminant1: D65
    ]
    tifffile.imwrite(
        path, np.clip(raw * _WHITE, 0, _WHITE).astype(np.uint16),
        photometric=32803, compression=None, extratags=extra,
    )
    return path


def _patch(rgb, i, size=64):
    return rgb[size // 2, i * size + size // 2]


@pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
class TestAgainstRealLibRaw:
    def test_exposure_is_the_sensors_not_the_frames(self, tmp_path):
        """The same patch must decode to the same linear value whatever else is in
        the frame. LibRaw's default would rescale each frame to its own brightest
        pixel, and an exposure setting would then mean something else on every
        frame."""
        bright = _write_dng(tmp_path / "a.dng", [(0.18, 0.18, 0.18), (0.9, 0.9, 0.9)])
        dim = _write_dng(tmp_path / "b.dng", [(0.18, 0.18, 0.18), (0.4, 0.4, 0.4)])
        a = engine._load_raw_linear(bright, 0)
        b = engine._load_raw_linear(dim, 0)
        assert _patch(a, 0)[1] == pytest.approx(0.18, abs=0.005)
        assert _patch(b, 0)[1] == pytest.approx(0.18, abs=0.005)
        assert _patch(a, 1)[1] == pytest.approx(0.9, abs=0.01)

    def test_a_channel_the_white_balance_pushes_past_white_is_kept(self, tmp_path):
        """A warm highlight: raw red 0.7 under a red gain of 2 is 1.4 of white.
        Clipped at white balance, that red is lost and the sky turns yellow; kept,
        it can be pulled back."""
        source = _write_dng(tmp_path / "w.dng", [(0.7, 0.9, 0.5)], neutral=(0.5, 1.0, 0.5),
                            noise=0.004)
        out = _patch(engine._load_raw_linear(source, 0), 0)
        assert out[0] == pytest.approx(1.4, abs=0.04)
        assert out[1] == pytest.approx(0.9, abs=0.03)
        assert out[2] == pytest.approx(1.0, abs=0.03)

    def test_a_pixel_the_sensor_clipped_is_white_and_not_magenta(self, tmp_path):
        """All three raw channels at the sensor's limit. Keeping the boosted red
        and blue would leave a magenta cast over every blown highlight.

        Neutral, and at least as bright as its brightest clipped channel: red and
        blue reached twice white before they clipped, so a neutral surface there
        was at least that bright in green too. Claiming less would leave the
        middle of a blown area darker than its partly clipped edges."""
        source = _write_dng(tmp_path / "c.dng", [(1.0, 1.0, 1.0)], neutral=(0.5, 1.0, 0.5))
        out = _patch(engine._load_raw_linear(source, 0), 0)
        assert out[0] == pytest.approx(out[1], rel=0.03)
        assert out[2] == pytest.approx(out[1], rel=0.03)
        assert out[1] >= 1.0

    def test_an_edit_on_a_real_raw_pulls_the_kept_highlight_back(self, tmp_path):
        from adjust.render import LinearFrame, develop_frame

        source = _write_dng(tmp_path / "w.dng", [(0.7, 0.9, 0.5)], neutral=(0.5, 1.0, 0.5),
                            noise=0.004)
        frame = LinearFrame(engine._load_raw_linear(source, 0))
        plain = np.asarray(develop_frame(frame, {"exposure": 0.01}), dtype=float)
        pulled = np.asarray(develop_frame(frame, {"highlights": -100}), dtype=float)
        # The red the camera's white balance pushed past white is a flat 255
        # without the edit, and a real value with it.
        assert plain[32, 32, 0] >= 253
        assert pulled[32, 32, 0] < 230

    def test_the_whole_path_exports_a_raw(self, tmp_path):
        """RAW in, edited JPEG out, through the real export function."""
        source = _write_dng(tmp_path / "IMG_0001.dng", [(0.02, 0.02, 0.02)] * 2)
        out = export_photo(
            source, tmp_path, engine.ExportTemplate(max_resolution=0, strip_exif=False),
            develop={"exposure": 3.0, "shadows": 40},
        )
        (tmp_path / "plain").mkdir()
        plain = export_photo(
            source, tmp_path / "plain", engine.ExportTemplate(max_resolution=0, strip_exif=False),
        )
        assert float(np.asarray(Image.open(out).convert("L")).mean()) > 3 * float(
            np.asarray(Image.open(plain).convert("L")).mean()
        )


class TestLinearCache:
    def test_a_frame_is_decoded_once_for_every_slider_movement(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        source = _grey_jpeg(tmp_path / "a.jpg", 100, size=64)
        first = engine.linear_base(source, 64)
        assert engine.linear_base(source, 64) is first

    def test_a_changed_file_is_not_served_from_the_old_decode(self, tmp_path):
        engine.clear_preview_cache()
        source = _grey_jpeg(tmp_path / "a.jpg", 100, size=64)
        first = engine.linear_base(source, 64)
        _grey_jpeg(source, 30, size=64)
        # The same size in bytes, and written within one tick of Windows' file
        # clock when the suite runs fast: a file changed for real is changed later.
        stamp = source.stat().st_mtime + 2
        os.utime(source, (stamp, stamp))
        assert engine.linear_base(source, 64) is not first

    def test_a_missing_file_is_none(self, tmp_path):
        assert engine.linear_base(tmp_path / "nope.jpg", 64) is None


class TestTheBlurNeedsNoOpenCV:
    """OpenCV is only a faster blur: without it the same Gaussian is computed in
    NumPy, so a machine without it renders what a machine with it does."""

    def _plane(self):
        """What a frame's colour plane looks like: structure at every scale, and a
        little noise on top."""
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:200, 0:300]
        structure = (0.5 + 0.2 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
                     + 0.1 * np.sin(xx / 7.0 + yy / 5.0) + 0.05 * (xx > 150))
        return (structure + 0.02 * rng.standard_normal((200, 300))).astype(np.float32)

    def _without_cv2(self, monkeypatch, plane, sigma):
        monkeypatch.setitem(sys.modules, "cv2", None)          # import cv2 raises
        return engine._blur_plane(plane, sigma)

    @pytest.mark.parametrize("sigma", [1.5, 3.0])
    def test_it_agrees_with_opencv(self, monkeypatch, sigma):
        cv2 = pytest.importorskip("cv2")
        plane = self._plane()
        expected = cv2.GaussianBlur(plane, (0, 0), sigma)
        got = self._without_cv2(monkeypatch, plane, sigma)
        # Exact where the blur is made at full scale; a block-reduced copy for a wide
        # one costs a little on a plane this busy (0.0004 on a real frame's colour).
        assert np.abs(got - expected)[8:-8, 8:-8].mean() < (5e-5 if sigma < 3 else 2e-3)

    @pytest.mark.parametrize("sigma", [1.5, 3.0])
    def test_it_smooths_and_keeps_the_level(self, monkeypatch, sigma):
        plane = self._plane()
        got = self._without_cv2(monkeypatch, plane, sigma)
        assert got.shape == plane.shape
        assert float(np.std(np.diff(got, axis=0))) < 0.5 * float(np.std(np.diff(plane, axis=0)))
        assert float(got.mean()) == pytest.approx(float(plane.mean()), abs=0.003)
