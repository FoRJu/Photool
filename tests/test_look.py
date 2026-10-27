"""
The camera's look: learned from the JPEG embedded beside the RAW data, so a RAW
frame starts from what the camera made and the edit is made on top of it.

The camera is simulated: a known scene in linear light is rendered the way a
camera's image processor does it — a colour matrix that adds saturation, then one
tone curve applied to every channel — to stand in for the embedded JPEG, so the
tests can say whether the fit recovers what was done to it. A second simulated
camera, which instead tones brightness alone and adds saturation on the display
side, checks that a rendering built differently is still followed closely.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import from_crs, normalize, to_crs
from adjust.look import STOPS, apply_look, fit_camera_look
from adjust.render import LUMA, LinearFrame, auto_develop, decode, develop_frame, encode
from export import engine
from test_develop_routes import client, FOLDER  # noqa: F401  (a fixture)
from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng

CAMERA_SATURATION = 1.3


def _tone(linear: np.ndarray) -> np.ndarray:
    """A punchy Picture Style's tone: lifted mid-tones, a firm contrast and a
    white that clips."""
    return np.clip(1.12 * encode(np.clip(linear, 0.0, 1.0)) ** 0.82 - 0.03, 0.0, 1.0)


def _camera_render(linear: np.ndarray) -> np.ndarray:
    """What a camera's image processor makes of linear light: a colour matrix
    that boosts saturation, then the same tone curve on every channel."""
    s = CAMERA_SATURATION
    boost = s * np.eye(3) - (s - 1.0) * np.outer(np.ones(3), LUMA)
    shown = _tone(np.maximum(linear @ boost.T, 0.0))
    return np.clip(np.rint(shown * 255.0), 0, 255).astype(np.uint8)


def _luma_camera_render(linear: np.ndarray) -> np.ndarray:
    """A rendering built the other way: tone on brightness alone with the colour
    ratios carried along, and saturation added to the finished picture."""
    luma = np.maximum(linear @ LUMA, 1e-6)
    shown = _tone(luma)
    enc = encode(linear * (decode(shown) / luma)[..., None])
    enc = enc + (shown - enc @ LUMA)[..., None]
    enc = shown[..., None] + (enc - shown[..., None]) * CAMERA_SATURATION
    return np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)


def _scene(h=90, w=140, seed=3) -> np.ndarray:
    """About ten stops of brightness, in colours all the way round the hue
    circle, with texture, and some of it above white."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    stops = -8.0 + 9.5 * (0.6 * x + 0.4 * y) + 0.6 * rng.standard_normal((h, w))
    luma = 0.18 * 2.0 ** (stops + 2.0)
    hue = 2.0 * np.pi * (3.0 * x + 2.0 * y)
    tint = 1.0 + 0.35 * np.stack(
        [np.cos(hue), np.cos(hue - 2.0 * np.pi / 3.0), np.cos(hue + 2.0 * np.pi / 3.0)], -1)
    tint = tint / (tint @ LUMA)[..., None] * (0.7 + 0.6 * x)[..., None]
    return np.clip(luma[..., None] * tint, 0, 6.0).astype(np.float32)


def _error(a: Image.Image, b: np.ndarray) -> tuple[float, float]:
    """Mean luma and colour disagreement, in fractions of full scale."""
    a = np.asarray(a, dtype=np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    luma = float(np.mean(np.abs(a @ LUMA - b @ LUMA)))
    ca, cb = a - (a @ LUMA)[..., None], b - (b @ LUMA)[..., None]
    return luma, float(np.mean(np.abs(ca - cb)))


def _hue_styled_render(linear: np.ndarray) -> np.ndarray:
    """A camera whose Picture Style also moves hues: greens turned towards yellow,
    paler and lighter, which no colour matrix can do without moving the rest."""
    from adjust import look as L
    base = _camera_render(linear).astype(np.float32) / 255.0
    lab = L._oklab(base)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    angle = np.arctan2(lab[..., 2], lab[..., 1])
    green = np.exp(-0.5 * ((np.angle(np.exp(1j * (angle - 2.4)))) / 0.5) ** 2)
    fade = chroma / (chroma + L._HUE_FADE)
    out = np.empty_like(lab)
    out[..., 0] = lab[..., 0] * (1.0 + 0.20 * green * fade)
    out[..., 1] = chroma * (1.0 - 0.40 * green * fade) * np.cos(angle - 0.40 * green * fade)
    out[..., 2] = chroma * (1.0 - 0.40 * green * fade) * np.sin(angle - 0.40 * green * fade)
    return np.clip(np.rint(L._from_oklab(out) * 255.0), 0, 255).astype(np.uint8)


class TestHue:
    def test_zero_terms_leave_a_picture_alone(self):
        from adjust.look import HUE_TERMS, shift_hue
        picture = np.random.default_rng(1).random((20, 30, 3)).astype(np.float32)
        out = shift_hue(picture, np.zeros((3, HUE_TERMS), dtype=np.float32))
        assert np.max(np.abs(out - picture)) < 2e-3

    def test_a_hue_the_matrix_cannot_make_is_learned_and_drawn(self):
        scene = _scene(seed=5)
        camera = _hue_styled_render(scene)
        look = fit_camera_look(scene, camera)
        assert look is not None and look.hue is not None
        frame = LinearFrame(scene)
        frame.look = look
        with_hue = _error(frame.encoded(), camera)

        look.hue = None
        frame = LinearFrame(scene)
        frame.look = look
        without = _error(frame.encoded(), camera)
        assert with_hue[0] + with_hue[1] < 0.8 * (without[0] + without[1])

    def test_a_colourless_picture_gets_no_hue_terms(self):
        scene = _scene(seed=2)
        grey = np.repeat((scene @ LUMA)[..., None], 3, axis=-1)
        camera = _camera_render(grey)
        look = fit_camera_look(grey, camera)
        assert look is None or look.hue is None

    def test_amount_dials_the_hue_back_with_the_rest(self):
        scene = _scene(seed=5)
        look = fit_camera_look(scene, _hue_styled_render(scene))
        full = apply_look(scene, look, 1.0)
        none = apply_look(scene, look, 0.0)
        half = apply_look(scene, look, 0.5)
        assert np.mean(np.abs(half - none)) < np.mean(np.abs(full - none))


class TestFit:
    def test_it_recovers_the_cameras_rendering(self):
        scene = _scene()
        camera = _camera_render(scene)
        look = fit_camera_look(scene, camera)
        assert look is not None and look.error < 0.03
        frame = LinearFrame(scene)
        frame.look = look
        luma_err, colour_err = _error(frame.encoded(), camera)
        assert luma_err < 0.02
        assert colour_err < 0.02

    def test_without_the_look_the_decode_is_far_from_the_camera(self):
        """The problem this solves: the flat decode is nowhere near the JPEG."""
        scene = _scene()
        camera = _camera_render(scene)
        flat = LinearFrame(scene).encoded()
        looked = LinearFrame(scene)
        looked.look = fit_camera_look(scene, camera)
        assert _error(flat, camera)[0] > 3 * _error(looked.encoded(), camera)[0]
        assert _error(flat, camera)[1] > 3 * _error(looked.encoded(), camera)[1]

    def test_a_rendering_built_differently_is_still_followed_closely(self):
        scene = _scene()
        camera = _luma_camera_render(scene)
        frame = LinearFrame(scene)
        frame.look = fit_camera_look(scene, camera)
        assert frame.look is not None
        luma_err, colour_err = _error(frame.encoded(), camera)
        assert luma_err < 0.02
        assert colour_err < 0.02

    def test_the_cameras_saturation_is_learned(self):
        scene = _scene()
        look = fit_camera_look(scene, _camera_render(scene))
        # A matrix that boosts colour has singular values above one in the colour plane.
        assert np.linalg.svd(look.matrix)[1].max() > 1.1

    def test_a_curve_never_reverses_the_picture(self):
        scene = _scene()
        look = fit_camera_look(scene, _camera_render(scene))
        assert np.all(np.diff(look.curve) >= 0)
        assert look.curve[0] >= 0 and look.curve[-1] <= 1.0

    def test_it_knows_where_the_camera_reached_white(self):
        scene = _scene()
        look = fit_camera_look(scene, _camera_render(scene))
        # The camera clips well below the scene's brightest value.
        assert look.white_stops < np.log2(scene.max() / 0.18) - 1.0


class TestAMonochromeCamera:
    """A black-and-white image control writes a JPEG with no colour in it."""

    def test_the_look_learns_to_remove_the_colour(self):
        scene = _scene()
        camera = _camera_render(scene)
        grey = np.repeat(np.rint((camera.astype(float) @ LUMA))[..., None], 3, axis=2).astype(np.uint8)
        look = fit_camera_look(scene, grey)
        assert look is not None
        chroma = lambda image: float(np.abs(
            (np.asarray(image, dtype=float) / 255.0)
            - ((np.asarray(image, dtype=float) / 255.0) @ LUMA)[..., None]).mean())
        flat = LinearFrame(scene)
        looked = LinearFrame(scene)
        looked.look = look
        assert chroma(looked.encoded()) < 0.005
        assert chroma(looked.encoded()) < 0.3 * chroma(flat.encoded())


class TestWhatItWillNotFit:
    def test_a_different_crop_is_not_the_same_picture(self):
        scene = _scene()
        camera = _camera_render(scene)[:, :100]
        assert fit_camera_look(scene, camera) is None

    def test_an_unrelated_picture_is_not_trusted(self):
        scene = _scene()
        noise = np.random.default_rng(0).integers(0, 256, (90, 140, 3), dtype=np.uint8)
        assert fit_camera_look(scene, noise) is None

    def test_a_flat_frame_gives_nothing_to_fit(self):
        flat = np.full((60, 90, 3), 0.18, dtype=np.float32)
        assert fit_camera_look(flat, np.full((60, 90, 3), 120, dtype=np.uint8)) is None


class TestRescueStillWorksUnderTheLook:
    """The point of applying the look last: the camera's white is a display
    limit, not a limit on the data."""

    def _frame(self):
        scene = _scene()
        frame = LinearFrame(scene)
        frame.look = fit_camera_look(scene, _camera_render(scene))
        return scene, frame

    def test_the_unedited_frame_clips_where_the_camera_did(self):
        scene, frame = self._frame()
        bright = scene @ LUMA > 1.8
        assert bright.sum() > 100
        assert np.asarray(frame.encoded())[bright][:, 1].min() >= 250

    def test_pulling_highlights_shows_what_the_camera_threw_away(self):
        scene, frame = self._frame()
        bright = scene @ LUMA > 1.8
        pulled = np.asarray(develop_frame(frame, {"highlights": -100}))[bright][:, 1].astype(float)
        assert pulled.mean() < 240
        assert pulled.std() > 3                    # texture, not a flat patch

    def test_lowering_exposure_does_the_same(self):
        scene, frame = self._frame()
        bright = scene @ LUMA > 1.8
        down = np.asarray(develop_frame(frame, {"exposure": -1.5}))[bright][:, 1].astype(float)
        assert down.mean() < 245 and down.std() > 2

    def test_the_shadows_open_from_the_cameras_starting_point(self):
        scene, frame = self._frame()
        dark = scene @ LUMA < 0.01
        before = np.asarray(frame.encoded())[dark][:, 1].mean()
        after = np.asarray(develop_frame(frame, {"shadows": 100}))[dark][:, 1].mean()
        assert after > before + 20


class TestDialingItBack:
    def test_flat_all_the_way_is_the_plain_decode(self):
        scene = _scene()
        looked, plain = LinearFrame(scene), LinearFrame(scene)
        looked.look = fit_camera_look(scene, _camera_render(scene))
        assert np.array_equal(
            np.asarray(develop_frame(looked, {"flat": 100})),
            np.asarray(develop_frame(plain, {"flat": 100})),
        )

    def test_halfway_is_between(self):
        scene = _scene()
        frame = LinearFrame(scene)
        frame.look = fit_camera_look(scene, _camera_render(scene))
        mean = lambda s: float(np.asarray(develop_frame(frame, s), dtype=float).mean())
        camera, half, plain = mean({}), mean({"flat": 50}), mean({"flat": 100})
        assert min(camera, plain) < half < max(camera, plain)

    def test_a_frame_with_no_look_ignores_the_control(self):
        frame = LinearFrame(_scene())
        assert np.array_equal(
            np.asarray(develop_frame(frame, {"flat": 40, "exposure": 0.01})),
            np.asarray(develop_frame(frame, {"exposure": 0.01})),
        )

    def test_a_frame_with_a_look_starts_from_it(self):
        scene = _scene()
        frame = LinearFrame(scene)
        flat = np.asarray(frame.encoded(), dtype=int)
        frame.look = fit_camera_look(scene, _camera_render(scene))
        assert np.abs(np.asarray(frame.encoded(), dtype=int) - flat).mean() > 10


class TestSettingsAndAuto:
    def test_the_control_is_stored_as_the_distance_from_the_camera(self):
        assert normalize({"flat": 0}) == {}
        assert normalize({"flat": 100}) == {"flat": 100}
        assert normalize({"flat": 250}) == {"flat": 100}

    def test_it_survives_the_sidecar(self):
        assert from_crs(to_crs({"flat": 35, "exposure": 0.5})) == {"flat": 35, "exposure": 0.5}

    def test_auto_leaves_a_frame_the_camera_exposed_well_alone_and_goes_for_the_highlights(self):
        scene = _scene()
        frame = LinearFrame(scene)
        frame.look = fit_camera_look(scene, _camera_render(scene))
        out = auto_develop(frame)
        assert "exposure" not in out
        assert out.get("highlights", 0) < 0

    def test_auto_brightens_a_frame_the_camera_left_dark(self):
        scene = _scene() * 0.02
        frame = LinearFrame(scene)
        frame.look = fit_camera_look(scene, _camera_render(scene))
        assert (auto_develop(frame) or {}).get("exposure", 0) > 0


# ── Through the engine, on a real RAW ────────────────────────────────

@pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
class TestOnARealRaw:
    LEVELS = np.geomspace(0.002, 0.95, 28)

    def _dng_and_jpeg(self, tmp_path):
        patches = [(v, v, v) for v in self.LEVELS]
        dng = _write_dng(tmp_path / "IMG_0009.dng", patches)
        linear = np.stack([np.repeat(self.LEVELS[None, :], 8, 0)] * 3, -1)
        linear = np.repeat(linear, 64, axis=1).astype(np.float32)
        linear = np.repeat(linear, 8, axis=0)
        camera = _camera_render(linear)
        buf = io.BytesIO()
        Image.fromarray(camera).save(buf, format="JPEG", quality=98)
        return dng, buf.getvalue(), camera

    def _patch_extractor(self, monkeypatch, jpeg):
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: jpeg)
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)

    def test_a_raw_frame_learns_its_look_from_its_embedded_jpeg(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        dng, jpeg, camera = self._dng_and_jpeg(tmp_path)
        self._patch_extractor(monkeypatch, jpeg)
        frame = engine.linear_base(dng, 1800)
        assert frame is not None and frame.look is not None
        shown = np.asarray(develop_frame(frame, {}).resize((camera.shape[1], camera.shape[0])))
        assert _error(Image.fromarray(shown), camera)[0] < 0.03

    def test_a_file_without_an_embedded_jpeg_is_rendered_flat(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        dng, _, _ = self._dng_and_jpeg(tmp_path)
        self._patch_extractor(monkeypatch, None)
        assert engine.linear_base(dng, 1800).look is None

    def test_the_answer_is_kept_per_file(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        dng, jpeg, _ = self._dng_and_jpeg(tmp_path)
        self._patch_extractor(monkeypatch, jpeg)
        first = engine.linear_base(dng, 1800).look
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg",
                            lambda s: (_ for _ in ()).throw(AssertionError("read again")))
        engine._LINEAR_CACHE.clear()
        assert engine.linear_base(dng, 1800).look is first

    def test_the_export_starts_from_the_camera_look_too(self, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        dng, jpeg, camera = self._dng_and_jpeg(tmp_path)
        self._patch_extractor(monkeypatch, jpeg)
        out = engine.export_photo(
            dng, tmp_path, engine.ExportTemplate(max_resolution=0, strip_exif=False),
            develop={"exposure": 0.01},
        )
        shown = Image.open(out).convert("RGB").resize((camera.shape[1], camera.shape[0]))
        assert _error(shown, camera)[0] < 0.04

    def test_an_export_with_no_edit_at_all_still_starts_from_the_camera_look(
        self, tmp_path, monkeypatch
    ):
        """No edit is not "no rendering". The plain decoder's render of the sensor
        data is a darker, differently coloured picture than the one the camera
        made, so a frame that asks for nothing has to be given the camera's own
        look first."""
        engine.clear_preview_cache()
        dng, jpeg, camera = self._dng_and_jpeg(tmp_path)
        self._patch_extractor(monkeypatch, jpeg)
        out = tmp_path / "out"
        out.mkdir()
        written = engine.export_photo(
            dng, out, engine.ExportTemplate(max_resolution=0, strip_exif=False),
        )
        shown = Image.open(written).convert("RGB").resize((camera.shape[1], camera.shape[0]))
        assert _error(shown, camera)[0] < 0.04
        # The flat decode this used to write is far from the camera's picture. How far
        # depends on the LibRaw version (0.047 with LibRaw 0.22, against 0.006 for the
        # export), so it is held to being several times further off, not to a number.
        flat = engine._load_image(dng, 0).resize((camera.shape[1], camera.shape[0]))
        assert _error(flat, camera)[0] > 5 * _error(shown, camera)[0]


@pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
class TestTheJpegBesideTheRaw:
    """RAW+JPEG: the camera's full rendering sits next to the file."""

    LEVELS = np.geomspace(0.002, 0.95, 28)

    def _pair(self, tmp_path, monkeypatch, embedded):
        engine.clear_preview_cache()
        dng = _write_dng(tmp_path / "GR_0001.dng", [(v, v, v) for v in self.LEVELS])
        linear = np.repeat(np.repeat(
            np.stack([np.repeat(self.LEVELS[None, :], 8, 0)] * 3, -1), 64, axis=1), 8, axis=0
        ).astype(np.float32)
        camera = _camera_render(linear)
        Image.fromarray(camera).save(tmp_path / "GR_0001.JPG", quality=98)
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: embedded)
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
        return dng, camera

    def test_it_is_used_when_the_raw_has_no_embedded_preview(self, tmp_path, monkeypatch):
        dng, camera = self._pair(tmp_path, monkeypatch, embedded=None)
        assert engine.linear_base(dng, 1800).look is not None

    def test_it_is_preferred_to_the_embedded_one(self, tmp_path, monkeypatch):
        """A preview of the wrong picture must not win over the real JPEG."""
        junk = io.BytesIO()
        Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(junk, "JPEG")
        dng, camera = self._pair(tmp_path, monkeypatch, embedded=junk.getvalue())
        frame = engine.linear_base(dng, 1800)
        assert frame.look is not None and frame.look.error < 0.03

    def test_one_that_does_not_match_is_passed_over_for_the_embedded_one(self, tmp_path, monkeypatch):
        dng, camera = self._pair(tmp_path, monkeypatch, embedded=None)
        # A square-cropped JPEG beside a 3:2 RAW is not the same picture.
        Image.fromarray(camera[:, :200]).save(tmp_path / "GR_0001.JPG", quality=98)
        good = io.BytesIO()
        Image.fromarray(camera).save(good, "JPEG", quality=98)
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: good.getvalue())
        assert engine.linear_base(dng, 1800).look is not None


class TestThroughTheApi:
    def test_every_control_the_model_knows_is_one_the_api_accepts(self):
        """A control missing from the request model is a slider that moves and
        saves nothing."""
        from adjust.develop import CURVE_KEYS, SETTINGS_SLIDERS, SLIDER_RANGES
        from api.routes import AdjustmentParams, _develop_payload

        sent = {key: SLIDER_RANGES.get(key, (0, 30))[1] // 2 or 30 for key in SETTINGS_SLIDERS}
        sent["exposure"] = 1.0
        sent.update({key: [[0, 0], [128, 150], [255, 255]] for key in CURVE_KEYS})
        sent["style"] = "a-learned-style"    # the style fade means nothing without one
        assert set(_develop_payload(AdjustmentParams(**sent))) == set(sent)

    def test_the_camera_look_control_is_stored(self, client):
        res = client.put("/api/photos/edit", json={
            "filepath": f"{FOLDER}/IMG_0001.JPG", "flat": 40,
        })
        assert res.json()["settings"] == {"flat": 40}

    def test_a_jpeg_has_no_camera_look_to_dial(self, client):
        res = client.post("/api/export/preview", json={
            "filepath": f"{FOLDER}/IMG_0001.JPG", "size": 128, "linear": True,
            "adjustment": {},
        })
        assert res.status_code == 200
        assert res.headers["x-camera-look"] == "0"
        assert res.headers["x-camera-look-state"] == "none"
        assert res.headers["x-camera-look-frames"] == "0"
        assert res.headers["x-camera-look-hint"] == ""

    @pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
    def test_a_raw_with_an_embedded_jpeg_reports_one(self, client, tmp_path, monkeypatch):
        engine.clear_preview_cache()
        library = Path(client.app.state.config.library_root)
        patches = [(v, v, v) for v in np.geomspace(0.002, 0.95, 28)]
        _write_dng(library / FOLDER / "IMG_0002.CR3", patches)
        linear = np.repeat(np.stack([np.repeat(np.geomspace(0.002, 0.95, 28)[None, :], 8, 0)] * 3, -1), 64, axis=1)
        linear = np.repeat(linear, 8, axis=0).astype(np.float32)
        buf = io.BytesIO()
        Image.fromarray(_camera_render(linear)).save(buf, format="JPEG", quality=98)
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: buf.getvalue())
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
        res = client.post("/api/export/preview", json={
            "filepath": f"{FOLDER}/IMG_0002.CR3", "size": 1800, "linear": True,
            "adjustment": {},
        })
        assert res.status_code == 200
        assert res.headers["x-camera-look"] == "1"
        # A synthetic DNG cannot be told apart from other cameras, so its look is
        # its own alone.
        assert res.headers["x-camera-look-state"] == "alone"
        assert res.headers["x-camera-look-frames"] == "1"


class TestLookStatus:
    """What the develop view says, quietly, about where a frame's look came from."""

    @staticmethod
    def _look(frames):
        from adjust.look import CameraLook
        return CameraLook(curve=np.zeros(len(STOPS), dtype=np.float32), matrix=np.eye(3, dtype=np.float32),
                          white_stops=1.0, error=0.0, frames=frames)

    def test_no_look_is_none(self):
        assert engine.look_status(Path("a.CR3"), None) == ("none", 0, "")

    def test_a_look_from_a_few_frames_is_still_learning(self):
        assert engine.look_status(Path("unseen.CR3"), self._look(1)) == ("learning", 1, "")
        assert engine.look_status(Path("unseen.CR3"), self._look(3))[0] == "learning"

    def test_a_look_from_a_pool_is_ready(self):
        assert engine.look_status(Path("unseen.CR3"), self._look(12)) == ("ready", 12, "")

    def test_a_frame_the_pool_did_not_describe_is_its_own(self, monkeypatch):
        monkeypatch.setitem(engine._DECISIONS, "b.CR3", {"pool": None, "joined": False,
                                                          "reason": "pooled colour draws the picture worse"})
        state, frames, _ = engine.look_status(Path("b.CR3"), self._look(1))
        assert state == "own" and frames >= 1

    def test_a_camera_that_cannot_be_told_apart_says_so(self, monkeypatch):
        monkeypatch.setitem(engine._DECISIONS, "c.CR3", {"pool": None, "joined": False,
                                                          "reason": "camera not identified"})
        assert engine.look_status(Path("c.CR3"), self._look(1)) == ("alone", 1, "")


    def test_a_picture_far_from_the_cameras_is_called_rough(self, monkeypatch):
        monkeypatch.setitem(engine._DECISIONS, "d.CR3", {"pool": None, "joined": True, "reason": "joined",
                                                          "picture_error": 0.05})
        assert engine.look_status(Path("d.CR3"), self._look(12))[2] == "rough"
        monkeypatch.setitem(engine._DECISIONS, "d.CR3", {"pool": None, "joined": True, "reason": "joined",
                                                          "picture_error": 0.02})
        assert engine.look_status(Path("d.CR3"), self._look(12))[2] == ""

    def test_a_scene_nothing_in_the_pool_resembles_suggests_more_frames(self, monkeypatch):
        monkeypatch.setitem(engine._DECISIONS, "e.CR3", {"pool": None, "joined": True, "reason": "joined",
                                                          "picture_error": 0.02})
        monkeypatch.setitem(engine._SCENE_FRAMES, "e.CR3", 0)
        assert engine.look_status(Path("e.CR3"), self._look(12))[2] == "scene"
        monkeypatch.setitem(engine._SCENE_FRAMES, "e.CR3", 3)
        assert engine.look_status(Path("e.CR3"), self._look(12))[2] == ""


class TestScenes:
    """One camera and style is not one tone curve: edits borrow tone from like scenes."""

    @staticmethod
    def _frame(stops, seed=0):
        """Samples and a probe of a scene whose brightness spreads as ``stops``."""
        from adjust.look import FrameSamples, scene_signature
        rng = np.random.default_rng(seed)
        luma = 0.18 * 2.0 ** (rng.normal(0.0, stops, (64, 64)))
        probe = np.repeat(luma[..., None], 3, axis=-1).astype(np.float32)
        lin = np.repeat(np.geomspace(0.003, 0.9, 300)[:, None], 3, axis=1).astype(np.float32)
        return FrameSamples(linear=lin, camera=np.clip(encode(lin) ** 0.9, 0, 1).astype(np.float32),
                            probe_linear=probe, probe_camera=probe, scene=scene_signature(probe, probe))

    def test_exposure_does_not_change_a_scene(self):
        from adjust.look import scene_distance, scene_signature
        a = self._frame(1.5, seed=1)
        brighter = scene_signature(a.probe_linear * 4.0, a.probe_camera)
        assert scene_distance(a.scene, brighter) < 0.05

    def test_a_flat_scene_and_a_contrasty_one_are_far_apart(self):
        from adjust.look import SCENE_RADIUS, scene_distance
        assert scene_distance(self._frame(0.5).scene, self._frame(3.0).scene) > SCENE_RADIUS
        assert scene_distance(self._frame(1.5, 1).scene, self._frame(1.5, 2).scene) < SCENE_RADIUS

    def test_neighbours_are_the_like_scenes_and_none_for_a_new_one(self):
        from adjust.look import nearest_scenes
        flat = [self._frame(0.5, seed=i) for i in range(3)]
        contrasty = [self._frame(3.0, seed=10 + i) for i in range(3)]
        assert set(map(id, nearest_scenes(flat + contrasty, self._frame(0.5, 99)))) == set(map(id, flat))
        assert nearest_scenes(flat, self._frame(3.0, 98)) == []

    def test_unknown_scenes_mean_the_whole_pool(self):
        from adjust.look import FrameSamples, nearest_scenes
        bare = FrameSamples(linear=np.ones((1, 3), np.float32), camera=np.ones((1, 3), np.float32))
        assert nearest_scenes([bare], bare) is None

    def test_a_new_scene_has_no_pool_curve_to_borrow(self):
        from adjust.look import fit_look, with_colour_of
        frames = [self._frame(1.5, seed=i) for i in range(3)]
        pooled = fit_look(frames)
        mine = self._frame(1.5, seed=7)
        assert with_colour_of(pooled, mine, []).pool_curve is None
        assert with_colour_of(pooled, mine, None).pool_curve is not None
        assert with_colour_of(pooled, mine, frames).pool_curve is not None

    def test_scenes_survive_the_pool_being_saved_and_loaded(self, tmp_path):
        engine.set_profile_dir(tmp_path)
        try:
            pool = engine._pool("scenekey")
            pool["a"] = self._frame(1.0, seed=1)
            pool["b"] = self._frame(2.0, seed=2)
            engine._save_pool("scenekey", pool)
            engine._POOLS.clear()
            loaded = engine._pool("scenekey")
            assert np.allclose(loaded["a"].scene, pool["a"].scene)
        finally:
            engine.set_profile_dir(None)

    def test_a_pool_saved_before_scenes_are_kept_still_loads(self, tmp_path):
        engine.set_profile_dir(tmp_path)
        try:
            lin = np.ones((5, 3), np.float32)
            np.savez_compressed(tmp_path / "oldkey.npz", frames=np.array(["a"]), counts=np.array([5]),
                                linear=lin, camera=lin)
            assert engine._pool("oldkey")["a"].scene is None
        finally:
            engine.set_profile_dir(None)


class TestLearningTheFolder:
    def test_the_endpoint_learns_the_raws_of_a_folder_in_the_background(self, client, monkeypatch):
        library = Path(client.app.state.config.library_root)
        (library / FOLDER / "X_0001.CR3").write_bytes(b"not a real raw")
        seen = []
        monkeypatch.setattr(engine, "linear_base",
                            lambda path, size, relearn=False: seen.append((path.name, relearn)))
        res = client.post("/api/camera-look/learn", json={"folder": FOLDER})
        assert res.status_code == 200 and res.json()["total"] >= 1
        for _ in range(100):
            state = client.get("/api/camera-look/learn", params={"folder": FOLDER}).json()
            if state["status"] != "running":
                break
            import time as _t; _t.sleep(0.05)
        # Every frame teaches its pool again, past the look kept for it.
        assert state["status"] == "done" and ("X_0001.CR3", True) in seen
        assert state["done"] == state["total"]

    def test_a_folder_that_does_not_exist_is_refused(self, client):
        assert client.post("/api/camera-look/learn", json={"folder": "nope/nowhere"}).status_code == 404


# ── Registration and pooling ─────────────────────────────────────────

def _textured_scene(h=400, w=600, seed=5) -> np.ndarray:
    """A frame with structure at several scales, for the lens registration to
    find, and colours all round the hue circle."""
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w))
    for k in (4, 16, 64):
        small = rng.standard_normal((h // k + 2, w // k + 2)).astype(np.float32)
        noise += np.asarray(Image.fromarray(small, "F").resize((w, h), Image.BICUBIC))[:h, :w]
    stops = -6.0 + 7.0 * (noise - noise.min()) / np.ptp(noise)
    luma = 0.18 * 2.0 ** (stops + 1.0)
    hue = rng.random() * 6.3 + 0.03 * np.arange(w)[None, :] + 0.02 * np.arange(h)[:, None]
    tint = 1.0 + 0.35 * np.stack(
        [np.cos(hue), np.cos(hue - 2.0 * np.pi / 3.0), np.cos(hue + 2.0 * np.pi / 3.0)], -1)
    tint = tint / (tint @ LUMA)[..., None]
    return np.clip(luma[..., None] * tint, 0, 6.0).astype(np.float32)


def _barrel(image: np.ndarray) -> np.ndarray:
    """The camera's JPEG of the same frame with the lens corrected: a few
    percent larger, and stretched more at the edges than in the middle."""
    from adjust.look import _lens_model

    h, w = image.shape[:2]
    centre, radius = np.array([w / 2.0, h / 2.0]), float(np.hypot(w / 2.0, h / 2.0))
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    src = _lens_model(pts, centre, radius, np.array([0.0, 0.0, 0.97, -0.02, 0.0]))
    x = np.clip(src[:, 0], 0, w - 1.001)
    y = np.clip(src[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    flat = image.reshape(-1, 3).astype(float)
    i = y0 * w + x0
    out = (flat[i] * (1 - fx) * (1 - fy) + flat[i + 1] * fx * (1 - fy)
           + flat[i + w] * (1 - fx) * fy + flat[i + w + 1] * fx * fy)
    return out.reshape(h, w, 3).astype(np.uint8)


class TestALensCorrectedJpeg:
    """The camera's JPEG is corrected for distortion and the RAW is not."""

    def test_it_is_registered_onto_the_raw_before_anything_is_compared(self, monkeypatch):
        import adjust.look as look_module

        scene = _textured_scene()
        camera = _barrel(_camera_render(scene))
        registered = fit_camera_look(scene, camera)
        monkeypatch.setattr(look_module, "_register", lambda a, b: None)
        unregistered = look_module.fit_look([look_module.frame_samples(scene, camera)])
        assert registered is not None
        assert registered.error < 0.02
        assert registered.error < 0.6 * unregistered.error


class TestManyFramesTogether:
    """A frame of a few colours cannot pin the colour matrix down on its own;
    frames of different colours together can."""

    def _one_colour(self, tint, seed):
        scene = _textured_scene(200, 300, seed)
        grey = (scene @ LUMA)[..., None]
        return (grey * np.asarray(tint) / (np.asarray(tint) @ LUMA)).astype(np.float32)

    def _colour_error(self, look, scene):
        frame = LinearFrame(scene)
        frame.look = look
        return _error(frame.encoded(), _camera_render(scene))[1]

    def test_pooled_frames_learn_the_colour_no_single_one_could(self):
        from adjust.look import fit_look, frame_samples

        tints = [(1.0, 0.75, 0.55), (0.6, 1.0, 0.8), (0.8, 0.7, 1.2)]
        frames = [self._one_colour(t, 10 + i) for i, t in enumerate(tints)]
        samples = [frame_samples(f, _camera_render(f)) for f in frames]
        pooled = fit_look(samples)
        alone = fit_look(samples[:1])
        test = _scene()                               # every hue, seen by neither fit
        assert pooled.frames == 3
        assert self._colour_error(pooled, test) < 0.6 * self._colour_error(alone, test)


class TestThePool:
    """Frames are pooled per camera and style, and a frame that does not belong
    is kept out."""

    def _samples(self, render, seed):
        from adjust.look import frame_samples

        scene = _textured_scene(200, 300, seed)
        return frame_samples(scene, render(scene))

    def _own(self, samples):
        from adjust.look import fit_look

        return fit_look([samples])

    @pytest.fixture
    def pool(self, tmp_path, monkeypatch):
        engine.set_profile_dir(tmp_path / "profiles")
        monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
        yield tmp_path / "profiles"
        engine.set_profile_dir(None)

    def test_a_frame_joins_and_the_next_one_gets_both(self, pool, tmp_path):
        a, b = self._samples(_camera_render, 1), self._samples(_camera_render, 2)
        first = engine._pooled_look(tmp_path / "A.CR3", a, self._own(a))
        second = engine._pooled_look(tmp_path / "B.CR3", b, self._own(b))
        assert first.frames == 1 and second.frames == 2

    def test_a_frame_rendered_otherwise_keeps_its_own_look_and_stays_out(self, pool, tmp_path):
        a, b = self._samples(_camera_render, 1), self._samples(_camera_render, 2)
        engine._pooled_look(tmp_path / "A.CR3", a, self._own(a))
        engine._pooled_look(tmp_path / "B.CR3", b, self._own(b))
        mono = lambda scene: np.repeat(
            np.rint(_camera_render(scene).astype(float) @ LUMA)[..., None], 3, -1).astype(np.uint8)
        c = self._samples(mono, 3)
        look = engine._pooled_look(tmp_path / "C.CR3", c, self._own(c))
        assert look.frames == 1
        assert len(engine._pool("camera-style")) == 2

    def test_a_frame_with_no_jpeg_still_gets_its_cameras_look(self, pool, tmp_path):
        for i, name in enumerate(("A", "B")):
            s = self._samples(_camera_render, i + 1)
            engine._pooled_look(tmp_path / f"{name}.CR3", s, self._own(s))
        look = engine._pooled_look(tmp_path / "C.CR3", None, None)
        assert look is not None and look.frames == 2

    def test_the_pool_is_kept_on_disk(self, pool, tmp_path):
        for i, name in enumerate(("A", "B")):
            s = self._samples(_camera_render, i + 1)
            engine._pooled_look(tmp_path / f"{name}.CR3", s, self._own(s))
        engine.set_profile_dir(pool)                  # a new session
        assert list(engine._pool("camera-style")) == [str(tmp_path / "A.CR3"), str(tmp_path / "B.CR3")]

    def test_reopening_a_frame_does_not_count_it_twice(self, pool, tmp_path):
        a = self._samples(_camera_render, 1)
        engine._pooled_look(tmp_path / "A.CR3", a, self._own(a))
        engine._pooled_look(tmp_path / "A.CR3", a, self._own(a))
        assert len(engine._pool("camera-style")) == 1


class TestAProfileDirectoryThatRefusesWrites:
    """Learned looks already there are used; nothing new is written, and the
    reason is reported once instead of on every frame."""

    def _samples(self, seed):
        from adjust.look import fit_look, frame_samples

        scene = _textured_scene(200, 300, seed)
        s = frame_samples(scene, _camera_render(scene))
        return s, fit_look([s])

    def test_it_is_read_but_never_written(self, tmp_path, monkeypatch):
        profiles = tmp_path / "profiles"
        monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
        engine.set_profile_dir(profiles)
        for i, name in enumerate(("A", "B")):
            s, own = self._samples(i + 1)
            engine._pooled_look(tmp_path / f"{name}.CR3", s, own)
        before = sorted(p.name for p in profiles.iterdir())

        import storage_health

        monkeypatch.setattr(storage_health, "probe_writable", lambda p: (False, "EACCES (13): denied"))
        writable, error = engine.set_profile_dir(profiles)
        assert not writable and "denied" in error
        assert len(engine._pool("camera-style")) == 2              # still read
        s, own = self._samples(3)
        look = engine._pooled_look(tmp_path / "C.CR3", s, own)
        assert look.frames == 3
        assert sorted(p.name for p in profiles.iterdir()) == before
        engine.set_profile_dir(None)

    def test_the_tool_stops_before_learning_into_nothing(self, tmp_path, monkeypatch):
        import storage_health
        from tools import camera_profile as tool

        monkeypatch.setattr(storage_health, "probe_writable", lambda p: (False, "EACCES (13): denied"))
        with pytest.raises(tool.ProfilesNotWritable, match="denied"):
            tool.learn([tmp_path], tmp_path / "profiles", out=io.StringIO())
        engine.set_profile_dir(None)

    def test_why_a_frame_was_kept_out_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
        engine.set_profile_dir(None)
        for i, name in enumerate(("A", "B")):
            s, own = self._samples(i + 1)
            engine._pooled_look(tmp_path / f"{name}.CR3", s, own)
        from adjust.look import fit_look, frame_samples

        scene = _textured_scene(200, 300, 3)
        mono = np.repeat(np.rint(_camera_render(scene).astype(float) @ LUMA)[..., None], 3, -1).astype(np.uint8)
        s = frame_samples(scene, mono)
        engine._pooled_look(tmp_path / "C.CR3", s, fit_look([s]))
        decision = engine.pooling_decision(tmp_path / "C.CR3")
        # kept out either way: by the whole picture, or by the colour of the patches
        assert decision["joined"] is False
        assert decision["reason"] in ("pooled colour draws the picture worse", "coloured unlike the pool")
        assert decision["pooled_error"] > decision["own_error"]
        assert engine.pooling_decision(tmp_path / "B.CR3")["joined"] is True


class TestThePoolKeepsEachFramesTone:
    """Colour is the camera's; tone the camera varies from frame to frame (Auto
    Lighting Optimizer lifts a dark frame more), so a frame toned differently but
    coloured the same still joins, and keeps its own tone."""

    def test_a_lifted_frame_joins_and_keeps_its_lift(self, tmp_path, monkeypatch):
        from adjust.look import fit_look, frame_samples, look_error

        monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
        engine.set_profile_dir(None)
        for i, name in enumerate(("A", "B")):
            scene = _textured_scene(200, 300, i + 1)
            s = frame_samples(scene, _camera_render(scene))
            engine._pooled_look(tmp_path / f"{name}.CR3", s, fit_look([s]))
        scene = _textured_scene(200, 300, 3)
        lifted = np.clip(255.0 * (_camera_render(scene) / 255.0) ** 0.7, 0, 255).astype(np.uint8)
        s = frame_samples(scene, lifted)
        own = fit_look([s])
        look = engine._pooled_look(tmp_path / "C.CR3", s, own)
        assert engine.pooling_decision(tmp_path / "C.CR3")["joined"] is True
        assert look.frames == 3
        assert look_error(look, [s]) < 1.5 * own.error + 0.003


class TestAnExposureEditMovesTowardsThePoolsTone:
    """A frame's own tone curve is what the camera did at that exposure; for an
    edit that moves the exposure, the pool's curve says more."""

    def _look(self):
        from adjust.look import CameraLook

        own = np.linspace(0.0, 1.0, len(STOPS)).astype(np.float32)
        pool = np.sqrt(own).astype(np.float32)
        return CameraLook(curve=own, matrix=np.eye(3, dtype=np.float32), white_stops=0.0,
                          error=0.0, frames=3, pool_curve=pool)

    def test_no_edit_is_exactly_the_frames_own_curve(self):
        from adjust.look import curve_for

        look = self._look()
        assert curve_for(look, 0.0) is look.curve

    def test_half_a_stop_is_half_way_and_a_stop_is_the_pools(self):
        from adjust.look import curve_for

        look = self._look()
        half = curve_for(look, -0.5)
        assert np.allclose(half, (look.curve + look.pool_curve) / 2, atol=1e-6)
        assert np.allclose(curve_for(look, 1.5), look.pool_curve)

    def test_a_look_of_its_own_ignores_the_edit(self):
        from adjust.look import curve_for

        look = self._look()
        look.pool_curve = None
        assert curve_for(look, 2.0) is look.curve

    def test_the_develop_pipeline_passes_the_edit_through(self):
        scene = _scene()
        frame = LinearFrame(scene)
        frame.look = self._look()
        plain = LinearFrame(scene)
        plain.look = self._look()
        plain.look.pool_curve = None
        # with the pool's (brighter) curve, an exposure edit shows brighter
        a = np.asarray(develop_frame(frame, {"exposure": -1.0}), dtype=float).mean()
        b = np.asarray(develop_frame(plain, {"exposure": -1.0}), dtype=float).mean()
        assert a > b + 5


class TestTheLensVignetting:
    """The camera brightens the corners its lens darkened; the RAW does not."""

    CORNER_STOPS = 1.0

    def _camera_with_correction(self, scene):
        from adjust.look import lens_gain

        gain = lens_gain(scene.shape[:2], (0.3 * self.CORNER_STOPS, 0.7 * self.CORNER_STOPS))
        return _camera_render(scene * gain[..., None])

    def test_the_cameras_correction_is_learned(self):
        from adjust.look import fit_with_lens

        scene = _textured_scene(400, 600, 7)
        samples, look = fit_with_lens(scene, self._camera_with_correction(scene))
        assert sum(look.vignette) == pytest.approx(self.CORNER_STOPS, abs=0.2)

    def test_with_it_the_look_matches_the_camera_better(self):
        from adjust.look import fit_camera_look, fit_with_lens, correct_lens

        scene = _textured_scene(400, 600, 7)
        camera = self._camera_with_correction(scene)
        plain = LinearFrame(scene)
        plain.look = fit_camera_look(scene, camera)
        _, look = fit_with_lens(scene, camera)
        lensed = LinearFrame(correct_lens(scene, look))
        lensed.look = look
        assert _error(lensed.encoded(), camera)[0] < 0.6 * _error(plain.encoded(), camera)[0]

    def test_a_camera_that_darkens_the_corners_itself_is_learned_too(self):
        """A Ricoh GR's Retro Image Control: a negative net correction."""
        from adjust.look import fit_with_lens, lens_gain

        scene = _textured_scene(400, 600, 7)
        gain = lens_gain(scene.shape[:2], (-0.3, -0.7))
        samples, look = fit_with_lens(scene, _camera_render(scene * gain[..., None]))
        assert sum(look.vignette) == pytest.approx(-1.0, abs=0.25)

    def test_a_camera_that_does_not_correct_is_left_alone(self):
        from adjust.look import fit_with_lens

        scene = _textured_scene(400, 600, 7)
        _, look = fit_with_lens(scene, _camera_render(scene))
        assert abs(sum(look.vignette)) < 0.15

    def test_a_loupe_patch_is_corrected_as_its_place_in_the_frame_is(self):
        from adjust.look import CameraLook, correct_lens

        look = CameraLook(curve=np.zeros(len(STOPS), np.float32), matrix=np.eye(3, dtype=np.float32),
                          white_stops=0.0, error=0.0, vignette=(0.3, 0.7))
        frame = np.full((300, 450, 3), 0.2, dtype=np.float32)
        whole = correct_lens(frame, look)
        patch = correct_lens(frame[30:130, 300:450], look, (300 / 450, 30 / 300, 1.0, 130 / 300))
        assert np.allclose(patch, whole[30:130, 300:450], rtol=2e-3)


class TestJudgedOnTheWholePicture:
    """A pool's colour can fit a frame's flat patches and still draw the frame
    wrongly (a bright frame beside darker ones); the picture decides."""

    def _pool_of_two(self, tmp_path, monkeypatch):
        from adjust.look import fit_look, frame_samples

        monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
        engine.set_profile_dir(None)
        for i, name in enumerate(("A", "B")):
            scene = _textured_scene(200, 300, i + 1)
            s = frame_samples(scene, _camera_render(scene))
            engine._pooled_look(tmp_path / f"{name}.CR3", s, fit_look([s]))
        scene = _textured_scene(200, 300, 3)
        s = frame_samples(scene, _camera_render(scene))
        return s, fit_look([s])

    def test_the_probe_prefers_the_look_that_draws_the_frame_better(self):
        from adjust.look import CameraLook, fit_look, frame_samples, probe_error

        scene = _textured_scene(200, 300, 1)
        s = frame_samples(scene, _camera_render(scene))
        good = fit_look([s])
        wrong = CameraLook(curve=np.sqrt(good.curve), matrix=good.matrix,
                           white_stops=good.white_stops, error=0.0)
        assert probe_error(good, s) < probe_error(wrong, s)
        assert probe_error(good, s) < 0.05

    def test_a_pooled_look_that_draws_the_picture_worse_is_not_used(self, tmp_path, monkeypatch):
        import adjust.look as look_module

        s, own = self._pool_of_two(tmp_path, monkeypatch)
        monkeypatch.setattr(look_module, "probe_error",
                            lambda look, samples: 0.03 if look.frames > 1 else 0.02)
        look = engine._pooled_look(tmp_path / "C.CR3", s, own)
        decision = engine.pooling_decision(tmp_path / "C.CR3")
        assert look.frames == 1
        assert decision["joined"] is False and decision["reason"] == "pooled colour draws the picture worse"
        assert decision["pooled_error"] == pytest.approx(0.03) and decision["own_error"] == pytest.approx(0.02)
        assert len(engine._pool("camera-style")) == 2

    def test_a_pooled_look_that_draws_it_as_well_joins(self, tmp_path, monkeypatch):
        import adjust.look as look_module

        s, own = self._pool_of_two(tmp_path, monkeypatch)
        monkeypatch.setattr(look_module, "probe_error", lambda look, samples: 0.02)
        look = engine._pooled_look(tmp_path / "C.CR3", s, own)
        assert look.frames == 3 and engine.pooling_decision(tmp_path / "C.CR3")["joined"] is True


class TestPoolingTheLens:
    """The camera's vignetting correction is fixed for a lens setting; frames of one
    setting agree on it better than any one frame finds it."""

    def test_agreeing_frames_give_their_common_correction(self):
        from adjust.look import pool_vignettes

        found = [(0.30, 0.70), (0.33, 0.67), (0.28, 0.74), (0.31, 0.71)]
        a, b = pool_vignettes(found)
        assert a + b == pytest.approx(1.0, abs=0.06)

    def test_too_few_frames_say_nothing(self):
        from adjust.look import pool_vignettes

        assert pool_vignettes([(0.3, 0.7), (0.3, 0.7)]) is None

    def test_frames_that_found_nothing_do_not_vote(self):
        from adjust.look import pool_vignettes

        a, b = pool_vignettes([(0.3, 0.7), (0.0, 0.0), (0.31, 0.7), (0.29, 0.7), (0.0, 0.0)])
        assert a + b == pytest.approx(1.0, abs=0.05)

    def test_frames_that_disagree_say_nothing(self):
        from adjust.look import pool_vignettes

        assert pool_vignettes([(0.3, 0.7), (0.5, 2.0), (0.1, 0.0), (0.2, 1.4), (0.6, 0.2)]) is None

    def test_a_known_correction_is_used_as_it_is(self):
        from adjust.look import fit_with_lens

        scene = _textured_scene(400, 600, 7)
        camera = _camera_render(scene)            # no vignetting in this camera
        _, look = fit_with_lens(scene, camera, vignette=(0.3, 0.7))
        assert look.vignette == (0.3, 0.7)

    def _frames(self, n=3):
        from adjust.look import fit_with_lens, lens_gain

        out = []
        for seed in range(n):
            scene = _textured_scene(300, 450, 20 + seed)
            camera = _camera_render(scene * lens_gain(scene.shape[:2], (0.3, 0.7))[..., None])
            out.append((scene, camera))
        return out

    def test_a_frame_that_finds_no_lens_takes_the_pools(self, tmp_path, monkeypatch):
        import adjust.look as look_module
        from adjust.look import fit_with_lens

        monkeypatch.setattr(engine, "_lens_key", lambda source: "lens-24-f4")
        engine.set_profile_dir(None)
        frames = self._frames(4)
        for i in range(3):
            s, own = fit_with_lens(*frames[i])
            engine._with_pooled_lens(tmp_path / f"{i}.CR3", frames[i][0], frames[i][1], s, own)
        scene, camera = frames[3]
        monkeypatch.setattr(look_module, "fit_vignette", lambda small, cam, look: (0.0, 0.0))
        s, own = fit_with_lens(scene, camera)
        assert sum(own.vignette) == 0.0                       # the search found nothing
        monkeypatch.undo()
        monkeypatch.setattr(engine, "_lens_key", lambda source: "lens-24-f4")
        s2, look = engine._with_pooled_lens(tmp_path / "3.CR3", scene, camera, s, own)
        assert sum(look.vignette) == pytest.approx(1.0, abs=0.25)
        assert engine.pooling_decision(tmp_path / "3.CR3")["lens"].startswith("pooled from 3 frames")

    def test_without_a_pool_the_frames_own_result_stands(self, tmp_path, monkeypatch):
        from adjust.look import fit_with_lens

        monkeypatch.setattr(engine, "_lens_key", lambda source: "lens-24-f4")
        engine.set_profile_dir(None)
        scene, camera = self._frames(1)[0]
        s, own = fit_with_lens(scene, camera)
        s2, look = engine._with_pooled_lens(tmp_path / "A.CR3", scene, camera, s, own)
        assert look is own and engine.pooling_decision(tmp_path / "A.CR3")["lens"] == "own"

    def test_an_unknown_lens_setting_is_left_alone(self, tmp_path, monkeypatch):
        from adjust.look import fit_with_lens

        monkeypatch.setattr(engine, "_lens_key", lambda source: None)
        engine.set_profile_dir(None)
        scene, camera = self._frames(1)[0]
        s, own = fit_with_lens(scene, camera)
        _, look = engine._with_pooled_lens(tmp_path / "A.CR3", scene, camera, s, own)
        assert look is own and "unknown" in engine.pooling_decision(tmp_path / "A.CR3")["lens"]

    def test_the_pool_is_kept_on_disk(self, tmp_path, monkeypatch):
        from adjust.look import fit_with_lens

        monkeypatch.setattr(engine, "_lens_key", lambda source: "lens-24-f4")
        engine.set_profile_dir(tmp_path / "profiles")
        for i, (scene, camera) in enumerate(self._frames(3)):
            s, own = fit_with_lens(scene, camera)
            engine._with_pooled_lens(tmp_path / f"{i}.CR3", scene, camera, s, own)
        engine.set_profile_dir(tmp_path / "profiles")         # a new session
        assert len(engine._lens_pool("lens-24-f4")) == 3
        engine.set_profile_dir(None)
