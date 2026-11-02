"""
What a frame teaches its camera look covers every tone. A night scene is mostly dark
with a little lit wall and some lamps: the wall must be sampled, the rims of the lamps
kept out where the tone has better places, and a whole-frame comparison must not let
the sky hide a wrong wall. The camera is simulated as in test_look.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import look as L
from adjust.look import CameraLook, STOPS, fit_look, frame_samples, probe_error, tone_detail, tone_error
from adjust.render import LUMA, encode
from export import engine
from test_develop_routes import client, FOLDER  # noqa: F401  (a fixture)
from test_look import _camera_render, _tone


def _night(h=540, w=810, seed=5) -> np.ndarray:
    """Dark textured sky and water, a smaller lit wall ramping through the mid-tones,
    and small lamps far above white."""
    rng = np.random.default_rng(seed)
    light = 0.003 * 2.0 ** (2.0 * rng.random((h, w)))
    light = light[..., None].repeat(3, 2)
    y0, y1, x0, x1 = int(h * .45), int(h * .75), int(w * .55), int(w * .95)
    ramp = 0.06 * 2.0 ** np.linspace(0.0, 4.5, x1 - x0)
    light[y0:y1, x0:x1] = (ramp[None, :, None] * np.array([1.0, 0.95, 0.8])) * (1 + .03 * rng.standard_normal((y1 - y0, x1 - x0, 1)))
    for k in range(40):
        cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
        light[cy - 3:cy + 4, cx - 3:cx + 4] = 16.0
    return np.maximum(light, 1e-5).astype(np.float32)


def _tones_of(samples):
    return L._to_stops(np.maximum(samples.linear @ LUMA, 1e-9))


class TestEveryToneIsTaught:
    def test_the_mid_tones_of_a_night_scene_are_well_sampled(self):
        scene = _night()
        samples = frame_samples(scene, _camera_render(scene))
        tones = _tones_of(samples)
        wall = (tones > -2.0) & (tones < 1.0)
        assert wall.sum() >= 200

    def test_the_mid_tones_are_taught_what_the_camera_really_wrote_there(self):
        scene = _night()
        samples = frame_samples(scene, _camera_render(scene))
        tones = _tones_of(samples)
        wall = (tones > -2.0) & (tones < 1.0)
        written = samples.camera[wall] @ LUMA
        assert np.median(np.abs(written - _tone(samples.linear[wall] @ LUMA))) < 0.05

    def test_both_pictures_are_smoothed_as_light(self):
        light = np.full((40, 40, 3), 0.01, np.float32)
        light[:, 20:] = 0.8
        smoothed = L._blur_as_light(encode(light))
        expected = encode(np.stack([L._blur(light[..., c]) for c in range(3)], axis=-1))
        assert np.abs(smoothed - expected).max() < 1e-3

    def test_a_lamps_steep_rim_is_left_out_where_its_tone_has_calmer_places(self):
        rng = np.random.default_rng(2)
        scene = np.full((300, 450, 3), 0.2, np.float32) * (1 + 0.01 * rng.standard_normal((300, 450, 1), dtype=np.float32))
        scene[:60] = 1.0                                    # a calm area, in tones its rim also passes through
        scene[150:250, 150:300] = 64.0                      # a hard-edged lamp, far above white
        samples = frame_samples(scene, _camera_render(scene))
        tones = _tones_of(samples)
        like_calm = np.abs(tones - L._to_stops(np.array([1.0]))[0]) < 0.25
        written = samples.camera[like_calm] @ LUMA
        assert np.median(np.abs(written - float(_tone(np.array([1.0]))[0]))) < 0.05


class TestTheWholePictureCounts:
    def _samples(self, wall_error):
        """Nine-tenths sky, one-tenth wall, with the look wrong on the wall alone."""
        h, w = 100, 100
        light = np.full((h, w, 3), 0.01, np.float32)
        light[:10] = 0.3
        look = CameraLook(curve=_tone(L.MID_GREY * 2.0 ** STOPS).astype(np.float32), matrix=np.eye(3, dtype=np.float32),
                          white_stops=3.0, error=0.0)
        camera = L.apply_look(light, look)
        camera[:10] += wall_error
        return look, L.FrameSamples(linear=light.reshape(-1, 3)[:10], camera=camera.reshape(-1, 3)[:10],
                                    probe_linear=light, probe_camera=camera)

    def test_a_wrong_wall_is_not_hidden_by_the_sky(self):
        look, samples = self._samples(0.12)
        whole = float(np.mean(np.abs(L.apply_look(samples.probe_linear, look) - samples.probe_camera)))
        assert whole < 0.02 and probe_error(look, samples) < 0.02
        assert tone_error(look, samples) > 0.05

    def test_a_right_picture_scores_nothing(self):
        look, samples = self._samples(0.0)
        assert tone_error(look, samples) < 1e-3

    def test_where_the_jpeg_does_not_reach_stays_out_of_the_probe(self):
        small = np.full((64, 256, 3), 0.2, np.float32)
        cam = np.full((64, 256, 3), 0.5, np.float32)
        cam[:, :8] = np.nan
        _, camera = L._probe(small, cam)
        assert np.isnan(camera[:, 0]).all() and np.isfinite(camera[:, -1]).all()


class TestFlattened:
    def _pair(self):
        rng = np.random.default_rng(1)
        h, w = 500, 500
        # Tones rise left to right; a pattern across them gives every tone its edges.
        cam = np.tile(np.linspace(0.05, 0.95, w, dtype=np.float32), (h, 1))
        ripple = 0.03 * np.sin(np.arange(h, dtype=np.float32) / 5.0)[:, None]
        return np.clip(cam + ripple, 0, 1)[..., None].repeat(3, 2)

    def test_a_right_picture_has_nothing_flattened(self):
        cam = self._pair()
        assert tone_detail(cam.copy(), cam)["flattened"] == 0.0

    def test_a_flattened_stretch_of_tones_counts_for_how_much_of_the_picture_it_is(self):
        cam = self._pair()
        shown = cam.copy()
        band = (cam[..., 0] > 0.4) & (cam[..., 0] < 0.8)
        shown[band] = 0.6 + (shown[band] - 0.6) * 0.2
        detail = tone_detail(shown, cam)
        assert detail["flattened"] > L.FLATTENED_MAX

    def test_one_tiny_flat_tone_is_not_posterization(self):
        cam = self._pair()
        shown = cam.copy()
        band = (cam[..., 0] > 0.86) & (cam[..., 0] < 0.885)
        shown[band] = 0.87
        assert tone_detail(shown, cam)["flattened"] <= L.FLATTENED_MAX


@pytest.fixture
def profiles(tmp_path):
    engine.set_profile_dir(tmp_path / "profiles")
    engine.clear_preview_cache()
    yield tmp_path / "profiles"
    engine.set_edit_check(None)
    engine.set_profile_dir(None)
    engine.clear_preview_cache()


class TestOlderLooks:
    def _kept(self, profiles, source, version):
        look = CameraLook(curve=np.linspace(0, 1, len(STOPS), dtype=np.float32), matrix=np.eye(3, dtype=np.float32),
                          white_stops=3.0, error=0.01, frames=9)
        (profiles / "looks").mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        import json

        np.savez(engine._look_path(source), meta=np.array(json.dumps(
            {"source": str(source), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "version": version,
             "pool_frames": 9, "decision": {"reason": "joined"}})),
                 curve=look.curve, matrix=look.matrix, numbers=np.array([3.0, 0.01, 9.0]),
                 vignette=np.zeros(2))
        return stat

    def _file(self, tmp_path):
        source = tmp_path / "IMG_0001.CR3"
        source.write_bytes(b"x")
        return source

    def test_an_unedited_frame_on_an_older_look_learns_again(self, profiles, tmp_path):
        source = self._file(tmp_path)
        stat = self._kept(profiles, source, engine._OLDER_LOOK_VERSION)
        engine.set_edit_check(lambda s: False)
        assert engine._stored_look(source, stat) is None

    def test_an_edited_frame_keeps_its_older_look_and_says_so(self, profiles, tmp_path):
        source = self._file(tmp_path)
        stat = self._kept(profiles, source, engine._OLDER_LOOK_VERSION)
        engine.set_edit_check(lambda s: True)
        look = engine._stored_look(source, stat)
        assert look is not None
        assert engine.look_status(source, look)[2] == "older"

    def test_not_knowing_whether_it_is_edited_leaves_it_as_drawn(self, profiles, tmp_path):
        source = self._file(tmp_path)
        stat = self._kept(profiles, source, engine._OLDER_LOOK_VERSION)

        def broken(_):
            raise RuntimeError("no catalog")

        engine.set_edit_check(broken)
        assert engine._stored_look(source, stat) is not None

    def test_a_current_look_is_used_edited_or_not(self, profiles, tmp_path):
        source = self._file(tmp_path)
        stat = self._kept(profiles, source, engine._LOOK_FILE_VERSION)
        engine.set_edit_check(lambda s: False)
        look = engine._stored_look(source, stat)
        assert look is not None and engine.look_status(source, look)[2] != "older"


class TestFrameHasEdit:
    def test_any_file_of_the_frame_counts_and_no_other(self, tmp_path):
        from catalog import CatalogDB

        db = CatalogDB(str(tmp_path / "catalog.db"))
        db.set_photo_edit("2026/a/IMG_0001.JPG", {"exposure": 0.5})
        assert db.frame_has_edit("2026/a", "IMG_0001")
        assert not db.frame_has_edit("2026/a", "IMG_000")
        assert not db.frame_has_edit("2026/b", "IMG_0001")


class TestUpdatingALook:
    def test_the_view_can_learn_one_frame_again(self, client, monkeypatch):
        import io

        from PIL import Image

        from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng

        if not _HAVE_REAL_RAW:
            pytest.skip("needs rawpy and tifffile to build a DNG")
        engine.clear_preview_cache()
        library = Path(client.app.state.config.library_root)
        levels = np.geomspace(0.002, 0.95, 28)
        _write_dng(library / FOLDER / "IMG_0002.CR3", [(v, v, v) for v in levels])
        linear = np.repeat(np.stack([np.repeat(levels[None, :], 8, 0)] * 3, -1), 64, axis=1)
        linear = np.repeat(linear, 8, axis=0).astype(np.float32)
        buf = io.BytesIO()
        Image.fromarray(_camera_render(linear)).save(buf, format="JPEG", quality=98)
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: buf.getvalue())
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
        res = client.post("/api/camera-look/relearn", json={"filepath": f"{FOLDER}/IMG_0002.CR3"})
        assert res.status_code == 200
        assert res.json()["state"] != "none" and res.json()["hint"] != "older"

    def test_a_path_outside_the_library_is_refused(self, client):
        assert client.post("/api/camera-look/relearn", json={"filepath": "../../secret.CR3"}).status_code >= 400


class TestAFlattenedFrameTakesThePoolsTone:
    def _look(self, frames=9):
        curve = np.clip(_tone(L.MID_GREY * 2.0 ** STOPS), 0, 1).astype(np.float32)
        return CameraLook(curve=curve, matrix=np.eye(3, dtype=np.float32), white_stops=3.0, error=0.0, frames=frames)

    def _setup(self, monkeypatch, tmp_path, details, pool_error=0.01):
        """A frame in a pool of 3, with ``details`` the flattened share each drawing scores."""
        source = tmp_path / "IMG_0001.CR3"
        engine._DECISIONS[str(source)] = {"pool": "p", "reason": "joined"}
        samples = L.FrameSamples(linear=np.zeros((1, 3), np.float32), camera=np.zeros((1, 3), np.float32))
        monkeypatch.setitem(engine._POOLS, "p", {"a": samples, "b": samples, "c": samples})
        monkeypatch.setattr(engine, "_embedded_camera_rgb", lambda s: np.zeros((8, 8, 3), np.uint8))
        monkeypatch.setattr(engine, "_sibling_jpeg_rgb", lambda s: None)
        monkeypatch.setattr(L, "align_camera", lambda linear, camera: np.full(linear.shape, 0.5, np.float32))
        pooled = self._look(frames=3)
        monkeypatch.setattr(L, "fit_look", lambda s: pooled)
        monkeypatch.setattr(L, "tone_error", lambda look, s: 0.01)
        drawn = iter(details)
        monkeypatch.setattr(L, "tone_detail", lambda shown, camera: {"flattened": next(drawn)})
        return source, samples, pooled

    def test_it_takes_the_pools_tone_and_says_so(self, monkeypatch, tmp_path):
        source, samples, pooled = self._setup(monkeypatch, tmp_path, [0.2, 0.0])
        rgb = np.full((1000, 1000, 3), 0.18, np.float32)
        look = engine._with_checked_tone(source, rgb, samples, self._look())
        assert look.curve is pooled.curve
        assert engine.look_status(source, look)[0] == "shared"

    def test_a_frame_that_is_not_flattened_keeps_its_own(self, monkeypatch, tmp_path):
        source, samples, pooled = self._setup(monkeypatch, tmp_path, [0.01])
        own = self._look()
        rgb = np.full((1000, 1000, 3), 0.18, np.float32)
        assert engine._with_checked_tone(source, rgb, samples, own) is own

    def test_a_pool_that_is_no_better_is_not_taken(self, monkeypatch, tmp_path):
        source, samples, pooled = self._setup(monkeypatch, tmp_path, [0.2, 0.3])
        own = self._look()
        rgb = np.full((1000, 1000, 3), 0.18, np.float32)
        assert engine._with_checked_tone(source, rgb, samples, own) is own

    def test_another_style_is_left_alone(self, monkeypatch, tmp_path):
        source, samples, pooled = self._setup(monkeypatch, tmp_path, [0.2, 0.0])
        engine._DECISIONS[str(source)]["reason"] = "coloured unlike the pool"
        own = self._look()
        rgb = np.full((1000, 1000, 3), 0.18, np.float32)
        assert engine._with_checked_tone(source, rgb, samples, own) is own
