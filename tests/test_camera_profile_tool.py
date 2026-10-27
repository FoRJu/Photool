"""
The offline camera-profile tool: finding brackets, measuring against the camera's
JPEG, and learning into (or only reading) the profile directory.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from export import engine
from tools import camera_profile as tool
from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng
from test_look import _camera_render


def _shot(name, exposure, taken, settings=(4.0, 24, 200)):
    return tool.Shot(path=Path(name), exposure=exposure, taken=taken, settings=settings)


class TestFindingBrackets:
    def test_a_burst_of_exposures_taken_together_is_a_bracket(self):
        shots = [_shot(f"{i}.CR3", e, 100 + i) for i, e in enumerate((0, -2, -1, 1, 2))]
        found = tool.brackets(shots)
        assert len(found) == 1 and len(found[0]) == 5

    def test_frames_far_apart_in_time_are_not_one_bracket(self):
        shots = [_shot("a", 0, 0), _shot("b", -1, 1), _shot("c", 1, 2),
                 _shot("d", 0, 500), _shot("e", -1, 501), _shot("f", 1, 502)]
        assert [len(g) for g in tool.brackets(shots)] == [3, 3]

    def test_a_change_of_aperture_or_iso_ends_a_bracket(self):
        shots = [_shot("a", 0, 0), _shot("b", -1, 1), _shot("c", 1, 2, settings=(5.6, 24, 200))]
        assert tool.brackets(shots) == []

    def test_a_burst_at_one_exposure_is_not_a_bracket(self):
        shots = [_shot(f"{i}", 0.0, i) for i in range(5)]
        assert tool.brackets(shots) == []


class TestComparing:
    def test_the_same_picture_scores_zero(self):
        img = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
        assert tool.compare(img, img)["delta_e"] == 0.0

    def test_pixels_the_camera_jpeg_does_not_reach_are_left_out(self):
        shown = np.full((10, 10, 3), 0.5, dtype=np.float32)
        camera = shown.copy()
        camera[:, :5] = np.nan
        assert tool.compare(shown, camera)["delta_e"] == 0.0

    def test_a_brighter_picture_is_further_away(self):
        a = np.full((10, 10, 3), 0.5, dtype=np.float32)
        assert tool.compare(a * 1.2, a)["delta_e"] > tool.compare(a * 1.05, a)["delta_e"] > 0


@pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")
class TestOnRawFiles:
    LEVELS = np.geomspace(0.002, 0.95, 28)

    @pytest.fixture
    def folder(self, tmp_path, monkeypatch):
        """Two DNGs whose embedded JPEG is a known camera rendering."""
        engine.clear_preview_cache()
        linear = np.repeat(np.repeat(
            np.stack([np.repeat(self.LEVELS[None, :], 8, 0)] * 3, -1), 64, axis=1), 8, axis=0
        ).astype(np.float32)
        buf = io.BytesIO()
        Image.fromarray(_camera_render(linear)).save(buf, format="JPEG", quality=98)
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: buf.getvalue())
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
        monkeypatch.setattr(engine, "_camera_key", lambda source: "test-camera")
        shots = tmp_path / "shots"
        shots.mkdir()
        for name in ("A.dng", "B.dng"):
            _write_dng(shots / name, [(v, v, v) for v in self.LEVELS])
        yield shots
        engine.set_profile_dir(None)

    def test_the_report_measures_each_frame_against_its_camera_jpeg(self, folder, tmp_path):
        out = io.StringIO()
        result = tool.report([folder], tmp_path / "profiles", size=1800, out=out)
        assert [Path(r["file"]).name for r in result["frames"]] == ["A.dng", "B.dng"]
        assert all(r["luma"] < 0.03 for r in result["frames"])
        assert "A.dng" in out.getvalue()

    def test_the_report_leaves_the_stored_profiles_alone(self, folder, tmp_path):
        profiles = tmp_path / "profiles"
        profiles.mkdir()
        tool.report([folder], profiles, size=1800, out=io.StringIO())
        assert list(profiles.iterdir()) == []

    def test_learning_pools_the_frames_where_the_app_will_find_them(self, folder, tmp_path):
        profiles = tmp_path / "profiles"
        rows = tool.learn([folder], profiles, size=1800, out=io.StringIO())
        assert [r["look_frames"] for r in rows] == [1, 2]
        listed = tool.list_profiles(profiles, out=io.StringIO())
        assert listed[0]["frames"] == ["A.dng", "B.dng"]


class TestReadingExposureSettings:
    def test_exiftool_is_asked_when_rawpy_cannot_say(self, monkeypatch, tmp_path):
        import subprocess
        import types

        def no_rawpy(path):
            raise AttributeError("'RawPy' object has no attribute 'other'")

        monkeypatch.setattr(tool, "_from_rawpy", no_rawpy)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
            stdout="0.005\n4.0\n200\n24\n2026:10:08 08:46:14\n"))
        shot = tool.read_shot(tmp_path / "A.CR3")
        assert shot is not None
        assert shot.exposure == pytest.approx(np.log2(0.005 * 200 / 16.0))
        assert shot.settings[:3] == (4.0, 24, 200)

    def test_a_picture_style_is_part_of_what_must_match(self):
        shots = [_shot(n, e, i, settings=(4.0, 24, 200, st))
                 for i, (n, e, st) in enumerate([("a", 0, "Standard"), ("b", -1, "Standard"),
                                                ("c", 1, "Standard"), ("d", 0, "Positive Film")])]
        groups = tool.brackets(shots)
        assert len(groups) == 1 and [s.path.name for s in groups[0]] == ["a", "b", "c"]

    def test_a_file_neither_can_read_has_no_settings(self, monkeypatch, tmp_path):
        def fail(path):
            raise OSError("no")

        monkeypatch.setattr(tool, "_from_rawpy", fail)
        monkeypatch.setattr(tool, "_from_exiftool", fail)
        assert tool.read_shot(tmp_path / "A.CR3") is None


class TestTheSameScene:
    def _frame(self, seed):
        from test_look import _textured_scene

        return _textured_scene(200, 300, seed)

    def test_a_bracket_frame_is_the_same_scene_once_its_exposure_is_taken_out(self):
        a = self._frame(1)
        assert tool.same_scene(a, a * 0.25, 2.0) > 0.98

    def test_another_picture_is_not(self):
        assert tool.same_scene(self._frame(1), self._frame(2), 0.0) < tool.SAME_SCENE


class TestCoarseError:
    def test_texture_alone_shows_in_the_fine_error_not_the_coarse_one(self):
        rng = np.random.default_rng(0)
        camera = np.full((64, 64, 3), 0.5, dtype=np.float32)
        textured = camera + rng.choice([-0.05, 0.05], size=(64, 64, 1)).astype(np.float32)
        m = tool.compare(textured, camera)
        assert m["delta_e"] > 2.0
        assert m["delta_e_coarse"] < 0.5 * m["delta_e"]


class TestTheBracketsMiddle:
    def test_an_extra_frame_of_another_scene_does_not_move_the_middle(self, monkeypatch, tmp_path):
        """Seven bracket frames and an eighth, brighter shot of something else:
        the reference is the bracket's own middle, not the middle of all eight."""
        from types import SimpleNamespace

        from test_look import _textured_scene

        scene, other = _textured_scene(120, 180, 1), _textured_scene(120, 180, 9)
        exposures = [-3, -2, -1, 0, 1, 2, 3]
        shots = [_shot(f"b{i}.CR3", e, i) for i, e in enumerate(exposures)] + [_shot("x.CR3", 4, 7)]
        frames = {s.path: SimpleNamespace(rgb=(other if s.path.name == "x.CR3" else scene) * 2.0 ** s.exposure)
                  for s in shots}
        monkeypatch.setattr(tool, "_frame", lambda path, size: frames[path])
        likeness = tool._likeness_to(shots[len(shots) // 2], shots, 100)
        members = [s for s in sorted(shots, key=lambda s: s.exposure) if likeness[s.path] >= tool.SAME_SCENE]
        assert [s.path.name for s in members] == [f"b{i}.CR3" for i in range(7)]
        assert members[len(members) // 2].exposure == 0


class TestWritingTheReport:
    def test_it_writes_json_that_reads_back(self, tmp_path):
        import json

        tool.write_json_checked(tmp_path / "r.json", {"a": [1, 2], "b": "x"})
        assert json.loads((tmp_path / "r.json").read_text()) == {"a": [1, 2], "b": "x"}
        assert list(tmp_path.iterdir()) == [tmp_path / "r.json"]          # no temp file left

    def test_content_that_never_reached_the_disk_is_an_error_not_a_blank_file(self, tmp_path, monkeypatch):
        """A write whose size is recorded and whose bytes are zeros: what a card that
        did not flush leaves behind."""
        from pathlib import Path

        real = Path.read_text
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "\x00" * 100
                            if self.name.endswith(".tmp") else real(self, *a, **k))
        (tmp_path / "r.json").write_text('{"old": true}')
        with pytest.raises(OSError, match="could not save the report intact"):
            tool.write_json_checked(tmp_path / "r.json", {"a": 1})
        assert (tmp_path / "r.json").read_text() == '{"old": true}'       # the previous report survives
        assert [p.name for p in tmp_path.iterdir()] == ["r.json"]

    def test_a_path_that_cannot_be_written_names_itself(self, tmp_path):
        with pytest.raises(OSError, match="nowhere"):
            tool.write_json_checked(tmp_path / "nowhere" / "r.json", {"a": 1})


# ── A broken look, seen ──────────────────────────────────────────────

def _smooth_curve():
    from adjust.look import STOPS

    linear = 0.18 * np.exp2(STOPS)
    return (1.0 - np.exp(-2.2 * linear)).astype(np.float32) ** 0.6


def _plateau_curve():
    """A camera curve with the kind of fault that posterized faces: flat across a
    stop of mid-tones, then steep to catch up."""
    from adjust.look import STOPS

    curve = _smooth_curve().astype(np.float64)
    low, high = np.searchsorted(STOPS, [-1.0, 0.0])
    top = np.searchsorted(STOPS, 1.0)
    curve[low:high] = curve[low]
    curve[high:top] = np.linspace(curve[low], curve[top], top - high)
    return curve.astype(np.float32)


def _ramp():
    """A soft picture of every tone: smooth gradients, as skin is."""
    y, x = np.mgrid[0:240, 0:360].astype(np.float32)
    triangle = np.abs((x % 120.0) - 60.0) / 60.0          # 0..1 and back, every 120 px
    tone = triangle * 0.88 + 0.05 + 0.02 * np.sin(y / 9.0)
    return np.repeat(np.clip(tone, 0, 1)[..., None], 3, axis=-1)


class TestSeeingABrokenLook:
    def test_a_camera_like_curve_rises_evenly(self):
        from adjust.look import EVEN_MIN, curve_evenness

        assert curve_evenness(_smooth_curve()) > EVEN_MIN

    def test_a_plateau_in_the_mid_tones_is_seen(self):
        from adjust.look import EVEN_MIN, curve_evenness

        assert curve_evenness(_plateau_curve()) < EVEN_MIN

    def test_the_camera_s_own_picture_has_no_banding(self):
        from adjust.look import tone_detail

        camera = _ramp()
        assert tone_detail(camera, camera)["banding"] == pytest.approx(0.0, abs=0.01)

    def test_a_posterized_band_of_tones_is_seen_against_the_camera(self):
        from adjust.look import BANDING_MAX, tone_detail

        camera = _ramp()
        # the band 0.4..0.55 drawn as one tone, the band above it stretched to catch up
        t = camera[..., 0]
        shown = np.where(t < 0.4, t, np.where(t < 0.55, 0.4, 0.4 + (t - 0.55) * (0.5 / 0.4)))
        shown = np.repeat(np.clip(shown, 0, 1)[..., None], 3, axis=-1).astype(np.float32)
        found = tone_detail(shown, camera)
        assert found["banding"] > BANDING_MAX
        assert 0.38 <= found["flattest"][1] <= 0.5

    def test_the_report_carries_the_banding(self):
        camera = _ramp()
        assert tool.compare(camera, camera)["banding"] == pytest.approx(0.0, abs=0.01)


class TestHealth:
    def _keep(self, profiles, source, curve):
        import json

        source.write_bytes(b"raw")
        stat = source.stat()
        meta = {"source": str(source), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size,
                "version": engine._LOOK_FILE_VERSION, "decision": {"reason": "joined"}}
        (profiles / "looks").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(profiles / "looks" / f"{source.stem}.npz", meta=np.array(json.dumps(meta)),
                            curve=curve, matrix=np.eye(3, dtype=np.float32),
                            numbers=np.array([2.5, 0.01, 1.0]), vignette=np.zeros(2))

    def test_the_broken_look_comes_first_and_is_marked(self, tmp_path):
        profiles = tmp_path / "profiles"
        self._keep(profiles, tmp_path / "good.CR3", _smooth_curve())
        self._keep(profiles, tmp_path / "bad.CR3", _plateau_curve())
        before = sorted(p.read_bytes() for p in (profiles / "looks").iterdir())
        out = io.StringIO()
        rows = tool.health([], profiles, out=out)
        assert [Path(r["file"]).name for r in rows] == ["bad.CR3", "good.CR3"]
        assert rows[0]["off"] and not rows[1]["off"]
        assert "1 of 2 kept looks are off" in out.getvalue()
        # read-only
        assert sorted(p.read_bytes() for p in (profiles / "looks").iterdir()) == before

    def test_only_the_photographs_asked_about(self, tmp_path):
        profiles = tmp_path / "profiles"
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        self._keep(profiles, tmp_path / "a" / "one.CR3", _plateau_curve())
        self._keep(profiles, tmp_path / "b" / "two.CR3", _plateau_curve())
        rows = tool.health([tmp_path / "a"], profiles, out=io.StringIO())
        assert [Path(r["file"]).name for r in rows] == ["one.CR3"]

    def test_a_file_changed_since_its_look_was_kept_is_stale(self, tmp_path):
        import os

        profiles = tmp_path / "profiles"
        source = tmp_path / "one.CR3"
        self._keep(profiles, source, _smooth_curve())
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        assert tool.health([], profiles, out=io.StringIO())[0]["stale"]

    def test_banding_alone_does_not_mark_a_look_off(self, tmp_path, monkeypatch):
        """A camera JPEG with heavy noise reduction is softer than any develop: the
        picture bands against it with nothing posterized."""
        import adjust.look as look

        profiles = tmp_path / "profiles"
        self._keep(profiles, tmp_path / "night.CR3", _smooth_curve())
        monkeypatch.setattr(tool, "_drawn_with", lambda path, kept, size: (_ramp(), _ramp()))
        monkeypatch.setattr(look, "tone_detail", lambda shown, camera: {
            "flattest": (-1.2, 0.56), "steepest": (0.2, 0.92), "banding": 1.4})
        row = tool.health([], profiles, render=True, out=io.StringIO())[0]
        assert not row["off"] and not row["shows"]

    def test_a_plateau_that_can_be_seen_shows(self, tmp_path, monkeypatch):
        import adjust.look as look

        profiles = tmp_path / "profiles"
        self._keep(profiles, tmp_path / "stage.CR3", _plateau_curve())
        monkeypatch.setattr(tool, "_drawn_with", lambda path, kept, size: (_ramp(), _ramp()))
        monkeypatch.setattr(look, "tone_detail", lambda shown, camera: {
            "flattest": (-0.8, 0.62), "steepest": (0.5, 0.86), "banding": 1.3})
        out = io.StringIO()
        row = tool.health([], profiles, render=True, out=out)[0]
        assert row["off"] and row["shows"]
        assert "1 visibly" in out.getvalue()
