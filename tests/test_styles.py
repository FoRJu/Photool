"""
Tests for styles: a look learned once and rendered on another camera's files.

Two simulated cameras look at the same scene. Camera A's engine is punchy (a
saturation boost and a contrasty curve); camera B records the scene a stop
darker in its raw numbers (another ISO calibration) and renders it plainly. A
style learned from A, put on B's files, should give A's character at the
brightness B's photographer chose — and not depend on where B's raw numbers sit.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from adjust import styles
from adjust.develop import from_crs, normalize, to_crs
from adjust.look import _oklab, fit_camera_look, frame_samples
from adjust.render import LUMA, LinearFrame, develop_frame, encode
from test_look import _camera_render, _scene, _tone


def _plain_render(linear: np.ndarray) -> np.ndarray:
    """Camera B's engine: the same kind of curve, no colour boost."""
    shown = _tone(np.maximum(linear, 0.0))
    return np.clip(np.rint(shown * 255.0), 0, 255).astype(np.uint8)


def _chroma(display: np.ndarray) -> float:
    lab = _oklab(np.clip(display, 0, 1).reshape(-1, 3))
    return float(np.median(np.hypot(lab[:, 1], lab[:, 2])))


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp:
        styles.set_dir(tmp)
        yield Path(tmp)
        styles.set_dir(None)


@pytest.fixture(scope="module")
def cameras():
    scene = _scene(h=120, w=180)
    a_samples = frame_samples(scene, _camera_render(scene))
    b_decode = scene * 0.5                                   # a stop lower in B's numbers
    b_look = fit_camera_look(b_decode, _plain_render(scene))
    b2_look = fit_camera_look(b_decode * 2.0, _plain_render(scene))
    assert a_samples is not None and b_look is not None and b2_look is not None
    return scene, a_samples, b_decode, b_look, b2_look


def _style(store, a_samples, name="Punchy"):
    style = styles.from_samples(name, [a_samples])
    styles.save(style, [a_samples])
    return style


class TestTransfer:
    def test_the_style_brings_its_colour_to_the_other_camera(self, store, cameras):
        scene, a_samples, b_decode, b_look, _ = cameras
        style = _style(store, a_samples)
        k = styles.anchor(b_decode, style, b_look)
        styled = styles.render(b_decode, style, k)
        own = styles.own_rendering(b_decode, b_look)
        camera_a = _camera_render(scene) / 255.0
        assert _chroma(styled) > _chroma(own) * 1.15
        assert abs(_chroma(styled) - _chroma(camera_a)) < abs(_chroma(own) - _chroma(camera_a))

    def test_the_frame_keeps_its_own_brightness(self, store, cameras):
        _scene_, a_samples, b_decode, b_look, _ = cameras
        style = _style(store, a_samples)
        styled = styles.render(b_decode, style, styles.anchor(b_decode, style, b_look))
        own = styles.own_rendering(b_decode, b_look)
        assert float(np.median(styled @ LUMA)) == pytest.approx(float(np.median(own @ LUMA)), abs=0.02)

    def test_the_anchor_undoes_where_the_raw_numbers_sit(self, store, cameras):
        """B's decode a stop up (another calibration, another look fitted to it)
        renders the same picture through the style: the anchor halves."""
        _scene_, a_samples, b_decode, b_look, b2_look = cameras
        style = _style(store, a_samples)
        k1 = styles.anchor(b_decode, style, b_look)
        k2 = styles.anchor(b_decode * 2.0, style, b2_look)
        assert k2 == pytest.approx(k1 / 2.0, rel=0.05)
        one = styles.render(b_decode, style, k1)
        two = styles.render(b_decode * 2.0, style, k2)
        assert float(np.mean(np.abs(one - two))) < 0.01

    def test_a_style_on_its_own_camera_changes_nothing(self, store, cameras):
        scene, a_samples, *_ = cameras
        style = _style(store, a_samples)
        own_a = fit_camera_look(scene, _camera_render(scene))
        assert styles.anchor(scene, style, own_a) == pytest.approx(1.0, abs=0.06)

    def test_without_a_camera_look_it_takes_the_styles_own_brightness(self, store, cameras):
        _scene_, a_samples, b_decode, _b, _b2 = cameras
        style = _style(store, a_samples)
        styled = styles.render(b_decode, style, styles.anchor(b_decode, style, None))
        assert float(np.mean(np.quantile(styled @ LUMA, (0.35, 0.5, 0.65)))) == \
            pytest.approx(style.typical, abs=0.03)


class TestInTheEdit:
    def test_an_edit_names_a_style_and_it_is_rendered(self, store, cameras):
        _scene_, a_samples, b_decode, b_look, _ = cameras
        style = _style(store, a_samples)
        frame = LinearFrame(b_decode.copy(), look=b_look)
        plain = np.asarray(develop_frame(frame, {"saturation": 1}), np.float32)
        styled = np.asarray(develop_frame(frame, {"style": style.id}), np.float32)
        assert float(np.abs(styled - plain).mean()) > 2.0

    def test_faded_all_the_way_is_the_cameras_own_look(self, store, cameras):
        _scene_, a_samples, b_decode, b_look, _ = cameras
        style = _style(store, a_samples)
        frame = LinearFrame(b_decode.copy(), look=b_look)
        own = np.asarray(develop_frame(frame, {"saturation": 1}), np.float32)
        faded = np.asarray(develop_frame(frame, {"style": style.id, "style_fade": 100, "saturation": 1}),
                           np.float32)
        assert float(np.abs(own - faded).mean()) < 0.6

    def test_a_style_this_machine_does_not_have_is_ignored(self, store, cameras):
        _scene_, _a, b_decode, b_look, _ = cameras
        frame = LinearFrame(b_decode.copy(), look=b_look)
        own = np.asarray(develop_frame(frame, {"saturation": 1}), np.float32)
        missing = np.asarray(develop_frame(frame, {"style": "not-here", "saturation": 1}), np.float32)
        assert float(np.abs(own - missing).mean()) < 0.6

    def test_the_style_is_kept_through_the_sidecar(self):
        edit = normalize({"style": "ricoh-positive", "style_fade": 30, "exposure": 0.5})
        assert edit["style"] == "ricoh-positive" and edit["style_fade"] == 30
        crs = to_crs(edit)
        assert crs["crs:PhotoolStyle"] == "ricoh-positive"
        assert from_crs(crs) == edit

    def test_a_fade_without_a_style_is_no_edit(self):
        assert normalize({"style_fade": 40}) == {}
        assert normalize({"style": "../etc/passwd"}) == {}


class TestStore:
    def test_save_load_list_rename_delete(self, store, cameras):
        _scene_, a_samples, *_ = cameras
        style = _style(store, a_samples, name="Ricoh Positive Film")
        assert style.id == "ricoh-positive-film"
        again = styles.from_samples("Ricoh Positive Film", [a_samples])
        assert again.id == "ricoh-positive-film-2"           # ids never collide
        loaded = styles.load(style.id)
        assert loaded.name == "Ricoh Positive Film" and loaded.frames == 1
        assert np.allclose(loaded.look.curve, style.look.curve)
        assert [s["id"] for s in styles.list_styles()] == [style.id]
        assert styles.rename(style.id, "Positive")
        assert styles.load(style.id).name == "Positive"
        assert styles.delete(style.id)
        assert styles.load(style.id) is None
        assert styles.list_styles() == []

    def test_a_deleted_style_is_put_aside_and_comes_back(self, store, cameras):
        _scene_, a_samples, *_ = cameras
        style = _style(store, a_samples, name="Positive")
        assert styles.delete(style.id)
        assert not styles.delete(style.id)                    # already out of the list
        aside = styles.style_dir() / "deleted" / f"{style.id}.npz"
        assert aside.exists()
        aside.replace(styles.style_dir() / aside.name)      # moving it back restores it
        assert styles.load(style.id).name == "Positive"
        assert styles.delete(style.id)
        again = _style(store, a_samples, name="Positive")     # relearned under the same id
        assert again.id == style.id
        assert styles.delete(again.id)                      # both deleted copies are kept
        assert len(list((styles.style_dir() / "deleted").glob("*.npz"))) == 2

    def test_learning_is_spread_over_the_whole_shoot(self):
        picked = styles.spread(list(range(100)), limit=5)
        assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5


class TestLearnFromAFolder:
    """Which frames a style is learned from, and what the panel is told before it
    runs. Reading the files and the fit are stubbed out; what is checked is the
    choice of files."""

    # Each file's rendering settings, as exiftool's Ricoh fields would give them.
    SETTINGS = {
        "R0001.DNG": "GR II|-|7|7|-|-|-|Positive Film|Off|Off|Off",
        "R0002.DNG": "GR II|-|7|7|-|-|-|Positive Film|Off|Off|Off",
        "R0004.DNG": "GR II|-|7|7|-|-|-|Positive Film|Off|Off|Medium",
        "R0005.DNG": "GR II|-|0|0|-|-|-|Standard|Off|Off|Off",
    }

    @pytest.fixture()
    def folder(self, store, monkeypatch):
        from export import engine

        def strings(files):
            return {Path(f): self.SETTINGS.get(Path(f).name, "") for f in files}

        # A RAW's key carries its sensor matrix and a JPEG's cannot, so the two
        # never match: the stub keeps that difference.
        def keys(files):
            return {Path(f): (("raw#" if Path(f).suffix.lower() in engine._RAW_EXTENSIONS else "jpeg#")
                              + self.SETTINGS.get(Path(f).name, "")) for f in files}

        monkeypatch.setattr(engine, "_style_strings", strings)
        monkeypatch.setattr(engine, "_camera_keys", keys)
        monkeypatch.setattr(engine, "_load_linear", lambda source, size: None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("R0001.DNG", "R0001.JPG", "R0002.DNG", "R0003.JPG", "R0004.DNG", "R0005.DNG"):
                (root / name).write_bytes(b"")
            yield engine, root

    def test_a_pair_shown_by_its_jpeg_stands_for_its_raw(self, folder):
        engine, root = folder
        job = engine.learn_style("GR", engine.raw_files_in(root), like=root / "R0001.JPG")
        assert job["total"] == 2

    def test_a_jpeg_with_no_raw_does_not_filter(self, folder):
        engine, root = folder
        job = engine.learn_style("GR", engine.raw_files_in(root), like=root / "R0003.JPG")
        assert job["total"] == 4

    def test_the_preview_says_what_is_learned_and_what_is_left_out(self, folder):
        engine, root = folder
        p = engine.style_preview(engine.raw_files_in(root), root / "R0001.JPG")
        assert (p["total"], p["matching"], p["filtered"]) == (4, 2, True)
        assert p["describe"] == "GR II · contrast 7 · saturation 7 · Positive Film"
        assert sorted(o["differs"] for o in p["others"]) == [
            "contrast 0, saturation 0, Standard", "highlight correction Medium"]

    def test_the_preview_of_a_jpeg_with_no_raw_leaves_nothing_out(self, folder):
        engine, root = folder
        p = engine.style_preview(engine.raw_files_in(root), root / "R0003.JPG")
        assert (p["total"], p["matching"], p["filtered"], p["others"]) == (4, 4, False, [])


def test_one_file_read_alone_has_no_header_and_is_still_read():
    from export import engine

    one = engine._style_blocks("Model : GR II\nPictureStyle : -\n")
    many = engine._style_blocks("======== W:/a/R1.DNG\nModel : GR II\n"
                                "======== W:/a/R1.JPG\nModel : GR II\n")
    assert one == [(None, ["GR II", "-"])]
    assert [p for p, _ in many] == [Path("W:/a/R1.DNG"), Path("W:/a/R1.JPG")]
