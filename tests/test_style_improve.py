"""
Improving a style with frames from other folders (adjust.styles.improve,
export.engine.improve_style): only frames shot with the style's camera and
settings are offered, and the result is kept only when it does no worse on the
style's own frames held out, with the version before kept to undo.

The frames are simulated: camera A's engine renders scenes of varied brightness
with a little noise, as a JPEG would; "other settings" are camera B's plain
engine. The style is learned from three dark frames of A, the "folder" of a
small shoot.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from adjust import styles  # noqa: E402
from adjust.look import frame_samples  # noqa: E402
from test_look import _camera_render, _scene  # noqa: E402
from test_styles import _plain_render  # noqa: E402


def _noisy(img, seed):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)


def _a(seed, light=1.0):
    scene = _scene(60, 90, seed=seed) * light
    return frame_samples(scene, _noisy(_camera_render(scene), seed))


def _b(seed):
    scene = _scene(60, 90, seed=seed)
    return frame_samples(scene, _noisy(_plain_render(scene), seed))


@pytest.fixture(scope="module")
def frames():
    return {
        "own": [_a(s, 0.25) for s in (1, 2, 3)],
        "same": [_a(s) for s in range(10, 18)],
        "other": [_b(s) for s in range(20, 28)],
    }


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp:
        styles.set_dir(tmp)
        yield Path(tmp)
        styles.set_dir(None)


def _saved(own, **kw):
    style = styles.from_samples("Punchy", own)
    for k, v in kw.items():
        setattr(style, k, v)
    styles.save(style, own)
    return styles.load(style.id)


class TestImprove:
    def test_frames_rendered_the_same_way_are_added(self, frames):
        style = styles.from_samples("Punchy", frames["own"])
        better, report, added = styles.improve(style, frames["own"], frames["same"])
        assert better is not None and len(added) == report["added"] == 8
        assert report["after"] <= report["before"]
        assert better.frames == 11 and better.id == style.id

    def test_frames_rendered_another_way_are_refused(self, frames):
        style = styles.from_samples("Punchy", frames["own"])
        better, report, added = styles.improve(style, frames["own"], frames["other"])
        assert better is None and added == []
        assert report["after"] > report["before"]
        assert "worse" in report["reason"]

    def test_mixed_in_only_the_ones_rendered_the_same_way_get_in(self, frames):
        # All of them make it worse; the better-screened half is tried next.
        style = styles.from_samples("Punchy", frames["own"])
        better, _report, added = styles.improve(style, frames["own"], frames["same"] + frames["other"])
        same = {id(s) for s in frames["same"]}
        assert better is not None and len(added) == 8
        assert all(id(s) in same for s in added)

    def test_nothing_offered_says_so(self, frames):
        style = styles.from_samples("Punchy", frames["own"])
        better, report, _ = styles.improve(style, frames["own"], [])
        assert better is None and "none of the frames" in report["reason"]

    def test_one_frame_cannot_be_checked(self, frames):
        style = styles.from_samples("Punchy", frames["own"][:1])
        better, report, _ = styles.improve(style, frames["own"][:1], frames["same"])
        assert better is None and "too few frames" in report["reason"]

    def test_never_more_than_a_style_keeps(self, frames, monkeypatch):
        monkeypatch.setattr(styles, "MAX_ADDED_FRAMES", 3)
        style = styles.from_samples("Punchy", frames["own"])
        better, report, added = styles.improve(style, frames["own"], frames["same"])
        assert better is not None and len(added) == 3


class TestStore:
    def test_what_it_was_learned_from_is_kept(self, store, frames):
        style = _saved(frames["own"], key="raw#GR", settings="GR II|-|7", folders=["/lib/a"])
        assert (style.key, style.settings, style.folders) == ("raw#GR", "GR II|-|7", ["/lib/a"])
        summary = style.summary()
        assert summary["improvable"] and not summary["has_previous"] and summary["folders"] == 1
        assert len(styles.samples_of(style.id)) == 3

    def test_a_style_from_before_cannot_be_improved(self, store, frames):
        assert _saved(frames["own"]).summary()["improvable"] is False

    def test_the_version_before_comes_back(self, store, frames):
        style = _saved(frames["own"], key="k")
        assert styles.keep_previous(style.id)
        better, _report, added = styles.improve(style, frames["own"], frames["same"])
        styles.save(better, frames["own"] + added)
        assert styles.load(style.id).frames == 11
        assert styles.load(style.id).summary()["has_previous"]
        assert styles.restore_previous(style.id)
        assert styles.load(style.id).frames == 3
        assert not styles.restore_previous(style.id)          # one undo

    def test_deleting_drops_the_version_before(self, store, frames):
        style = _saved(frames["own"], key="k")
        styles.keep_previous(style.id)
        assert styles.delete(style.id)
        assert not styles.has_previous(style.id)


class TestFromOtherFolders:
    """improve_style end to end, with reading the files stubbed: which folders
    and frames it looks at, what it saves, and what it says."""

    @pytest.fixture()
    def library(self, store, frames, monkeypatch):
        import background
        from export import engine

        monkeypatch.setattr(background, "_last_use", 0.0)
        background.resume()
        root = store / "lib"
        # "home" is where the style came from; "match" has frames like it,
        # "plain" frames shot with other settings, "none" no RAW at all.
        layout = {"home": ["H1.DNG", "H2.DNG"], "match": ["M1.DNG", "M2.DNG", "M3.DNG", "M4.DNG"],
                  "plain": ["P1.DNG", "P2.DNG"], "none": ["N1.JPG"]}
        for folder, names in layout.items():
            (root / folder).mkdir(parents=True)
            for n in names:
                (root / folder / n).write_bytes(b"")
        samples = {f"M{i + 1}.DNG": s for i, s in enumerate(frames["same"][:4])}
        samples.update({f"P{i + 1}.DNG": s for i, s in enumerate(frames["other"][:2])})
        read = []

        def sample(source, size=1000):
            read.append(Path(source).name)
            return samples.get(Path(source).name)

        monkeypatch.setattr(engine, "_camera_keys",
                            lambda files: {Path(f): "plain" if Path(f).name[0] == "P" else "punchy"
                                           for f in files})
        monkeypatch.setattr(engine, "_style_sample", sample)
        style = _saved(frames["own"], key="punchy", settings="Canon|Standard",
                       folders=[str(root / "home")])
        engine._STYLE_JOB.clear()
        engine._STYLE_JOB.update(status="idle")
        return engine, root, style, read

    def test_only_same_settings_frames_from_other_folders_are_read_and_added(self, library):
        engine, root, style, read = library
        job = engine.improve_style(style.id, [root / f for f in ("home", "match", "plain", "none")])
        assert job["status"] == "done" and job["kind"] == "improve"
        assert sorted(read) == ["M1.DNG", "M2.DNG", "M3.DNG"]   # three per folder, not home, not plain
        report = job["report"]
        assert report["added"] == 3 and report["folders"] == 1
        saved = styles.load(style.id)
        assert saved.frames == 6 and str(root / "match") in saved.folders
        assert saved.improved["added"] == 3
        assert job["style"]["has_previous"]

    def test_no_matching_folder_leaves_it_as_it_was(self, library):
        engine, root, style, read = library
        job = engine.improve_style(style.id, [root / "plain", root / "none"])
        assert job["status"] == "done" and job["report"]["added"] == 0
        assert "no other folder" in job["report"]["reason"]
        assert read == [] and styles.load(style.id).frames == 3
        assert not styles.has_previous(style.id)

    def test_a_style_from_before_says_to_learn_it_again(self, library, frames):
        engine, root, _style, _read = library
        old = _saved(frames["own"][:2])
        job = engine.improve_style(old.id, [root / "match"])
        assert job["status"] == "failed" and "learn it again" in job["error"]


def test_learning_remembers_the_camera_settings_and_folder(store, frames, monkeypatch):
    from export import engine

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for n in ("R1.DNG", "R2.DNG"):
            (root / n).write_bytes(b"")
        monkeypatch.setattr(engine, "_style_strings", lambda files: {Path(f): "GR II|-|7" for f in files})
        monkeypatch.setattr(engine, "_camera_keys", lambda files: {Path(f): "raw#GR" for f in files})
        it = iter(frames["own"])
        monkeypatch.setattr(engine, "_style_sample", lambda source, size=1000: next(it))
        engine._STYLE_JOB.clear()
        job = engine.learn_style("GR", engine.raw_files_in(root), like=root / "R1.DNG")
        assert job["status"] == "done"
        style = styles.load(job["style"]["id"])
        assert (style.key, style.settings, style.folders) == ("raw#GR", "GR II|-|7", [str(root)])
        assert job["style"]["improvable"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    library = tmp_path / "lib"
    library.mkdir()
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app) as c:
        yield c


def test_through_the_api(client, frames):
    assert client.post("/api/styles/nope/improve").status_code == 404
    old = styles.from_samples("Old", frames["own"])
    styles.save(old, frames["own"])
    res = client.post(f"/api/styles/{old.id}/improve")
    assert res.status_code == 400 and "learn it again" in res.json()["detail"]
    assert client.post(f"/api/styles/{old.id}/restore").status_code == 404
    listed = client.get("/api/styles").json()["styles"]
    assert listed[0]["improvable"] is False and listed[0]["has_previous"] is False
