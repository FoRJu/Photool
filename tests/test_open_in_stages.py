"""
A frame with no look yet opens at once on the look its camera's pool knows, while its own
is learned; exports wait for the real one. And the looks most likely to be wanted are
learned ahead, while nobody is using the app, without touching anything else.
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

from export import engine, look_ahead
from test_develop_routes import client, FOLDER  # noqa: F401  (a fixture)
from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng
from test_look import _camera_render, _scene
from adjust.look import frame_samples

pytestmark = pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")

LEVELS = np.geomspace(0.002, 0.95, 28)


def _dng_and_jpeg(path: Path):
    dng = _write_dng(path, [(v, v, v) for v in LEVELS])
    linear = np.repeat(np.stack([np.repeat(LEVELS[None, :], 8, 0)] * 3, -1), 64, axis=1)
    linear = np.repeat(linear, 8, axis=0).astype(np.float32)
    buf = io.BytesIO()
    Image.fromarray(_camera_render(linear)).save(buf, format="JPEG", quality=98)
    return dng, buf.getvalue()


@pytest.fixture
def clean(tmp_path, monkeypatch):
    import thumbnails.extract as extract

    engine.set_profile_dir(tmp_path / "profiles")
    engine.clear_preview_cache()
    monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
    monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
    yield tmp_path
    engine.set_profile_dir(None)
    engine.clear_preview_cache()


def _fill_pool(n=3):
    pool = engine._pool("camera-style")
    for i in range(n):
        scene = _scene(120, 180, seed=i + 1)
        pool[f"other{i}"] = frame_samples(scene, _camera_render(scene))


def _slow_learning(monkeypatch):
    """Learning that waits to be told, so a test can see the time before it finishes."""
    gate = threading.Event()
    real = engine._pooled_look

    def waiting(source, samples, own):
        assert gate.wait(30)
        return real(source, samples, own)

    monkeypatch.setattr(engine, "_pooled_look", waiting)
    return gate


class TestStarter:
    def test_a_new_frame_is_drawn_at_once_on_its_pools_look_then_gets_its_own(self, clean, monkeypatch):
        import thumbnails.extract as extract

        dng, jpeg = _dng_and_jpeg(clean / "IMG_0001.dng")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: jpeg)
        _fill_pool()
        gate = _slow_learning(monkeypatch)

        frame, pending = engine.starter_base(dng, 512)
        assert pending and frame is not None and frame.look is not None
        assert frame.look.frames == 3                      # the pool's: it hasn't this frame in it

        again, still = engine.starter_base(dng, 512)       # a slider moved: same frame, no decode
        assert still and again is frame

        gate.set()
        finished = engine.linear_base(dng, 512)            # waits for the look; never the starter
        assert finished.look is not None and finished is not frame
        latest, pending = engine.starter_base(dng, 512)
        assert not pending and latest is finished

    def test_a_kept_look_opens_finished(self, clean, monkeypatch):
        import thumbnails.extract as extract

        dng, jpeg = _dng_and_jpeg(clean / "IMG_0001.dng")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: jpeg)
        _fill_pool(4)
        monkeypatch.setattr(engine, "_LOOK_SETTLED_FRAMES", 1)
        engine.linear_base(dng, 512)
        engine.clear_preview_cache()
        frame, pending = engine.starter_base(dng, 512)
        assert not pending and frame.look is not None

    def test_with_no_pool_it_waits_as_it_did(self, clean, monkeypatch):
        import thumbnails.extract as extract

        dng, jpeg = _dng_and_jpeg(clean / "IMG_0001.dng")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: jpeg)
        frame, pending = engine.starter_base(dng, 512)
        assert not pending and frame is not None

    def test_an_export_never_gets_the_starter(self, clean, monkeypatch):
        import thumbnails.extract as extract

        dng, jpeg = _dng_and_jpeg(clean / "IMG_0001.dng")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: jpeg)
        _fill_pool()
        gate = _slow_learning(monkeypatch)
        starter, _ = engine.starter_base(dng, 512)
        got = []
        t = threading.Thread(target=lambda: got.append(engine.linear_base(dng, 512)))
        t.start()
        time.sleep(0.3)
        assert not got                                     # still waiting for the look
        gate.set()
        t.join(30)
        assert got[0] is not starter

    def test_a_frame_learned_by_two_callers_at_once_is_learned_once(self, clean, monkeypatch):
        import thumbnails.extract as extract

        dng, jpeg = _dng_and_jpeg(clean / "IMG_0001.dng")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: jpeg)
        calls = []
        real = engine._reference_samples

        def counted(source, rgb):
            calls.append(1)
            time.sleep(0.2)
            return real(source, rgb)

        monkeypatch.setattr(engine, "_reference_samples", counted)
        rgb = engine._load_linear(dng, 512)
        threads = [threading.Thread(target=engine.camera_look_for, args=(dng, rgb)) for _ in range(3)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        assert len(calls) == 1


class TestThroughTheRoute:
    def _raw(self, client, monkeypatch):
        import thumbnails.extract as extract

        library = Path(client.app.state.config.library_root)
        _, jpeg = _dng_and_jpeg(library / FOLDER / "IMG_0002.CR3")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: jpeg)
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
        monkeypatch.setattr(engine, "_camera_key", lambda source: "camera-style")
        engine.clear_preview_cache()
        return f"{FOLDER}/IMG_0002.CR3"

    def test_it_says_the_look_is_pending_until_it_is_not(self, client, monkeypatch):
        filepath = self._raw(client, monkeypatch)
        _fill_pool()
        gate = _slow_learning(monkeypatch)
        body = {"filepath": filepath, "size": 512, "linear": True, "adjustment": {}}
        res = client.post("/api/export/preview", json={**body, "starter": True})
        assert res.status_code == 200
        assert res.headers["x-camera-look-pending"] == "1"
        assert res.headers["x-camera-look-state"] == "starter"
        gate.set()
        res = client.post("/api/export/preview", json=body)      # waits for the look
        assert res.headers["x-camera-look-pending"] == "0"
        assert res.headers["x-camera-look-state"] != "starter"
        res = client.post("/api/export/preview", json={**body, "starter": True})
        assert res.headers["x-camera-look-pending"] == "0"

    def test_the_first_picture_of_an_edited_frame_does_not_develop_it(self, client, monkeypatch):
        import thumbnails.edited as edited

        filepath = f"{FOLDER}/IMG_0001.JPG"
        assert client.put("/api/photos/edit", json={"filepath": filepath, "exposure": 1.0}).status_code == 200
        monkeypatch.setattr(edited, "get_edited_thumbnail",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("developed it")))
        res = client.get(f"/api/thumb/{filepath}?size=preview&camera=1")
        assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"

    def test_a_jpeg_has_nothing_to_wait_for(self, client):
        res = client.post("/api/export/preview", json={"filepath": f"{FOLDER}/IMG_0001.JPG", "size": 128,
                                                       "linear": True, "adjustment": {}, "starter": True})
        assert res.status_code == 200 and res.headers["x-camera-look-pending"] == "0"


class TestLookAhead:
    def _db(self, tmp_path):
        from catalog import CatalogDB

        db = CatalogDB(str(tmp_path / "catalog.db"))
        rows = [("2026/old/a.DNG", 3), ("2026/old/b.DNG", 0)] + [(f"2026/n{i}/c{i}.DNG", 0) for i in range(5)]
        with db._connect() as conn:
            for path, rating in rows:
                folder, name = path.rsplit("/", 1)
                conn.execute("INSERT INTO photos (filepath, filename, folder, extension, base_name, rating) "
                             "VALUES (?, ?, ?, '.DNG', ?, ?)", (path, name, folder, name.split(".")[0], rating))
        return db

    def test_marked_frames_come_first_then_the_newest_folders_and_not_the_rest(self, tmp_path):
        got = look_ahead.wanted(self._db(tmp_path))
        assert got == ["2026/old/a.DNG", "2026/old/b.DNG", "2026/n4/c4.DNG", "2026/n3/c3.DNG"][:1]\
            + sorted(["2026/old/b.DNG", "2026/n4/c4.DNG", "2026/n3/c3.DNG"], reverse=True)

    def test_it_waits_its_turn_and_stops_when_told(self, tmp_path, monkeypatch):
        import background

        db = self._db(tmp_path)
        library = tmp_path / "lib"
        for row in look_ahead.wanted(db):
            (library / row).parent.mkdir(parents=True, exist_ok=True)
            (library / row).write_bytes(b"x")
        look_ahead._seen.clear()
        learned, asked = [], []
        monkeypatch.setattr(engine, "learn_look_ahead", lambda s, size: learned.append(s.name) or True)
        monkeypatch.setattr(background, "wait_turn", lambda name, stop=None: asked.append(name) or True)
        result = look_ahead.run_once(library, db)
        assert result["status"] == "completed" and result["learned"] == len(learned) >= 1
        assert set(asked) == {"Camera looks"} and len(asked) == len(learned)

        again = []
        monkeypatch.setattr(engine, "learn_look_ahead", lambda s, size: again.append(s) or True)
        look_ahead.run_once(library, db)
        assert again == []                                 # not asked twice in a session

        look_ahead._seen.clear()
        stopped = look_ahead.run_once(library, db, should_stop=lambda: True)
        assert stopped["status"] == "stopped" and again == []
        look_ahead._seen.clear()

    def test_one_bad_frame_does_not_stop_the_rest(self, tmp_path, monkeypatch):
        import background

        db = self._db(tmp_path)
        library = tmp_path / "lib"
        for row in look_ahead.wanted(db):
            (library / row).parent.mkdir(parents=True, exist_ok=True)
            (library / row).write_bytes(b"x")
        look_ahead._seen.clear()
        tried = []

        def flaky(source, size):
            tried.append(source.name)
            raise RuntimeError("cannot read")

        monkeypatch.setattr(engine, "learn_look_ahead", flaky)
        monkeypatch.setattr(background, "wait_turn", lambda name, stop=None: True)
        assert look_ahead.run_once(library, db)["status"] == "completed"
        assert len(tried) > 1
        look_ahead._seen.clear()

    def test_new_photographs_wake_it_and_stop_ends_it_promptly(self, tmp_path, monkeypatch):
        db = self._db(tmp_path)
        passes = []
        monkeypatch.setattr(look_ahead, "run_once", lambda root, d, stop: passes.append(1))
        look_ahead.start(tmp_path, db)
        end = time.time() + 5
        while not passes and time.time() < end:
            time.sleep(0.02)
        assert len(passes) == 1
        look_ahead.wake()
        end = time.time() + 5
        while len(passes) < 2 and time.time() < end:
            time.sleep(0.02)
        assert len(passes) == 2
        thread = look_ahead._thread
        look_ahead.stop()
        thread.join(5)
        assert not thread.is_alive()
