"""
A photograph's camera look is kept on disk once settled, so opening it after a
restart does not learn it again; and the develop view's neighbours are decoded
in the background before they are asked for.
"""

import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from export import engine
from test_edited_thumbnails import _HAVE_REAL_RAW, _write_dng
from test_look import _camera_render

pytestmark = pytest.mark.skipif(not _HAVE_REAL_RAW, reason="needs rawpy and tifffile to build a DNG")

LEVELS = np.geomspace(0.002, 0.95, 28)


def _dng_and_jpeg(folder: Path, name: str = "IMG_0009.dng"):
    dng = _write_dng(folder / name, [(v, v, v) for v in LEVELS])
    linear = np.stack([np.repeat(LEVELS[None, :], 8, 0)] * 3, -1)
    linear = np.repeat(np.repeat(linear, 64, axis=1), 8, axis=0).astype(np.float32)
    buf = io.BytesIO()
    Image.fromarray(_camera_render(linear)).save(buf, format="JPEG", quality=98)
    return dng, buf.getvalue()


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """A profile directory of its own, the camera unidentified (a settled look of
    one frame) unless a test says otherwise, and every cache empty."""
    monkeypatch.setattr(engine, "_camera_key", lambda source: None)
    engine.set_profile_dir(tmp_path / "profiles")
    engine.clear_preview_cache()
    yield tmp_path / "profiles"
    engine.set_profile_dir(None)
    engine.clear_preview_cache()


def _jpeg_is(monkeypatch, jpeg):
    import thumbnails.extract as extract

    monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: jpeg)
    monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)


def _jpeg_must_not_be_read(monkeypatch):
    import thumbnails.extract as extract

    monkeypatch.setattr(extract, "extract_embedded_jpeg",
                        lambda s: (_ for _ in ()).throw(AssertionError("learned again")))


def _restart():
    """What a restart leaves: the pools on disk, nothing in memory."""
    engine.set_profile_dir(engine._PROFILE_DIR)
    engine.clear_preview_cache()


class TestKeptLooks:
    def test_a_settled_look_is_not_learned_again_after_a_restart(self, profiles, tmp_path, monkeypatch):
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        first = engine.linear_base(dng, 1800).look
        status = engine.look_status(dng, first)
        assert first is not None and list((profiles / "looks").glob("*.npz"))

        _restart()
        _jpeg_must_not_be_read(monkeypatch)
        kept = engine.linear_base(dng, 1800).look
        assert np.array_equal(kept.curve, first.curve)
        assert np.array_equal(kept.matrix, first.matrix)
        assert kept.vignette == first.vignette and kept.frames == first.frames
        assert (kept.hue is None) == (first.hue is None)
        # The develop view's note on the look reads the decision kept with it.
        assert engine.look_status(dng, kept) == status

    def test_a_file_changed_since_is_learned_again(self, profiles, tmp_path, monkeypatch):
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        engine.linear_base(dng, 1800)
        _restart()
        stat = dng.stat()
        os.utime(dng, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        read = []
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: read.append(s) or jpeg)
        assert engine.linear_base(dng, 1800).look is not None
        assert read

    def test_a_look_made_while_its_pool_was_learning_is_not_kept(self, profiles, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_camera_key", lambda source: "learningpool")
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        look = engine.linear_base(dng, 1800).look
        assert engine.look_status(dng, look)[0] == "learning"
        assert not list((profiles / "looks").glob("*.npz"))

    def test_a_look_from_a_pool_that_has_since_doubled_is_learned_again(self, profiles, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_camera_key", lambda source: "growingpool")
        monkeypatch.setattr(engine, "_LOOK_SETTLED_FRAMES", 1)
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        engine.linear_base(dng, 1800)
        assert list((profiles / "looks").glob("*.npz"))

        _restart()
        stat = dng.stat()
        assert engine._stored_look(dng, stat) is not None
        pool = engine._pool("growingpool")
        pool["another frame"] = next(iter(pool.values()))
        assert engine._stored_look(dng, stat) is None

    def test_learning_the_folder_learns_every_frame_again(self, profiles, tmp_path, monkeypatch):
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        engine.linear_base(dng, 1800)
        _restart()
        read = []
        import thumbnails.extract as extract

        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda s: read.append(s) or jpeg)
        engine.learn_folder(str(tmp_path), [str(dng)], size=1000)
        assert read

    def test_a_small_first_decode_learns_from_one_the_size_the_look_is_fitted_at(
            self, profiles, tmp_path, monkeypatch):
        from adjust.look import FIT_SIZE

        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        seen = []
        real = engine._reference_samples
        monkeypatch.setattr(engine, "_reference_samples",
                            lambda source, rgb: seen.append(rgb.shape) or real(source, rgb))
        small = engine.linear_base(dng, 256)
        assert max(small.rgb.shape[:2]) <= 256 and small.look is not None
        assert max(seen[0][:2]) == FIT_SIZE
        # and that look is the one every size after is drawn with
        assert engine.linear_base(dng, 1800).look is small.look

    def test_nothing_is_kept_without_a_profile_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(engine, "_camera_key", lambda source: None)
        engine.set_profile_dir(None)
        engine.clear_preview_cache()
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        assert engine.linear_base(dng, 1800).look is not None
        assert not (tmp_path / "looks").exists()


class TestPrefetch:
    def test_the_next_frames_are_decoded_in_the_background(self, profiles, tmp_path, monkeypatch):
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        engine.prefetch([dng], 1000, full=dng)
        engine._PREFETCH_THREAD.join(timeout=60)
        key = (str(dng), 1000, dng.stat().st_mtime_ns, dng.stat().st_size)
        assert key in engine._LINEAR_CACHE
        assert engine._FULL_CACHE

    def test_a_frame_asked_for_while_it_is_being_decoded_is_decoded_once(self, profiles, tmp_path, monkeypatch):
        dng, jpeg = _dng_and_jpeg(tmp_path)
        _jpeg_is(monkeypatch, jpeg)
        calls = []
        real = engine._load_linear
        started = threading.Event()

        def slow(source, size):
            calls.append(source)
            started.set()
            threading.Event().wait(0.3)
            return real(source, size)

        monkeypatch.setattr(engine, "_load_linear", slow)
        results = []
        first = threading.Thread(target=lambda: results.append(engine.linear_base(dng, 1000)))
        first.start()
        started.wait(10)
        results.append(engine.linear_base(dng, 1000))
        first.join(30)
        assert len(calls) == 1
        assert results[0] is results[1]
