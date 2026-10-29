"""
Background work stepping aside while the app is in use (backend/background.py),
the page's pause and resume, and one render per thumbnail however many ask.
"""

import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import background  # noqa: E402
from thumbnails import extract  # noqa: E402
from thumbnails.extract import ThumbnailCache, ThumbnailSize  # noqa: E402


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(background, "_last_use", 0.0)
    monkeypatch.setattr(background, "QUIET_SECONDS", 0.3)
    monkeypatch.setattr(background, "_POLL_SECONDS", 0.02)
    background.resume()
    yield
    background.resume()


class TestWhatCountsAsUse:
    @pytest.mark.parametrize("path", [
        "/api/thumb/2026/2026-01-01/IMG_0001.JPG",
        "/api/catalog/photos/2026/2026-01-01",
        "/api/library/folders",
        "/api/export/preview",
        "/api/photos/edit",
    ])
    def test_browsing_and_editing_count(self, path):
        assert background.counts_as_use("GET", path)

    @pytest.mark.parametrize("path", [
        "/api/activity", "/api/background", "/api/faces/scan/status", "/api/places/status",
        "/api/export/progress/3", "/api/import/progress/2", "/api/thumbnails/queue/status",
        "/api/camera-look/learn", "/api/styles/learn", "/api/health", "/api/auth/status",
        "/api/export/basket/count", "/api/library/smart-search", "/",
    ])
    def test_the_pages_own_polling_does_not(self, path):
        assert not background.counts_as_use("GET", path)


class TestWaitTurn:
    def test_idle_app_runs_at_once(self):
        start = time.monotonic()
        assert background.wait_turn("Faces")
        assert time.monotonic() - start < 0.05

    def test_waits_until_the_app_has_been_quiet(self):
        background.touch()
        assert background.status()["yielding"]
        start = time.monotonic()
        assert background.wait_turn("Faces")
        assert time.monotonic() - start >= 0.25
        assert not background.status()["yielding"]

    def test_pause_holds_until_resume_and_says_who_waits(self):
        background.pause()
        done = threading.Event()
        worker = threading.Thread(target=lambda: (background.wait_turn("Smart search"), done.set()))
        worker.start()
        time.sleep(0.1)
        assert not done.is_set()
        state = background.status()
        assert state["paused"] and state["waiting"] == ["Smart search"]
        background.resume()
        assert done.wait(1.0)
        worker.join()
        assert background.status()["waiting"] == []

    def test_a_stop_while_waiting_returns_false(self):
        background.pause()
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        assert background.wait_turn("Faces", stop.is_set) is False


class TestOneRenderPerThumbnail:
    def test_callers_at_the_same_moment_share_one_render(self, tmp_path, monkeypatch):
        source = tmp_path / "IMG_0001.JPG"
        Image.new("RGB", (800, 600), (120, 90, 60)).save(source)
        cache = ThumbnailCache(str(tmp_path / "thumbs"))
        calls = []
        real = extract._extract_source_jpeg

        def slow(path):
            calls.append(path)
            time.sleep(0.2)
            return real(path)

        monkeypatch.setattr(extract, "_extract_source_jpeg", slow)
        results = []
        threads = [threading.Thread(target=lambda: results.append(
            extract.get_thumbnail(source, cache, ThumbnailSize.GRID))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert len(results) == 4 and all(results) and len(set(results)) == 1

    def test_a_file_without_a_picture_is_not_tried_again_by_every_caller(self, tmp_path, monkeypatch):
        source = tmp_path / "IMG_0002.JPG"
        source.write_bytes(b"not a picture")
        cache = ThumbnailCache(str(tmp_path / "thumbs"))
        calls = []

        def nothing(path):
            calls.append(path)
            time.sleep(0.1)
            return None

        monkeypatch.setattr(extract, "_extract_source_jpeg", nothing)
        threads = [threading.Thread(target=extract.get_thumbnail, args=(source, cache)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert extract._rendering == {}


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


def test_through_the_api(client):
    background._last_use = 0.0
    assert client.get("/api/background").json()["yielding"] is False
    client.get("/api/activity")                        # polling is not use
    assert client.get("/api/background").json()["yielding"] is False
    client.get("/api/library/folders")                 # browsing is
    assert client.get("/api/background").json()["yielding"] is True

    state = client.post("/api/background/pause").json()
    assert state["paused"] and state["work"] == []
    assert client.get("/api/activity").json()["background"]["paused"]
    assert client.post("/api/background/resume").json()["paused"] is False
