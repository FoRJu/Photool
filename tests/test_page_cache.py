"""
Tests for how the built page is cached: the page is checked each time, so a
rebuild shows on a plain refresh, and the content-named assets are kept.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from main import PageFiles


def _client(tmp_path) -> TestClient:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log(1)")
    app = FastAPI()
    app.mount("/", PageFiles(directory=str(tmp_path), html=True), name="page")
    return TestClient(app)


def test_page_is_checked_each_time(tmp_path):
    client = _client(tmp_path)
    for url in ("/", "/index.html"):
        assert client.get(url).headers["cache-control"] == "no-cache"


def test_unchanged_page_comes_back_not_modified(tmp_path):
    client = _client(tmp_path)
    etag = client.get("/").headers["etag"]
    assert client.get("/", headers={"if-none-match": etag}).status_code == 304


def test_assets_are_kept(tmp_path):
    client = _client(tmp_path)
    assert "immutable" in client.get("/assets/index-abc123.js").headers["cache-control"]
