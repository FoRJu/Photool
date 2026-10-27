"""
Photool on a computer where exiftool is not on the PATH: found where Homebrew
puts it (exiftool_path), and without it at all, the camera's JPEG and the
orientation still read from the RAW by LibRaw, so the camera's look is learned.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import exiftool_path  # noqa: E402
from samples import sample  # noqa: E402
from thumbnails import extract  # noqa: E402


@pytest.fixture()
def fresh_lookup(monkeypatch):
    monkeypatch.setattr(exiftool_path, "_looked", False)
    monkeypatch.setattr(exiftool_path, "_found", None)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))


def test_it_is_found_where_homebrew_puts_it(tmp_path, monkeypatch, fresh_lookup):
    brew = tmp_path / "homebrew" / "bin"
    brew.mkdir(parents=True)
    tool = brew / "exiftool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setattr(exiftool_path.shutil, "which", lambda name: None)
    monkeypatch.setattr(exiftool_path, "_PLACES", (str(tmp_path / "nowhere"), str(brew)))
    assert exiftool_path.ensure_on_path() == str(tool)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(brew)
    assert exiftool_path.available()


def test_nowhere_is_said(tmp_path, monkeypatch, fresh_lookup):
    monkeypatch.setattr(exiftool_path.shutil, "which", lambda name: None)
    monkeypatch.setattr(exiftool_path, "_PLACES", (str(tmp_path),))
    assert exiftool_path.ensure_on_path() is None and not exiftool_path.available()


def test_the_health_check_says_whether_it_is_there(tmp_path, monkeypatch, fresh_lookup):
    from fastapi.testclient import TestClient

    (tmp_path / "lib").mkdir()
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{tmp_path / 'lib'}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exiftool_path.shutil, "which", lambda name: None)
    monkeypatch.setattr(exiftool_path, "_PLACES", (str(tmp_path),))
    import main

    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["tools"] == {"exiftool": False}


def test_a_raw_gives_up_its_camera_jpeg_without_exiftool(monkeypatch):
    raw = sample("lens-24mm.CR3")
    monkeypatch.setattr(extract, "_exiftool_available", False)
    data = extract.extract_embedded_jpeg(raw)
    assert data is not None and data[:2] == b"\xff\xd8" and len(data) > 100_000
    assert extract._get_raw_orientation(raw) in (1, 3, 6, 8)
