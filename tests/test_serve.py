"""
Starting the server on its own (backend/serve.py): a stale page is rebuilt, and a
second start while one is running leaves quietly.
"""

import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import serve  # noqa: E402


def _frontend(tmp_path, built_after_source=True):
    front = tmp_path / "frontend"
    (front / "src").mkdir(parents=True)
    (front / "dist").mkdir()
    source = front / "src" / "App.jsx"
    source.write_text("x")
    built = front / "dist" / "index.html"
    built.write_text("<html>")
    now = time.time()
    os.utime(source, (now - 100, now - 100) if built_after_source else (now, now))
    os.utime(built, (now - 50, now - 50))
    return front


def test_a_page_built_after_its_source_is_fresh(tmp_path):
    assert not serve.page_is_stale(_frontend(tmp_path))


def test_a_source_changed_after_the_build_makes_the_page_stale(tmp_path):
    assert serve.page_is_stale(_frontend(tmp_path, built_after_source=False))


def test_no_built_page_is_stale(tmp_path):
    front = _frontend(tmp_path)
    (front / "dist" / "index.html").unlink()
    assert serve.page_is_stale(front)


def test_a_port_in_use_is_seen_as_running():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert serve.already_running(port)
    assert not serve.already_running(port)


def test_npm_is_found_as_a_program_windows_can_start(monkeypatch):
    """Node puts an extension-less ``npm`` shell script beside ``npm.cmd``; started
    as a program it fails with WinError 193, so on Windows npm.cmd is the one."""
    found = {"npm": r"C:\Program Files\nodejs\npm", "npm.cmd": r"C:\Program Files\nodejs\npm.cmd"}
    monkeypatch.setattr(serve.shutil, "which", lambda name: found.get(name))
    monkeypatch.setattr(serve.os, "name", "nt")
    assert serve.find_npm().endswith("npm.cmd")
    monkeypatch.setattr(serve.os, "name", "posix")
    assert serve.find_npm().endswith("npm")


def test_the_real_npm_here_can_be_started():
    import subprocess

    npm = serve.find_npm()
    if npm is None:
        import pytest
        pytest.skip("npm is not installed here")
    out = subprocess.run([npm, "--version"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip()
