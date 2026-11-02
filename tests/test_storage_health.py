"""The write probe: it reports what the disk does, and is not fooled by its own other runs."""

import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from storage_health import probe_writable  # noqa: E402


def test_probes_at_the_same_time_do_not_fail_each_other(tmp_path):
    """Two requests at once, as a phone and a computer reconnecting after a restart do. Sharing
    one scratch file, they reported a healthy disk as read-only."""
    target = tmp_path / "catalog.db"
    target.write_bytes(b"x")
    failures = []

    def probes():
        for _ in range(100):
            ok, message = probe_writable(target)
            if not ok:
                failures.append(message)

    threads = [threading.Thread(target=probes) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == []


def test_it_leaves_nothing_behind(tmp_path):
    target = tmp_path / "catalog.db"
    target.write_bytes(b"x")
    assert probe_writable(target) == (True, "")
    assert [p.name for p in tmp_path.iterdir()] == ["catalog.db"]


def test_a_folder_that_refuses_writes_is_still_reported(tmp_path, monkeypatch):
    def refuse(self, data):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    ok, message = probe_writable(tmp_path / "catalog.db")
    assert not ok and "EACCES" in message and str(tmp_path) in message


@pytest.mark.skipif(os.name != "nt", reason="Windows holds a file open against opening it for writing")
def test_a_missing_folder_is_made(tmp_path):
    assert probe_writable(tmp_path / "new" / "catalog.db") == (True, "")
    assert (tmp_path / "new").is_dir()
