"""
The face scan that starts on its own: which photographs it thinks nobody has
looked at, that it scans only their folders, and that an import puts its
photographs in the catalog so the sweep can find them.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB
from catalog.service import CatalogService
from config import Config
from faces import jobs


def _library(tmp: Path, folders: dict[str, list[str]]):
    library = tmp / "library"
    for folder, names in folders.items():
        (library / folder).mkdir(parents=True)
        for name in names:
            (library / folder / name).write_bytes(b"x")
    db = CatalogDB(str(tmp / "catalog.db"))
    service = CatalogService(db, str(library))
    for folder in folders:
        service.index_folder(folder, [".cr3", ".jpg"])
    return library, service


def _app(tmp: Path, library: Path, service):
    config = Config(library_root=str(library), models_dir=str(tmp / "models"))
    return SimpleNamespace(state=SimpleNamespace(config=config, catalog=service, thumb_cache=None))


@pytest.fixture(autouse=True)
def _idle_job():
    jobs.JOB.update(status="idle", automatic=False)
    jobs.STOP.clear()
    yield
    jobs.JOB.update(status="idle", automatic=False)


def test_a_pair_is_looked_at_when_either_half_was(tmp_path):
    library, service = _library(tmp_path, {
        "2026/2026-10-17": ["A.CR3", "A.JPG", "B.CR3", "B.JPG"],
        "2026/2026-10-18": ["C.JPG"],
    })
    service.db.replace_faces("2026/2026-10-17/A.JPG", [], 1, 1, model="arcface")
    assert jobs.folders_needing_scan(service.db, "arcface") == {
        "2026/2026-10-17": 1, "2026/2026-10-18": 1}


def test_a_scan_by_another_model_does_not_count(tmp_path):
    library, service = _library(tmp_path, {"f": ["A.JPG"]})
    service.db.replace_faces("f/A.JPG", [], 1, 1, model="sface")
    assert jobs.folders_needing_scan(service.db, "arcface") == {"f": 1}
    assert jobs.folders_needing_scan(service.db, "sface") == {}


def _stub_scan(monkeypatch, seen):
    import faces.models
    import faces.service

    monkeypatch.setattr("faces.detect.available", lambda *a, **k: (True, ""))
    monkeypatch.setattr(faces.models, "ensure_models", lambda *a, **k: None)

    def scan_folder(db, cache, root, folder, models_dir, **kwargs):
        seen.append(folder)
        db.replace_faces(f"{folder}/A.JPG", [], 1, 1, model="arcface")
        return {"scanned": 1, "skipped": 0, "failed": 0, "faces": 0}

    monkeypatch.setattr(faces.service, "scan_folder", scan_folder)
    monkeypatch.setattr(faces.service, "rebuild_people", lambda *a, **k: {"people": 0})


def test_the_sweep_scans_only_the_folders_with_gaps(tmp_path, monkeypatch):
    library, service = _library(tmp_path, {"old": ["A.JPG"], "new": ["A.JPG"]})
    service.db.replace_faces("old/A.JPG", [], 1, 1, model="arcface")
    seen = []
    _stub_scan(monkeypatch, seen)

    result = jobs.sweep_once(_app(tmp_path, library, service))
    assert result["started"] and seen == ["new"]
    assert jobs.JOB["status"] == "completed" and jobs.JOB["automatic"]
    assert jobs.JOB["scope"] == jobs.SWEEP_SCOPE

    # Nothing left: the next look starts nothing.
    assert jobs.sweep_once(_app(tmp_path, library, service)) == {"started": False, "folders": 0}


def test_the_sweep_never_fetches_the_models(tmp_path, monkeypatch):
    library, service = _library(tmp_path, {"new": ["A.JPG"]})
    monkeypatch.setattr("faces.detect.available", lambda *a, **k: (False, "not installed"))
    assert jobs.sweep_once(_app(tmp_path, library, service)) == {"started": False, "reason": "models"}
    assert jobs.JOB["status"] == "idle"


def test_the_sweep_waits_for_a_scan_someone_started(tmp_path, monkeypatch):
    library, service = _library(tmp_path, {"new": ["A.JPG"]})
    seen = []
    _stub_scan(monkeypatch, seen)
    jobs.JOB["status"] = "running"
    waits = []

    def wait(seconds):
        waits.append(seconds)
        jobs.JOB["status"] = "completed"     # the clicked scan finishes

    jobs.sweep_once(_app(tmp_path, library, service), wait=wait)
    assert waits and seen == ["new"]


def test_a_second_start_is_refused(tmp_path):
    library, service = _library(tmp_path, {"new": ["A.JPG"]})
    jobs.JOB["status"] = "running"
    assert not jobs.start(_app(tmp_path, library, service), ["new"], "new")


def test_an_import_puts_its_photos_in_the_catalog(tmp_path, monkeypatch):
    from api import routes
    from importer.copy import ImportResult, ImportSummary

    library, service = _library(tmp_path, {"2026/2026-10-17": ["A.JPG"]})
    (library / "2026" / "2026-10-19").mkdir(parents=True)
    copied = library / "2026" / "2026-10-19" / "DJI_0001.JPG"
    copied.write_bytes(b"x")
    woken = []
    monkeypatch.setattr(routes, "_new_photos", lambda app, folders: woken.append(folders))
    monkeypatch.setattr(routes, "get_exif_summary", lambda path: {})

    summary = ImportSummary(results=[
        ImportResult(source=Path("F:/DCIM/DJI_001/DJI_0001.JPG"), destination=copied, status="copied"),
        ImportResult(source=Path("F:/DCIM/DJI_001/DJI_0000.JPG"),
                     destination=library / "2026" / "2026-10-17" / "A.JPG", status="skipped_duplicate"),
    ])
    assert routes._index_imported(_app(tmp_path, library, service), summary) == ["2026/2026-10-19"]
    assert [p.filename for p in service.get_photos_in_folder("2026/2026-10-19")] == ["DJI_0001.JPG"]
    assert woken == [["2026/2026-10-19"]]
