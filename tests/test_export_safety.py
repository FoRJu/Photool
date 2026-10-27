"""
Where an export may write, and who gets what it wrote.

This computer writes only inside <library>/export and never over a file. A
paired device writes nothing here: its export is a download that only it can
fetch, made from a basket of its own.
"""

import io
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

FOLDER = "2024/trip"
PHOTO = f"{FOLDER}/IMG_0001.JPG"
LAN = ("192.168.1.20", 50000)
OTHER_LAN = ("192.168.1.21", 50000)
HOME = ("127.0.0.1", 50001)


@pytest.fixture
def env(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    rgb = np.full((300, 400, 3), 80, dtype=np.uint8)
    Image.fromarray(rgb, "RGB").save(library / PHOTO, format="JPEG", quality=95)
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
        "lan_access: true\n"
    )
    monkeypatch.chdir(tmp_path)
    import main
    from export import jobs

    from api import routes

    jobs._progress.clear()   # jobs live for the process, not for one test
    routes._coverage_cache["value"] = None

    with TestClient(main.app, client=HOME) as home:
        assert home.post(f"/api/catalog/index/{FOLDER}").status_code == 200
        yield main.app, home, library


def _pair(app, address=LAN, name="Mom's iPad"):
    lan, home = TestClient(app, client=address), TestClient(app, client=HOME)
    asked = lan.post("/api/auth/pair/request", json={"name": name}).json()
    assert home.post(f"/api/auth/pair/{asked['id']}/approve").status_code == 200
    assert lan.post("/api/auth/pair/claim",
                    json={"id": asked["id"], "claim": asked["claim"]}).json()["status"] == "approved"
    return lan


def _basket(c):
    c.post("/api/export/basket/add", json={"filepath": PHOTO, "folder": FOLDER, "filename": "IMG_0001.JPG"})


def _run(c, **body):
    started = c.post("/api/export/run", json=body)
    assert started.status_code == 200, started.text
    job = started.json()
    for _ in range(100):
        prog = c.get(f"/api/export/progress/{job['job_id']}").json()
        if prog["status"] != "running":
            break
        time.sleep(0.1)
    assert prog["status"] == "completed" and prog["failed"] == 0, prog
    return job, prog


class TestThisComputer:
    @pytest.mark.parametrize("name", ["../outside", "C:/Windows", "/etc", "a/../../b", "nul", "what?"])
    def test_a_path_is_refused(self, env, name):
        _, home, _ = env
        _basket(home)
        res = home.post("/api/export/run", json={"folder": name})
        assert res.status_code == 400
        assert "export folder" in res.json()["detail"]

    def test_exports_land_in_the_export_folder_and_never_overwrite(self, env):
        _, home, library = env
        _basket(home)
        job, _ = _run(home, folder="Trip")
        assert job["kind"] == "folder" and Path(job["folder"]) == (library / "export" / "Trip").resolve()
        _run(home, folder="Trip")
        assert sorted(p.name for p in (library / "export" / "Trip").iterdir()) == ["IMG_0001-2.jpg", "IMG_0001.jpg"]
        # The original is untouched.
        assert (library / PHOTO).stat().st_size > 0

    def test_no_name_means_now(self, env):
        _, home, library = env
        _basket(home)
        job, _ = _run(home)
        assert Path(job["folder"]).parent == (library / "export").resolve()


class TestShowingTheFolder:
    """After an export on this computer, its folder can be opened in Explorer, to
    drag the photos into Google Photos. Only that folder, and only from here."""

    def test_the_folder_written_is_opened(self, env, monkeypatch):
        import os

        _, home, library = env
        _basket(home)
        job, _ = _run(home, folder="Trip")
        opened = []
        monkeypatch.setattr(os, "startfile", lambda path: opened.append(path), raising=False)
        monkeypatch.setattr(sys, "platform", "win32")
        res = home.post(f"/api/export/open/{job['job_id']}")
        assert res.status_code == 200, res.text
        assert [Path(p) for p in opened] == [(library / "export" / "Trip").resolve()]

    def test_an_unknown_export_opens_nothing(self, env):
        _, home, _ = env
        assert home.post("/api/export/open/nosuchjob").status_code == 404

    def test_another_device_cannot_open_a_window_here(self, env):
        app, home, _ = env
        _basket(home)
        job, _ = _run(home, folder="Trip")
        lan = _pair(app)
        assert lan.post(f"/api/export/open/{job['job_id']}").status_code == 403


class TestHowAPhotoIsWritten:
    def test_the_old_batch_style_and_matching_are_ignored(self, env):
        """A request from an older page (or agent) that still names them writes the
        photo as it stands, the same bytes as one that does not."""
        _, home, library = env
        _basket(home)
        _run(home, folder="plain")
        _run(home, folder="asked", harmonize=True, resize_only=False,
             style={"saturation": 100, "vignette": 100})
        plain = (library / "export" / "plain" / "IMG_0001.jpg").read_bytes()
        asked = (library / "export" / "asked" / "IMG_0001.jpg").read_bytes()
        assert plain == asked

    def test_the_old_style_presets_are_gone(self, env):
        _, home, _ = env
        assert home.get("/api/export/presets").status_code in (404, 405)


class TestAPairedDevice:
    def test_its_basket_is_its_own(self, env):
        app, home, _ = env
        lan = _pair(app)
        _basket(lan)
        assert lan.get("/api/export/basket/count").json()["count"] == 1
        assert home.get("/api/export/basket/count").json()["count"] == 0
        assert lan.get("/api/catalog/search", params={"q": "is:basket"}).json()["total"] == 1
        assert home.get("/api/catalog/search", params={"q": "is:basket"}).json()["total"] == 0

    def test_it_gets_a_download_and_writes_nothing_here(self, env):
        app, home, library = env
        lan = _pair(app)
        _basket(lan)
        # Asking for a place on this computer's disk is ignored, not obeyed.
        job, prog = _run(lan, folder="../../somewhere", strip_exif=False)
        assert job["kind"] == "download" and job["folder"] == ""
        assert not (library / "export").exists()
        names = lan.get(f"/api/export/download/{job['job_id']}").json()["files"]
        assert names == ["IMG_0001.jpg"]
        one = lan.get(f"/api/export/download/{job['job_id']}/file/IMG_0001.jpg")
        assert one.status_code == 200 and one.headers["content-type"] == "image/jpeg"
        z = lan.get(f"/api/export/download/{job['job_id']}/photos.zip")
        assert z.status_code == 200
        assert zipfile.ZipFile(io.BytesIO(z.content)).namelist() == ["IMG_0001.jpg"]
        # This computer sees that it happened, but the download is the device's.
        assert home.get(f"/api/export/progress/{job['job_id']}").json()["owner_name"] == "Mom's iPad"
        assert home.get(f"/api/export/download/{job['job_id']}/photos.zip").status_code == 404

    def test_nobody_else_can_fetch_it(self, env):
        app, _, _ = env
        mine, theirs = _pair(app), _pair(app, OTHER_LAN, "Laptop")
        _basket(mine)
        job, _ = _run(mine)
        assert theirs.get(f"/api/export/progress/{job['job_id']}").status_code == 404
        assert theirs.get(f"/api/export/download/{job['job_id']}/photos.zip").status_code == 404
        assert theirs.get(f"/api/export/download/{job['job_id']}/file/IMG_0001.jpg").status_code == 404

    def test_only_written_names_are_served(self, env):
        app, _, _ = env
        lan = _pair(app)
        _basket(lan)
        job, _ = _run(lan)
        assert lan.get(f"/api/export/download/{job['job_id']}/file/..%2F..%2Fcatalog.db").status_code == 404

    def test_a_download_has_a_size_limit(self, env):
        app, _, _ = env
        app.state.config.remote_export_limit = 0
        lan = _pair(app)
        _basket(lan)
        res = lan.post("/api/export/run", json={})
        assert res.status_code == 400 and "more than one download" in res.json()["detail"]


def test_free_path_never_reuses_a_name(tmp_path):
    from export.engine import free_path

    (tmp_path / "IMG_0001.JPG").write_bytes(b"original")
    path = free_path(tmp_path, "IMG_0001", ".JPG")
    assert path.name == "IMG_0001-2.JPG"


def test_a_shared_basket_becomes_this_computers(tmp_path):
    """A catalog from before baskets had owners keeps its basket, as this
    computer's."""
    db_path = tmp_path / "catalog.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE export_basket (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                folder TEXT NOT NULL DEFAULT '',
                filename TEXT NOT NULL DEFAULT '',
                added_at TEXT DEFAULT ''
            );
            INSERT INTO export_basket (filepath, folder, filename) VALUES ('a/b.jpg', 'a', 'b.jpg');
        """)
    from catalog import CatalogDB

    db = CatalogDB(str(db_path))
    assert [r.filepath for r in db.list_basket()] == ["a/b.jpg"]
    db.add_to_basket("a/b.jpg", "a", "b.jpg", owner="device:1")
    assert db.basket_count("device:1") == 1 and db.basket_count() == 1


def test_activity_shows_every_export_here_and_only_its_own_elsewhere(env):
    app, home, _ = env
    lan, other = _pair(app), _pair(app, OTHER_LAN, "Laptop")
    _basket(lan)
    _run(lan)
    seen = home.get("/api/activity").json()
    assert seen["coverage"]["photos"] == 1
    assert [e["owner_name"] for e in seen["exports"]] == ["Mom's iPad"]
    assert len(lan.get("/api/activity").json()["exports"]) == 1
    assert other.get("/api/activity").json()["exports"] == []
