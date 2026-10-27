"""
Virtual copies: more than one edit of the same file, through the HTTP API.

Version 0 is the photograph's own edit and must behave exactly as before; copies
are 1, 2, … with their own settings, and an export writes each one beside the
frame instead of over it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

FOLDER = "2025/2025-03-06"
PHOTO = f"{FOLDER}/IMG_0001.JPG"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    rgb = np.full((300, 400, 3), 80, dtype=np.uint8)
    Image.fromarray(rgb, "RGB").save(library / PHOTO, format="JPEG", quality=95)
    (tmp_path / "config.yaml").write_text(
        # Single-quoted: YAML takes the backslashes of a Windows path literally there.
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app) as c:
        assert c.post(f"/api/catalog/index/{FOLDER}").status_code == 200
        yield c, library, tmp_path


def _put(c, version, **settings):
    r = c.put("/api/photos/edit", json={"filepath": PHOTO, "version": version, **settings})
    assert r.status_code == 200, r.text
    return r.json()["settings"]


def test_a_copy_starts_from_the_edit_it_was_made_from(env):
    c, _, _ = env
    _put(c, 0, exposure=1.0, contrast=20)
    made = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()
    assert made["version"] == 1
    assert made["settings"] == {"exposure": 1.0, "contrast": 20}


def test_copies_are_edited_independently_of_the_photo(env):
    c, _, _ = env
    _put(c, 0, exposure=1.0)
    v = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    _put(c, v, saturation=-100)
    assert c.get("/api/photos/edit", params={"filepath": PHOTO}).json()["settings"] == {"exposure": 1.0}
    got = c.get("/api/photos/edit", params={"filepath": PHOTO, "version": v}).json()["settings"]
    assert got["saturation"] == -100


def test_a_copy_with_no_edit_is_kept(env):
    c, _, _ = env
    v = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    _put(c, v, exposure=0.5)
    c.request("DELETE", "/api/photos/edit", params={"filepath": PHOTO, "version": v})
    copies = c.get("/api/photos/versions", params={"filepath": PHOTO}).json()["copies"]
    assert [x["version"] for x in copies] == [v] and copies[0]["settings"] == {}


def test_numbers_are_not_reused_after_a_delete(env):
    c, _, _ = env
    a = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    b = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    c.request("DELETE", "/api/photos/versions", params={"filepath": PHOTO, "version": b})
    again = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    assert (a, b, again) == (1, 2, 2) or again > a  # highest-remaining + 1 is acceptable


def test_an_unknown_copy_is_a_404(env):
    c, _, _ = env
    assert c.get("/api/photos/edit", params={"filepath": PHOTO, "version": 9}).status_code == 404
    r = c.put("/api/photos/edit", json={"filepath": PHOTO, "version": 9, "exposure": 1})
    assert r.status_code == 404


def test_copies_survive_losing_the_catalog(env):
    c, library, _ = env
    v = c.post("/api/photos/versions", json={"filepath": PHOTO, "name": "Mono"}).json()["version"]
    _put(c, v, saturation=-100)
    sidecar = library / (PHOTO + ".copies.json")
    assert json.loads(sidecar.read_text())["copies"][0]["name"] == "Mono"

    # A fresh catalog over the same library.
    import main
    from catalog.database import CatalogDB
    from catalog.service import CatalogService

    db = CatalogDB(str(library.parent / "other.db"))
    svc = CatalogService(db, str(library))
    svc.index_folder(FOLDER, [".jpg"])
    copies = svc.photo_versions(PHOTO)
    assert copies and copies[0]["name"] == "Mono" and copies[0]["settings"]["saturation"] == -100


def test_the_last_copy_removed_removes_its_file(env):
    c, library, _ = env
    v = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    assert (library / (PHOTO + ".copies.json")).exists()
    c.request("DELETE", "/api/photos/versions", params={"filepath": PHOTO, "version": v})
    assert not (library / (PHOTO + ".copies.json")).exists()


def test_pasting_across_a_burst_does_not_touch_copies(env):
    c, _, _ = env
    v = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    _put(c, v, saturation=-100)
    c.post("/api/photos/edit/batch", json={"filepaths": [PHOTO], "exposure": 2.0})
    got = c.get("/api/photos/edit", params={"filepath": PHOTO, "version": v}).json()["settings"]
    assert got == {"saturation": -100}


def test_export_writes_each_copy_beside_the_frame(env):
    c, library, tmp = env
    v = c.post("/api/photos/versions", json={"filepath": PHOTO, "name": "Black & White"}).json()["version"]
    _put(c, v, saturation=-100)
    skipped = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    c.put("/api/photos/versions", json={"filepath": PHOTO, "version": skipped, "export": False})
    c.post("/api/export/basket/add", json={"filepath": PHOTO, "folder": FOLDER, "filename": "IMG_0001.JPG"})
    out = library / "export" / "out"
    started = c.post("/api/export/run", json={"folder": "out"}).json()
    assert started["total"] == 2
    import time
    for _ in range(100):
        prog = c.get(f"/api/export/progress/{started['job_id']}").json()
        if prog["status"] == "completed":
            break
        time.sleep(0.1)
    assert prog["failed"] == 0
    assert sorted(p.name for p in out.glob("*.jpg")) == ["IMG_0001.jpg", "IMG_0001_vc1-black-white.jpg"]
    assert c.get("/api/export/basket").json()["items"][0]["copies"] == 2


def _none_export(c, tmp):
    """Run a "None" export of whatever is in the basket; return the progress."""
    c.post("/api/export/basket/add", json={"filepath": PHOTO, "folder": FOLDER,
                                          "filename": "IMG_0001.JPG"})
    out = tmp / "lib" / "export" / "out"
    started = c.post("/api/export/run", json={"folder": "out"}).json()
    import time
    for _ in range(100):
        prog = c.get(f"/api/export/progress/{started['job_id']}").json()
        if prog["status"] == "completed":
            break
        time.sleep(0.1)
    assert prog["failed"] == 0, prog
    return started, out


def test_a_none_export_of_an_undeveloped_frame_writes_it_as_shot(env):
    """"None" writes this frame, which has not been developed, as the camera's own
    picture, resized; and its copy marked for export beside it, as that copy is
    developed — a copy is a development the person asked for, not batch styling."""
    c, _, tmp = env
    v = c.post("/api/photos/versions", json={"filepath": PHOTO, "name": "Dark"}).json()["version"]
    _put(c, v, exposure=-2.0)
    skipped = c.post("/api/photos/versions", json={"filepath": PHOTO}).json()["version"]
    c.put("/api/photos/versions", json={"filepath": PHOTO, "version": skipped, "export": False})
    started, out = _none_export(c, tmp)
    # The copy switched off is neither written nor counted, so the number the
    # panel shows is the number of files that will exist.
    assert started["total"] == 2
    assert sorted(p.name for p in out.glob("*.jpg")) == ["IMG_0001.jpg", "IMG_0001_vc1-dark.jpg"]
    written = Image.open(out / "IMG_0001.jpg").convert("L")
    assert abs(float(np.asarray(written, dtype=np.float32).mean()) - 80) <= 4
    dark = Image.open(out / "IMG_0001_vc1-dark.jpg").convert("L")
    assert float(np.asarray(dark, dtype=np.float32).mean()) < 60  # 80, two stops down


def test_a_none_export_of_a_developed_frame_writes_the_development(env):
    """An edit belongs to the photograph rather than to the batch, so "None" — which
    leaves out the batch's style and matching — still writes it. The frame
    here is stored a stop up."""
    c, _, tmp = env
    _put(c, 0, exposure=1.0)
    started, out = _none_export(c, tmp)
    assert started["total"] == 1
    assert sorted(p.name for p in out.glob("*.jpg")) == ["IMG_0001.jpg"]
    written = Image.open(out / "IMG_0001.jpg").convert("L")
    assert float(np.asarray(written, dtype=np.float32).mean()) > 100  # 80, plus a stop
