"""
Videos in the library: shown beside the photographs with a frame for a thumbnail,
their date and length, played as they are, culled and exported like a photograph,
and never developed or counted in a burst.
"""

import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from config import is_video  # noqa: E402
from thumbnails import video  # noqa: E402

FOLDER = "2026/2026-07-04"
# A DJI Pocket names its clips with the local time it began, to the second, and
# writes no DateTimeOriginal.
CLIP = f"{FOLDER}/DJI_20260704100836_0206_D.MP4"
PHOTO = f"{FOLDER}/IMG_0001.JPG"

needs_ffmpeg = pytest.mark.skipif(not video.available(), reason="ffmpeg is not installed")


def test_what_is_a_video():
    assert is_video(".mp4") and is_video("CLIP.MOV") and is_video(Path("a/b/c.m4v"))
    assert not is_video(".jpg") and not is_video("IMG_0001.CR3")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    subprocess.run(
        [video._tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(library / CLIP)],
        check=True, capture_output=True,
    )
    Image.fromarray(np.full((240, 320, 3), 128, np.uint8), "RGB").save(library / PHOTO, format="JPEG")
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
        assert c.post(f"/api/catalog/index/{FOLDER}").status_code == 200
        yield c, library


def _row(c, filepath):
    rows = c.get(f"/api/catalog/photos/{FOLDER}").json()["photos"]
    return next(r for r in rows if r["filepath"] == filepath)


@needs_ffmpeg
class TestAVideoInTheLibrary:
    def test_it_is_listed_with_its_length_and_the_time_in_its_name(self, client):
        c, _ = client
        row = _row(c, CLIP)
        assert row["video"] is True and row["duration"] == pytest.approx(2.0, abs=0.2)
        assert row["capture_date"] == "2026:07:04 10:08:36"
        assert _row(c, PHOTO)["video"] is False

    def test_its_thumbnail_is_a_frame_of_it(self, client):
        c, _ = client
        res = c.get(f"/api/thumb/{CLIP}?size=grid")
        assert res.status_code == 200
        assert Image.open(io.BytesIO(res.content)).size[0] > 0

    def test_it_is_played_as_it_is_a_part_at_a_time(self, client):
        c, library = client
        res = c.get(f"/api/video/{CLIP}", headers={"Range": "bytes=0-99"})
        assert res.status_code == 206 and res.content == (library / CLIP).read_bytes()[:100]
        assert c.get(f"/api/video/{PHOTO}").status_code == 404

    def test_a_copy_any_browser_plays_is_made_when_asked_for(self, client):
        c, _ = client
        for _ in range(300):
            res = c.get(f"/api/video/{CLIP}?compatible=1")
            if res.status_code != 202:
                break
            assert res.json()["state"] == "working"
            time.sleep(0.2)
        assert res.status_code == 200 and res.content[4:8] == b"ftyp"

    def test_it_is_rated_like_a_photograph(self, client):
        c, _ = client
        assert c.put("/api/catalog/rate", json={"filepath": CLIP, "rating": 3}).status_code == 200
        assert _row(c, CLIP)["rating"] == 3

    def test_it_is_never_developed(self, client):
        c, _ = client
        res = c.put("/api/photos/edit", json={"filepath": CLIP, "exposure": 1.0})
        assert res.status_code == 400
        assert _row(c, CLIP)["edited"] is False

    def test_it_is_exported_as_it_is(self, client):
        c, library = client
        c.post("/api/export/basket/add", json={"filepath": CLIP, "folder": FOLDER,
                                               "filename": Path(CLIP).name})
        job = c.post("/api/export/run", json={"folder": "clips"}).json()
        for _ in range(100):
            prog = c.get(f"/api/export/progress/{job['job_id']}").json()
            if prog["status"] != "running":
                break
            time.sleep(0.1)
        assert prog["status"] == "completed" and prog["failed"] == 0, prog
        written = library / "export" / "clips" / Path(CLIP).name
        assert written.read_bytes() == (library / CLIP).read_bytes()
