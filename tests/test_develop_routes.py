"""
The develop feature through the HTTP API, against a real (temporary) library:
the edit is stored, the library thumbnail shows it, the preview and Auto work
from the linear decode, and taking the edit off puts the camera's rendering back.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

FOLDER = "2025/2025-03-06"
PHOTO = f"{FOLDER}/IMG_0001.JPG"


def _mean(data: bytes) -> float:
    return _mean_of(Image.open(io.BytesIO(data)))


def _mean_of(picture: Image.Image) -> float:
    return float(np.asarray(picture.convert("L")).mean())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    # A dark frame with a bright strip: something for shadows and Auto to do.
    rgb = np.full((300, 400, 3), 22, dtype=np.uint8)
    rgb[:60] = 245
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
        yield c


def _edit(client, **settings):
    res = client.put("/api/photos/edit", json={"filepath": PHOTO, **settings})
    assert res.status_code == 200
    return res.json()["settings"]


class TestThumbnailsShowTheEdit:
    def test_an_edited_frame_is_drawn_with_its_edit(self, client):
        plain = client.get(f"/api/thumb/{PHOTO}?size=grid").content
        _edit(client, exposure=2.0, shadows=80)
        edited = client.get(f"/api/thumb/{PHOTO}?size=grid&v=x").content
        assert _mean(edited) > _mean(plain) + 20

    def test_the_cull_view_preview_shows_it_too(self, client):
        plain = client.get(f"/api/thumb/{PHOTO}?size=preview").content
        _edit(client, exposure=2.0)
        assert _mean(client.get(f"/api/thumb/{PHOTO}?size=preview&v=x").content) > _mean(plain) + 20

    def test_a_versioned_edit_may_be_cached_and_an_unversioned_one_may_not(self, client):
        _edit(client, exposure=1.0)
        versioned = client.get(f"/api/thumb/{PHOTO}?size=grid&v=abc")
        bare = client.get(f"/api/thumb/{PHOTO}?size=grid")
        assert "immutable" in versioned.headers["cache-control"]
        assert bare.headers["cache-control"] == "no-cache"

    def test_the_plain_thumbnail_stays_cacheable(self, client):
        res = client.get(f"/api/thumb/{PHOTO}?size=grid")
        assert "immutable" in res.headers["cache-control"]

    def test_taking_the_edit_off_puts_the_camera_rendering_back(self, client):
        before = client.get(f"/api/thumb/{PHOTO}?size=grid").content
        _edit(client, exposure=2.0)
        client.get(f"/api/thumb/{PHOTO}?size=grid&v=x")
        assert client.delete(f"/api/photos/edit?filepath={PHOTO}").status_code == 200
        assert client.get(f"/api/thumb/{PHOTO}?size=grid").content == before

    def test_the_full_size_file_is_never_an_edit(self, client):
        plain = client.get(f"/api/thumb/{PHOTO}?size=full").content
        _edit(client, exposure=2.0)
        assert client.get(f"/api/thumb/{PHOTO}?size=full").content == plain

    def test_changing_the_edit_changes_the_thumbnail(self, client):
        _edit(client, exposure=1.0)
        a = _mean(client.get(f"/api/thumb/{PHOTO}?size=grid&v=a").content)
        _edit(client, exposure=2.5)
        b = _mean(client.get(f"/api/thumb/{PHOTO}?size=grid&v=b").content)
        assert b > a + 5


class TestTheNewControlsAreStored:
    def test_every_control_round_trips_through_the_catalog(self, client):
        sent = {
            "exposure": 1.25, "highlights": -60, "shadows": 45, "compress": 35,
            "clarity": 20, "vibrance": 15, "saturation": -5, "fade": 10,
            "shadow_hue": 215, "shadow_sat": 25, "highlight_hue": 40, "highlight_sat": 20,
        }
        stored = _edit(client, **sent)
        assert stored == sent
        again = client.get(f"/api/photos/edit?filepath={PHOTO}").json()["settings"]
        assert again == sent

    def test_the_edit_is_written_to_the_sidecar(self, client, tmp_path):
        _edit(client, exposure=1.0, compress=40, clarity=10)
        xmp = (tmp_path / "lib" / (PHOTO + ".xmp")).read_text()
        assert 'crs:Exposure2012="+1.00"' in xmp
        assert 'crs:ToneCompress="40"' in xmp
        assert 'crs:Clarity2012="+10"' in xmp

    def test_exposure_reaches_five_stops(self, client):
        assert _edit(client, exposure=9.0)["exposure"] == 5.0

    def test_the_edit_appears_in_the_folder_listing_for_the_grid(self, client):
        _edit(client, exposure=1.0)
        photos = client.get(f"/api/catalog/photos/{FOLDER}").json()["photos"]
        assert photos[0]["edited"] is True and photos[0]["edit"]["exposure"] == 1.0


class TestPreviewAndAuto:
    def _preview(self, client, **adjustment):
        res = client.post("/api/export/preview", json={
            "filepath": PHOTO, "size": 256, "adjustment": adjustment,
        })
        assert res.status_code == 200
        return res.content

    def test_the_preview_applies_the_edit(self, client):
        assert _mean(self._preview(client, exposure=2.0)) > _mean(self._preview(client)) + 20

    def test_the_preview_applies_the_new_controls(self, client):
        plain = _mean(self._preview(client))
        assert _mean(self._preview(client, shadows=100)) > plain + 5
        assert _mean(self._preview(client, fade=100)) > plain

    def test_a_draft_is_the_same_picture_at_the_same_size(self, client):
        """While a slider is held the view asks for a draft: drawn from a smaller
        copy of the frame, but sent at the preview's size, and the same edit."""
        def picture(draft):
            res = client.post("/api/export/preview", json={
                "filepath": PHOTO, "size": 256, "linear": True, "draft": draft,
                "adjustment": {"exposure": 1.0, "shadows": 40},
            })
            assert res.status_code == 200
            return Image.open(io.BytesIO(res.content)).convert("RGB")

        full, draft = picture(False), picture(True)
        assert draft.size == full.size
        assert abs(_mean_of(draft) - _mean_of(full)) < 2

    def test_auto_suggests_light_for_a_dark_frame(self, client):
        res = client.post("/api/export/develop/auto", json={"filepath": PHOTO, "size": 256})
        assert res.status_code == 200
        suggestion = res.json()["settings"]
        assert suggestion.get("exposure", 0) > 0 or suggestion.get("shadows", 0) > 0

    def test_the_none_preview_of_an_undeveloped_frame_is_the_camera_picture(
        self, client, tmp_path
    ):
        """"None" promises the export writes the picture the camera made when the
        frame has not been developed, so the preview has to show that picture."""
        res = client.post("/api/export/preview", json={
            "filepath": PHOTO, "size": 256, "resize_only": True,
        })
        assert res.status_code == 200
        camera = Image.open(tmp_path / "lib" / PHOTO).convert("L")
        assert abs(_mean(res.content) - float(np.asarray(camera, dtype=np.float32).mean())) < 4

    def test_the_none_preview_of_a_developed_frame_shows_the_development(self, client):
        """An edit belongs to the photograph rather than to the batch, so "None"
        still shows it; what the tag leaves out is style and matching.
        The panel sends the frame's own edit with the request, as it does here."""
        res = client.post("/api/export/preview", json={
            "filepath": PHOTO, "size": 256, "resize_only": True,
            "adjustment": {"exposure": 2.0, "shadows": 80},
        })
        assert res.status_code == 200
        assert _mean(res.content) > _mean(self._preview(client, exposure=0.01)) + 40
