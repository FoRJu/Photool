"""
Pictures that are not a camera's (thumbnails/raster.py): film scans in 16-bit
TIFF, Photoshop documents, PNGs with transparency; and the clips of older
cameras. They are shown in the library like any photograph.
"""

import io
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import Config, is_video  # noqa: E402
from thumbnails.extract import _source_jpeg  # noqa: E402
from thumbnails.raster import open_raster  # noqa: E402


def _psd(path: Path, rgb: tuple, size=(40, 30)) -> None:
    """A minimal flattened Photoshop document: one RGB composite, no layers."""
    w, h = size
    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, h, w, 8, 3)
    body = struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">I", 0)
    planes = b"".join(bytes([c]) * (w * h) for c in rgb)
    path.write_bytes(header + body + struct.pack(">H", 0) + planes)


def _mean(img: Image.Image) -> tuple:
    return tuple(int(v) for v in np.asarray(img.convert("RGB")).reshape(-1, 3).mean(axis=0))


def test_a_photoshop_document_is_read_for_its_composite(tmp_path):
    path = tmp_path / "scan0001.psd"
    _psd(path, (200, 40, 10))
    img = open_raster(path)
    assert img.size == (40, 30) and img.mode == "RGB"
    assert _mean(img) == (200, 40, 10)
    jpeg = _source_jpeg(path)
    assert jpeg[:2] == b"\xff\xd8"


def test_a_16_bit_scan_is_read_at_8_bits(tmp_path):
    import cv2

    path = tmp_path / "roll1.tif"
    a = np.zeros((20, 30, 3), np.uint16)
    a[..., 2] = 50000            # red, in OpenCV's BGR order
    cv2.imwrite(str(path), a)
    img = open_raster(path)
    r, g, b = _mean(img)
    assert abs(r - 50000 // 257) <= 2 and g == 0 and b == 0


def test_opencv_reads_what_pillow_cannot(tmp_path, monkeypatch):
    import cv2

    path = tmp_path / "old.tif"
    cv2.imwrite(str(path), np.full((10, 10, 3), 30000, np.uint16))

    def refuse(*a, **k):
        raise OSError("cannot identify image file")
    monkeypatch.setattr(Image, "open", refuse)
    img = open_raster(path)
    assert img is not None and abs(_mean(img)[0] - 30000 // 257) <= 2


def test_transparency_is_laid_on_white(tmp_path):
    path = tmp_path / "logo.png"
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(path)
    assert _mean(open_raster(path)) == (255, 255, 255)


def test_a_scan_beside_a_raw_of_the_same_name_is_developed_as_itself(tmp_path):
    from export.engine import resolve_export_source

    (tmp_path / "scan0001.dng").write_bytes(b"raw")
    psd = tmp_path / "scan0001.psd"
    _psd(psd, (1, 2, 3))
    assert resolve_export_source(psd) == psd


def test_the_library_shows_scans_and_old_clips_whatever_the_import_list_says():
    config = Config(library_root="/tmp/x", import_extensions=[".cr3", ".jpg"])
    shown = set(config.photo_extensions)
    assert {".tif", ".psd", ".png", ".avi", ".wmv", ".mpg"} <= shown
    assert config.import_extensions == [".cr3", ".jpg"]      # a card import is unchanged
    assert is_video("CLIP0001.WMV") and is_video("mvi_0001.mpg") and not is_video("x.psd")


def test_a_folder_of_scans_is_indexed_and_drawn(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    folder = "2003/2003-09-01 Lakeside"
    lib = tmp_path / "lib" / folder
    lib.mkdir(parents=True)
    _psd(lib / "scan0001.psd", (10, 120, 200))
    Image.new("RGB", (30, 20), (90, 90, 90)).save(lib / "scan0002.tif")
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{tmp_path / 'lib'}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\nanalysis_auto_enabled: false\n")
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app) as c:
        assert c.post(f"/api/catalog/index/{folder}").status_code == 200
        rows = c.get(f"/api/catalog/photos/{folder}").json()["photos"]
        assert sorted(r["filename"] for r in rows) == ["scan0001.psd", "scan0002.tif"]
        r = c.get(f"/api/thumb/{folder}/scan0001.psd", params={"size": "grid"})
        assert r.status_code == 200 and "X-Photool-No-Picture" not in r.headers
        b, g, rr = _mean(Image.open(io.BytesIO(r.content)))[::-1]
        assert rr < 40 and b > 150
