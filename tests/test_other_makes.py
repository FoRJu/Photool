"""
Cameras other than Canon: what a Ricoh GR II (DNG + JPG, folder 100RICOH, Image
Control in its maker notes) needs from import to look learning.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import Config
from export import engine
from importer.sd_detect import detect_sd_cards, find_importable_files


def test_the_default_import_takes_dng():
    assert ".dng" in Config().import_extensions


def test_the_example_config_lists_dng():
    example = (Path(__file__).resolve().parent.parent / "backend" / "config.example.yaml").read_text()
    assert '".dng"' in example


def test_a_ricoh_card_is_found_with_its_dngs_and_jpegs(monkeypatch):
    """A GR II writes DCIM/100RICOH/R0000001.DNG with R0000001.JPG beside it."""
    # Only the mount path below: on Windows every drive letter is scanned too, and a
    # real card in the reader would be found beside this one (see test_importer).
    monkeypatch.setattr("importer.sd_detect.platform.system", lambda: "Test")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dcim = tmp / "sd" / "DCIM" / "100RICOH"
        dcim.mkdir(parents=True)
        for name in ("R0000001.DNG", "R0000001.JPG", "R0000002.DNG", "R0000002.JPG"):
            (dcim / name).write_bytes(b"x")
        config = Config(library_root=str(tmp / "library"), sd_mount_paths=[str(tmp / "sd")])
        cards = detect_sd_cards(config)
        assert len(cards) == 1 and cards[0].total_files == 4
        names = sorted(f.name for f in find_importable_files(cards[0], config.import_extensions))
        assert names == ["R0000001.DNG", "R0000001.JPG", "R0000002.DNG", "R0000002.JPG"]


class TestStyleOfOtherMakes:
    @staticmethod
    def _exiftool(monkeypatch, lines):
        def fake(cmd, **kwargs):
            assert cmd[0] == "exiftool"
            return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(lines) + "\n", stderr="")
        monkeypatch.setattr(engine.subprocess, "run", fake)

    @staticmethod
    def _lines(values):
        names = engine._STYLE_TAGS + engine._STYLE_TAGS_OTHER
        return [f"{n:<24}: {values.get(n, '-')}" for n in names]

    def test_a_canon_files_style_is_what_it_always_was(self, monkeypatch):
        """Pools already learned on the PC must keep their key."""
        self._exiftool(monkeypatch, self._lines({"Model": "Canon EOS R6m2", "PictureStyle": "Neutral",
                                                  "Contrast": "0", "Saturation": "0"}))
        assert engine._style_of(Path("a.CR3")) == "Canon EOS R6m2|Neutral|0|0|-|-|-"

    def test_a_ricoh_image_control_is_part_of_the_style(self, monkeypatch):
        self._exiftool(monkeypatch, self._lines({"Model": "RICOH GR II", "Contrast": "+1", "Saturation": "0",
                                                  "ImageEffects": "Positive Film"}))
        style = engine._style_of(Path("R0000001.DNG"))
        assert "RICOH GR II" in style and "Positive Film" in style

    def test_a_setting_only_the_jpeg_records_is_taken_from_it(self, tmp_path, monkeypatch):
        raw, jpg = tmp_path / "R0000001.DNG", tmp_path / "R0000001.JPG"
        raw.write_bytes(b"x")
        jpg.write_bytes(b"y")
        raw_lines = self._lines({"Model": "RICOH GR II"})
        jpg_lines = self._lines({"Model": "RICOH GR II", "ImageEffects": "Positive Film"})
        text = "\n".join([f"======== {raw}", *raw_lines, f"======== {jpg}", *jpg_lines]) + "\n"
        monkeypatch.setattr(engine.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=text, stderr=""))
        assert "Positive Film" in engine._style_of(raw)

    def test_the_jpegs_settings_do_not_change_a_canon_key(self, tmp_path, monkeypatch):
        raw, jpg = tmp_path / "A.CR3", tmp_path / "A.JPG"
        raw.write_bytes(b"x")
        jpg.write_bytes(b"y")
        canon = self._lines({"Model": "Canon EOS R6m2", "PictureStyle": "Neutral"})
        text = "\n".join([f"======== {raw}", *canon, f"======== {jpg}",
                           *self._lines({"Model": "Canon EOS R6m2", "PictureStyle": "Standard"})]) + "\n"
        monkeypatch.setattr(engine.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=text, stderr=""))
        assert engine._style_of(raw) == "Canon EOS R6m2|Neutral|-|-|-|-|-"

    def test_two_image_controls_are_two_pools(self, monkeypatch):
        self._exiftool(monkeypatch, self._lines({"Model": "RICOH GR II", "ImageEffects": "Standard"}))
        a = engine._style_of(Path("a.DNG"))
        self._exiftool(monkeypatch, self._lines({"Model": "RICOH GR II", "ImageEffects": "Hi-Contrast B&W"}))
        assert a != engine._style_of(Path("b.DNG"))


class TestThumbnailsOfARawWithoutAPreview:
    def test_the_jpeg_beside_it_is_used_before_decoding_the_raw(self, tmp_path, monkeypatch):
        import thumbnails.extract as extract

        raw = tmp_path / "R0000001.DNG"
        raw.write_bytes(b"not decoded")
        Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(tmp_path / "R0000001.JPG", quality=90)
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: None)
        decoded = []
        monkeypatch.setattr(extract, "_decode_raw_with_rawpy", lambda source: decoded.append(source) or None)
        data = extract._extract_source_jpeg(raw)
        assert data is not None and data[:2] == b"\xff\xd8"
        assert decoded == []

    def test_without_one_it_decodes_the_raw_as_before(self, tmp_path, monkeypatch):
        import thumbnails.extract as extract

        raw = tmp_path / "R0000002.DNG"
        raw.write_bytes(b"x")
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: None)
        monkeypatch.setattr(extract, "_decode_raw_with_rawpy", lambda source: b"\xff\xd8decoded")
        assert extract._extract_source_jpeg(raw) == b"\xff\xd8decoded"


@pytest.mark.skipif(not __import__("test_edited_thumbnails")._HAVE_REAL_RAW,
                    reason="needs rawpy and tifffile to build a DNG")
class TestInspect:
    """`inspect` says what the app sees in a file from a camera it has not met."""

    @staticmethod
    def _pair(tmp_path, monkeypatch, jpeg_shape=None, jpeg=True):
        from test_edited_thumbnails import _write_dng
        import thumbnails.extract as extract

        levels = np.geomspace(0.002, 0.95, 28)
        dng = _write_dng(tmp_path / "R0000001.DNG", [(v, v, v) for v in levels])
        if jpeg:
            h, w = jpeg_shape or (64, 64 * 28)
            Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(tmp_path / "R0000001.JPG", quality=90)
        monkeypatch.setattr(extract, "extract_embedded_jpeg", lambda source: None)
        monkeypatch.setattr(extract, "_get_raw_orientation", lambda source: 1)
        return dng

    def test_a_dng_with_its_jpeg_can_be_learned_from(self, tmp_path, monkeypatch):
        from tools.camera_profile import inspect_file
        info = inspect_file(self._pair(tmp_path, monkeypatch))
        # LibRaw gives no matrix for this synthetic DNG; the look can still be learned
        # (highlights are then rebuilt whole), and the file says so.
        assert info["learnable"]
        assert info["colour_matrix"] is False and any("no usable colour matrix" in n for n in info["notes"])
        assert info["references"][0] == {"kind": "sibling JPEG", "found": True,
                                         "size": [64 * 28, 64], "same_shape": True}

    def test_a_jpeg_of_another_shape_is_said_not_to_be_used(self, tmp_path, monkeypatch):
        from tools.camera_profile import inspect_file
        info = inspect_file(self._pair(tmp_path, monkeypatch, jpeg_shape=(64, 64)))
        assert not info["learnable"]
        assert any("another shape" in n for n in info["notes"])

    def test_a_dng_alone_cannot_be_learned_from(self, tmp_path, monkeypatch):
        from tools.camera_profile import inspect_file
        info = inspect_file(self._pair(tmp_path, monkeypatch, jpeg=False))
        assert not info["learnable"] and any("nothing to learn" in n for n in info["notes"])

    def test_it_prints_a_readable_summary(self, tmp_path, monkeypatch):
        import io
        from tools.camera_profile import inspect
        out = io.StringIO()
        inspect([tmp_path.__class__(self._pair(tmp_path, monkeypatch))], out)
        text = out.getvalue()
        assert "R0000001.DNG" in text and "sibling JPEG" in text and "can be learned" in text


class TestLensKeyAndPictureStyle:
    """The vignetting correction is the camera's per lens setting, and per Image Control
    where the control adds an effect of its own (Ricoh Retro)."""

    @staticmethod
    def _key(monkeypatch, tmp_path, style, name="a.DNG"):
        import json

        f = tmp_path / name
        f.write_bytes(b"x")
        monkeypatch.setattr(engine, "_style_of", lambda source: style)
        monkeypatch.setattr(engine.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps([{"Model": "GR II", "FocalLength": 18.3, "FNumber": 2.8}]), stderr=""))
        engine._LENS_KEYS.clear()
        return engine._lens_key(f)

    def test_two_image_controls_do_not_share_a_lens_pool(self, monkeypatch, tmp_path):
        base = "GR II|-|7|7|-|-|-"
        standard = self._key(monkeypatch, tmp_path, base + "|Standard|Off|Off|Off|-|-", "a.DNG")
        retro = self._key(monkeypatch, tmp_path, base + "|Retro|Off|Off|Off|-|-", "b.DNG")
        assert standard and retro and standard != retro

    def test_the_same_image_control_shares_one(self, monkeypatch, tmp_path):
        s = "GR II|-|7|7|-|-|-|Retro|Off|Off|Off|-|-"
        assert self._key(monkeypatch, tmp_path, s, "a.DNG") == self._key(monkeypatch, tmp_path, s, "b.DNG")

    def test_a_canon_lens_pool_is_shared_across_picture_styles_as_before(self, monkeypatch, tmp_path):
        """No Ricoh/Sony/Fuji settings are recorded, so the style string is the Canon base
        alone and the key is the one it always was."""
        import hashlib

        neutral = self._key(monkeypatch, tmp_path, "Canon EOS R6m2|Neutral|N|N|N|Standard|On", "a.CR3")
        standard = self._key(monkeypatch, tmp_path, "Canon EOS R6m2|Standard|N|N|N|Standard|On", "b.CR3")
        assert neutral == standard
        assert neutral == hashlib.sha1(b"GR II|None|18|2.8").hexdigest()[:16]
