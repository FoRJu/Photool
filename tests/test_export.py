"""
Tests for what an export is developed from, and for the EXIF that has to survive
it.

Both matter for the same reason: a school event is delivered as a set, so the
set has to start from comparable files and still carry the day the photos were
taken.
"""

import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from export import engine
from export.engine import MatchSettings, export_photo, resolve_export_source


class TestResolveExportSource:
    def test_prefers_the_raw_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_0001.JPG").write_bytes(b"x")
            (root / "IMG_0001.CR3").write_bytes(b"x")
            resolved = resolve_export_source(root / "IMG_0001.JPG")
            assert resolved.name == "IMG_0001.CR3"

    def test_falls_back_to_the_basked_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jpg = root / "IMG_0002.JPG"
            jpg.write_bytes(b"x")
            assert resolve_export_source(jpg) == jpg

    def test_a_raw_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cr3 = root / "IMG_0003.CR3"
            cr3.write_bytes(b"x")
            (root / "IMG_0003.JPG").write_bytes(b"x")
            assert resolve_export_source(cr3) == cr3

    def test_can_be_turned_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jpg = root / "IMG_0004.JPG"
            jpg.write_bytes(b"x")
            (root / "IMG_0004.CR3").write_bytes(b"x")
            assert resolve_export_source(jpg, prefer_raw=False) == jpg

    def test_prefers_the_camera_raw_over_a_dng(self):
        """Order matters: a sidecar DNG is not what the camera shot."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_0005.JPG").write_bytes(b"x")
            (root / "IMG_0005.DNG").write_bytes(b"x")
            (root / "IMG_0005.CR3").write_bytes(b"x")
            resolved = resolve_export_source(root / "IMG_0005.JPG")
            assert resolved.name == "IMG_0005.CR3"

    def test_name_with_dots_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_0006 (1).JPG").write_bytes(b"x")
            (root / "IMG_0006 (1).CR3").write_bytes(b"x")
            resolved = resolve_export_source(root / "IMG_0006 (1).JPG")
            assert resolved.name == "IMG_0006 (1).CR3"


class TestExifCarryOver:
    """rawpy writes pixels and nothing else, so a RAW-sourced export has no
    dates in it until they are copied from the file the photo was shot into."""

    def _capture_args(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            engine.subprocess, "run",
            lambda args, **kwargs: calls.append(args),
        )
        return calls

    def test_dates_come_from_the_raw_when_the_pixels_do(self, monkeypatch):
        calls = self._capture_args(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "IMG_0001.CR3"
            raw.write_bytes(b"x")
            out = root / "IMG_0001.jpg"
            out.write_bytes(b"y")
            engine._write_exif(out, "", True, exif_source=raw)

        args = calls[0]
        assert "-tagsFromFile" in args
        assert args[args.index("-tagsFromFile") + 1] == str(raw)
        assert "-DateTimeOriginal" in args
        assert "-CreateDate" in args
        assert args[-1] == str(out)

    def test_a_jpeg_export_keeps_reading_its_own_tags(self, monkeypatch):
        calls = self._capture_args(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "IMG_0002.jpg"
            out.write_bytes(b"y")
            engine._write_exif(out, "", True, exif_source=out)
        assert calls[0][calls[0].index("-tagsFromFile") + 1] == "@"

    def test_a_missing_source_falls_back_to_the_output(self, monkeypatch):
        calls = self._capture_args(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "IMG_0003.jpg"
            out.write_bytes(b"y")
            engine._write_exif(out, "", True, exif_source=Path(tmp) / "gone.CR3")
        assert calls[0][calls[0].index("-tagsFromFile") + 1] == "@"

    def test_photographer_still_written(self, monkeypatch):
        calls = self._capture_args(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "IMG_0004.jpg"
            out.write_bytes(b"y")
            engine._write_exif(out, "A. Photographer", True)
        assert "-Artist=A. Photographer" in calls[0]

    def test_nothing_to_do(self, monkeypatch):
        calls = self._capture_args(monkeypatch)
        engine._write_exif(Path("out.jpg"), "", False)
        assert calls == []


class TestRawDecodeBaseline:
    """Matching a set only works if every frame starts from the same kind of
    decode. Auto-bright normalises each frame against its own histogram, so it
    adds the very drift the export is meant to remove: measured over eight
    frames of one event it widened the brightness spread from 0.8 EV to 1.4 EV
    and lifted frames by an unpredictable 0 to 0.9 EV."""

    def _fake_rawpy(self, monkeypatch):
        captured = {}

        class _Raw:
            sizes = types.SimpleNamespace(width=6000, height=4000)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def postprocess(self, **kwargs):
                captured.update(kwargs)
                return np.zeros((16, 16, 3), dtype=np.uint8)

        monkeypatch.setitem(sys.modules, "rawpy", types.SimpleNamespace(imread=lambda path: _Raw()))
        return captured

    def test_the_decode_is_fixed_not_auto_brightened(self, monkeypatch):
        captured = self._fake_rawpy(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            image = engine._load_raw(Path(tmp) / "IMG_0001.CR3", 2048)
        assert image is not None
        assert captured["no_auto_bright"] is True
        assert captured["use_camera_wb"] is True

    def test_a_small_output_still_uses_the_half_size_decode(self, monkeypatch):
        captured = self._fake_rawpy(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            engine._load_raw(Path(tmp) / "IMG_0001.CR3", 2048)
        assert captured["half_size"] is True


class TestResizeOnlyExport:
    """How the export panel writes every photo: as it stands, resized.

    The picture is already what it should be. For a frame nobody has developed
    that means the camera's own JPEG — the sensor data is not the picture, and the
    camera's rendering of it (its white balance, its Picture Style, its lens
    correction and its noise reduction) is. For a frame that *has* been developed
    it means the developed frame: an edit belongs to the photograph, not to the
    batch, so it is written as developed.

    Measured on one R6 II frame against its camera JPEG: the JPEG resized is
    identical, a developed decode is 8.5 levels of 255 away, and the plain decode
    the old unedited path wrote is 48 levels away and a stop dark.
    """

    def _template(self, max_resolution=64):
        return engine.ExportTemplate(max_resolution=max_resolution, quality=95, strip_exif=False)

    @staticmethod
    def _solid(path: Path, value: int, size=(160, 100)):
        rgb = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        Image.fromarray(rgb, "RGB").save(path, format="JPEG", quality=95)

    @staticmethod
    def _mean(path: Path) -> float:
        return float(np.asarray(Image.open(path).convert("L"), dtype=np.float32).mean())

    def test_a_raw_is_written_from_the_camera_jpeg_beside_it(self, tmp_path):
        raw = tmp_path / "IMG_0001.CR3"
        raw.write_bytes(b"not really a raw file")
        self._solid(tmp_path / "IMG_0001.JPG", 210)
        out = tmp_path / "out"
        out.mkdir()
        written = export_photo(raw, out, self._template(), resize_only=True)
        assert max(Image.open(written).size) == 64
        assert abs(self._mean(written) - 210) <= 6

    def test_a_developed_frame_is_written_as_developed_and_nothing_else(self, tmp_path):
        """The edit reaches the file; matching does not — proven by writing the
        same edited frame twice, once with a match asked for and once without, and
        getting the same bytes."""
        source = tmp_path / "IMG_0002.JPG"
        self._solid(source, 90)
        out = tmp_path / "out"
        out.mkdir()
        bare = export_photo(
            source, out, self._template(), develop={"exposure": 3.0}, resize_only=True
        )
        assert self._mean(bare) > 180  # three stops of light, in the file

        match = MatchSettings(
            target={"count": 2, "lum": 10.0, "wb_r": 1.6, "wb_b": 0.6}, strength=1.0
        )
        dressed_dir = out / "dressed"
        dressed_dir.mkdir()
        dressed = export_photo(
            source, dressed_dir, self._template(),
            match=match,
            develop={"exposure": 3.0},
            resize_only=True,
        )
        assert match.applied == 0
        assert dressed.read_bytes() == bare.read_bytes()

    def test_the_embedded_jpeg_is_used_when_there_is_none_beside_the_raw(
        self, tmp_path, monkeypatch
    ):
        raw = tmp_path / "IMG_0003.CR3"
        raw.write_bytes(b"x")
        embedded = Image.fromarray(np.full((40, 60, 3), 170, dtype=np.uint8), "RGB")
        monkeypatch.setattr(engine, "_embedded_camera_image", lambda path: embedded.copy())
        monkeypatch.setattr(engine, "_load_image", lambda path, res=0: None)
        out = tmp_path / "out"
        out.mkdir()
        written = export_photo(raw, out, self._template(32), resize_only=True)
        assert max(Image.open(written).size) == 32
        assert abs(self._mean(written) - 170) <= 6

    def test_a_raw_with_no_camera_jpeg_falls_back_to_the_decode(
        self, tmp_path, monkeypatch
    ):
        """A RAW that carries no JPEG at all (some DNGs) still exports: the decode
        is the frame, just rendered rather than as the camera saw it."""
        raw = tmp_path / "IMG_0004.CR3"
        raw.write_bytes(b"x")
        decoded = Image.fromarray(np.full((40, 60, 3), 40, dtype=np.uint8), "RGB")
        monkeypatch.setattr(engine, "_embedded_camera_image", lambda path: None)
        monkeypatch.setattr(engine, "_load_image", lambda path, res=0: decoded.copy())
        out = tmp_path / "out"
        out.mkdir()
        written = export_photo(raw, out, self._template(32), resize_only=True)
        assert max(Image.open(written).size) == 32
        assert abs(self._mean(written) - 40) <= 6

    def test_an_unreadable_file_still_fails_rather_than_writing_nothing(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        try:
            export_photo(tmp_path / "gone.CR3", out, self._template(), resize_only=True)
        except RuntimeError as e:
            assert "gone.CR3" in str(e)
        else:  # pragma: no cover - the export must not report success
            raise AssertionError("expected a RuntimeError")


class TestEveryRawExportStartsFromTheCameraLook:
    """A RAW is not a picture, so its export must never be the plain decoder's
    render of the sensor data.

    That render carries none of the learned in-camera look: measured against the
    camera's own JPEG of the same frame it is 48 levels of 255 away and a stop
    dark, where the developed render is 8.5 away.
    """

    @staticmethod
    def _marker():
        return Image.fromarray(np.full((40, 60, 3), 200, dtype=np.uint8), "RGB")

    def test_a_raw_with_no_edit_is_developed_rather_than_plainly_decoded(
        self, tmp_path, monkeypatch
    ):
        raw = tmp_path / "IMG_0001.CR3"
        raw.write_bytes(b"x")
        seen = []

        def fake_develop(source, template, develop, match):
            seen.append(develop)
            return self._marker()

        monkeypatch.setattr(engine, "_develop_linear", fake_develop)
        monkeypatch.setattr(
            engine, "_load_image",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("the plain decode was used")),
        )
        out = tmp_path / "out"
        out.mkdir()
        written = export_photo(
            raw, out, engine.ExportTemplate(max_resolution=64, strip_exif=False),
        )
        assert written.exists()
        assert seen == [{}]  # the develop pipeline, with no edit on it

    def test_a_jpeg_with_no_edit_keeps_its_own_short_path(self, tmp_path, monkeypatch):
        """A JPEG is already the camera's rendering; making it go through a linear
        decode would be work for nothing."""
        source = tmp_path / "IMG_0002.JPG"
        TestResizeOnlyExport._solid(source, 120)
        monkeypatch.setattr(
            engine, "_develop_linear",
            lambda *a: (_ for _ in ()).throw(AssertionError("developed a JPEG with no edit")),
        )
        out = tmp_path / "out"
        out.mkdir()
        written = export_photo(
            source, out, engine.ExportTemplate(max_resolution=64, strip_exif=False),
        )
        assert written.exists()


def test_the_camera_picture_is_resized_to_the_size_asked_for():
    # 2048/6000 * 6000 truncates to 2047 in floating point.
    img = Image.new("RGB", (6000, 4000))
    assert engine._resize_image(img, 2048).size == (2048, 1365)
