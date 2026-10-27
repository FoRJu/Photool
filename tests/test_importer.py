"""
Tests for the importer module.

Run with:
    cd backend && python -m pytest ../tests/ -v
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import Config, load_config
from importer.copy import (
    _get_date_from_filename,
    build_dest_path,
    build_library_index,
    import_files,
    parse_exiftool_json,
    sha256_file,
)
from importer.sd_detect import detect_sd_cards, find_importable_files, SDCardInfo
from importer.sd_detect import mounted_volumes as real_mounted_volumes  # before the fixture stubs it


@pytest.fixture(autouse=True)
def _no_host_drive_scan(monkeypatch):
    """
    detect_sd_cards() also scans every drive or volume the system has. On a dev
    machine with a real card inserted that leaks into every test, so rely only on
    the explicit sd_mount_paths each test configures.
    """
    monkeypatch.setattr("importer.sd_detect.mounted_volumes", lambda: [])


def test_a_mac_finds_a_card_under_volumes_by_any_name(tmp_path, monkeypatch):
    """A card is named by the camera that formatted it, so on a Mac every volume
    is looked at, not only the configured ones."""
    from importer import sd_detect

    volumes = tmp_path / "Volumes"
    for name, dcim in (("NIKON D750", True), ("Untitled", True), ("Backup", False), (".timemachine", True)):
        (volumes / name / ("DCIM/100NIKON" if dcim else "Photos")).mkdir(parents=True)
    monkeypatch.setattr(sd_detect, "MAC_VOLUMES", volumes)
    monkeypatch.setattr(sd_detect.platform, "system", lambda: "Darwin")
    assert sorted(p.name for p in real_mounted_volumes()) == ["Backup", "NIKON D750", "Untitled"]


def _make_fake_sd(tmp: Path, files: dict[str, bytes]) -> Path:
    """Create a fake SD card DCIM structure."""
    dcim = tmp / "sd" / "DCIM" / "100CANON"
    dcim.mkdir(parents=True)
    for name, content in files.items():
        (dcim / name).write_bytes(content)
    return tmp / "sd"


def _make_config(tmp: Path, sd_path: Path) -> Config:
    return Config(
        library_root=str(tmp / "library"),
        folder_pattern="{year}/{month}/{day}",
        sd_mount_paths=[str(sd_path)],
        import_extensions=[".cr3", ".jpg"],
        duplicate_strategy="skip",
    )


class TestSDDetect:
    def test_detects_canon_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sd = _make_fake_sd(tmp, {"IMG_0001.CR3": b"fake raw data"})
            config = _make_config(tmp, sd)

            cards = detect_sd_cards(config)
            assert len(cards) == 1
            assert cards[0].total_files == 1

    def test_no_card_when_no_dcim(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sd = tmp / "sd"
            sd.mkdir()
            config = _make_config(tmp, sd)

            cards = detect_sd_cards(config)
            assert len(cards) == 0

    def test_finds_importable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sd = _make_fake_sd(tmp, {
                "IMG_0001.CR3": b"raw1",
                "IMG_0001.JPG": b"jpg1",
                "IMG_0002.CR3": b"raw2",
                "readme.txt": b"ignore me",
            })
            config = _make_config(tmp, sd)
            cards = detect_sd_cards(config)

            files = find_importable_files(cards[0], config.import_extensions)
            names = [f.name for f in files]
            assert "IMG_0001.CR3" in names
            assert "IMG_0001.JPG" in names
            assert "IMG_0002.CR3" in names
            assert "readme.txt" not in names

    def test_detects_5d_mark_ii_card_with_cr2_and_jpg(self):
        """
        A 5D Mark II writes 100EOS5D rather than 100CANON, and CR2 rather than
        CR3. Both halves have to be recognised: a full card whose RAW extension
        is not configured still reports the few camera JPEGs sitting beside the
        RAWs, so it looks like a nearly empty card rather than a missing one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dcim = tmp / "sd" / "DCIM" / "100EOS5D"
            dcim.mkdir(parents=True)
            (dcim / "IMG_0114.CR2").write_bytes(b"raw1")
            (dcim / "IMG_0115.CR2").write_bytes(b"raw2")
            (dcim / "IMG_5067.JPG").write_bytes(b"jpg1")

            config = Config(
                library_root=str(tmp / "library"),
                folder_pattern="{year}/{month}/{day}",
                sd_mount_paths=[str(tmp / "sd")],
                import_extensions=[".cr3", ".cr2", ".jpg", ".jpeg"],
                duplicate_strategy="skip",
            )

            cards = detect_sd_cards(config)
            assert len(cards) == 1
            assert [f.name for f in cards[0].image_folders] == ["100EOS5D"]
            assert cards[0].total_files == 3

            names = [
                f.name
                for f in find_importable_files(cards[0], config.import_extensions)
            ]
            assert names == ["IMG_0114.CR2", "IMG_0115.CR2", "IMG_5067.JPG"]

    def test_default_extensions_include_cr2(self):
        """
        Nothing downstream filters CR2 out — the thumbnail pipeline, the catalog
        and the export engine all list it — so the default extension list is the
        only thing standing between a 5D Mark II's RAWs and being imported.
        """
        assert ".cr2" in Config().import_extensions

    def test_cr2_is_dropped_when_the_extension_is_not_configured(self):
        """The failure the 5D Mark II user hit: right card, wrong extension list."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dcim = tmp / "sd" / "DCIM" / "100EOS5D"
            dcim.mkdir(parents=True)
            for i in range(5):
                (dcim / f"IMG_01{i:02d}.CR2").write_bytes(b"raw")

            config = Config(
                library_root=str(tmp / "library"),
                sd_mount_paths=[str(tmp / "sd")],
                import_extensions=[".cr3", ".jpg", ".jpeg"],
            )

            cards = detect_sd_cards(config)
            assert len(cards) == 1
            assert cards[0].total_files == 0
            assert find_importable_files(cards[0], config.import_extensions) == []


class TestOtherDevices:
    """A card is any device's: the DCIM rule is the same for all of them, and
    only the folder names differ."""

    def _card(self, tmp: Path, folder: str, names: list[str]) -> Path:
        dcim = tmp / "sd" / "DCIM" / folder
        dcim.mkdir(parents=True)
        for name in names:
            (dcim / name).write_bytes(b"data")
        return tmp / "sd"

    def _config(self, tmp: Path, sd: Path) -> Config:
        return Config(library_root=str(tmp / "library"), folder_pattern="{year}/{month}/{day}",
                      sd_mount_paths=[str(sd)], import_extensions=[".jpg", ".mp4", ".mov"])

    def test_a_dji_pocket_is_found_by_its_own_folder_name(self, tmp_path):
        """DJI_001 has no number in front, which the old rule required."""
        sd = self._card(tmp_path, "DJI_001", [
            "DJI_20251018143005_0001_D.MP4", "DJI_20251018143005_0001_D.LRF",
            "DJI_20251018143100_0002_D.JPG"])
        (card,) = detect_sd_cards(self._config(tmp_path, sd))
        assert card.device == "DJI"
        assert card.total_files == 2 and card.videos == 1    # the .LRF proxy stays

    def test_a_gopro_card_says_so(self, tmp_path):
        sd = self._card(tmp_path, "100GOPRO", [
            "GOPR0001.JPG", "GX010001.MP4", "GX010001.LRV", "GX010001.THM"])
        (card,) = detect_sd_cards(self._config(tmp_path, sd))
        assert card.device == "GoPro"
        assert card.total_files == 2 and card.videos == 1

    def test_an_older_camera_is_named_by_its_folder(self, tmp_path):
        sd = self._card(tmp_path, "100NIKON", ["DSC_0001.JPG"])
        assert detect_sd_cards(self._config(tmp_path, sd))[0].device == "Nikon"

    def test_a_macs_dot_files_on_a_card_are_not_copied(self, tmp_path):
        sd = self._card(tmp_path, "100CANON", ["IMG_0001.JPG", "._IMG_0001.JPG"])
        config = self._config(tmp_path, sd)
        (card,) = detect_sd_cards(config)
        assert [p.name for p in find_importable_files(card, config.import_extensions)] == ["IMG_0001.JPG"]

    def test_a_video_is_dated_by_its_name_without_asking_exiftool_again(self, tmp_path, monkeypatch):
        sd = self._card(tmp_path, "DJI_001", ["DJI_20251018143005_0001_D.MP4"])
        config = self._config(tmp_path, sd)
        monkeypatch.setattr("importer.copy.batch_capture_dates", lambda paths: ({}, True))

        def no_retry(path):
            raise AssertionError(f"asked exiftool again for {path.name}")

        monkeypatch.setattr("importer.copy._get_date_from_exif", no_retry)
        (card,) = detect_sd_cards(config)
        summary = import_files(find_importable_files(card, config.import_extensions), config)
        assert summary.copied == 1
        assert (tmp_path / "library" / "2025" / "10" / "18" / "DJI_20251018143005_0001_D.MP4").exists()

    @pytest.mark.parametrize("name, expected", [
        ("DJI_20251018143005_0001_D.MP4", datetime(2025, 10, 18, 14, 30, 5)),
        ("20251018_143005.jpg", datetime(2025, 10, 18, 14, 30, 5)),
        ("2024-06-15_001.CR3", datetime(2024, 6, 15)),
        ("6U4A1234.CR3", None),
        ("GX010042.MP4", None),
        ("R0020560.DNG", None),
        ("IMG_20230230.jpg", None),          # no 30th of February
    ])
    def test_dates_in_file_names(self, name, expected):
        assert _get_date_from_filename(Path(name)) == expected

    def test_list_settings_in_the_config_file_are_read(self, tmp_path):
        """They have default factories, so they are not class attributes, and a
        hasattr() check used to drop them without a word."""
        path = tmp_path / "config.yaml"
        path.write_text('import_extensions: [".cr3", ".mp4"]\nsd_mount_paths: ["X:/"]\n')
        config = load_config(str(path))
        assert config.import_extensions == [".cr3", ".mp4"]
        assert config.sd_mount_paths == ["X:/"]
        assert config.photo_extensions == [".cr3", ".mp4"]


class TestBuildDestPath:
    def test_date_based_path(self):
        config = Config(
            library_root="/photos",
            folder_pattern="{year}/{month}/{day}",
        )
        date = datetime(2024, 6, 15, 14, 30, 0)
        dest = build_dest_path(Path("/sd/DCIM/100CANON/IMG_0001.CR3"), date, config)
        assert dest == Path("/photos/2024/06/15/IMG_0001.CR3")


class TestChecksum:
    def test_sha256_consistency(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world" * 1000)
            f.flush()
            path = Path(f.name)

        try:
            h1 = sha256_file(path)
            h2 = sha256_file(path)
            assert h1 == h2
            assert len(h1) == 64  # SHA-256 hex length
        finally:
            path.unlink()


class TestImport:
    def test_basic_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sd = _make_fake_sd(tmp, {
                "IMG_0001.CR3": b"fake raw " * 100,
                "IMG_0001.JPG": b"fake jpg " * 50,
            })
            config = _make_config(tmp, sd)
            cards = detect_sd_cards(config)
            files = find_importable_files(cards[0], config.import_extensions)

            summary = import_files(files, config, verify_checksums=True)

            assert summary.copied == 2
            assert summary.failed == 0
            # Files should exist in library (using mtime-based date folders)
            library = tmp / "library"
            assert library.exists()
            # Find the copied files
            all_files = list(library.rglob("IMG_*"))
            assert len(all_files) == 2

    def test_skip_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sd = _make_fake_sd(tmp, {"IMG_0001.CR3": b"fake raw"})
            config = _make_config(tmp, sd)
            cards = detect_sd_cards(config)
            files = find_importable_files(cards[0], config.import_extensions)

            # Import twice
            import_files(files, config, verify_checksums=False)
            summary2 = import_files(files, config, verify_checksums=False)

            assert summary2.skipped == 1
            assert summary2.copied == 0


class TestAlreadyBackedUp:
    """
    The fast path: a card file that is already in the library must be skipped
    without paying for EXIF. This is what makes a mostly-imported card finish
    in seconds instead of hours.
    """

    def test_library_index_maps_names_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lib = tmp / "library" / "2024" / "2024-06-15"
            lib.mkdir(parents=True)
            (lib / "A.CR3").write_bytes(b"one")
            (lib / "B.CR3").write_bytes(b"two2")
            (lib / "notes.txt").write_bytes(b"skip me")
            (lib / ".hidden.CR3").write_bytes(b"skip me too")

            index = build_library_index(tmp / "library", [".cr3"])
            assert sorted(s for s, _ in index["a.cr3"]) == [3]
            assert index["b.cr3"][0][0] == 4
            # non-image and hidden entries are not indexed
            assert "notes.txt" not in index
            assert ".hidden.cr3" not in index

    def test_existing_photo_skipped_without_exif(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("EXIF must not be read for an already-backed-up file")

        monkeypatch.setattr("importer.copy.batch_capture_dates", _boom)
        monkeypatch.setattr("importer.copy._get_date_from_exif", _boom)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload = b"identical photo bytes " * 40
            # Pre-seed the library with the same photo under a date folder.
            lib = tmp / "library" / "2024" / "2024-06-15"
            lib.mkdir(parents=True)
            (lib / "IMG_0001.CR3").write_bytes(payload)

            sd = _make_fake_sd(tmp, {"IMG_0001.CR3": payload})
            config = _make_config(tmp, sd)
            files = find_importable_files(detect_sd_cards(config)[0], config.import_extensions)

            summary = import_files(files, config, verify_checksums=True)
            assert summary.skipped == 1
            assert summary.copied == 0
            assert summary.results[0].destination == lib / "IMG_0001.CR3"

    def test_same_name_different_size_is_still_imported(self, monkeypatch):
        """
        Canon reuses filenames across sessions, so a name match alone must not
        be treated as a duplicate — otherwise a new photo is silently dropped.
        """
        monkeypatch.setattr("importer.copy.batch_capture_dates", lambda paths: ({}, False))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lib = tmp / "library" / "2024" / "2024-06-15"
            lib.mkdir(parents=True)
            (lib / "IMG_0001.CR3").write_bytes(b"old photo, different bytes")

            sd = _make_fake_sd(tmp, {"IMG_0001.CR3": b"brand new photo!!!"})
            config = _make_config(tmp, sd)
            files = find_importable_files(detect_sd_cards(config)[0], config.import_extensions)

            summary = import_files(files, config, verify_checksums=True)
            assert summary.copied == 1
            assert summary.skipped == 0
            # Same name, same folder pattern (dates fall back to mtime here).
            assert len(list((tmp / "library").rglob("IMG_0001.CR3"))) == 2

    def test_probe_mode_catches_same_size_different_content(self, monkeypatch):
        """
        Name + size is the default test. Opting into a byte probe additionally
        rejects a same-name, same-size file whose contents differ.
        """
        monkeypatch.setattr("importer.copy.batch_capture_dates", lambda paths: ({}, False))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lib = tmp / "library" / "2024" / "2024-06-15"
            lib.mkdir(parents=True)
            (lib / "IMG_0002.CR3").write_bytes(b"A" * 5000)

            sd = _make_fake_sd(tmp, {"IMG_0002.CR3": b"B" * 5000})
            config = _make_config(tmp, sd)
            files = find_importable_files(detect_sd_cards(config)[0], config.import_extensions)

            # Default: name + size match is trusted, so it is treated as backed up.
            default = import_files(files, config, verify_checksums=True)
            assert default.skipped == 1 and default.copied == 0

            # With the probe on, the differing bytes are caught and it imports.
            probed = import_files(
                files, config, verify_checksums=True, duplicate_probe_bytes=1024
            )
            assert probed.copied == 1 and probed.skipped == 0

    def test_reuse_existing_off_forces_full_pass(self, monkeypatch):
        monkeypatch.setattr("importer.copy.batch_capture_dates", lambda paths: ({}, False))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload = b"same bytes"
            lib = tmp / "library" / "2024" / "2024-06-15"
            lib.mkdir(parents=True)
            (lib / "IMG_0003.CR3").write_bytes(payload)

            sd = _make_fake_sd(tmp, {"IMG_0003.CR3": payload})
            config = _make_config(tmp, sd)
            files = find_importable_files(detect_sd_cards(config)[0], config.import_extensions)

            # Fast path recognises it as already backed up…
            fast = import_files(files, config, verify_checksums=False)
            assert fast.skipped == 1 and fast.copied == 0

            # …and turning the fast path off does the full EXIF/copy pass.
            full = import_files(files, config, verify_checksums=False, reuse_existing=False)
            assert full.copied == 1 and full.skipped == 0


class TestBatchExifParsing:
    """Batched exiftool is the difference between ~770ms and ~32ms per file."""

    def test_parses_entries_and_normalises_paths(self):
        payload = """[
          {"SourceFile": "E:/DCIM/100CANON/A.CR3", "DateTimeOriginal": "2024-06-15T14:30:00"},
          {"SourceFile": "E:/DCIM/100CANON/B.CR3", "DateTimeOriginal": "2024-06-16T09:00:00"},
          {"SourceFile": "E:/DCIM/100CANON/C.MP4"}
        ]"""
        dates = parse_exiftool_json(payload)
        assert len(dates) == 2  # no DateTimeOriginal → not included
        key = "e:/dcim/100canon/a.cr3" if os.name == "nt" else "E:/DCIM/100CANON/A.CR3"
        assert dates[key] == datetime(2024, 6, 15, 14, 30, 0)

    def test_tolerates_garbage_and_empty_output(self):
        assert parse_exiftool_json("") == {}
        assert parse_exiftool_json("not json") == {}
        assert parse_exiftool_json("[]") == {}
        assert parse_exiftool_json('[{"SourceFile": "x"}]') == {}

    def test_batches_stay_within_limits(self):
        from importer.copy import _batches, EXIFTOOL_BATCH_SIZE

        paths = [Path(f"E:/DCIM/100CANON/{'x' * 30}_{i:05d}.CR3") for i in range(500)]
        batches = list(_batches(paths))
        assert sum(len(b) for b in batches) == 500
        assert all(len(b) <= EXIFTOOL_BATCH_SIZE for b in batches)
        assert all(sum(len(str(p)) + 1 for p in b) <= 12000 for b in batches)
        # flat, in order
        assert [p for b in batches for p in b] == paths
