"""
Tests for per-photograph, non-destructive edits.

The edit belongs to the picture: it is held in the catalog so a grid can be
drawn without opening two thousand files, and in the XMP sidecar so it travels
with the photographs, survives losing the catalog, and can be read by another
editor. These tests are about the two copies agreeing — and about the sidecar
being shared without being wrecked.
"""

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from adjust.develop import CRS_KEYS, from_crs, to_crs
from catalog.database import CatalogDB
from catalog.service import CatalogService
from catalog.xmp import read_xmp, sidecar_path, sync_develop_to_xmp


def _setup(tmp: Path, names=("IMG_0001", "IMG_0002", "IMG_0003")):
    lib = tmp / "library"
    folder = lib / "2024" / "06" / "15"
    folder.mkdir(parents=True)
    for name in names:
        (folder / f"{name}.CR3").write_bytes(b"fake raw " * 100)
    db = CatalogDB(str(tmp / "catalog.db"))
    return CatalogService(db, str(lib)), lib


class TestSidecarRoundTrip:
    """The sidecar is the durable copy, so reading back what was written has to
    give the same edit. A value that drifts by a unit on every save is a slider
    that walks itself."""

    def test_every_control_survives_a_round_trip(self):
        settings = {
            "exposure": 0.7,
            "temperature": -30,
            "tint": 40,
            "contrast": 15,
            "highlights": -25,
            "shadows": 30,
            "whites": 12,
            "blacks": -10,
        }
        assert from_crs(to_crs(settings)) == settings

    @pytest.mark.parametrize("tint", [t for t in range(-100, 101) if t])
    def test_tint_is_exact_at_every_step(self, tint):
        """Lightroom's tint is -150..150 and this app's is -100..100, so the
        converted value is a whole number only sometimes: 7 becomes 10.5, which
        rounds to 10, which reads back as 6. That is why the exact value is
        written under its own name as well.

        Zero is left out: a control at zero is not an edit, so it is not written
        and not read back — that is the same state as never having been touched.
        """
        assert from_crs(to_crs({"tint": tint})) == {"tint": tint}

    def test_a_sidecar_from_before_the_exact_tag_still_reads(self):
        """A file written by the earlier build has only the converted value."""
        assert from_crs({"crs:Tint2012": "+60"}) == {"tint": 40}

    def test_another_editors_white_balance_is_not_taken_for_this_apps(self):
        """Lightroom's crs:Temperature is kelvin. Reading it as this app's
        relative warmth would silently re-edit every frame it was opened on."""
        assert from_crs({"crs:Temperature": "5200"}) == {}
        assert from_crs({"crs:Exposure2012": "+0.70"}) == {"exposure": 0.7}

    def test_nothing_to_read_is_no_edit(self):
        assert from_crs(None) == {}
        assert from_crs({}) == {}
        assert from_crs({"crs:Exposure2012": "not a number"}) == {}


class TestSidecarIsShared:
    """A slider movement writes the sidecar immediately now, so it has to be
    safe to write: a file another editor also keeps settings in must not lose
    half its contents because someone nudged an exposure here."""

    def test_another_editors_settings_survive_an_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            # What Lightroom would have in the same block.
            sync_develop_to_xmp(img, {
                "crs:Temperature": "5200",
                "crs:CropTop": "0.05",
                "crs:ProcessVersion": "15.4",
            })

            from catalog.xmp import sync_develop_to_xmp as sync
            sync(img, to_crs({"exposure": 1.0}), owned=CRS_KEYS)

            develop = read_xmp(img).develop
            assert develop["crs:Exposure2012"] == "+1.00"
            assert develop["crs:Temperature"] == "5200"
            assert develop["crs:CropTop"] == "0.05"
            assert develop["crs:ProcessVersion"] == "15.4"

    def test_clearing_an_edit_leaves_another_editors_settings_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            sync_develop_to_xmp(img, {
                "crs:Exposure2012": "+0.70",
                "crs:CropLeft": "0.1",
            })
            # Clearing writes an empty edit set, owning only this app's names.
            sync_develop_to_xmp(img, {}, owned=CRS_KEYS)

            develop = read_xmp(img).develop
            assert "crs:Exposure2012" not in develop
            assert develop["crs:CropLeft"] == "0.1"


class TestOnePhotographsEdit:
    def test_a_save_lands_in_both_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            filepath = "2024/06/15/IMG_0001.CR3"

            service.set_photo_edit(filepath, {"exposure": 0.5, "shadows": 30})

            assert service.photo_edit(filepath) == {"exposure": 0.5, "shadows": 30}
            assert read_xmp(lib / filepath).develop == {
                "crs:Exposure2012": "+0.50", "crs:Shadows2012": "+30",
            }

    def test_a_reset_clears_both_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            filepath = "2024/06/15/IMG_0001.CR3"

            service.set_photo_edit(filepath, {"exposure": 0.5})
            service.set_photo_edit(filepath, {})

            assert service.photo_edit(filepath) == {}
            assert read_xmp(lib / filepath).develop == {}

    def test_a_folder_reports_which_frames_are_edited(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            service.set_photo_edit("2024/06/15/IMG_0001.CR3", {"exposure": 1.0})

            edits = service.photo_edits("2024/06/15")
            assert list(edits) == ["2024/06/15/IMG_0001.CR3"]

    def test_one_edit_can_be_put_on_a_whole_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])

            result = service.apply_edit_to_many(
                ["2024/06/15/IMG_0001.CR3", "2024/06/15/IMG_0003.CR3"],
                {"contrast": 20},
            )

            assert sorted(result["copied"]) == [
                "2024/06/15/IMG_0001.CR3", "2024/06/15/IMG_0003.CR3",
            ]
            assert service.photo_edit("2024/06/15/IMG_0002.CR3") == {}

    def test_a_file_that_is_not_in_the_catalog_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            result = service.apply_edit_to_many(
                ["2024/06/15/IMG_0001.CR3", "2024/06/15/NOPE.CR3"], {"contrast": 20}
            )
            assert result["copied"] == ["2024/06/15/IMG_0001.CR3"]


class TestChoosingWhatAPasteCarries:
    """The Copy checklist: a paste carries the sections ticked and leaves the
    rest of each photo's edit alone; a tripod session ticks Level."""

    SOURCE = "2024/06/15/IMG_0001.CR3"
    TARGET = "2024/06/15/IMG_0002.CR3"
    CROP = {"crop_x": 0.1, "crop_y": 0.1, "crop_w": 0.6, "crop_h": 0.6}

    def _service(self, tmp):
        service, _ = _setup(Path(tmp))
        service.index_folder("2024/06/15", [".cr3"])
        service.set_photo_edit(self.TARGET, {"contrast": 30, **self.CROP})
        return service

    def test_with_no_choice_the_tone_goes_and_the_framing_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.apply_edit_to_many([self.TARGET], {"exposure": 0.5, "angle": 2.0})
            assert service.photo_edit(self.TARGET) == {"exposure": 0.5, **self.CROP}

    def test_only_the_level_leaves_the_tone_and_crop_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.apply_edit_to_many([self.TARGET], {"exposure": 0.5, "angle": 2.0}, keys=["angle"])
            assert service.photo_edit(self.TARGET) == {"contrast": 30, "angle": 2.0, **self.CROP}

    def test_a_ticked_section_that_the_source_does_not_set_is_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.apply_edit_to_many([self.TARGET], {"exposure": 0.5}, keys=["exposure", "contrast"])
            assert service.photo_edit(self.TARGET) == {"exposure": 0.5, **self.CROP}

    def test_keys_that_are_not_an_edits_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.apply_edit_to_many([self.TARGET], {"exposure": 0.5}, keys=["exposure", "filepath", "rating"])
            assert service.photo_edit(self.TARGET) == {"exposure": 0.5, "contrast": 30, **self.CROP}

    def test_a_crop_is_not_put_on_a_frame_turned_the_other_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            other = "2024/06/15/IMG_0003.CR3"
            service._turned = lambda paths: {self.SOURCE: False, self.TARGET: True, other: False}
            crop = {"crop_x": 0.2, "crop_y": 0.2, "crop_w": 0.5, "crop_h": 0.5}
            result = service.apply_edit_to_many([self.TARGET, other], {"angle": 1.0, **crop},
                                                keys=["angle", *crop], source=self.SOURCE)
            assert result["kept_framing"] == [self.TARGET]
            assert service.photo_edit(self.TARGET) == {"contrast": 30, "angle": 1.0, **self.CROP}
            assert service.photo_edit(other) == {"angle": 1.0, **crop}

    def test_a_paste_can_be_taken_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            untouched = "2024/06/15/IMG_0003.CR3"
            result = service.apply_edit_to_many([self.TARGET, untouched], {"exposure": 0.5})
            assert result["edits"][untouched] == {"exposure": 0.5}
            service.restore_edits(result["previous"])
            assert service.photo_edit(self.TARGET) == {"contrast": 30, **self.CROP}
            assert service.photo_edit(untouched) == {}
            assert untouched not in service.photo_edits("2024/06/15")


class TestAdoptingFromTheSidecar:
    """The sidecar is the durable copy, so a catalog that has lost its edits —
    or never had them — gets them back from the files."""

    def test_an_edit_in_the_sidecar_comes_back_on_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = _setup(Path(tmp))
            img = lib / "2024" / "06" / "15" / "IMG_0002.CR3"
            sync_develop_to_xmp(img, to_crs({"exposure": -0.3, "tint": 7}))

            service.index_folder("2024/06/15", [".cr3"])

            # tint 7 is the case the converted tag cannot carry exactly.
            assert service.photo_edit("2024/06/15/IMG_0002.CR3") == {
                "exposure": -0.3, "tint": 7,
            }

    def test_an_edit_survives_a_rebuild_from_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            service.set_photo_edit(
                "2024/06/15/IMG_0001.CR3", {"exposure": 1.0, "highlights": -20}
            )

            # Lose the catalog entirely, as a disk failure or a move would.
            fresh = CatalogService(CatalogDB(str(Path(tmp) / "rebuilt.db")), str(lib))
            fresh.index_folder("2024/06/15", [".cr3"])

            assert fresh.photo_edit("2024/06/15/IMG_0001.CR3") == {
                "exposure": 1.0, "highlights": -20,
            }

    def test_a_sidecar_without_an_edit_does_not_delete_one(self):
        """A sidecar that failed to write, or was never created, is not the user
        asking for the edit to go — only clearing it is."""
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = _setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            filepath = "2024/06/15/IMG_0001.CR3"
            service.set_photo_edit(filepath, {"exposure": 1.0})

            sidecar_path(lib / filepath).unlink()
            (lib / "2024" / "06" / "15" / "IMG_0004.CR3").write_bytes(b"new")
            service.index_folder("2024/06/15", [".cr3"])

            assert service.photo_edit(filepath) == {"exposure": 1.0}
