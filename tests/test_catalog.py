"""
Tests for the catalog module: SQLite DB, XMP sidecars, and service layer.
"""

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord, Flag, ColorLabel
from catalog.xmp import read_xmp, write_xmp, XMPData, sidecar_path, sync_develop_to_xmp, sync_rating_to_xmp
from catalog.service import CatalogService


# ── Database tests ───────────────────────────────────────────────────

class TestCatalogDB:
    def _make_db(self, tmp: Path) -> CatalogDB:
        return CatalogDB(str(tmp / "test.db"))

    def test_create_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            assert db.count() == 0

    def test_upsert_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            photo = PhotoRecord(
                filepath="2024/06/15/IMG_0001.CR3",
                filename="IMG_0001.CR3",
                folder="2024/06/15",
                extension=".cr3",
                file_size=25000000,
                rating=3,
                flag=1,
            )
            db.upsert_photo(photo)

            result = db.get_photo("2024/06/15/IMG_0001.CR3")
            assert result is not None
            assert result.rating == 3
            assert result.flag == 1
            assert result.filename == "IMG_0001.CR3"

    def test_set_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            photo = PhotoRecord(
                filepath="2024/06/15/IMG_0001.CR3",
                filename="IMG_0001.CR3",
                folder="2024/06/15",
            )
            db.upsert_photo(photo)

            db.set_rating("2024/06/15/IMG_0001.CR3", 4)
            result = db.get_photo("2024/06/15/IMG_0001.CR3")
            assert result.rating == 4

    def test_set_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            photo = PhotoRecord(
                filepath="2024/06/15/IMG_0001.CR3",
                filename="IMG_0001.CR3",
                folder="2024/06/15",
            )
            db.upsert_photo(photo)

            db.set_flag("2024/06/15/IMG_0001.CR3", -1)
            result = db.get_photo("2024/06/15/IMG_0001.CR3")
            assert result.flag == -1

    def test_set_color_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            photo = PhotoRecord(
                filepath="2024/06/15/IMG_0001.CR3",
                filename="IMG_0001.CR3",
                folder="2024/06/15",
            )
            db.upsert_photo(photo)

            db.set_color_label("2024/06/15/IMG_0001.CR3", 3)  # Green
            result = db.get_photo("2024/06/15/IMG_0001.CR3")
            assert result.color_label == 3

    def test_rating_clamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            photo = PhotoRecord(
                filepath="test.cr3", filename="test.cr3", folder="test",
            )
            db.upsert_photo(photo)

            db.set_rating("test.cr3", 99)
            assert db.get_photo("test.cr3").rating == 5

            db.set_rating("test.cr3", -3)
            assert db.get_photo("test.cr3").rating == 0

    def test_folder_query_with_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            folder = "2024/06/15"

            for i in range(5):
                db.upsert_photo(PhotoRecord(
                    filepath=f"{folder}/IMG_{i:04d}.CR3",
                    filename=f"IMG_{i:04d}.CR3",
                    folder=folder,
                    rating=i,
                    flag=1 if i >= 3 else (-1 if i == 0 else 0),
                ))

            # All photos
            all_photos = db.get_photos_in_folder(folder)
            assert len(all_photos) == 5

            # Min rating
            good = db.get_photos_in_folder(folder, min_rating=3)
            assert len(good) == 2  # rating 3 and 4

            # Flag filter
            picks = db.get_photos_in_folder(folder, flag_filter=1)
            assert len(picks) == 2

            rejects = db.get_photos_in_folder(folder, flag_filter=-1)
            assert len(rejects) == 1

    def test_folder_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            folder = "2024/06/15"

            db.upsert_photo(PhotoRecord(
                filepath=f"{folder}/a.cr3", filename="a.cr3",
                folder=folder, flag=1, rating=3,
            ))
            db.upsert_photo(PhotoRecord(
                filepath=f"{folder}/b.cr3", filename="b.cr3",
                folder=folder, flag=-1, rating=0,
            ))
            db.upsert_photo(PhotoRecord(
                filepath=f"{folder}/c.cr3", filename="c.cr3",
                folder=folder, flag=0, rating=0,
            ))

            stats = db.get_folder_stats(folder)
            assert stats["total"] == 3
            assert stats["picks"] == 1
            assert stats["rejects"] == 1
            assert stats["unflagged"] == 1
            assert stats["rated"] == 1

    def test_bulk_upsert(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            photos = [
                PhotoRecord(
                    filepath=f"test/IMG_{i}.CR3",
                    filename=f"IMG_{i}.CR3",
                    folder="test",
                )
                for i in range(100)
            ]
            db.bulk_upsert(photos)
            assert db.count() == 100


class TestDevelopPresets:
    """Saved edit sets: one photo's own eight controls, kept under a name.

    The database stores the payload verbatim, so these tests are about the
    round trip and about a row this build cannot read not taking the whole
    list down with it.
    """

    def test_a_preset_comes_back_as_it_went_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            db.save_develop_preset("Warm interior", {"exposure": 0.4, "shadows": 25})
            assert db.list_develop_presets() == [
                {"name": "Warm interior", "settings": {"exposure": 0.4, "shadows": 25}}
            ]

    def test_a_look_says_which_sections_it_carries(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            db.save_develop_preset("Night", {"exposure": 0.4, "contrast": 20}, ["light", "curve"])
            assert db.list_develop_presets() == [
                {"name": "Night", "settings": {"contrast": 20, "exposure": 0.4}, "sections": ["light", "curve"]}
            ]
            db.save_develop_preset("Night", {"exposure": 0.2})
            assert "sections" not in db.list_develop_presets()[0]

    def test_saving_the_same_name_again_replaces_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            db.save_develop_preset("Interior", {"exposure": 0.4})
            db.save_develop_preset("Interior", {"exposure": -0.4})
            assert [p["settings"] for p in db.list_develop_presets()] == [
                {"exposure": -0.4}
            ]

    def test_the_list_is_ordered_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            for name in ("Zoo", "Aviary", "Museum"):
                db.save_develop_preset(name, {"exposure": 0.1})
            assert [p["name"] for p in db.list_develop_presets()] == [
                "Aviary", "Museum", "Zoo"
            ]

    def test_deleting_reports_whether_there_was_anything_to_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            db.save_develop_preset("Interior", {"exposure": 0.4})
            assert db.delete_develop_preset("Interior") is True
            assert db.delete_develop_preset("Interior") is False
            assert db.list_develop_presets() == []

    def test_a_row_that_cannot_be_read_is_left_out_rather_than_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CatalogDB(str(Path(tmp) / "test.db"))
            db.save_develop_preset("Good", {"exposure": 0.4})
            with closing(sqlite3.connect(Path(tmp) / "test.db")) as conn:
                conn.execute(
                    "INSERT INTO develop_presets (name, settings) VALUES ('Bad', '{oops')"
                )
                conn.commit()
            assert [p["name"] for p in db.list_develop_presets()] == ["Good"]

    def test_an_older_database_gains_the_table(self):
        """The DDL lives in the migration list, which runs on every startup, so
        a catalog written before this feature existed gains the table the next
        time it is opened."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            CatalogDB(str(db_path))
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("DROP TABLE develop_presets")
                conn.commit()

            reopened = CatalogDB(str(db_path))
            reopened.save_develop_preset("Interior", {"shadows": 10})
            assert reopened.list_develop_presets() == [
                {"name": "Interior", "settings": {"shadows": 10}}
            ]


# ── XMP sidecar tests ───────────────────────────────────────────────

class TestXMP:
    def test_sidecar_path(self):
        p = sidecar_path(Path("/photos/2024/06/15/IMG_0001.CR3"))
        assert p == Path("/photos/2024/06/15/IMG_0001.CR3.xmp")

    def test_read_nonexistent_returns_defaults(self):
        data = read_xmp(Path("/nonexistent/IMG_0001.CR3"))
        assert data.rating == 0
        assert data.flag == 0
        assert data.color_label == 0

    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            data = XMPData(rating=4, flag=1, color_label=2)
            write_xmp(img, data)

            xmp_file = sidecar_path(img)
            assert xmp_file.exists()

            result = read_xmp(img)
            assert result.rating == 4
            assert result.flag == 1
            assert result.color_label == 2

    def test_update_preserves_and_modifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            # Initial write
            write_xmp(img, XMPData(rating=3, flag=1, color_label=1))

            # Update just rating
            write_xmp(img, XMPData(rating=5, flag=0, color_label=0))

            result = read_xmp(img)
            assert result.rating == 5
            assert result.flag == 0
            assert result.color_label == 0

    def test_reject_flag_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "test.CR3"
            img.write_bytes(b"fake")

            write_xmp(img, XMPData(flag=-1))
            result = read_xmp(img)
            assert result.flag == -1

    # ── Develop settings ─────────────────────────────────────────────
    # The same edit the export applied, written where another editor can read
    # it. These tests are about the sidecar carrying it, not about the numbers.

    def test_develop_settings_come_back_from_a_new_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            write_xmp(img, XMPData(
                rating=3,
                develop={"crs:Exposure2012": "+0.70", "crs:Shadows2012": "+30"},
            ))

            result = read_xmp(img)
            assert result.develop == {
                "crs:Exposure2012": "+0.70", "crs:Shadows2012": "+30",
            }
            # Lightroom reads by namespace, so the declaration has to be there
            # and the attributes have to carry the prefix.
            text = sidecar_path(img).read_text(encoding="utf-8")
            assert 'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"' in text
            assert 'crs:Exposure2012="+0.70"' in text

    def test_develop_settings_survive_a_cull_update(self):
        """Rating a frame after exporting it must not wipe the edit: a star
        click is the most frequent write in the app."""
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            sync_develop_to_xmp(img, {"crs:Exposure2012": "-0.30"})
            sync_rating_to_xmp(img, rating=5, flag=1)

            result = read_xmp(img)
            assert result.rating == 5
            assert result.flag == 1
            assert result.develop == {"crs:Exposure2012": "-0.30"}

    def test_the_cull_state_survives_an_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            sync_rating_to_xmp(img, rating=2, color_label=3, tags=["Zoo"])
            sync_develop_to_xmp(img, {"crs:Contrast2012": "+20"})

            result = read_xmp(img)
            assert (result.rating, result.color_label, result.tags) == (2, 3, ["Zoo"])
            assert result.develop == {"crs:Contrast2012": "+20"}

    def test_clearing_the_edit_takes_the_attributes_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            sync_develop_to_xmp(img, {"crs:Exposure2012": "+0.70"})
            sync_develop_to_xmp(img, {})

            assert read_xmp(img).develop == {}
            assert "crs:Exposure2012" not in sidecar_path(img).read_text(encoding="utf-8")

    def test_an_unusual_value_does_not_break_the_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            sync_develop_to_xmp(img, {"crs:Exposure2012": '1" & <two>'})
            assert read_xmp(img).develop == {"crs:Exposure2012": '1" & <two>'}


# ── CatalogService tests ────────────────────────────────────────────

class TestCatalogService:
    def _setup(self, tmp: Path):
        lib = tmp / "library"
        folder = lib / "2024" / "06" / "15"
        folder.mkdir(parents=True)

        # Create fake image files
        for i in range(3):
            (folder / f"IMG_{i:04d}.CR3").write_bytes(b"fake raw " * 100)
            (folder / f"IMG_{i:04d}.JPG").write_bytes(b"fake jpg " * 50)

        db = CatalogDB(str(tmp / "catalog.db"))
        service = CatalogService(db, str(lib))
        return service, lib

    def test_index_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))

            result = service.index_folder("2024/06/15", [".cr3", ".jpg"])
            assert result["indexed"] == 6  # 3 CR3 + 3 JPG

            photos = service.get_photos_in_folder("2024/06/15")
            assert len(photos) == 6

    def test_a_macs_dot_files_are_not_photographs(self):
        """A Mac leaves "._IMG_0001.JPG" beside each file it touches on a
        non-Mac drive: an image extension, and nothing a photo reader can open."""
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            folder = lib / "2024" / "06" / "15"
            (folder / "._IMG_0000.JPG").write_bytes(b"Mac OS X sidecar")
            (folder / ".DS_Store").write_bytes(b"")

            assert service.index_folder("2024/06/15", [".cr3", ".jpg"])["indexed"] == 6
            names = {p.filename for p in service.get_photos_in_folder("2024/06/15")}
            assert "._IMG_0000.JPG" not in names

    def test_rows_that_were_never_photographs_are_forgotten(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3", ".jpg"])
            db = service.db
            for name in ("._IMG_0000.JPG", "CLIP_0001.MP4"):
                db.upsert_photo(PhotoRecord(filepath=f"2024/06/15/{name}", filename=name,
                                            folder="2024/06/15", extension=Path(name).suffix.lower()))
            assert db.forget_non_photos({".mp4", ".mov"}) == 2
            assert len(service.get_photos_in_folder("2024/06/15")) == 6
            assert db.forget_non_photos({".mp4", ".mov"}) == 0

    def test_rate_and_sync_xmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])

            filepath = "2024/06/15/IMG_0000.CR3"

            # Set rating
            service.set_rating(filepath, 4)
            photo = service.get_photo(filepath)
            assert photo.rating == 4

            # Check XMP was written
            xmp = read_xmp(lib / filepath)
            assert xmp.rating == 4

    def test_flag_and_sync_xmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])

            filepath = "2024/06/15/IMG_0001.CR3"

            service.set_flag(filepath, 1)  # pick
            assert service.get_photo(filepath).flag == 1
            assert read_xmp(lib / filepath).flag == 1

            service.set_flag(filepath, -1)  # reject
            assert service.get_photo(filepath).flag == -1
            assert read_xmp(lib / filepath).flag == -1

    def test_toggle_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])

            filepath = "2024/06/15/IMG_0000.CR3"

            # Toggle pick on
            new = service.toggle_flag(filepath, 1)
            assert new == 1

            # Toggle pick off (same key again)
            new = service.toggle_flag(filepath, 1)
            assert new == 0

    def test_an_edit_reaches_the_sidecar_as_well_as_the_catalog(self):
        """The catalog row is what this app reads back; the sidecar is what
        Lightroom reads and what survives losing the database, so an edit made
        here is neither invisible to another editor nor trapped in the catalog."""
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            filepath = "2024/06/15/IMG_0000.CR3"

            stored = service.set_photo_edit(filepath, {"exposure": 0.7, "shadows": 30})

            assert stored == {"exposure": 0.7, "shadows": 30}
            assert service.photo_edit(filepath) == stored
            assert read_xmp(lib / filepath).develop == {
                "crs:Exposure2012": "+0.70", "crs:Shadows2012": "+30",
            }

    def test_the_edit_and_the_cull_state_end_up_in_one_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            filepath = "2024/06/15/IMG_0001.CR3"

            service.set_rating(filepath, 4)
            service.set_photo_edit(filepath, {"highlights": -20})
            service.set_flag(filepath, 1)

            result = read_xmp(lib / filepath)
            assert (result.rating, result.flag) == (4, 1)
            assert result.develop == {"crs:Highlights2012": "-20"}

    def test_clearing_an_edit_takes_it_off_the_sidecar_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            filepath = "2024/06/15/IMG_0002.CR3"

            service.set_photo_edit(filepath, {"exposure": 1.0})
            assert service.clear_photo_edit(filepath) is True

            assert service.photo_edit(filepath) == {}
            assert read_xmp(lib / filepath).develop == {}

    def test_editing_a_photo_that_is_not_there_is_not_an_error(self):
        """A catalog entry whose file has gone must not fail a save: the edit is
        still the edit the catalog will report."""
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])
            assert service._write_edit_to_sidecar("2024/06/15/NOPE.CR3", {"exposure": 1}) is False

    def test_index_preserves_xmp_ratings(self):
        """Re-indexing should read existing XMP sidecars."""
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))

            # First index
            service.index_folder("2024/06/15", [".cr3"])
            service.set_rating("2024/06/15/IMG_0000.CR3", 5)
            service.set_flag("2024/06/15/IMG_0000.CR3", 1)

            # Rebuild from scratch (simulates DB loss)
            db2 = CatalogDB(str(Path(tmp) / "catalog2.db"))
            service2 = CatalogService(db2, str(lib))
            service2.index_folder("2024/06/15", [".cr3"])

            photo = service2.get_photo("2024/06/15/IMG_0000.CR3")
            assert photo.rating == 5  # preserved from XMP
            assert photo.flag == 1    # preserved from XMP

    def test_folder_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024/06/15", [".cr3"])

            service.set_flag("2024/06/15/IMG_0000.CR3", 1)
            service.set_flag("2024/06/15/IMG_0001.CR3", -1)
            service.set_rating("2024/06/15/IMG_0002.CR3", 3)

            stats = service.get_folder_stats("2024/06/15")
            assert stats["total"] == 3
            assert stats["picks"] == 1
            assert stats["rejects"] == 1
            assert stats["rated"] == 1
