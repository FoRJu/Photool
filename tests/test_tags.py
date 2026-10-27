"""
Tests for event tags: the tag tables, tag filtering (including pair-aware
matching), dc:subject sidecar round-trips, and bulk basket/tag operations.
"""

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord
from catalog.xmp import read_xmp, write_xmp, XMPData
from catalog.service import CatalogService


# ── Database: tag tables and filtering ───────────────────────────────

class TestTags:
    def _make_db(self, tmp: Path) -> CatalogDB:
        return CatalogDB(str(tmp / "test.db"))

    def _seed(self, db: CatalogDB):
        for i in range(3):
            db.upsert_photo(PhotoRecord(
                filepath=f"2024-06-15/IMG_{i:04d}.CR3",
                filename=f"IMG_{i:04d}.CR3",
                folder="2024-06-15",
                extension=".cr3",
            ))

    def test_create_and_list_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            tag_id = db.create_tag("School - Grade 5 Concert")
            assert tag_id is not None
            # The same name returns the existing tag instead of duplicating it
            assert db.create_tag("School - Grade 5 Concert") == tag_id

            tags = db.list_tags()
            assert len(tags) == 1
            assert tags[0]["name"] == "School - Grade 5 Concert"
            assert tags[0]["photo_count"] == 0

    def test_blank_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            assert db.create_tag("") is None
            assert db.create_tag("   ") is None

    def test_set_photo_tags_and_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            self._seed(db)

            db.set_photo_tags("2024-06-15/IMG_0000.CR3", ["School", "Grade 5"])
            db.set_photo_tags("2024-06-15/IMG_0001.CR3", ["Family"])

            assert db.get_photo_tags("2024-06-15/IMG_0000.CR3") == ["Grade 5", "School"]

            tagged = db.get_photos_in_folder("2024-06-15", tag_filter="School")
            assert [p.filepath for p in tagged] == ["2024-06-15/IMG_0000.CR3"]

            # Replacing with an empty list clears the tags
            db.set_photo_tags("2024-06-15/IMG_0000.CR3", [])
            assert db.get_photo_tags("2024-06-15/IMG_0000.CR3") == []

    def test_tags_come_back_with_a_folder_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            self._seed(db)
            db.set_photo_tags("2024-06-15/IMG_0000.CR3", ["School"])

            tag_map = db.get_tags_for_filepaths(
                ["2024-06-15/IMG_0000.CR3", "2024-06-15/IMG_0001.CR3"]
            )
            assert tag_map == {"2024-06-15/IMG_0000.CR3": ["School"]}

    def test_tag_filter_combines_with_rating_and_color(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            self._seed(db)
            db.set_rating("2024-06-15/IMG_0000.CR3", 4)
            db.set_rating("2024-06-15/IMG_0001.CR3", 1)
            db.set_color_label("2024-06-15/IMG_0000.CR3", 2)
            db.add_tag_to_photos(
                ["2024-06-15/IMG_0000.CR3", "2024-06-15/IMG_0001.CR3"], "School"
            )

            matches = db.get_photos_in_folder(
                "2024-06-15", min_rating=3, color_filter=2, tag_filter="School"
            )
            assert [p.filepath for p in matches] == ["2024-06-15/IMG_0000.CR3"]

    def test_tag_filter_matches_the_other_half_of_a_pair(self):
        """Tagging the JPG must not hide the pair when RAWs are shown too."""
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            for ext in (".CR3", ".JPG"):
                db.upsert_photo(PhotoRecord(
                    filepath=f"2024-06-15/IMG_0000{ext}",
                    filename=f"IMG_0000{ext}",
                    folder="2024-06-15",
                    extension=ext.lower(),
                ))
            db.set_photo_tags("2024-06-15/IMG_0000.JPG", ["School"])

            all_files = db.get_photos_in_folder(
                "2024-06-15", tag_filter="School", pair_mode="all"
            )
            assert len(all_files) == 2

            paired = db.get_photos_in_folder(
                "2024-06-15", tag_filter="School", pair_mode="paired"
            )
            assert [p.filepath for p in paired] == ["2024-06-15/IMG_0000.JPG"]

    def test_add_basket_from_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            self._seed(db)
            db.set_rating("2024-06-15/IMG_0000.CR3", 4)
            db.set_rating("2024-06-15/IMG_0001.CR3", 5)
            db.set_rating("2024-06-15/IMG_0002.CR3", 1)

            matches = db.get_photos_in_folder("2024-06-15", min_rating=4)
            assert db.add_basket_from_photos(matches) == 2
            assert db.basket_count() == 2

            # Re-running the same selection adds nothing new
            assert db.add_basket_from_photos(matches) == 0
            assert db.basket_count() == 2

    def test_delete_tag_detaches_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            self._seed(db)
            db.set_photo_tags("2024-06-15/IMG_0000.CR3", ["School"])
            assert db.delete_tag("School") is True
            assert db.get_photo_tags("2024-06-15/IMG_0000.CR3") == []
            assert db.list_tags() == []

    def test_rename_tag_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(Path(tmp))
            self._seed(db)
            db.set_photo_tags("2024-06-15/IMG_0000.CR3", ["School"])
            db.set_photo_tags("2024-06-15/IMG_0001.CR3", ["School - copy"])
            assert db.rename_tag("School - copy", "School") is True

            tags = db.list_tags()
            assert [t["name"] for t in tags] == ["School"]
            assert tags[0]["photo_count"] == 2

    def test_old_catalog_upgrades_in_place(self):
        """A catalog created before tags existed gains the tables on reopen."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test.db")
            db = CatalogDB(db_path)
            self._seed(db)
            db.set_photo_tags("2024-06-15/IMG_0000.CR3", ["School"])

            reopened = CatalogDB(db_path)
            assert reopened.get_photo_tags("2024-06-15/IMG_0000.CR3") == ["School"]


# ── XMP: dc:subject keywords ─────────────────────────────────────────

class TestXMPKeywords:
    def test_subject_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            write_xmp(img, XMPData(rating=3, tags=["School", "Grade 5 Concert"]))
            result = read_xmp(img)
            assert result.tags == ["School", "Grade 5 Concert"]
            assert result.rating == 3

    def test_special_characters_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            write_xmp(img, XMPData(tags=["Tom & Jerry's <show>"]))
            assert read_xmp(img).tags == ["Tom & Jerry's <show>"]

    def test_keywords_survive_a_rating_update(self):
        """Updating through the parsed-data path must not drop keywords."""
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            write_xmp(img, XMPData(rating=2, tags=["School"]))
            data = read_xmp(img)
            data.rating = 5
            write_xmp(img, data)

            result = read_xmp(img)
            assert result.rating == 5
            assert result.tags == ["School"]

    def test_clearing_keywords_removes_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")

            write_xmp(img, XMPData(tags=["School"]))
            write_xmp(img, XMPData(rating=1, tags=[]))
            assert read_xmp(img).tags == []

    def test_parses_a_lightroom_style_sidecar(self):
        """Keywords written by another tool are readable."""
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "IMG_0001.CR3"
            img.write_bytes(b"fake")
            sidecar = img.with_suffix(img.suffix + ".xmp")
            sidecar.write_text(
                """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmp:Rating="4">
   <dc:subject>
    <rdf:Bag>
     <rdf:li>School Trip</rdf:li>
     <rdf:li>Year 6</rdf:li>
    </rdf:Bag>
   </dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>""",
                encoding="utf-8",
            )

            result = read_xmp(img)
            assert result.tags == ["School Trip", "Year 6"]
            assert result.rating == 4


# ── Service: dual-write and bulk operations ──────────────────────────

class TestServiceTags:
    def _setup(self, tmp: Path):
        lib = tmp / "library"
        folder = lib / "2024-06-15"
        folder.mkdir(parents=True)
        for i in range(3):
            (folder / f"IMG_{i:04d}.CR3").write_bytes(b"fake raw " * 100)
            (folder / f"IMG_{i:04d}.JPG").write_bytes(b"fake jpg " * 50)
        db = CatalogDB(str(tmp / "catalog.db"))
        return CatalogService(db, str(lib)), lib

    def test_set_photo_tags_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])

            filepath = "2024-06-15/IMG_0000.CR3"
            service.set_photo_tags(filepath, ["School", "Grade 5"])
            assert service.get_photo_tags(filepath) == ["Grade 5", "School"]
            # The sidecar is written from the DB, so it carries the same order
            assert read_xmp(lib / filepath).tags == ["Grade 5", "School"]

    def test_rating_write_keeps_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])

            filepath = "2024-06-15/IMG_0000.CR3"
            service.set_photo_tags(filepath, ["School"])
            service.set_rating(filepath, 4)

            assert read_xmp(lib / filepath).tags == ["School"]
            assert service.get_photo_tags(filepath) == ["School"]

    def test_tag_matching_photos_bulk(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            service.set_rating("2024-06-15/IMG_0000.CR3", 4)
            service.set_rating("2024-06-15/IMG_0001.CR3", 5)

            result = service.tag_matching_photos("2024-06-15", "School", min_rating=4)
            assert result["matched"] == 2
            assert result["tagged"] == 2

            matches = service.get_photos_in_folder("2024-06-15", tag_filter="School")
            assert len(matches) == 2
            assert read_xmp(lib / "2024-06-15/IMG_0001.CR3").tags == ["School"]

            # Untagging the same selection reverses it
            result = service.untag_matching_photos("2024-06-15", "School", min_rating=4)
            assert result["untagged"] == 2
            assert service.get_photos_in_folder("2024-06-15", tag_filter="School") == []

    def test_add_matching_to_basket(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3", ".jpg"])
            service.set_rating("2024-06-15/IMG_0000.CR3", 4)
            service.set_flag("2024-06-15/IMG_0001.CR3", -1)

            result = service.add_matching_to_basket(
                "2024-06-15", min_rating=4, pair_mode="all"
            )
            assert result["matched"] == 1
            assert result["added"] == 1
            assert result["basket"] == 1

            # Everything that is not rejected — the already-basked photo is skipped
            result = service.add_matching_to_basket(
                "2024-06-15", flag_filter=0, pair_mode="all"
            )
            assert result["matched"] == 5
            assert result["added"] == 4
            assert result["basket"] == 5

    def test_index_imports_keywords_from_sidecar(self):
        """Tags written by another tool come back when the folder is indexed."""
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            write_xmp(
                lib / "2024-06-15/IMG_0000.CR3",
                XMPData(rating=2, tags=["School"]),
            )

            service.index_folder("2024-06-15", [".cr3"])

            assert service.get_photo_tags("2024-06-15/IMG_0000.CR3") == ["School"]
            assert service.get_photo("2024-06-15/IMG_0000.CR3").rating == 2

    def test_reindex_keeps_db_only_tags(self):
        """A photo with no sidecar keywords keeps the tags stored in the DB."""
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            service.db.set_photo_tags("2024-06-15/IMG_0000.CR3", ["Family"])

            service.index_folder("2024-06-15", [".cr3"])

            assert service.get_photo_tags("2024-06-15/IMG_0000.CR3") == ["Family"]
