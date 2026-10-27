"""
Tests for the review flow: the undecided-only filter, the decision counts behind
the progress bar, and remembering where the user stopped in a folder.

The definition that matters is what counts as *decided*. A star, a flag or a
colour label is a decision about the photo. An event tag is not — it says which
event a photo belongs to, and tagging a whole event in bulk must not make it look
reviewed.
"""

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analysis.image_stats import STATS_VERSION
from catalog.database import CatalogDB, PhotoRecord
from catalog.service import CatalogService

FOLDER = "2024-06-15"


def _photo(name: str, folder: str = FOLDER, **overrides) -> PhotoRecord:
    fields = {
        "filepath": f"{folder}/{name}",
        "filename": name,
        "folder": folder,
        "extension": Path(name).suffix.lower(),
        "base_name": Path(name).stem,
    }
    fields.update(overrides)
    return PhotoRecord(**fields)


class TestUndecidedFilter:
    def _db(self, tmp: Path) -> CatalogDB:
        db = CatalogDB(str(tmp / "test.db"))
        for i in range(5):
            db.upsert_photo(_photo(f"IMG_{i:04d}.CR3"))
        return db

    def _paths(self, db: CatalogDB, **filters) -> list[str]:
        return [p.filepath for p in db.get_photos_in_folder(FOLDER, **filters)]

    def test_any_cull_state_counts_as_decided(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(Path(tmp))
            db.set_rating(f"{FOLDER}/IMG_0000.CR3", 3)
            db.set_flag(f"{FOLDER}/IMG_0001.CR3", 1)
            db.set_flag(f"{FOLDER}/IMG_0002.CR3", -1)
            db.set_color_label(f"{FOLDER}/IMG_0003.CR3", 2)

            # Only the untouched frame is still asking to be reviewed. A reject
            # is a decision, and so is a colour label with no rating on it.
            assert self._paths(db, unreviewed_only=True) == [f"{FOLDER}/IMG_0004.CR3"]

    def test_an_event_tag_is_not_a_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(Path(tmp))
            db.set_photo_tags(f"{FOLDER}/IMG_0000.CR3", ["Grade 5 Concert"])

            # Labelling a whole event must not hide it from the review pass.
            assert f"{FOLDER}/IMG_0000.CR3" in self._paths(db, unreviewed_only=True)

    def test_a_decided_twin_hides_the_whole_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db = CatalogDB(str(tmp / "test.db"))
            db.upsert_photo(_photo("IMG_0001.CR3"))
            db.upsert_photo(_photo("IMG_0001.JPG"))
            db.upsert_photo(_photo("IMG_0002.CR3"))

            db.set_flag(f"{FOLDER}/IMG_0001.CR3", -1)

            # Viewing every file, the JPG of a rejected pair must not come back
            # asking for a second decision.
            remaining = self._paths(db, unreviewed_only=True, pair_mode="all")
            assert remaining == [f"{FOLDER}/IMG_0002.CR3"]

    def test_rating_and_undecided_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(Path(tmp))
            db.set_rating(f"{FOLDER}/IMG_0000.CR3", 4)
            # Anything rated is decided, so the two filters cannot both match.
            assert self._paths(db, unreviewed_only=True, min_rating=1) == []

    def test_untouched_folder_is_all_undecided(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(Path(tmp))
            assert len(self._paths(db, unreviewed_only=True)) == 5
            assert len(self._paths(db, unreviewed_only=False)) == 5


class TestDecisionCounts:
    def test_folder_stats_report_progress(self):
        with tempfile.TemporaryDirectory() as raw:
            db = CatalogDB(str(Path(raw) / "test.db"))
            for i in range(4):
                db.upsert_photo(_photo(f"IMG_{i:04d}.CR3"))
            db.set_rating(f"{FOLDER}/IMG_0000.CR3", 2)
            db.set_flag(f"{FOLDER}/IMG_0001.CR3", -1)
            db.set_photo_tags(f"{FOLDER}/IMG_0002.CR3", ["Concert"])

            stats = db.get_folder_stats(FOLDER)
            assert stats["total"] == 4
            assert stats["decided"] == 2
            assert stats["undecided"] == 2

    def test_progress_counts_images_not_files(self):
        with tempfile.TemporaryDirectory() as raw:
            db = CatalogDB(str(Path(raw) / "test.db"))
            db.upsert_photo(_photo("IMG_0000.CR3"))
            db.upsert_photo(_photo("IMG_0000.JPG"))
            db.upsert_photo(_photo("IMG_0001.CR3"))
            db.upsert_photo(_photo("IMG_0001.JPG"))

            # Deciding one half of a pair decides the pair, and the grid shows
            # one frame per pair — so progress is counted per image.
            db.set_flag(f"{FOLDER}/IMG_0000.CR3", -1)
            stats = db.get_folder_stats(FOLDER)
            assert stats["total"] == 4
            assert stats["unique_images"] == 2
            assert stats["decided"] == 1
            assert stats["undecided"] == 1

    def test_folder_list_reports_progress(self):
        with tempfile.TemporaryDirectory() as raw:
            db = CatalogDB(str(Path(raw) / "test.db"))
            db.upsert_photo(_photo("IMG_0000.CR3"))
            db.upsert_photo(_photo("IMG_0001.CR3"))
            db.set_flag(f"{FOLDER}/IMG_0000.CR3", 1)

            folders = {f["folder"]: f for f in db.get_all_folders()}
            assert folders[FOLDER]["decided"] == 1


class TestFolderState:
    def _db(self, raw: str) -> CatalogDB:
        return CatalogDB(str(Path(raw) / "test.db"))

    def test_a_folder_without_a_saved_position_reports_none(self):
        with tempfile.TemporaryDirectory() as raw:
            db = self._db(raw)
            state = db.get_folder_state(FOLDER)
            assert state["last_filepath"] is None
            assert state["updated_at"] == 0

    def test_position_round_trips_and_overwrites(self):
        with tempfile.TemporaryDirectory() as raw:
            db = self._db(raw)
            db.set_folder_state(FOLDER, f"{FOLDER}/IMG_0100.CR3")
            assert db.get_folder_state(FOLDER)["last_filepath"] == f"{FOLDER}/IMG_0100.CR3"

            db.set_folder_state(FOLDER, f"{FOLDER}/IMG_0250.CR3")
            state = db.get_folder_state(FOLDER)
            assert state["last_filepath"] == f"{FOLDER}/IMG_0250.CR3"
            # One row per folder, not one row per visit.
            assert state["updated_at"] > 0

    def test_positions_are_per_folder(self):
        with tempfile.TemporaryDirectory() as raw:
            db = self._db(raw)
            db.set_folder_state("2024-06-15", "a/one.CR3")
            db.set_folder_state("2024-06-16", "b/two.CR3")
            assert db.get_folder_state("2024-06-15")["last_filepath"] == "a/one.CR3"
            assert db.get_folder_state("2024-06-16")["last_filepath"] == "b/two.CR3"

    def test_an_empty_position_clears_it(self):
        with tempfile.TemporaryDirectory() as raw:
            db = self._db(raw)
            db.set_folder_state(FOLDER, f"{FOLDER}/IMG_0100.CR3")
            for blank in ("", None):
                assert db.set_folder_state(FOLDER, blank)["last_filepath"] is None
                assert db.get_folder_state(FOLDER)["last_filepath"] is None
                db.set_folder_state(FOLDER, f"{FOLDER}/IMG_0100.CR3")

    def test_service_passes_it_through(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            lib = tmp / "library"
            lib.mkdir()
            service = CatalogService(CatalogDB(str(tmp / "catalog.db")), str(lib))
            service.set_folder_state(FOLDER, f"{FOLDER}/IMG_0007.CR3")
            assert service.get_folder_state(FOLDER)["last_filepath"] == f"{FOLDER}/IMG_0007.CR3"


class TestBurstsRespectTheReviewFilter:
    """
    Reviewing only undecided frames must still group them, and must not offer a
    frame that has already been decided as the one to reject.
    """

    def _setup(self, tmp: Path):
        lib = tmp / "library"
        (lib / FOLDER).mkdir(parents=True)
        service = CatalogService(CatalogDB(str(tmp / "catalog.db")), str(lib))
        for i in range(3):
            name = f"IMG_{i:04d}.CR3"
            (lib / FOLDER / name).write_bytes(b"x")
            service.db.upsert_photo(_photo(name))
            service.db.upsert_photo_stats(
                f"{FOLDER}/{name}",
                {"stats_version": STATS_VERSION, "dhash": "0" * 16, "sharpness": 10.0 + i},
                source_mtime=1_700_000_000 + i,
                source_size=1,
            )
        return service

    def test_a_decided_frame_leaves_the_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._setup(Path(tmp))
            # The sharpest frame is already rated, so the review pass should be
            # left with the other two and pick the sharpest of those.
            service.set_rating(f"{FOLDER}/IMG_0002.CR3", 4)

            result = service.find_bursts(FOLDER, unreviewed_only=True, pair_mode="all")
            assert len(result["groups"]) == 1
            group = result["groups"][0]
            assert [m["filepath"] for m in group["members"]] == [
                f"{FOLDER}/IMG_0000.CR3",
                f"{FOLDER}/IMG_0001.CR3",
            ]
            assert group["keep"] == f"{FOLDER}/IMG_0001.CR3"

    def test_without_the_filter_the_decided_frame_is_still_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._setup(Path(tmp))
            service.set_rating(f"{FOLDER}/IMG_0002.CR3", 4)
            result = service.find_bursts(FOLDER, pair_mode="all")
            assert result["groups"][0]["keep"] == f"{FOLDER}/IMG_0002.CR3"
