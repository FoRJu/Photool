"""
Tests for person clustering: the arithmetic that decides who is who, the rows
that remember it, and the name that turns a group into a tag.

No OpenCV and no model files here. Detection is the one part of this feature
that cannot be tested without both, so it is kept thin and everything that
decides anything is arithmetic over vectors.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB
from catalog.service import CatalogService
from catalog.xmp import read_xmp
from faces.clustering import MERGE_MARGIN, centroid, group, unit
from faces.models import MODELS, ModelSpec, ModelsUnavailable, ensure_models, model_path, verify
from faces.service import scan_folder
from PIL import Image


def _vector(*values) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _face(face_id: int, embedding, px: float = 100.0):
    """One face, in both the shape the grouping reads and the shape the rows
    store, so a single fixture serves the arithmetic and the database tests."""
    return SimpleNamespace(
        face_id=face_id,
        embedding=unit(np.asarray(embedding, dtype=np.float32)),
        px=px,
        x=0.1,
        y=0.1,
        w=0.2,
        h=0.2,
        score=0.9,
    )


# ── The arithmetic ───────────────────────────────────────────────────

class TestUnit:
    def test_a_vector_of_any_length_becomes_a_direction(self):
        assert unit(_vector(3.0, 4.0)).tolist() == pytest.approx([0.6, 0.8])

    def test_a_vector_that_is_all_zero_has_no_direction(self):
        assert unit(_vector(0.0, 0.0)).tolist() == [0.0, 0.0]

    def test_the_average_of_one_face_is_that_face(self):
        face = unit(_vector(1.0, 2.0, 3.0))
        assert centroid([face]).tolist() == pytest.approx(face.tolist(), abs=1e-6)


class TestGrouping:
    """The rule is one line — a face joins a group when it points the same way
    as the group's average — so these tests are about the three decisions around
    it: the order, the average, and the seeds."""

    def test_faces_of_one_person_land_together(self):
        a = _vector(1.0, 0.0)
        b = _vector(0.98, 0.2)
        settings = group([_face(1, a), _face(2, b)], threshold=0.5)
        assignments, _averages, counts = settings
        assert assignments[0] == assignments[1]
        assert counts[assignments[0]] == 2

    def test_faces_of_two_people_stay_apart(self):
        assignments, _averages, counts = group(
            [_face(1, _vector(1.0, 0.0)), _face(2, _vector(0.0, 1.0))], threshold=0.5
        )
        assert assignments[0] != assignments[1]
        assert sorted(counts.values()) == [1, 1]

    def test_similarity_just_under_the_bar_is_two_groups(self):
        cosine = 0.49
        assignments, _averages, _counts = group(
            [_face(1, _vector(1.0, 0.0)),
             _face(2, _vector(cosine, (1 - cosine ** 2) ** 0.5))],
            threshold=0.5,
        )
        assert assignments[0] != assignments[1]

    def test_the_biggest_face_defines_the_group(self):
        """A clear face arrives first, so a marginal one is measured against it
        rather than the other way round."""
        clear = _vector(1.0, 0.0)
        marginal = unit(_vector(0.52, 0.854))       # ~0.52 against clear
        assignments, _a, _c = group(
            [_face(1, clear, px=300), _face(2, marginal, px=45)], threshold=0.5
        )
        assert assignments[0] == assignments[1]

    def test_a_chain_of_lookalikes_does_not_join_two_people(self):
        """A resembles B and B resembles C, but A and C are far enough apart that
        nobody should be in a group with both. Comparing against the average
        rather than the nearest member is what stops that."""
        a = _vector(1.0, 0.0)
        b = _vector(0.0, 1.0)
        assignments, _a, counts = group(
            [
                _face(1, _vector(1.0, 0.0), px=300),
                _face(2, _vector(0.55, 0.835), px=200),
                _face(3, _vector(0.0, 1.0), px=100),
            ],
            threshold=0.5,
        )
        assert len(set(assignments)) == 3 or counts[assignments[2]] == 1

    def test_a_face_too_small_to_trust_is_found_but_not_grouped(self):
        settings = group(
            [_face(1, _vector(1.0, 0.0), px=300), _face(2, _vector(1.0, 0.0), px=12)],
            threshold=0.5,
            min_px=40,
        )
        assignments, _averages, counts = settings
        assert assignments[0] != 0
        assert assignments[1] == 0
        assert counts[assignments[0]] == 1

    def test_the_same_order_always_gives_the_same_answer(self):
        faces = [
            _face(1, _vector(1.0, 0.0), px=100),
            _face(2, _vector(0.9, 0.4), px=100),
            _face(3, _vector(0.0, 1.0), px=100),
        ]
        first = group(faces, threshold=0.5)[0]
        second = group(list(reversed(faces)), threshold=0.5)[0]
        by_id = {f.face_id: a for f, a in zip(faces, first)}
        by_id_reversed = {f.face_id: a for f, a in zip(list(reversed(faces)), second)}
        assert {k: (v != 0) for k, v in by_id.items()} == {
            k: (v != 0) for k, v in by_id_reversed.items()
        }

    def test_a_named_group_keeps_its_number(self):
        assignments, averages, counts = group(
            [_face(7, _vector(1.0, 0.0))],
            threshold=0.5,
            seeds=[(42, _vector(1.0, 0.0), 3)],
        )
        assert assignments[0] == 42
        assert counts[42] == 4
        assert 42 in averages

    def test_a_named_group_can_still_grow(self):
        face = unit(_vector(0.97, 0.24))
        assignments, _averages, counts = group(
            [_face(1, face)], threshold=0.5, seeds=[(5, _vector(1.0, 0.0), 2)]
        )
        assert assignments[0] == 5
        assert counts[5] == 3

    def test_a_named_group_is_not_renumbered_by_later_faces(self):
        assignments, _averages, _counts = group(
            [_face(1, _vector(0.0, 1.0)), _face(2, _vector(1.0, 0.0))],
            threshold=0.5,
            seeds=[(9, _vector(1.0, 0.0), 1)],
        )
        assert assignments[1] == 9
        assert assignments[0] not in (0, 9)

    def test_two_groups_that_are_one_person_get_merged(self):
        """Fragmentation is the failure the threshold cannot avoid: the same
        person split in two because the first photographs of them were unalike.
        Here the first two faces are far enough apart to make two groups, the
        later ones drift the first group's average toward the second, and the
        closing pass sees that the two averages agree."""
        assignments, _averages, counts = group(
            [
                _face(1, _vector(1.0, 0.0), px=300),
                _face(2, _vector(0.45, 0.893), px=250),
                _face(3, _vector(0.95, 0.31), px=200),
                _face(4, _vector(0.5, 0.866), px=150),
            ],
            threshold=0.5,
        )
        assert len(set(a for a in assignments if a)) == 1
        assert counts[assignments[0]] == 4

    def test_a_named_group_is_never_absorbed_by_a_rebuild(self):
        """A name is a decision, and it may have been the decision to keep two
        groups apart. Merging them away would undo the user's own answer."""
        assignments, averages, counts = group(
            [_face(1, _vector(1.0, 0.0), px=300)],
            threshold=0.5,
            seeds=[
                (4, _vector(1.0, 0.0), 3),
                (5, _vector(0.999, 0.045), 2),
            ],
        )
        assert {4, 5}.issubset(set(averages))
        # The new face joins one of them; neither group is absorbed by the other.
        assert counts[4] + counts[5] == 6
        assert assignments[0] in (4, 5)

    def test_two_different_people_are_not_merged_by_that_pass(self):
        assignments, _averages, _counts = group(
            [_face(1, _vector(1.0, 0.0)), _face(2, _vector(0.0, 1.0))],
            threshold=0.5,
        )
        assert assignments[0] != assignments[1]

    def test_the_merge_bar_is_stricter_than_the_join_bar(self):
        assert MERGE_MARGIN > 0

    def test_no_faces_is_no_groups(self):
        assignments, averages, counts = group([], threshold=0.5)
        assert assignments == [] and averages == {} and counts == {}

    def test_many_people_each_keep_their_own_faces(self):
        """More groups than the pass makes room for at first: the rows grow, and
        every face still finds its own person among them."""
        people = np.eye(64, dtype=np.float32)
        faces = [_face(i, people[i % 64] + 0.05 * people[(i + 1) % 64]) for i in range(128)]
        assignments, _averages, counts = group(faces, threshold=0.5)
        assert len(counts) == 64
        assert all(assignments[i] == assignments[i + 64] for i in range(64))


# ── The rows ─────────────────────────────────────────────────────────

def _db(tmp: Path) -> CatalogDB:
    return CatalogDB(str(tmp / "catalog.db"))


def _add_photo(db: CatalogDB, filepath: str) -> None:
    from catalog.database import PhotoRecord

    folder, _, filename = filepath.rpartition("/")
    db.upsert_photo(
        PhotoRecord(filepath=filepath, filename=filename, folder=folder, base_name=filename)
    )


class TestFaceRows:
    def test_a_photo_with_faces_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0001.CR3")
            faces = [_face(0, _vector(1.0, 0.0), px=120)]
            faces[0].x, faces[0].y, faces[0].w, faces[0].h = 0.1, 0.2, 0.3, 0.25
            faces[0].score = 0.91
            assert db.replace_faces("2024-12-26/IMG_0001.CR3", faces, 111, 222) == 1

            rows = db.list_faces("2024-12-26/IMG_0001.CR3")
            assert len(rows) == 1
            assert (rows[0]["x"], rows[0]["w"]) == (0.1, 0.3)
            assert rows[0]["score"] == pytest.approx(0.91)
            vector = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
            assert vector.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)

    def test_a_photo_with_no_faces_is_still_remembered_as_done(self):
        """Otherwise every scan would look at it again, forever."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0002.CR3")
            db.replace_faces("2024-12-26/IMG_0002.CR3", [], 10, 20)
            assert db.faces_are_current("2024-12-26/IMG_0002.CR3", 10, 20)
            assert db.list_faces("2024-12-26/IMG_0002.CR3") == []

    def test_a_file_that_changed_is_not_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0003.CR3")
            db.replace_faces("2024-12-26/IMG_0003.CR3", [], 10, 20)
            assert not db.faces_are_current("2024-12-26/IMG_0003.CR3", 10, 21)
            assert not db.faces_are_current("2024-12-26/IMG_0003.CR3", 11, 20)
            assert not db.faces_are_current("2024-12-26/NOPE.CR3", 10, 20)

    def test_scanning_again_replaces_the_old_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0004.CR3")
            db.replace_faces(
                "2024-12-26/IMG_0004.CR3",
                [_face(0, _vector(1.0, 0.0)), _face(1, _vector(0.0, 1.0))],
                1, 2,
            )
            db.replace_faces("2024-12-26/IMG_0004.CR3", [_face(0, _vector(1.0, 0.0))], 1, 3)
            assert len(db.list_faces("2024-12-26/IMG_0004.CR3")) == 1

    def test_faces_come_back_one_folder_at_a_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0001.CR3")
            _add_photo(db, "2024-12-27/IMG_0009.CR3")
            db.replace_faces("2024-12-26/IMG_0001.CR3", [_face(0, _vector(1.0, 0.0))], 1, 1)
            db.replace_faces("2024-12-27/IMG_0009.CR3", [_face(0, _vector(1.0, 0.0))], 1, 1)
            assert len(db.list_faces_for_folder("2024-12-26")) == 1
            assert len(db.list_all_faces()) == 2

    def test_status_counts_what_has_been_looked_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0001.CR3")
            _add_photo(db, "2024-12-26/IMG_0002.CR3")
            db.replace_faces(
                "2024-12-26/IMG_0001.CR3",
                [_face(0, _vector(1.0, 0.0), px=100), _face(1, _vector(0.0, 1.0), px=10)],
                1, 1,
            )
            status = db.face_scan_status()
            assert status["photos"] == 2
            assert status["scanned"] == 1
            assert status["faces"] == 2


class TestPeopleRows:
    def _prepared(self, tmp: Path) -> tuple[CatalogDB, int, int]:
        db = _db(tmp)
        for name in ("IMG_0001", "IMG_0002", "IMG_0003"):
            _add_photo(db, f"2024-12-26/{name}.CR3")
            db.replace_faces(f"2024-12-26/{name}.CR3", [_face(0, _vector(1.0, 0.0), px=90)], 1, 1)
        first, second = db.create_cluster(), db.create_cluster()
        rows = db.list_all_faces()
        db.set_face_clusters(
            {rows[0]["id"]: first, rows[1]["id"]: first, rows[2]["id"]: second}
        )
        return db, first, second

    def test_a_group_knows_its_faces_and_its_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, first, _second = self._prepared(Path(tmp))
            clusters = {c["id"]: c for c in db.list_clusters()}
            assert clusters[first]["faces"] == 2
            assert clusters[first]["photos"] == 2
            assert clusters[first]["sample"]["filepath"].endswith(".CR3")

    def test_the_sample_is_the_largest_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2024-12-26/IMG_0001.CR3")
            _add_photo(db, "2024-12-26/IMG_0002.CR3")
            db.replace_faces("2024-12-26/IMG_0001.CR3", [_face(0, _vector(1.0, 0.0), px=40)], 1, 1)
            db.replace_faces("2024-12-26/IMG_0002.CR3", [_face(0, _vector(1.0, 0.0), px=200)], 1, 1)
            cluster = db.create_cluster()
            db.set_face_clusters({r["id"]: cluster for r in db.list_all_faces()})
            assert db.list_clusters()[0]["sample"]["filepath"].endswith("IMG_0002.CR3")

    def test_a_named_group_comes_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, first, second = self._prepared(Path(tmp))
            db.set_cluster_name(second, "Ada")
            assert db.list_clusters()[0]["id"] == second
            assert db.list_clusters()[0]["name"] == "Ada"

    def test_merging_moves_the_faces_and_forgets_the_empty_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, first, second = self._prepared(Path(tmp))
            assert db.merge_clusters(first, [second]) == 1
            assert db.get_cluster(second) is None
            assert len(db.list_faces("2024-12-26/IMG_0003.CR3")) == 1
            assert db.list_clusters()[0]["faces"] == 3

    def test_taking_a_group_apart_leaves_the_faces_ungrouped(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, first, _second = self._prepared(Path(tmp))
            assert db.detach_cluster(first) == 2
            assert db.get_cluster(first) is None
            grouped = [r for r in db.list_all_faces() if r["cluster_id"] > 0]
            assert len(grouped) == 1

    def test_one_face_can_be_taken_out_on_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, first, _second = self._prepared(Path(tmp))
            assert db.detach_face("2024-12-26/IMG_0002.CR3", 0) is True
            assert db.detach_face("2024-12-26/IMG_0002.CR3", 0) is False
            assert db.get_cluster(first) is not None
            assert db.list_clusters()[0]["faces"] == 1

    def test_empty_groups_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            db.create_cluster("Ada")
            assert db.drop_empty_clusters() == 1
            assert db.list_clusters() == []


# ── The name becomes a tag ───────────────────────────────────────────

class TestNamingAPerson:
    def _service(self, tmp: Path) -> tuple[CatalogService, Path]:
        library = tmp / "library"
        folder = library / "2024-12-26"
        folder.mkdir(parents=True)
        for name in ("IMG_0001", "IMG_0002", "IMG_0003"):
            (folder / f"{name}.CR3").write_bytes(b"fake raw")
        db = CatalogDB(str(tmp / "catalog.db"))
        service = CatalogService(db, str(library))
        service.index_folder("2024-12-26", [".cr3"])
        return service, library

    def _grouped(self, service: CatalogService) -> tuple[int, int]:
        for name in ("IMG_0001", "IMG_0002"):
            service.db.replace_faces(
                f"2024-12-26/{name}.CR3", [_face(0, _vector(1.0, 0.0), px=90)], 1, 1
            )
        service.db.replace_faces(
            "2024-12-26/IMG_0003.CR3", [_face(0, _vector(0.0, 1.0), px=90)], 1, 1
        )
        rows = service.db.list_all_faces()
        first, second = service.db.create_cluster(), service.db.create_cluster()
        service.db.set_face_clusters(
            {rows[0]["id"]: first, rows[1]["id"]: first, rows[2]["id"]: second}
        )
        return first, second

    def test_naming_tags_the_photographs(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _library = self._service(Path(tmp))
            first, _second = self._grouped(service)
            result = service.name_person(first, "Ada")
            assert result["photos"] == 2
            assert service.db.get_photo_tags("2024-12-26/IMG_0001.CR3") == ["Ada"]
            assert service.db.get_photo_tags("2024-12-26/IMG_0003.CR3") == []

    def test_the_name_reaches_the_sidecar(self):
        """The tag is the whole point: dc:subject is what another editor reads,
        and the photographs travel with it."""
        with tempfile.TemporaryDirectory() as tmp:
            service, library = self._service(Path(tmp))
            first, _second = self._grouped(service)
            service.name_person(first, "Ada")
            assert read_xmp(library / "2024-12-26/IMG_0001.CR3").tags == ["Ada"]

    def test_renaming_takes_the_old_name_off_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, library = self._service(Path(tmp))
            first, _second = self._grouped(service)
            service.name_person(first, "Ada")
            service.name_person(first, "Ada Lovelace")
            assert service.db.get_photo_tags("2024-12-26/IMG_0001.CR3") == ["Ada Lovelace"]
            assert read_xmp(library / "2024-12-26/IMG_0001.CR3").tags == ["Ada Lovelace"]

    def test_clearing_the_name_takes_the_tag_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _library = self._service(Path(tmp))
            first, _second = self._grouped(service)
            service.name_person(first, "Ada")
            service.name_person(first, "")
            assert service.db.get_photo_tags("2024-12-26/IMG_0001.CR3") == []
            assert service.db.get_cluster(first)["name"] == ""

    def test_merging_keeps_one_name_and_tags_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _library = self._service(Path(tmp))
            first, second = self._grouped(service)
            service.name_person(first, "Ada")
            service.merge_people(second, first)
            assert service.db.get_cluster(first)["name"] == "Ada"
            assert service.db.get_photo_tags("2024-12-26/IMG_0003.CR3") == ["Ada"]

    def test_merging_into_an_unnamed_group_keeps_the_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _library = self._service(Path(tmp))
            first, second = self._grouped(service)
            service.name_person(second, "Ada")
            service.merge_people(second, first)
            assert service.db.get_cluster(first)["name"] == "Ada"
            assert service.db.get_photo_tags("2024-12-26/IMG_0001.CR3") == ["Ada"]

    def test_taking_a_group_apart_takes_its_name_with_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _library = self._service(Path(tmp))
            first, _second = self._grouped(service)
            service.name_person(first, "Ada")
            service.detach_person(first)
            assert service.db.get_cluster(first) is None
            assert service.db.get_photo_tags("2024-12-26/IMG_0001.CR3") == []

    def test_a_named_group_can_be_filtered_like_any_other_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _library = self._service(Path(tmp))
            first, _second = self._grouped(service)
            service.name_person(first, "Ada")
            found = service.get_photos_in_folder("2024-12-26", tag_filter="Ada")
            assert sorted(p.filename for p in found) == ["IMG_0001.CR3", "IMG_0002.CR3"]


# ── The models ───────────────────────────────────────────────────────

# ── Scanning a folder ────────────────────────────────────────────────
# The detector and the models are the two things these cannot have, so both are
# replaced: what is under test is the pass itself — which frames it picks, how
# much work it hands over at once, and what it reports while doing it.

def _jpeg() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _synthetic_faces(px: float = 100.0):
    from faces.detect import Face

    return [
        Face(x=0.1, y=0.1, w=0.2, h=0.2, score=0.9, px=px,
             embedding=unit(_vector(1.0, 0.0)))
    ]


class TestScanning:
    def _library(self, tmp: Path, names=("IMG_0001",)):
        library = tmp / "library"
        folder = library / "2026-06-12"
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.CR3").write_bytes(b"fake raw")
            (folder / f"{name}.JPG").write_bytes(b"fake jpg")
        db = CatalogDB(str(tmp / "catalog.db"))
        service = CatalogService(db, str(library))
        service.index_folder("2026-06-12", [".cr3", ".jpg"])
        return db, library

    def _stub(self, monkeypatch, seen=None, faces=None):
        """Replace the rendering and the detector with something countable."""
        import thumbnails

        calls = {"batches": [], "detections": 0, "min_px": []}

        def fake_batch(sources, cache, size, **kwargs):
            calls["batches"].append(list(sources))
            for source in sources:
                cache.put(source, size, _jpeg())
            return {"total": len(sources), "generated": len(sources), "cached": 0, "failed": 0}

        def fake_find(image, min_px=40.0):
            calls["detections"] += 1
            calls["min_px"].append(min_px)
            return _synthetic_faces() if faces is None else faces

        monkeypatch.setattr(thumbnails, "generate_thumbnails_batch", fake_batch)
        monkeypatch.setattr("faces.detect.find_faces", fake_find)
        return calls

    def _cache(self, tmp: Path):
        from thumbnails import ThumbnailCache

        return ThumbnailCache(str(tmp / "thumbs"))

    def test_one_photograph_is_looked_at_once_however_many_files_it_has(self, tmp_path, monkeypatch):
        """A CR3 and the JPG beside it are one moment. Looking at both would put
        every person in the library into the list twice."""
        db, library = self._library(tmp_path, ("IMG_0001", "IMG_0002"))
        self._stub(monkeypatch)
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")
        assert result["total"] == 2          # two photographs, not four files
        assert result["scanned"] == 2
        assert len(db.list_all_faces()) == 2

    def test_only_one_half_of_each_pair_ends_up_with_faces(self, tmp_path, monkeypatch):
        db, library = self._library(tmp_path, ("IMG_0001",))
        self._stub(monkeypatch)
        scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")
        stored = [row["filepath"] for row in db.list_all_faces()]
        assert len(stored) == 1
        assert stored[0].startswith("2026-06-12/IMG_0001.")

    def test_faces_stored_against_the_other_half_are_dropped(self, tmp_path, monkeypatch):
        """From a run made before the pairing rule existed, or when it changes."""
        db, library = self._library(tmp_path, ("IMG_0001",))
        db.replace_faces("2026-06-12/IMG_0001.CR3", _synthetic_faces(), 1, 1)
        assert len(db.list_all_faces()) == 1

        self._stub(monkeypatch)
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")
        assert result["dropped"] == 1
        stored = [row["filepath"] for row in db.list_all_faces()]
        assert len(stored) == 1
        assert not stored[0].endswith(".CR3")

    def test_frames_already_looked_at_are_skipped(self, tmp_path, monkeypatch):
        db, library = self._library(tmp_path, ("IMG_0001", "IMG_0002"))
        calls = self._stub(monkeypatch)
        cache = self._cache(tmp_path)
        scan_folder(db, cache, str(library), "2026-06-12", "models")
        assert calls["detections"] == 2

        second = scan_folder(db, cache, str(library), "2026-06-12", "models")
        assert second["skipped"] == 2 and second["scanned"] == 0
        assert calls["detections"] == 2      # nothing was looked at twice

    def test_the_rendering_is_handed_over_in_chunks(self, tmp_path, monkeypatch):
        """One batch for a whole folder is what made a large folder crawl: it
        writes every embedded JPEG to a temporary directory at once, holds them
        all in memory, and takes longer than the extraction timeout."""
        monkeypatch.setattr("faces.service.RENDER_CHUNK", 2)
        db, library = self._library(tmp_path, ("A", "B", "C", "D", "E"))
        calls = self._stub(monkeypatch)
        scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")
        assert [len(batch) for batch in calls["batches"]] == [2, 2, 1]

    def test_the_floor_is_passed_to_the_detector(self, tmp_path, monkeypatch):
        """A face that cannot be grouped is not worth describing, and describing
        one is the expensive part."""
        db, library = self._library(tmp_path, ("IMG_0001",))
        calls = self._stub(monkeypatch)
        scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models",
                    min_group_px=55.0)
        assert calls["min_px"] == [55.0]

    def test_it_reports_reading_then_looking(self, tmp_path, monkeypatch):
        db, library = self._library(tmp_path, ("IMG_0001", "IMG_0002"))
        self._stub(monkeypatch)
        seen = []
        scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models",
                    progress=lambda phase, done, total: seen.append((phase, done, total)))
        phases = [name for name, _done, _total in seen]
        assert phases[0] == "rendering"
        assert "scanning" in phases
        assert all(total == 2 for _name, _done, total in seen)
        assert seen[-1][0] == "scanning" and seen[-1][1] == 2

    def test_a_frame_with_no_faces_is_still_remembered_as_done(self, tmp_path, monkeypatch):
        db, library = self._library(tmp_path, ("IMG_0001",))
        self._stub(monkeypatch, faces=[])
        cache = self._cache(tmp_path)
        scan_folder(db, cache, str(library), "2026-06-12", "models")
        assert db.list_all_faces() == []
        assert scan_folder(db, cache, str(library), "2026-06-12", "models")["skipped"] == 1

    def test_a_batch_that_fails_does_not_end_the_folder(self, tmp_path, monkeypatch):
        """The batch is only a head start; each frame is still rendered on its own."""
        import thumbnails

        db, library = self._library(tmp_path, ("IMG_0001", "IMG_0002"))
        self._stub(monkeypatch)

        def broken_batch(sources, cache, size, **kwargs):
            raise RuntimeError("exiftool went away")

        monkeypatch.setattr(thumbnails, "generate_thumbnails_batch", broken_batch)
        monkeypatch.setattr(thumbnails, "get_thumbnail", lambda source, cache, size: _jpeg())
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")
        assert result["scanned"] == 2 and result["failed"] == 0

    def test_a_raw_on_its_own_is_looked_at_through_the_raw(self, tmp_path, monkeypatch):
        """No JPEG beside it: the RAW's own rendering (its embedded JPEG) is read."""
        library = tmp_path / "library"
        folder = library / "2026-06-12"
        folder.mkdir(parents=True)
        (folder / "IMG_0001.CR3").write_bytes(b"fake raw")
        db = CatalogDB(str(tmp_path / "catalog.db"))
        CatalogService(db, str(library)).index_folder("2026-06-12", [".cr3", ".jpg"])
        calls = self._stub(monkeypatch)
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")
        assert result["scanned"] == 1
        assert [source.name for batch in calls["batches"] for source in batch] == ["IMG_0001.CR3"]
        assert [row["filepath"] for row in db.list_all_faces()] == ["2026-06-12/IMG_0001.CR3"]

    def test_it_stops_between_chunks(self, tmp_path, monkeypatch):
        """Stop is checked between frames and between chunks, so a stopped scan
        ends on a frame boundary and leaves consistent rows behind."""
        monkeypatch.setattr("faces.service.RENDER_CHUNK", 1)
        db, library = self._library(tmp_path, ("A", "B", "C"))
        calls = self._stub(monkeypatch)
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models",
                             should_stop=lambda: calls["detections"] >= 1)
        assert len(calls["batches"]) == 1
        assert result["scanned"] == 1
        assert result["total"] == 3          # the frames it never reached are still counted

    def test_a_missing_folder_is_reported_not_raised(self, tmp_path):
        db, _library = self._library(tmp_path)
        result = scan_folder(db, self._cache(tmp_path), str(tmp_path / "library"), "nope", "models")
        assert result["error"] == "Folder not found"


class TestScanningInParallel:
    """
    A scan used to look at one frame at a time, which pinned it to a single
    core however many the machine had: measured on a real folder, a serial pass
    held 0.99 of a core and took 7.4s for 96 frames. These pin the two halves
    of the fix — that frames really are looked at concurrently, and that the
    database is still written from one thread.
    """

    def _library(self, tmp: Path, names=("A", "B", "C", "D", "E", "F")):
        library = tmp / "library"
        folder = library / "2026-06-12"
        folder.mkdir(parents=True)
        db = CatalogDB(str(tmp / "catalog.db"))
        for name in names:
            (folder / f"{name}.CR3").write_bytes(b"fake raw")
        CatalogService(db, str(library)).index_folder("2026-06-12", [".cr3"])
        return db, library

    def _stub(self, monkeypatch, workers=4, on_find=None):
        """Count which thread looked at each frame, and re-render per frame."""
        import threading
        import time

        import thumbnails

        import faces.service as service

        monkeypatch.setattr(service, "FACE_WORKERS", workers)

        state = {"threads": [], "detections": 0}
        lock = threading.Lock()

        def fake_batch(sources, cache, size, **kwargs):
            for source in sources:
                cache.put(source, size, _jpeg())
            return {"total": len(sources), "generated": len(sources), "cached": 0, "failed": 0}

        def fake_find(image, min_px=40.0):
            # The counter is shared by the pool, so the increment and the read
            # have to be one step: otherwise two frames can be handed the same
            # number and a test that raises for exactly one frame becomes flaky.
            with lock:
                state["threads"].append(threading.current_thread().name)
                state["detections"] += 1
                nth = state["detections"]
            # Long enough that the pool cannot finish a frame before the next
            # one starts, so the overlap this test asserts is real and not luck.
            time.sleep(0.01)
            if on_find:
                on_find(nth)
            return _synthetic_faces()

        monkeypatch.setattr(thumbnails, "generate_thumbnails_batch", fake_batch)
        monkeypatch.setattr("faces.detect.find_faces", fake_find)
        return state

    def _cache(self, tmp: Path):
        from thumbnails import ThumbnailCache

        return ThumbnailCache(str(tmp / "thumbs"))

    def test_frames_are_looked_at_on_more_than_one_thread(self, tmp_path, monkeypatch):
        """The whole point: with a serial loop every name here is the same."""
        db, library = self._library(tmp_path, tuple(f"IMG_{i:04d}" for i in range(12)))
        state = self._stub(monkeypatch, workers=4)
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")

        assert result["scanned"] == 12
        assert len(set(state["threads"])) > 1, state["threads"]

    def test_the_database_is_written_from_one_thread(self, tmp_path, monkeypatch):
        """Detectors run concurrently; the rows must not. One writer keeps the
        SQLite write lock out of the pool and every face row landing from one
        place."""
        import threading

        db, library = self._library(tmp_path, tuple(f"IMG_{i:04d}" for i in range(12)))
        self._stub(monkeypatch, workers=4)

        writers = []
        real = db.replace_faces
        monkeypatch.setattr(
            db,
            "replace_faces",
            lambda *a, **kw: (writers.append(threading.current_thread().name), real(*a, **kw))[1],
        )

        scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")

        assert writers
        assert set(writers) == {threading.current_thread().name}, set(writers)

    def test_parallel_and_serial_store_the_same_rows(self, tmp_path, monkeypatch):
        """Chunking the work across threads must not change what is stored, or
        which frame a face belongs to."""
        names = tuple(f"IMG_{i:04d}" for i in range(12))

        def stored_rows(db):
            # Row ids and timestamps are the database's own bookkeeping; what
            # matters is that the same faces land against the same frames.
            return sorted(
                (
                    row["filepath"], row["face_index"], row["x"], row["y"],
                    row["w"], row["h"], row["px"], bytes(row["embedding"]),
                )
                for row in db.list_all_faces()
            )

        stored = []
        for workers in (1, 4):
            db, library = self._library(tmp_path / f"w{workers}", names)
            self._stub(monkeypatch, workers=workers)
            result = scan_folder(db, self._cache(tmp_path / f"w{workers}"), str(library),
                                 "2026-06-12", "models")
            stored.append((result["scanned"], result["faces"], stored_rows(db)))
        assert stored[0] == stored[1]
        assert stored[0][0] == 12 and stored[0][1] == 12

    def test_one_frame_that_cannot_be_described_does_not_stop_the_rest(self, tmp_path, monkeypatch):
        """A detector that throws for one frame is a failure of that frame, not
        of the folder — the scan counts it and carries on."""
        db, library = self._library(tmp_path, tuple(f"IMG_{i:04d}" for i in range(6)))

        def boom(nth):
            if nth == 2:
                raise RuntimeError("no faces today")

        self._stub(monkeypatch, workers=4, on_find=boom)
        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")

        assert result["failed"] == 1
        assert result["scanned"] == 5
        assert len(db.list_all_faces()) == 5

    def test_a_frame_with_no_rendering_is_counted_not_raised(self, tmp_path, monkeypatch):
        import thumbnails

        db, library = self._library(tmp_path, ("A", "B", "C"))
        state = self._stub(monkeypatch, workers=4)

        real_get = thumbnails.get_thumbnail
        monkeypatch.setattr(
            thumbnails,
            "get_thumbnail",
            lambda source, cache, size: None if source.name.startswith("B") else real_get(source, cache, size),
        )

        result = scan_folder(db, self._cache(tmp_path), str(library), "2026-06-12", "models")

        assert result["failed"] == 1
        assert result["scanned"] == 2


class TestCleaningUpRows:
    def test_forgetting_a_file_takes_its_faces_and_its_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2026-06-12/IMG_0001.CR3")
            db.replace_faces("2026-06-12/IMG_0001.CR3", _synthetic_faces(), 5, 6)
            assert db.forget_faces("2026-06-12/IMG_0001.CR3") == 1
            assert db.list_faces("2026-06-12/IMG_0001.CR3") == []
            assert not db.faces_are_current("2026-06-12/IMG_0001.CR3", 5, 6)

    def test_only_faces_below_the_floor_are_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(Path(tmp))
            _add_photo(db, "2026-06-12/IMG_0001.CR3")
            db.replace_faces(
                "2026-06-12/IMG_0001.CR3",
                [_synthetic_faces(px=120)[0], _synthetic_faces(px=9)[0], _synthetic_faces(px=30)[0]],
                1, 1,
            )
            assert db.prune_small_faces(40) == 2
            kept = db.list_faces("2026-06-12/IMG_0001.CR3")
            assert len(kept) == 1 and kept[0]["px"] == 120


class TestModels:
    def test_a_file_that_is_not_the_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ModelSpec("x.onnx", "http://example.invalid/x", "00" * 32, 4)
            path = Path(tmp) / "x.onnx"
            path.write_bytes(b"nope")
            assert verify(path, spec) is False

    def test_a_truncated_download_is_rejected(self):
        """A half file fails at load time with a message about the network, and
        the retry that would fix it never happens."""
        with tempfile.TemporaryDirectory() as tmp:
            body = b"1234"
            import hashlib
            spec = ModelSpec(
                "x.onnx", "http://example.invalid/x", hashlib.sha256(body).hexdigest(), 8
            )
            path = Path(tmp) / "x.onnx"
            path.write_bytes(body)
            assert verify(path, spec) is False

    def test_a_matching_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = b"1234"
            import hashlib
            spec = ModelSpec(
                "x.onnx", "http://example.invalid/x", hashlib.sha256(body).hexdigest(), 4
            )
            path = Path(tmp) / "x.onnx"
            path.write_bytes(body)
            assert verify(path, spec) is True

    def test_the_expected_models_are_the_ones_the_code_names(self):
        from faces.detect import DETECTOR_MODEL as DET
        from faces.detect import RECOGNIZER_MODEL as REC

        assert DET in MODELS and REC in MODELS
        assert MODELS[DET].size == 232589
        assert MODELS[REC].size == 38696376 or MODELS[REC].size == 38696353

    def test_nothing_is_downloaded_when_the_caller_says_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ModelsUnavailable):
                ensure_models(tmp, download=False)

    def test_a_model_that_is_missing_is_named_in_the_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ensure_models(tmp, download=False)
            except ModelsUnavailable as exc:
                assert "face_detection_yunet_2023mar.onnx" in str(exc)
            else:  # pragma: no cover
                pytest.fail("expected ModelsUnavailable")

    def test_the_path_is_the_directory_plus_the_name(self):
        assert model_path("models", "x.onnx") == Path("models") / "x.onnx"
