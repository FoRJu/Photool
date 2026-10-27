"""
Tests for the second version of people: quality, exemplars, "not this person",
names that survive a re-scan with another model, and "is this Alice?".

Like test_faces.py, nothing here needs a model file: the arithmetic is tested
on small hand-made vectors, and the one test that runs OpenCV is skipped when
OpenCV is missing.
"""

import hashlib
import io
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord, parse_refused
from config import Config
from faces.clustering import group, pick_exemplars, unit
from faces.models import (ARCFACE_MODEL, DETECTOR_MODEL, MODELS, RECOGNIZER_MODEL, ModelSpec,
                          _download, recognizer_file, recognizer_name, required, verify)
from faces.service import rebuild_people


def _v(*values) -> np.ndarray:
    return unit(np.array(values, dtype=np.float32))


def _face(face_id, embedding, px=100.0, quality=1.0, not_people=()):
    return SimpleNamespace(face_id=face_id, embedding=unit(np.asarray(embedding, np.float32)),
                           px=px, quality=quality, not_people=set(not_people),
                           x=0.1, y=0.1, w=0.2, h=0.2, score=0.9)


def _db(tmp) -> CatalogDB:
    return CatalogDB(str(Path(tmp) / "catalog.db"))


def _photo(db, filepath):
    folder, _, filename = filepath.rpartition("/")
    db.upsert_photo(PhotoRecord(filepath=filepath, filename=filename, folder=folder,
                                base_name=filename))


# ── Quality ──────────────────────────────────────────────────────────

class TestQuality:
    def test_a_poor_face_does_not_start_a_person(self):
        """A blurred profile that looks like nobody stays ungrouped rather than
        becoming a one-photo "person"."""
        assignments, _a, counts = group(
            [_face(1, _v(1, 0, 0)), _face(2, _v(0, 1, 0), quality=0.1)],
            threshold=0.5, min_seed_quality=0.3,
        )
        assert assignments[0] != 0
        assert assignments[1] == 0

    def test_a_poor_face_still_joins_a_clear_match(self):
        assignments, _a, _c = group(
            [_face(1, _v(1, 0, 0)), _face(2, _v(0.95, 0.31, 0), quality=0.1)],
            threshold=0.5, min_seed_quality=0.3,
        )
        assert assignments[0] == assignments[1] != 0

    def test_a_poor_face_needs_a_little_more_to_join(self):
        """Just over the bar is enough for a clear face, not for a poor one."""
        clear, _a, _c = group([_face(1, _v(1, 0)), _face(2, _v(0.52, 0.854), quality=1.0)],
                              threshold=0.5, min_seed_quality=0.3)
        poor, _a, _c = group([_face(1, _v(1, 0)), _face(2, _v(0.52, 0.854), quality=0.1)],
                             threshold=0.5, min_seed_quality=0.3)
        assert clear[0] == clear[1]
        assert poor[1] == 0

    def test_the_clearest_face_goes_first_not_the_largest(self):
        """The order of the pass is the algorithm's character: the sharp frontal
        face is looked at before a bigger blurred one."""
        assignments, _a, _c = group(
            [_face(1, _v(0, 1), px=400, quality=0.35), _face(2, _v(1, 0), px=120, quality=0.95)],
            threshold=0.5, merge_margin=None,
        )
        assert 0 < assignments[1] < assignments[0]

    @pytest.mark.skipif(__import__("importlib").util.find_spec("cv2") is None,
                        reason="OpenCV not installed")
    def test_quality_falls_for_a_profile_and_for_blur(self):
        import cv2

        from faces.detect import quality

        frontal = np.array([[38, 52], [74, 52], [56, 72], [42, 92], [71, 92]], np.float32)
        profile = frontal.copy()
        profile[2, 0] = 74  # the nose out past one eye
        rng = np.random.default_rng(1)
        sharp = (rng.random((112, 112, 3)) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
        assert quality(120, 0.95, profile, sharp) < quality(120, 0.95, frontal, sharp)
        assert quality(120, 0.95, frontal, blurred) < quality(120, 0.95, frontal, sharp)
        assert quality(30, 0.95, frontal, sharp) < quality(120, 0.95, frontal, sharp)


# ── Exemplars ────────────────────────────────────────────────────────

class TestExemplars:
    def test_exemplars_are_unlike_each_other(self):
        ex = pick_exemplars([_v(1, 0), _v(0.99, 0.14), _v(0.5, 0.866)])
        assert len(ex) == 2

    def test_the_best_quality_face_is_the_first_exemplar(self):
        ex = pick_exemplars([_v(1, 0), _v(0, 1)], qualities=[0.2, 0.9])
        assert float(ex[0] @ _v(0, 1)) == pytest.approx(1.0)

    def test_a_profile_joins_through_the_exemplar_it_resembles(self):
        """Frontal faces A, a three-quarter view B that joined, then a profile C
        that is close to B but under the bar on the group's average: it joins
        through B — and the group stays one person."""
        named = (7, _v(1, 0.3, 0), 10, np.stack([_v(1, 0, 0), _v(0.6, 0.8, 0)]))
        assignments, _a, _c = group([_face(1, _v(0.42, 0.9, 0.12))], threshold=0.5, seeds=[named])
        assert assignments[0] == 7

    def test_a_stranger_to_one_exemplar_does_not_join_through_another(self):
        """The chain guard: C resembles exemplar B but is orthogonal to exemplar
        A, so it is not the same person as the group."""
        named = (7, _v(1, 0.3, 0), 10, np.stack([_v(1, 0, 0), _v(0.55, 0.835, 0)]))
        assignments, _a, _c = group([_face(1, _v(0.0, 1.0, 0.0))], threshold=0.5, seeds=[named])
        assert assignments[0] != 7


# ── "Not this person" ────────────────────────────────────────────────

class TestRefusal:
    def test_a_face_never_joins_a_group_it_was_taken_out_of(self):
        assignments, _a, _c = group(
            [_face(1, _v(1, 0), not_people={9})], threshold=0.5, seeds=[(9, _v(1, 0), 4)]
        )
        assert assignments[0] not in (0, 9)

    def test_detaching_a_face_is_remembered_through_a_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(tmp)
            for i in range(3):
                _photo(db, f"f/IMG_{i}.CR3")
                db.replace_faces(f"f/IMG_{i}.CR3", [_face(0, _v(1, 0))], 1, 1, model="arcface")
            alice = db.create_cluster("Alice")
            rows = db.list_all_faces()
            db.set_face_clusters({r["id"]: alice for r in rows})

            assert db.detach_face("f/IMG_2.CR3", 0)
            rebuild_people(db, 0.5, min_group_px=0, model="arcface")

            row = db.list_faces("f/IMG_2.CR3")[0]
            assert row["cluster_id"] != alice
            assert alice in parse_refused(row["not_people"])
            assert all(r["cluster_id"] == alice for r in db.list_faces("f/IMG_0.CR3"))


    def test_a_face_more_like_one_taken_out_stays_out(self):
        """The sibling taken out of Alice keeps the next photograph of that
        sibling out, in the pass and in the closing merge; a face that is
        plainly Alice still joins."""
        sibling = np.stack([_v(0.7, 0.7, 0)])
        named = (9, _v(1, 0, 0), 10, None, sibling)
        assignments, _a, _c = group(
            [_face(1, _v(0.75, 0.66, 0)), _face(2, _v(0.95, 0.31, 0))],
            threshold=0.5, seeds=[named],
        )
        assert assignments[0] != 9
        assert assignments[1] == 9

    def test_faces_taken_out_teach_the_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(tmp)
            vectors = {"A0": _v(1, 0, 0), "A1": _v(0.98, 0.2, 0),
                       "S0": _v(0.6, 0.8, 0), "S1": _v(0.62, 0.78, 0.05)}
            for name, vector in vectors.items():
                _photo(db, f"f/{name}.CR3")
                db.replace_faces(f"f/{name}.CR3", [_face(0, vector)], 1, 1, model="arcface")
            alice = db.create_cluster("Alice")
            rows = {r["filepath"]: r for r in db.list_all_faces()}
            db.set_face_clusters({rows[f"f/{n}.CR3"]["id"]: alice for n in ("A0", "A1", "S0")})

            assert db.detach_face("f/S0.CR3", 0)
            rebuild_people(db, 0.5, min_group_px=0, model="arcface")

            assert db.list_faces("f/S1.CR3")[0]["cluster_id"] != alice
            assert db.list_faces("f/A1.CR3")[0]["cluster_id"] == alice

# ── Names survive a better model ─────────────────────────────────────

class TestRescan:
    def test_a_frame_described_by_another_model_is_not_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(tmp)
            _photo(db, "f/IMG_0.CR3")
            db.replace_faces("f/IMG_0.CR3", [], 10, 20)                 # legacy: SFace
            assert db.faces_are_current("f/IMG_0.CR3", 10, 20)
            assert db.faces_are_current("f/IMG_0.CR3", 10, 20, model="sface")
            assert not db.faces_are_current("f/IMG_0.CR3", 10, 20, model="arcface")
            assert db.face_scans_by_other_model("arcface") == 1

    def test_a_named_face_keeps_its_name_when_found_again(self):
        """Re-describing a library with a new model replaces every row; a new
        face over the same box keeps the old face's named person."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(tmp)
            _photo(db, "f/IMG_0.CR3")
            old = _face(0, _v(1, 0))
            other = _face(1, _v(0, 1))
            other.x, other.y = 0.6, 0.5
            db.replace_faces("f/IMG_0.CR3", [old, other], 1, 1)
            alice, unnamed = db.create_cluster("Alice"), db.create_cluster()
            rows = db.list_faces("f/IMG_0.CR3")
            db.set_face_clusters({rows[0]["id"]: alice, rows[1]["id"]: unnamed})

            again = _face(0, np.ones(512, np.float32))
            again.x, again.y = 0.11, 0.105            # the same face, a little off
            stranger = _face(1, np.ones(512, np.float32))
            stranger.x, stranger.y = 0.6, 0.5
            db.replace_faces("f/IMG_0.CR3", [stranger, again], 1, 1, model="arcface")

            by_box = {round(r["x"], 2): r for r in db.list_faces("f/IMG_0.CR3")}
            assert by_box[0.11]["cluster_id"] == alice
            assert by_box[0.6]["cluster_id"] == 0     # unnamed groups are rebuilt
            assert by_box[0.11]["model"] == "arcface"

    def test_a_rebuild_reads_only_the_current_models_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _db(tmp)
            _photo(db, "f/A.CR3")
            _photo(db, "f/B.CR3")
            db.replace_faces("f/A.CR3", [_face(0, _v(1, 0))], 1, 1)                  # sface, 2 numbers
            db.replace_faces("f/B.CR3", [_face(0, _v(1, 0, 0))], 1, 1, model="arcface")
            summary = rebuild_people(db, 0.5, min_group_px=0, model="arcface")
            assert summary["faces"] == 1


# ── "Is this Alice?" ─────────────────────────────────────────────────

class TestSuggestions:
    def _library(self, tmp):
        db = _db(tmp)
        vectors = {"A0": _v(1, 0, 0), "A1": _v(0.98, 0.2, 0),
                   "X0": _v(0.3, 0.95, 0), "X1": _v(0.25, 0.96, 0.1)}
        for name, vector in vectors.items():
            _photo(db, f"f/{name}.CR3")
            db.replace_faces(f"f/{name}.CR3", [_face(0, vector)], 1, 1, model="arcface")
        alice = db.create_cluster("Alice")
        rows = {r["filepath"]: r for r in db.list_all_faces()}
        db.set_face_clusters({rows["f/A0.CR3"]["id"]: alice, rows["f/A1.CR3"]["id"]: alice})
        return db, alice

    def test_a_group_close_to_a_named_person_is_offered_as_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, alice = self._library(tmp)
            summary = rebuild_people(db, 0.5, min_group_px=0, model="arcface")
            assert summary["suggestions"] == 1
            unnamed = [c for c in db.list_clusters() if not c["name"]]
            assert len(unnamed) == 1
            assert unnamed[0]["suggest_id"] == alice
            assert unnamed[0]["suggest_name"] == "Alice"

    def test_no_is_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, alice = self._library(tmp)
            rebuild_people(db, 0.5, min_group_px=0, model="arcface")
            unnamed = [c for c in db.list_clusters() if not c["name"]][0]
            assert db.refuse_person(unnamed["id"], alice) == 2
            rebuild_people(db, 0.5, min_group_px=0, model="arcface")
            again = [c for c in db.list_clusters() if not c["name"]][0]
            assert again["suggest_id"] == 0


# ── Models and settings ──────────────────────────────────────────────

class TestModelChoice:
    def test_each_setting_names_its_file(self):
        assert recognizer_file("arcface") == ARCFACE_MODEL
        assert recognizer_file("sface") == RECOGNIZER_MODEL
        assert recognizer_file("nonsense") == ARCFACE_MODEL
        assert recognizer_name("ArcFace_Lite") == "arcface_lite"

    def test_only_the_chosen_recogniser_is_needed(self):
        assert required("arcface") == [DETECTOR_MODEL, ARCFACE_MODEL]
        assert required("sface") == [DETECTOR_MODEL, RECOGNIZER_MODEL]

    def test_a_model_inside_a_release_pack_is_unpacked_and_checked(self, tmp_path):
        body = b"onnx bytes"
        pack = tmp_path / "pack.zip"
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("buffalo/other.onnx", b"not this one")
            archive.writestr("buffalo/model.onnx", body)
        spec = ModelSpec("model.onnx", pack.as_uri(), hashlib.sha256(body).hexdigest(),
                         len(body), member="model.onnx")
        target = tmp_path / "models" / "model.onnx"
        _download(spec, target)
        assert target.read_bytes() == body and verify(target, spec)
        assert not list((tmp_path / "models").glob("*.part"))

    def test_the_threshold_follows_the_model(self):
        assert Config(face_model="sface", face_match_threshold=0.55).effective_face_threshold() == 0.55
        assert Config(face_model="arcface").effective_face_threshold() == pytest.approx(0.42)
        # An old config that still says 0.5 for SFace does not hold ArcFace to it.
        assert Config(face_model="arcface", face_match_threshold=0.5).effective_face_threshold() < 0.5
        assert Config(face_model="arcface", face_threshold=0.37).effective_face_threshold() == 0.37

    def test_the_arcface_files_are_pinned(self):
        assert MODELS[ARCFACE_MODEL].member and len(MODELS[ARCFACE_MODEL].sha256) == 64
