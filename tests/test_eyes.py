"""
Faces in a burst (analysis/eyes.py): whose eyes closed, and which frame keeps the
people at their best.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analysis import eyes  # noqa: E402
from catalog.database import CatalogDB  # noqa: E402


def _face(x, eyes_value, quality=0.8, y=0.4, w=0.1, pose=1.0, seen=0.9):
    return {"x": x, "y": y, "w": w, "px": 200, "score": 0.95, "quality": quality,
            "pose": pose, "eyes": eyes_value, "seen": seen}


class TestAssess:
    def test_a_blink_is_judged_against_the_same_person_a_moment_earlier(self):
        out = eyes.assess({
            "a": [_face(0.3, 1.2), _face(0.7, 1.1)],
            "b": [_face(0.31, 0.6), _face(0.7, 1.15)],      # the left person blinked
        })
        assert (out["a"]["closed"], out["b"]["closed"]) == (0, 1)
        assert out["a"]["score"] > out["b"]["score"]

    def test_someone_who_never_opens_their_eyes_is_not_called_closed(self):
        # Dark room or a downward glance throughout: nothing to compare with.
        out = eyes.assess({"a": [_face(0.5, 0.5)], "b": [_face(0.5, 0.45)]})
        assert out["a"]["closed"] == out["b"]["closed"] == 0

    def test_two_people_far_apart_are_not_one_person(self):
        out = eyes.assess({"a": [_face(0.2, 1.3)], "b": [_face(0.8, 0.5)]})
        assert out["b"]["closed"] == 0

    def test_weak_faces_are_not_judged(self):
        # The back of a head can look like a face to the detector.
        out = eyes.assess({"a": [_face(0.5, 1.3, quality=0.2)], "b": [_face(0.5, 0.3, quality=0.2)]})
        assert out == {"a": {"faces": 0, "closed": 0, "score": 0.0},
                       "b": {"faces": 0, "closed": 0, "score": 0.0}}


    def test_a_head_turned_aside_is_not_judged(self):
        # A sideways glance shows lid and lashes: it read as closed (05-25, 3957).
        out = eyes.assess({"a": [_face(0.8, 0.85)], "b": [_face(0.8, 0.59, pose=0.5)]})
        assert out["b"]["closed"] == 0

    def test_eyes_behind_sunglasses_are_not_judged(self):
        # The patch is the lens, and its contrast moves with the light (05-25, 3951).
        out = eyes.assess({"a": [_face(0.6, 0.91, seen=0.35)], "b": [_face(0.6, 0.68, seen=0.33)]})
        assert out["b"]["closed"] == 0

    def test_a_face_measured_before_pose_was_kept_is_not_judged(self):
        old = {"x": 0.5, "y": 0.4, "w": 0.1, "px": 200, "score": 0.95, "quality": 0.8, "eyes": 0.5}
        out = eyes.assess({"a": [_face(0.5, 1.2)], "b": [old]})
        assert out["b"]["closed"] == 0


class TestKeeper:
    def test_the_people_outweigh_the_background(self):
        group = [{"filepath": "a", "sharpness": 900, "mtime": 1},
                 {"filepath": "b", "sharpness": 400, "mtime": 2}]
        assessed = {"a": {"closed": 1, "score": 0.3}, "b": {"closed": 0, "score": 0.9}}
        assert eyes.best_frame(group, assessed)["filepath"] == "b"

    def test_with_nothing_known_of_faces_the_sharpest_is_kept(self):
        group = [{"filepath": "a", "sharpness": 400, "mtime": 1},
                 {"filepath": "b", "sharpness": 900, "mtime": 2}]
        assert eyes.best_frame(group, {"a": {"score": 0.0}, "b": {"score": 0.0}})["filepath"] == "b"


def test_an_open_eye_has_more_contrast_than_a_closed_one():
    from faces.detect import ALIGN_TEMPLATE

    # A face of its own, the same in both: light from one side, and a mouth.
    face = np.tile(np.linspace(130, 210, 112), (112, 1))
    face[88:96, 42:70] = 95
    rng = np.random.default_rng(0)
    closed = np.clip(face + rng.normal(0, 6, face.shape), 0, 255).astype(np.uint8)
    opened = closed.copy()
    for cx, cy in ALIGN_TEMPLATE[:2]:
        x, y = int(cx), int(cy)
        opened[y - 3:y + 3, x - 9:x + 9] = 235              # sclera
        opened[y - 3:y + 3, x - 3:x + 3] = 30               # iris
        closed[y:y + 1, x - 9:x + 9] = 90                   # a line of lashes
    assert eyes.eye_contrast(opened) > 1.5 * eyes.eye_contrast(closed)


def test_measurements_are_kept_per_file(tmp_path):
    db = CatalogDB(str(tmp_path / "catalog.db"))
    eyes.store(db, [("f/a.jpg", 100, [_face(0.5, 1.0)]), ("f/b.jpg", 200, [])])
    assert eyes.stored(db, ["f/a.jpg", "f/b.jpg", "f/c.jpg"]) == {
        "f/a.jpg": (100, [_face(0.5, 1.0)]), "f/b.jpg": (200, [])}
    assert eyes.frames_of(db, ["f/b.jpg"]) == {"f/b.jpg": []}
