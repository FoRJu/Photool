"""
People masks: everyone in the frame, their faces, and one person of them.

The models are not needed for most of this: the face boxes and the people
selection are put on the frame as the models would leave them, so what is
tested is what is made from them. A real photograph is used when the models
are here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import segment  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.masks import weight  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402
from samples import sample  # noqa: E402
from test_develop_routes import client  # noqa: E402,F401  (a fixture)

H, W = 120, 240
# Three people side by side, each a column of the people selection with a face
# at its top: as the models would leave them on a family photograph.
FACES = [(0.10, 0.10, 0.10, 0.15), (0.45, 0.10, 0.10, 0.15), (0.80, 0.10, 0.10, 0.15)]


def _frame(faces=FACES, crowd=False):
    rgb = np.full((H, W, 3), 0.2, dtype=np.float32)
    frame = LinearFrame(rgb)
    people = np.zeros((H, W), dtype=np.float32)
    for x, y, w, h in faces:
        people[int(y * H):, int((x - 0.03) * W):int((x + w + 0.03) * W)] = 1.0
    if crowd:
        people[60:, int(0.97 * W):] = 1.0      # someone at the edge with no face found
    frame._mattes["_rgb8"] = (rgb * 255).astype(np.uint8)
    frame._mattes["people"] = people
    frame._mattes["_faces"] = list(faces)
    return frame


def _centre(box):
    x, y, w, h = box
    return int((y + h / 2) * H), int((x + w / 2) * W)


class TestTheEdit:
    def test_a_person_is_kept_for_people_and_faces_only(self):
        clean = normalize({"masks": [{"kind": "faces", "person": 2, "exposure": 0.5},
                                     {"kind": "sky", "person": 2, "exposure": -0.5},
                                     {"kind": "people", "person": 0, "invert": 1, "saturation": -20}]})
        assert clean["masks"] == [
            {"kind": "faces", "invert": False, "person": 2, "exposure": 0.5},
            {"kind": "sky", "invert": False, "exposure": -0.5},
            {"kind": "people", "invert": True, "saturation": -20},
        ]
        assert from_crs(to_crs(clean)) == clean


class TestOnePerson:
    def test_each_person_is_their_own_column(self):
        frame = _frame()
        for n, box in enumerate(FACES, start=1):
            mine = segment.matte(frame, "people", n)
            r, c = _centre(box)
            assert mine[r + 30, c] > 0.9
            for other, obox in enumerate(FACES, start=1):
                if other != n:
                    assert mine[_centre(obox)[0] + 30, _centre(obox)[1]] < 0.05

    def test_everyone_is_the_whole_selection(self):
        frame = _frame()
        assert np.array_equal(segment.matte(frame, "people"), frame._mattes["people"])

    def test_someone_whose_face_was_not_found_is_nobodys(self):
        # Faces 12 px across: the crowd at the edge is three face widths from the
        # nearest one, further than anyone's arms reach.
        small = [(x, y, 0.05, h) for x, y, _, h in FACES]
        frame = _frame(faces=small, crowd=True)
        third = segment.matte(frame, "people", 3)
        assert third[100, W - 2] < 0.05            # the crowd at the edge
        assert third[100, _centre(small[2])[1]] > 0.9

    def test_a_person_who_is_not_there_is_nothing(self):
        assert segment.matte(_frame(), "people", 7).max() == 0.0


class TestFaces:
    def test_every_face_is_lit_and_the_bodies_are_not(self):
        frame = _frame()
        faces = segment.matte(frame, "faces")
        for box in FACES:
            r, c = _centre(box)
            assert faces[r, c] > 0.8
            assert faces[r + 40, c] < 0.05          # the body below
        assert faces[5, 5] < 0.05

    def test_one_persons_face_only(self):
        faces = segment.matte(_frame(), "faces", 2)
        assert faces[_centre(FACES[1])] > 0.8
        assert faces[_centre(FACES[0])] < 0.05 and faces[_centre(FACES[2])] < 0.05

    def test_the_oval_is_kept_inside_the_person(self):
        frame = _frame()
        faces = segment.matte(frame, "faces")
        # Beside the first face, past the person's edge, is the wall.
        r, _ = _centre(FACES[0])
        assert faces[r, int(0.05 * W)] < 0.05

    def test_a_face_the_people_model_missed_keeps_its_oval(self):
        frame = _frame()
        frame._mattes["people"] = np.zeros((H, W), dtype=np.float32)
        faces = segment.matte(frame, "faces")
        assert faces[_centre(FACES[0])] > 0.8

    def test_no_faces_is_nothing(self):
        assert segment.matte(_frame(faces=[]), "faces").max() == 0.0


class TestFaceBoxes:
    def test_a_crowd_behind_is_left_out_and_a_row_alike_is_kept(self, monkeypatch):
        class Detector:
            def __init__(self, rows):
                self.rows = np.array(rows, dtype=np.float32)

            def setInputSize(self, size):
                pass

            def detect(self, bgr):
                return 1, self.rows

        def row(x, w):
            return [x, 10, w, w] + [0] * 10 + [0.9]

        frame = LinearFrame(np.full((H, W, 3), 0.2, dtype=np.float32))
        frame._mattes["_rgb8"] = np.zeros((H, W, 3), dtype=np.uint8)
        monkeypatch.setitem(segment._nets, "faces", Detector([row(150, 40), row(20, 40), row(100, 12)]))
        boxes = segment.face_boxes(frame)
        # Left to right, and the 12 px face behind (under 40% of the largest) is gone.
        assert [round(b[0] * W) for b in boxes] == [20, 150]


class TestInTheDevelop:
    def test_a_faces_mask_lifts_the_faces(self, monkeypatch):
        frame = _frame()
        monkeypatch.setattr(segment, "whole_matte", lambda f, kind, person=None: segment.matte(frame, kind, person))
        out = np.asarray(develop_frame(frame, {"masks": [{"kind": "faces", "exposure": 1.0}]}), dtype=np.float32)
        plain = np.asarray(frame.encoded(), dtype=np.float32)
        r, c = _centre(FACES[1])
        assert out[r, c].mean() > plain[r, c].mean() + 20
        assert abs(out[100, c].mean() - plain[100, c].mean()) < 2      # the body

    def test_a_mask_weight_reads_the_persons_selection(self):
        frame = _frame()
        mine = segment.matte(frame, "people", 1)
        w = weight({"kind": "people", "person": 1}, (H, W), matte=mine)
        assert w[_centre(FACES[0])[0] + 30, _centre(FACES[0])[1]] > 0.9


def test_a_real_group_is_found_when_the_models_are_here():
    from config import load_config

    segment.configure(load_config().models_dir)
    if not (segment.available("people") and segment.available("faces")):
        pytest.skip("the people and face models are not downloaded here")
    photo = sample("group-of-three.CR3")
    from export.engine import linear_base

    frame = linear_base(photo, 1024)
    boxes = segment.face_boxes(frame)
    assert len(boxes) == 3                     # the three people, not the crowd behind
    people = segment.matte(frame, "people")
    assert people.mean() > 0.3


class TestThroughTheApi:
    def test_the_people_a_mask_can_choose_are_listed_left_to_right(self, client, monkeypatch):
        from test_develop_routes import PHOTO

        monkeypatch.setattr(segment, "face_boxes", lambda frame: [(0.1, 0.2, 0.05, 0.08), (0.6, 0.2, 0.05, 0.08)])
        res = client.post("/api/export/develop/people", json={"filepath": PHOTO, "size": 256})
        assert res.status_code == 200, res.text
        assert [f["x"] for f in res.json()["faces"]] == [0.1, 0.6]

    def test_one_persons_selection_is_drawn(self, client, monkeypatch):
        from test_develop_routes import PHOTO

        asked = []

        def fake(frame, kind, person=None):
            asked.append((kind, person))
            return np.zeros((8, 8), dtype=np.float32)

        monkeypatch.setattr(segment, "matte", fake)
        res = client.post("/api/export/develop/mask-image",
                          json={"filepath": PHOTO, "kind": "faces", "person": 2, "size": 256})
        assert res.status_code == 200 and res.headers["content-type"] == "image/png"
        assert asked == [("faces", 2)]
