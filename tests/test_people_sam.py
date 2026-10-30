"""
People outlined by SAM 2.1 with the parsers' guidance, and the skin a retouch
reaches through a lens and round the mouth.

No model is needed: the parsers' reading, SAM's candidates and the old people
model's answer are put in place as they would be, so what is tested is what is
made of them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import objects, parts, segment  # noqa: E402
from adjust.render import LinearFrame  # noqa: E402

H, W = 200, 400
FACES = [(0.10, 0.10, 0.10, 0.15), (0.60, 0.10, 0.10, 0.15)]


def _probs(n_classes, h, w, on):
    """A parser's answer over a crop: background everywhere but ``on``, which
    is ``(class, mask)`` pairs."""
    p = np.zeros((n_classes, h, w), np.float32)
    p[0] = 1.0
    for cls, mask in on:
        p[cls][mask] = 1.0
        p[0][mask] = 0.0
    return p


class TestPeopleBySam:
    def _frame(self, monkeypatch, picked):
        frame = LinearFrame(np.full((H, W, 3), 0.2, np.float32))
        frame._mattes["_faces"] = list(FACES)
        frame._mattes["_parts_rgb8"] = np.full((H, W, 3), 60, np.uint8)
        # Each person a block the body parser reads as theirs.
        reads = []
        for i, (x, y, w, h) in enumerate(FACES):
            x0, x1 = int((x - 0.05) * W), int((x + w + 0.05) * W)
            crop = (x0, 0, x1, H)
            on = np.zeros((H, x1 - x0), bool)
            on[int(y * H):, :] = True
            reads.append({"body": (crop, _probs(len(parts.BODY_CLASSES), H, x1 - x0, [(4, on)]))})
        monkeypatch.setattr(objects, "available", lambda model: True)
        monkeypatch.setattr(parts, "available", lambda: True)
        monkeypatch.setattr(parts, "people_read", lambda f: reads)

        def predict(rgb8, part, model, key=None):
            # Three candidates over the frame at 64 x 128: the person's block
            # (from the box), the face alone, and everything.
            logits = np.full((3, 64, 128), -10.0, np.float32)
            x0, y0, x1, y1 = part["box"]
            logits[0, int(y0 * 64):int(y1 * 64), int(x0 * 128):int(x1 * 128)] = 10.0
            fx, fy = part["points"][0][:2]
            logits[1, int(fy * 64) - 2:int(fy * 64) + 2, int(fx * 128) - 2:int(fx * 128) + 2] = 10.0
            logits[2] = 10.0
            return logits, np.array([0.5, 0.9, 0.7], np.float32)

        def part_matte(rgb8, grey, part, model, key):
            picked.append(part["pick"])
            m = np.zeros((H, W), np.float32)
            x0, y0, x1, y1 = part["box"]
            m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 1.0
            return m

        monkeypatch.setattr(objects, "predict", predict)
        monkeypatch.setattr(objects, "_part_matte", part_matte)
        old = np.zeros((H, W), np.float32)
        old[:, :] = 0.0
        old[100:, 350:] = 1.0                    # someone at the edge whose face was not found
        old[100:, 40:80] = 1.0                   # the people model's blob over person 1 too
        monkeypatch.setattr(segment, "_model_matte", lambda f, kind: old)
        return frame

    def test_the_candidate_like_the_parsers_is_taken_and_not_the_best_scored(self, monkeypatch):
        picked = []
        frame = self._frame(monkeypatch, picked)
        segment.matte(frame, "people")
        assert picked == [0, 0]                  # the block, though SAM scored the face higher

    def test_one_person_is_theirs_alone(self, monkeypatch):
        frame = self._frame(monkeypatch, [])
        second = segment.matte(frame, "people", 2)
        assert second[150, 280] == 1.0 and second[150, 60] == 0.0

    def test_someone_with_no_face_found_still_counts(self, monkeypatch):
        frame = self._frame(monkeypatch, [])
        everyone = segment.matte(frame, "people")
        assert everyone[150, 380] == 1.0         # the faceless person, from the people model
        assert everyone[150, 60] == 1.0 and everyone[150, 280] == 1.0

    def test_without_sam_the_people_model_as_before(self, monkeypatch):
        monkeypatch.setattr(objects, "available", lambda model: False)
        frame = LinearFrame(np.full((H, W, 3), 0.2, np.float32))
        frame._mattes["people"] = np.ones((50, 100), np.float32)
        assert segment._people_by_sam(frame) is None


class TestSkinThroughLensesAndRoundTheMouth:
    def test_what_looks_like_skin_is_skin(self):
        size = 220                                # the face crop: a 100 px face
        frame = LinearFrame(np.full((size, size, 3), 0.2, np.float32))
        skin_rgb = np.array([200, 150, 120], np.uint8)
        pic = np.tile(skin_rgb, (size, size, 1))
        yy, xx = np.mgrid[0:size, 0:size]
        # Glasses: a dark frame round a lens through which the skin shows.
        lens = (np.abs(xx - 70) < 20) & (np.abs(yy - 80) < 12)
        rim = (np.abs(xx - 70) < 24) & (np.abs(yy - 80) < 16) & ~lens
        pic[rim] = (20, 20, 25)
        # Lips: a red body, and their labelled outer edge still skin.
        lips = (np.abs(xx - 110) < 30) & (np.abs(yy - 160) < 10)
        body = (np.abs(xx - 110) < 27) & (np.abs(yy - 160) < 7)
        pic[body] = (190, 70, 80)
        skin = ~(lens | rim | lips)
        cls = parts.FACE_CLASSES.index
        probs = _probs(len(parts.FACE_CLASSES), size, size,
                       [(cls("skin"), skin), (cls("glasses"), lens | rim), (cls("u_lip"), lips)])
        frame._mattes["_parts_rgb8"] = pic
        frame._mattes["_parts_read"] = [{"face": ((0, 0, size, size), probs)}]
        frame._mattes[("_parts", ("body_skin", "face_skin"), 0)] = skin.astype(np.float32)
        out = parts.skin_matte(frame)
        assert out[80, 70] > 0.8                 # through the lens
        assert out[80 - 14, 70] < 0.2            # the frame
        assert out[160, 110] < 0.2               # the lips themselves
        assert out[160 - 9, 110] > 0.5           # their labelled edge, which is skin


class TestASecondOpinionOnFaces:
    def test_a_box_the_face_parser_sees_no_face_in_is_no_one(self):
        size = 100
        cls = parts.FACE_CLASSES.index
        face = np.zeros((size, size), bool)
        face[30:70, 30:70] = True
        real = {"face": ((0, 0, size, size), _probs(len(parts.FACE_CLASSES), size, size, [(cls("skin"), face)]))}
        hand = {"face": ((0, 0, size, size), _probs(len(parts.FACE_CLASSES), size, size, []))}
        box = (0.3, 0.3, 0.4, 0.4)
        assert parts.is_a_face(real, box, size, size)
        assert not parts.is_a_face(hand, box, size, size)
        assert parts.is_a_face({"body": ((0, 0, 1, 1), None)}, box, size, size)   # not read: trusted
