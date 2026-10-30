"""
Parts of people: a people mask that names areas (hair, face skin, eyes…) reaches
those parts of the people whose faces are found.

The parsers are not needed: a stand-in answers every crop with the same layout
(the top third hair, the middle face skin, a band of eyes, the bottom clothes),
so what is tested is how crops are read and put back on the frame, and how an
area reaches the picture.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import parts, segment  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402
from test_develop_routes import PHOTO, client  # noqa: E402,F401  (a fixture)

H, W = 200, 400
# Two people, one on each half.
FACES = [(0.15, 0.20, 0.10, 0.15), (0.70, 0.20, 0.10, 0.15)]


class _Output:
    name = "logits"


class _Parser:
    """Every crop alike: hair at the top, face skin (or the body's face) in the
    middle with a band of eyes across it, clothes at the bottom."""

    def __init__(self, classes):
        self.classes = classes
        self.runs = 0

    def get_inputs(self):
        return [type("I", (), {"name": "pixel_values"})()]

    def run(self, _names, feeds):
        self.runs += 1
        assert feeds["pixel_values"].shape == (1, 3, parts.INPUT, parts.INPUT)
        n = 32
        logits = np.full((len(self.classes), n, n), -10.0, np.float32)
        logits[0] = 0.0
        hair = self.classes.index("hair")
        skin = self.classes.index("skin" if "skin" in self.classes else "face")
        cloth = self.classes.index("cloth" if "cloth" in self.classes else "upper")
        logits[hair, : n // 3] = 10.0
        logits[skin, n // 3: 2 * n // 3] = 10.0
        if "l_eye" in self.classes:
            logits[self.classes.index("l_eye"), n // 2 - 1: n // 2 + 1] = 20.0
        logits[cloth, 2 * n // 3:] = 10.0
        return [logits[None]]


@pytest.fixture()
def parsers(monkeypatch):
    found = {"face": _Parser(parts.FACE_CLASSES), "body": _Parser(parts.BODY_CLASSES)}
    monkeypatch.setattr(parts, "_session", lambda key: found[key])
    return found


def _frame():
    frame = LinearFrame(np.full((H, W, 3), 0.2, np.float32))
    frame._mattes["_faces"] = list(FACES)
    frame._mattes["_parts_rgb8"] = np.full((H, W, 3), 60, np.uint8)
    return frame


def _crop_centre(box):
    x0, y0, x1, y1 = parts._face_crop(box, H, W)
    return (y0 + y1) // 2, (x0 + x1) // 2, (y0, y1)


class TestTheEdit:
    def test_areas_are_kept_for_people_only_in_their_own_order(self):
        clean = normalize({"masks": [
            {"kind": "people", "areas": ["hair", "nonsense", "eyes", "hair"], "exposure": 0.3},
            {"kind": "faces", "areas": ["hair"], "exposure": 0.3},
            {"kind": "people", "areas": [], "exposure": 0.3},
        ]})
        assert clean["masks"][0]["areas"] == ["eyes", "hair"]
        assert "areas" not in clean["masks"][1] and "areas" not in clean["masks"][2]
        assert from_crs(to_crs(clean)) == clean


class TestReading:
    def test_each_face_is_read_once_for_every_area(self, parsers):
        frame = _frame()
        for area in ("hair", "eyes", "clothes"):
            parts.matte(frame, [area])
        assert parsers["face"].runs == 2 and parsers["body"].runs == 2

    def test_an_area_reaches_its_part_of_each_person(self, parsers):
        frame = _frame()
        hair = parts.matte(frame, ["hair"])
        eyes = parts.matte(frame, ["eyes"])
        for box in FACES:
            cy, cx, (y0, y1) = _crop_centre(box)
            top = y0 + (y1 - y0) // 8
            assert hair[top, cx] > 0.9                    # the crop's top third
            assert eyes[cy, cx] > 0.9 and eyes[top, cx] < 0.1
        assert hair[H - 2, W // 2] < 0.1                  # between and below the people

    def test_one_person_is_their_parts_only(self, parsers):
        frame = _frame()
        mine = parts.matte(frame, ["eyes"], person=2)
        (cy1, cx1, _), (cy2, cx2, _) = _crop_centre(FACES[0]), _crop_centre(FACES[1])
        assert mine[cy2, cx2] > 0.9 and mine[cy1, cx1] < 0.1

    def test_a_people_mask_with_areas_reads_the_parts(self, parsers):
        frame = _frame()
        assert np.array_equal(segment.matte(frame, "people", None, ["eyes"]), parts.matte(frame, ["eyes"]))

    def test_no_faces_no_parts(self, parsers):
        frame = _frame()
        frame._mattes["_faces"] = []
        assert parts.matte(frame, ["hair"]).max() == 0.0


class TestRender:
    def test_the_edit_lights_the_part(self, parsers):
        frame = _frame()
        cy, cx, (y0, y1) = _crop_centre(FACES[0])
        plain = np.asarray(develop_frame(frame, {}), np.float32)
        lit = np.asarray(develop_frame(frame, {"masks": [{"kind": "people", "areas": ["eyes"], "exposure": 1.0}]}),
                         np.float32)
        assert lit[cy, cx].mean() > plain[cy, cx].mean() + 15
        assert abs(lit[H - 2, W // 2].mean() - plain[H - 2, W // 2].mean()) < 1.0


class TestRoutes:
    def test_without_the_parsers_the_route_says_so(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(parts, "_models_dir", str(tmp_path / "empty"))
        monkeypatch.setattr(parts, "_failed", {})
        monkeypatch.setattr(parts, "_download", lambda spec, target: (_ for _ in ()).throw(
            parts.ModelsUnavailable("no network here")))
        info = client.get("/api/export/develop/parts-model").json()
        assert info["available"] is False and info["download_mb"] == parts.DOWNLOAD_MB
        monkeypatch.setattr(segment, "ensure", lambda kind, download=True: None)
        res = client.post("/api/export/develop/mask-model", json={"kind": "parts"})
        assert res.status_code == 503 and "no network" in res.json()["detail"]
