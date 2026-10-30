"""
Object masks: one thing pointed at — a box round it, clicks on it — chosen by a
Segment Anything model, kept as a picture and read back at render time.

The model is not needed: a stand-in with the same inputs and outputs selects the
box it is given (or a square round the first click), and one for SAM 3's words
finds the same two squares whatever is named, so what is tested is how a prompt
reaches the model and how its answer reaches the frame, not the model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import objects  # noqa: E402
from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.masks import weight  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402
from test_develop_routes import PHOTO, client  # noqa: E402,F401  (a fixture)


class _Output:
    def __init__(self, name):
        self.name = name


class _Encoder:
    def __init__(self, n, names=("image_embeddings.0", "image_embeddings.1", "image_embeddings.2")):
        self.n, self.names = n, names
        self.runs = 0

    def get_outputs(self):
        return [_Output(n) for n in self.names]

    def run(self, _names, feeds):
        self.runs += 1
        assert feeds["pixel_values"].shape == (1, 3, self.n, self.n)
        return [np.zeros((1, 1, 1, 1), np.float32)] * len(self.names)


class _Text:
    def __init__(self):
        self.runs = 0

    def run(self, _names, feeds):
        self.runs += 1
        assert feeds["input_ids"].shape == feeds["attention_mask"].shape == (1, 32)
        return [np.zeros((1, 32, 4), np.float32)]


class _Detector:
    """Finds a square at the view's top left (score 0.9) and one at its bottom
    right (0.8), and a third it is not sure of (0.2), whatever is named."""

    side = 64

    def get_outputs(self):
        return [_Output(n) for n in ("pred_logits", "pred_boxes", "presence_logits", "pred_masks", "semantic_seg")]

    def run(self, _names, feeds):
        assert {"features_288", "features_144", "features_72", "text_features", "text_mask"} <= set(feeds)
        n = self.side
        masks = np.full((1, 3, n, n), -20.0, np.float32)
        masks[0, 0, : n // 4, : n // 4] = 20.0
        masks[0, 1, n // 2:, : n // 4] = 20.0
        masks[0, 2, 3 * n // 4:, 3 * n // 4:] = 20.0
        logit = lambda p: np.log(p / (1 - p))  # noqa: E731
        scores = np.array([[logit(0.9), logit(0.2), logit(0.8)]], np.float32)
        return scores, np.zeros((1, 3, 4), np.float32), np.full((1, 1), 20.0, np.float32), masks, masks[:, :1]


class _Decoder:
    """Selects what it was given, in the model's own square: the box, or a square
    of side 0.2 round the first click. Its three candidates are that, half of it
    and nothing, scored so the first is best."""

    def __init__(self, n, side=256):
        self.n, self.side = n, side

    def run(self, _names, feeds):
        s = self.side / self.n
        if feeds["input_boxes"].shape[1]:
            x0, y0, x1, y1 = feeds["input_boxes"][0, 0] * s
        else:
            x, y = feeds["input_points"][0, 0, 0] * s
            r = 0.1 * self.side
            x0, y0, x1, y1 = x - r, y - r, x + r, y + r
        logits = np.full((3, self.side, self.side), -20.0, np.float32)
        ys, xs = np.mgrid[0:self.side, 0:self.side] + 0.5
        inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        logits[0][inside] = 20.0
        logits[1][inside & (xs <= (x0 + x1) / 2)] = 20.0
        scores = np.array([[[0.9, 0.6, 0.1]]], np.float32)
        return scores, logits[None, None], np.zeros((1, 1, 1), np.float32)


@pytest.fixture()
def model(monkeypatch, tmp_path):
    n = objects.MODELS["sam2.1"]["input"]
    pair = (_Encoder(n), _Decoder(n))
    monkeypatch.setattr(objects, "_session_pair", lambda _model: pair)
    objects._embeds.clear()
    objects.configure(str(tmp_path / "models"), str(tmp_path / "mattes"))
    yield pair
    objects._embeds.clear()


@pytest.fixture()
def sam3(monkeypatch, tmp_path):
    n = objects.MODELS["sam3"]["input"]
    names = ("features_288", "features_144", "features_72",
             "image_embeddings.0", "image_embeddings.1", "image_embeddings.2")
    encoder, text = _Encoder(n, names), _Text()
    monkeypatch.setattr(objects, "_session_pair", lambda _model: (encoder, _Decoder(n)))
    monkeypatch.setattr(objects, "_find_pair", lambda _model: (text, _Detector()))
    monkeypatch.setattr(objects, "_tokens", lambda phrase: (np.zeros((1, 32), np.int64), np.ones((1, 32), np.int64)))
    objects._embeds.clear()
    objects._texts.clear()
    objects.configure(str(tmp_path / "models"), str(tmp_path / "mattes"))
    yield encoder, text
    objects._embeds.clear()
    objects._texts.clear()


def _picture(h=200, w=300):
    # Flat grey: the guided filter has no edge to pull a selection to.
    return np.full((h, w, 3), 128, np.uint8)


class TestTheEdit:
    def test_parts_are_cleaned(self):
        clean = normalize({"masks": [{
            "kind": "object", "model": "nonsense", "matte": "0123456789abcdef", "exposure": 0.5,
            "parts": [
                {"view": [0.2, 0.1, 0.6, 0.5], "points": [[0.3, 0.2, 1], [9, 0.2, 0], ["x", 1, 1]], "pick": 7},
                {"points": [[0.5, 0.5, 0]]},                       # only "not this": nothing chosen
                {"view": [0, 0, 0, 0], "box": [0.8, 0.9, 0.4, 0.3]},  # a box drawn up and left
            ]}]})
        assert clean["masks"] == [{
            "kind": "object", "invert": False, "model": "sam2.1", "matte": "0123456789abcdef",
            "exposure": 0.5,
            "parts": [
                {"view": [0.2, 0.1, 0.6, 0.5], "points": [[0.3, 0.2, 1], [1.0, 0.2, 0]], "pick": 2},
                {"view": [0.0, 0.0, 1.0, 1.0], "box": [0.4, 0.3, 0.8, 0.9]},
            ]}]
        assert from_crs(to_crs(clean)) == clean

    def test_a_named_part_keeps_its_words_and_choices(self):
        clean = normalize({"masks": [{"kind": "object", "model": "sam3", "parts": [
            {"view": [0, 0, 1, 1], "text": "  the   dog\n", "keep": [2, 0, 2, 99, "x"], "points": [[0.5, 0.5, 1]]},
            {"text": "   "},                                     # named nothing: nothing chosen
        ]}]})
        assert clean["masks"][0]["parts"] == [{"view": [0.0, 0.0, 1.0, 1.0], "text": "the dog", "keep": [0, 2]}]
        assert from_crs(to_crs(clean)) == clean

    def test_a_matte_name_is_only_hex(self):
        clean = normalize({"masks": [{"kind": "object", "matte": "../../etc/passwd", "exposure": 1}]})
        assert "matte" not in clean["masks"][0]


class TestChoosing:
    def test_a_box_selects_inside_it(self, model):
        rgb8 = _picture()
        part = {"view": [0.0, 0.0, 1.0, 1.0], "box": [0.2, 0.3, 0.5, 0.8]}
        sel = objects.selection(rgb8, [part], "sam2.1")
        assert sel.shape == (200, 300)
        assert sel[110, 105] > 0.95                        # the middle of the box
        assert sel[20, 20] < 0.05 and sel[110, 250] < 0.05

    def test_a_part_on_a_zoomed_view_lands_where_it_was_pointed(self, model):
        # Zoomed into the right half's lower part: a click there selects round it
        # on the whole frame, and nothing outside the view.
        rgb8 = _picture()
        part = {"view": [0.5, 0.5, 1.0, 1.0], "points": [[0.75, 0.75, 1]]}
        sel = objects.selection(rgb8, [part], "sam2.1")
        assert sel[150, 225] > 0.95
        assert sel[:100].max() < 0.05 and sel[:, :150].max() < 0.05
        # The square is a fifth of the view each way, not of the whole frame.
        rows = np.where(sel[:, 225] > 0.5)[0]
        assert 15 <= len(rows) <= 25

    def test_parts_add_up(self, model):
        rgb8 = _picture()
        parts = [{"view": [0, 0, 1, 1], "box": [0.05, 0.05, 0.25, 0.25]},
                 {"view": [0, 0, 1, 1], "box": [0.7, 0.7, 0.9, 0.9]}]
        sel = objects.selection(rgb8, parts, "sam2.1")
        assert sel[30, 45] > 0.95 and sel[160, 240] > 0.95 and sel[100, 150] < 0.05

    def test_a_picked_candidate_is_used(self, model):
        rgb8 = _picture()
        part = {"view": [0, 0, 1, 1], "box": [0.2, 0.2, 0.8, 0.8], "pick": 1}
        sel = objects.selection(rgb8, [part], "sam2.1")
        assert sel[100, 100] > 0.95 and sel[100, 200] < 0.05     # the left half only

    def test_a_view_is_read_once(self, model):
        encoder, _ = model
        rgb8 = _picture()
        for x in (0.3, 0.5, 0.7):
            objects.predict(rgb8, {"view": [0, 0, 1, 1], "points": [[x, 0.5, 1]]}, "sam2.1", key="frame")
        assert encoder.runs == 1


class TestFinding:
    def test_every_instance_named_is_found_best_first(self, sam3):
        logits, scores = objects.find(_picture(), {"view": [0, 0, 1, 1], "text": "dog"}, "sam3")
        assert np.allclose(scores, [0.9, 0.8], atol=1e-3)       # the unsure one is left out
        assert logits.shape == (2, 64, 64) and logits[1, 63, 63] > 0

    def test_a_named_part_selects_what_it_keeps(self, sam3):
        rgb8 = _picture()
        every = objects.selection(rgb8, [{"view": [0, 0, 1, 1], "text": "dog"}], "sam3")
        assert every[10, 10] > 0.95 and every[190, 290] > 0.95 and every[100, 150] < 0.05
        second = objects.selection(rgb8, [{"view": [0, 0, 1, 1], "text": "dog", "keep": [1]}], "sam3")
        assert second[10, 10] < 0.05 and second[190, 290] > 0.95

    def test_words_and_clicks_share_one_reading(self, sam3):
        encoder, text = sam3
        rgb8 = _picture()
        objects.predict(rgb8, {"view": [0, 0, 1, 1], "points": [[0.5, 0.5, 1]]}, "sam3", key="frame")
        for word in ("dog", "hair", "dog"):
            objects.find(rgb8, {"view": [0, 0, 1, 1], "text": word}, "sam3", key="frame")
        assert encoder.runs == 1
        assert text.runs == 2                    # each phrase read once

    def test_words_on_an_older_view_read_it_again(self, sam3):
        # The detector's features are kept only for the latest views; a click's
        # embeddings are kept longer.
        encoder, _ = sam3
        rgb8 = _picture()
        views = [[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]
        for view in views:
            objects.predict(rgb8, {"view": view, "points": [[view[0] + 0.1, view[1] + 0.1, 1]]}, "sam3", key="f")
        objects.predict(rgb8, {"view": views[0], "points": [[0.2, 0.2, 1]]}, "sam3", key="f")
        assert encoder.runs == 3
        objects.find(rgb8, {"view": views[0], "text": "dog"}, "sam3", key="f")   # the oldest read
        assert encoder.runs == 4

    def test_sam21_does_not_find(self):
        assert objects.finds("sam3") and not objects.finds("sam2.1")
        with pytest.raises(objects.ModelsUnavailable):
            objects._find_pair("sam2.1")


class TestOneAtATime:
    def test_the_model_is_never_run_twice_at_once(self, model, monkeypatch):
        # DirectML crashes the process when one session runs on two threads, as
        # it does when clicks come faster than the answers.
        import threading
        import time

        encoder, decoder = model
        busy, overlaps = [0], [0]

        def guarded(run):
            def wrapper(*args):
                busy[0] += 1
                overlaps[0] += busy[0] > 1
                time.sleep(0.01)
                try:
                    return run(*args)
                finally:
                    busy[0] -= 1
            return wrapper

        monkeypatch.setattr(encoder, "run", guarded(encoder.run))
        monkeypatch.setattr(decoder, "run", guarded(decoder.run))
        rgb8 = _picture()
        threads = [threading.Thread(target=objects.predict,
                                    args=(rgb8, {"view": [0, 0, 1, 1], "points": [[0.1 * k + 0.1, 0.5, 1]]},
                                          "sam2.1", "frame"))
                   for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert overlaps[0] == 0
        assert encoder.runs == 1                 # the view was read once, for all eight


class TestKeeping:
    def test_a_kept_selection_reads_back(self, model):
        matte = np.zeros((40, 60), np.float32)
        matte[10:30, 20:50] = 1.0
        matte_id = objects.save(matte)
        assert objects.save(matte) == matte_id               # its own hash: once
        objects._mattes.clear()
        assert np.array_equal(objects.load(matte_id) > 0.5, matte > 0.5)
        assert objects.load("f" * 32) is None

    def test_the_render_reads_the_kept_selection(self, model):
        matte = np.zeros((100, 150), np.float32)
        matte[:, 75:] = 1.0                                  # the right half
        matte_id = objects.save(matte)
        frame = LinearFrame(np.full((100, 150, 3), 0.18, np.float32))
        plain = develop_frame(frame, {})
        lit = develop_frame(frame, {"masks": [{"kind": "object", "matte": matte_id, "exposure": 1.0,
                                               "parts": [{"box": [0.5, 0, 1, 1]}]}]})
        a, b = np.asarray(plain, np.float32), np.asarray(lit, np.float32)
        assert abs(b[50, 20].mean() - a[50, 20].mean()) < 1.0        # left untouched
        assert b[50, 130].mean() > a[50, 130].mean() + 20             # right a stop lighter

    def test_a_missing_selection_reaches_nothing(self):
        mask = {"kind": "object", "matte": "e" * 32, "invert": True}
        assert weight(mask, (10, 10), matte=None).max() == 0.0


class TestRoutes:
    def test_point_then_keep(self, client, model):
        res = client.post("/api/export/develop/object-predict",
                          json={"filepath": PHOTO, "part": {"view": [0, 0, 1, 1], "points": [[0.5, 0.5, 1]]}})
        assert res.status_code == 200
        found = res.json()
        assert found["best"] == 0 and len(found["candidates"]) == 3
        assert found["candidates"][0]["image"].startswith("data:image/png;base64,")

        res = client.post("/api/export/develop/object-commit",
                          json={"filepath": PHOTO, "parts": [{"box": [0.25, 0.25, 0.75, 0.75]}]})
        assert res.status_code == 200
        kept = res.json()
        assert kept["model"] == "sam2.1" and kept["parts"][0]["view"] == [0.0, 0.0, 1.0, 1.0]
        png = client.get(f"/api/export/develop/object-matte/{kept['matte']}")
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"

    def test_find_then_keep(self, client, sam3):
        res = client.post("/api/export/develop/object-find",
                          json={"filepath": PHOTO, "model": "sam3", "part": {"text": "dog"}})
        assert res.status_code == 200
        found = res.json()
        assert found["text"] == "dog" and [f["score"] for f in found["found"]] == [0.9, 0.8]
        assert found["found"][0]["box"] == [0.0, 0.0, 0.25, 0.25]
        res = client.post("/api/export/develop/object-commit",
                          json={"filepath": PHOTO, "model": "sam3", "parts": [{"text": "dog", "keep": [1]}]})
        assert res.status_code == 200 and res.json()["parts"][0]["keep"] == [1]

    def test_finding_needs_sam3(self, client, model):
        res = client.post("/api/export/develop/object-find",
                          json={"filepath": PHOTO, "model": "sam2.1", "part": {"text": "dog"}})
        assert res.status_code == 400

    def test_nothing_pointed_at_is_refused(self, client, model):
        res = client.post("/api/export/develop/object-commit",
                          json={"filepath": PHOTO, "parts": [{"points": [[0.5, 0.5, 0]]}]})
        assert res.status_code == 400

    def test_a_paste_leaves_object_masks_with_their_frame(self, client, model):
        from test_develop_routes import _edit

        own = {"kind": "object", "matte": "a" * 32, "exposure": 1.0, "parts": [{"box": [0, 0, 0.5, 0.5]}]}
        _edit(client, masks=[own])
        theirs = {**own, "matte": "b" * 32}
        other = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.2, "exposure": -1.0}
        carried = client.app.state.catalog._carried(PHOTO, normalize({"masks": [other, theirs]}))
        assert [(m["kind"], m.get("matte")) for m in carried["masks"]] == [("radial", None), ("object", "a" * 32)]
