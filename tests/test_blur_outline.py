"""Lens blur whose edge is taken from an outline (adjust.outline): two depths either side of
it, each pixel shared between them, which of SAM's answers to take, and how it is stored."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from adjust import depth, objects, outline  # noqa: E402
from adjust.develop import OWN_KEYS, from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402

FOCUS, RADIUS = 1.0, 12.0
MATTE = "0123456789abcdef0123456789abcdef"


def scene(w=400, h=300):
    """Blue stripes on the left half (far), red stripes on the right (near); the outline is
    the crisp right half, and the depth is that step softened as a depth map's edge is."""
    import cv2

    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:, :, 1] = 0.02
    for x in range(0, w // 2, 10):
        rgb[:, x:x + 2, 2] = 0.6
    for x in range(w // 2, w, 10):
        rgb[:, x:x + 2, 0] = 0.6
    matte = np.zeros((h, w), np.float32)
    matte[:, w // 2:] = 1.0
    d = cv2.GaussianBlur(matte, (0, 0), 3 * w / 400)
    return rgb, d, matte


def detail(rgb):
    return float(np.abs(np.diff(rgb[..., 2], axis=1)).mean())


def framed(rgb, d):
    """A frame whose depth is already known, so no model is needed."""
    frame = LinearFrame(rgb.copy())
    frame._mattes["depth"] = d
    return frame


@pytest.fixture()
def kept(tmp_path, monkeypatch):
    """Selections kept in a folder of the test's own."""
    monkeypatch.setattr(objects, "_mattes_dir", str(tmp_path))
    objects._mattes.clear()
    yield objects.save
    objects._mattes.clear()


def _with_edge(rgb, d, matte):
    inside, outside = outline.sides(d, matte)
    return depth.lens_blur(rgb, outside, FOCUS, depth.DEFAULT_SHARP, RADIUS, edge=(matte, inside))


class TestSides:
    def test_each_side_keeps_its_own_distance_up_to_the_edge(self):
        _, d, matte = scene()
        inside, outside = outline.sides(d, matte)
        assert inside.min() >= 0.99
        assert outside.max() <= 0.01

    def test_a_slope_is_kept(self):
        import cv2

        _, _, matte = scene()
        ramp = np.linspace(0.6, 1.0, 300, dtype=np.float32)[:, None]
        d = cv2.GaussianBlur(np.where(matte > 0.5, ramp, 0.0).astype(np.float32), (0, 0), 3)
        inside, _ = outline.sides(d, matte)
        # Where the outline is and the band beside it, which is all inside is read for.
        assert np.abs(inside - ramp)[10:-10, 180:].max() < 0.05

    def test_a_matte_of_any_size(self):
        import cv2

        _, d, matte = scene()
        big = cv2.resize(matte, (800, 600), interpolation=cv2.INTER_NEAREST)
        inside, outside = outline.sides(d, big)
        assert inside.shape == d.shape and inside.min() >= 0.99 and outside.max() <= 0.01

    def test_no_edge_is_nothing(self):
        _, d, _ = scene()
        assert outline.sides(d, np.ones_like(d)) is None
        assert outline.sides(d, np.zeros_like(d)) is None

    def test_a_thin_thing_is_kept_whole(self):
        _, d, _ = scene()
        thin = np.zeros_like(d)
        thin[:, 300:303] = 1.0
        assert outline.sides(d, thin) is not None


class TestEdge:
    def test_without_it_the_map_softens_the_subjects_edge(self):
        """What happens today, kept as the comparison."""
        rgb, d, _ = scene()
        out = depth.lens_blur(rgb, d, FOCUS, depth.DEFAULT_SHARP, RADIUS)
        assert np.abs(out - rgb)[:, 200:215].max() > 0.1

    def test_with_it_the_subject_stays_sharp_to_its_edge(self):
        rgb, d, matte = scene()
        out = _with_edge(rgb, d, matte)
        assert np.abs(out - rgb)[:, 200:215].max() < 0.02

    def test_with_it_the_background_beside_the_edge_is_blurred_like_the_rest(self):
        rgb, d, matte = scene()
        out = _with_edge(rgb, d, matte)
        assert detail(out[:, 170:200]) < detail(out[:, 20:80]) * 1.5

    def test_with_it_none_of_the_subject_is_smeared_into_the_blur(self):
        rgb, d, matte = scene()
        out = _with_edge(rgb, d, matte)
        assert out[:, 170:200, 0].max() < 0.01
        # Without, it is.
        before = depth.lens_blur(rgb, d, FOCUS, depth.DEFAULT_SHARP, RADIUS)
        assert before[:, 170:200, 0].max() > 0.05

    def test_no_edge_is_exactly_as_before(self):
        rgb, d, _ = scene()
        a = depth.lens_blur(rgb, d, FOCUS, depth.DEFAULT_SHARP, RADIUS)
        b = depth.lens_blur(rgb, d, FOCUS, depth.DEFAULT_SHARP, RADIUS, edge=None)
        assert np.array_equal(a, b)


class TestBlurred:
    def _plain(self, rgb, d, **kw):
        frame = framed(rgb, d)
        return depth.blurred_pixels(frame, frame.rgb, frame.window, 60, 100, **kw)

    def test_an_outline_that_is_not_here_or_has_no_edge_changes_nothing(self, kept):
        rgb, d, _ = scene()
        plain = self._plain(rgb, d)
        assert np.array_equal(self._plain(rgb, d, outline={"matte": MATTE}), plain)
        assert np.array_equal(self._plain(rgb, d, outline={"matte": kept(np.ones_like(d))}), plain)
        assert np.array_equal(self._plain(rgb, d, outline={"matte": kept(np.zeros_like(d))}), plain)

    def test_without_the_folder_it_changes_nothing(self, monkeypatch):
        monkeypatch.setattr(objects, "_mattes_dir", None)
        rgb, d, _ = scene()
        assert np.array_equal(self._plain(rgb, d, outline={"matte": MATTE}), self._plain(rgb, d))

    def test_same_blur_at_any_preview_size(self, kept):
        import cv2

        rgb, d, matte = scene(800, 600)
        matte_id = kept(matte)
        large = framed(rgb, d)
        small = framed(cv2.resize(rgb, (400, 300), interpolation=cv2.INTER_AREA), d)
        a = depth.blurred(large, 60, 100, outline={"matte": matte_id}).rgb
        b = depth.blurred(small, 60, 100, outline={"matte": matte_id}).rgb
        a = cv2.resize(a, (400, 300), interpolation=cv2.INTER_AREA)
        assert np.abs(a - b).mean() < 0.02

    def test_kept_for_the_same_outline(self, kept):
        rgb, d, matte = scene()
        frame = framed(rgb, d)
        one, two = kept(matte), kept(1.0 - matte)
        first = depth.blurred(frame, 50, 100, outline={"matte": one})
        assert depth.blurred(frame, 50, 100, outline={"matte": one}) is first
        assert depth.blurred(frame, 50, 100, outline={"matte": two}) is not first
        assert depth.blurred(frame, 50, 100) is not first

    def test_through_the_pipeline_only_near_the_edge(self, kept):
        rgb, d, matte = scene()
        frame = framed(rgb, d)
        edit = {"lens_blur": 60, "blur_focus": 100}
        plain = np.asarray(develop_frame(frame, edit), np.float32)
        frame = framed(rgb, d)
        edged = np.asarray(develop_frame(frame, {**edit, "blur_outline": {"matte": kept(matte)}}), np.float32)
        change = np.abs(edged - plain)
        assert change[:, 190:230].max() > 20
        assert change[:, :130].max() <= 1 and change[:, 270:].max() <= 1


class TestPick:
    def test_the_largest_that_holds_together(self):
        h = w = 64
        near = np.full((h, w), 0.1, np.float32)
        near[10:50, 10:50] = 0.5
        logits = np.full((3, h, w), -5.0, np.float32)
        logits[0, 20:25, 20:25] = 5.0        # small, at the click's distance
        logits[1, 10:50, 10:50] = 5.0        # the thing clicked
        logits[2, :, :] = 5.0                # it and all behind it
        scores = np.array([0.9, 0.5, 0.95], np.float32)
        assert outline.pick(logits, scores, near, 0.5) == 1

    def test_none_holding_together_is_the_best_score(self):
        h = w = 64
        near = np.linspace(0, 1, w, dtype=np.float32)[None].repeat(h, 0)
        logits = np.full((3, h, w), 5.0, np.float32)
        scores = np.array([0.2, 0.7, 0.5], np.float32)
        assert outline.pick(logits, scores, near, 0.5) == 1


class TestSettings:
    OUTLINE = {"matte": MATTE, "parts": [{"view": [0, 0, 1, 1], "points": [[0.5, 0.5, 1]], "pick": 2}],
               "model": "sam2.1"}

    def test_kept_beside_lens_blur(self):
        clean = normalize({"lens_blur": 40, "blur_outline": self.OUTLINE})
        assert clean["blur_outline"] == self.OUTLINE

    def test_dropped_without_it(self):
        assert normalize({"blur_outline": self.OUTLINE}) == {}

    def test_dropped_without_a_matte(self):
        for matte in (None, "", "not hex!", "abc", 7):
            assert "blur_outline" not in normalize({"lens_blur": 40, "blur_outline": {**self.OUTLINE, "matte": matte}})

    def test_its_parts_cleaned(self):
        raw = {"matte": MATTE, "parts": [{"points": [[2, -1, 1]], "pick": 9, "extra": 1}, "nonsense",
                                         {"points": [[0.5, 0.5, 0]]}], "model": "sam3"}
        clean = normalize({"lens_blur": 40, "blur_outline": raw})["blur_outline"]
        assert clean == {"matte": MATTE, "model": "sam2.1",
                         "parts": [{"view": [0.0, 0.0, 1.0, 1.0], "points": [[1.0, 0.0, 1]], "pick": 2}]}

    def test_it_belongs_to_its_frame(self):
        assert "blur_outline" in OWN_KEYS

    def test_survives_the_sidecar(self):
        edit = {"lens_blur": 35, "blur_focus": 70, "blur_outline": self.OUTLINE}
        crs = to_crs(edit)
        assert "crs:PhotoolBlurOutline" in crs
        assert from_crs(crs) == edit


class TestRoute:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        from PIL import Image
        from fastapi.testclient import TestClient

        folder = "2025/2025-03-06"
        library = tmp_path / "lib"
        (library / folder).mkdir(parents=True)
        Image.fromarray(np.full((300, 400, 3), 90, np.uint8), "RGB").save(library / folder / "IMG_0001.JPG")
        (tmp_path / "config.yaml").write_text(
            f"library_root: '{library}'\n"
            f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
            f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
            "sd_mount_paths: []\n"
            "analysis_auto_enabled: false\n"
        )
        monkeypatch.chdir(tmp_path)
        import main

        with TestClient(main.app) as c:
            assert c.post(f"/api/catalog/index/{folder}").status_code == 200
            monkeypatch.setattr(objects, "_mattes_dir", str(tmp_path / "mattes"))
            yield c, f"{folder}/IMG_0001.JPG"

    def test_an_outline_is_made_kept_and_added_to(self, client, monkeypatch):
        c, photo = client
        seen = {}
        near = np.zeros((64, 64), np.float32)
        near[:, 32:] = 1.0

        def predict(rgb8, part, model, key=None):
            assert model == "sam2.1"
            logits = np.full((3, 64, 64), -5.0, np.float32)
            logits[0, :, 40:] = 5.0
            logits[1, :, 32:] = 5.0
            logits[2] = 5.0
            return logits, np.array([0.9, 0.1, 0.95], np.float32)

        def selection(rgb8, parts, model, key=None):
            seen["parts"] = parts
            out = np.zeros(rgb8.shape[:2], np.float32)
            out[:, out.shape[1] // 2:] = 1.0
            return out

        monkeypatch.setattr(objects, "predict", predict)
        monkeypatch.setattr(objects, "selection", selection)
        monkeypatch.setattr(depth, "whole_map", lambda frame: near)
        earlier = {"view": [0, 0, 1, 1], "points": [[0.2, 0.3, 1]]}
        res = c.post("/api/export/develop/blur-outline",
                     json={"filepath": photo, "u": 0.8, "v": 0.5, "settings": {}, "parts": [earlier]})
        assert res.status_code == 200
        body = res.json()
        assert body["depth"] == 100
        made = body["outline"]
        assert made["model"] == "sam2.1"
        assert objects.load(made["matte"]) is not None
        assert len(made["parts"]) == 2 and made["parts"][0]["points"] == [[0.2, 0.3, 1]]
        assert made["parts"][1]["points"] == [[0.8, 0.5, 1]]
        assert made["parts"][1]["pick"] == 1      # the larger of the two at one distance
        assert seen["parts"] == made["parts"]

    def test_a_frame_that_is_not_there(self, client):
        c, _ = client
        res = c.post("/api/export/develop/blur-outline", json={"filepath": "2025/nothing.JPG", "u": 0.5, "v": 0.5})
        assert res.status_code in (400, 404)
