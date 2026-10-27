"""
Paste and match: one edit on many frames, each adjusted towards the frame it was
copied from (export.paste_match), through the grid's batch paste, the develop
view's paste, To burst and the develop.paste command.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from export import anchors, paste_match  # noqa: E402

FOLDER = "2025/2025-03-06"
OTHER = "2025/2025-04-01"
HERO = f"{FOLDER}/IMG_0001.JPG"
DRIFTED = f"{FOLDER}/IMG_0002.JPG"
RELIT = f"{FOLDER}/IMG_0003.JPG"
ELSEWHERE = f"{OTHER}/IMG_0100.JPG"


class TestChoosingTheExposure:
    def test_capture_settings_are_used_when_the_picture_agrees(self):
        found = paste_match.choose_exposure(capture=0.33, picture=0.28)
        assert found["used"] == "capture" and found["stops"] == 0.33

    def test_light_that_changed_is_left_as_pasted(self):
        # The photographer opened up 2/3 stop for a darker stage: the picture is as
        # bright as before, and correcting by the settings would undo that.
        found = paste_match.choose_exposure(capture=-0.68, picture=0.05)
        assert found["used"] is None and found["stops"] == 0.0
        assert "light changed" in found["reason"]

    def test_the_same_settings_need_nothing(self):
        found = paste_match.choose_exposure(capture=0.0, picture=0.15)
        assert found["stops"] == 0.0 and found["used"] == "capture"

    def test_unreadable_settings_change_nothing(self):
        assert paste_match.choose_exposure(capture=None, picture=0.3)["stops"] == 0.0

    def test_the_agreement_is_a_quarter_stop(self):
        assert paste_match.choose_exposure(0.5, 0.5 - paste_match.AGREE_STOPS + 0.01)["used"] == "capture"
        assert paste_match.choose_exposure(0.5, 0.5 - paste_match.AGREE_STOPS - 0.01)["used"] is None


@pytest.fixture()
def env(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    for rel, level in ((HERO, 90), (DRIFTED, 70), (RELIT, 90), (ELSEWHERE, 200)):
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        # warm, and graded across the frame: a colour picture, not a grey card
        ramp = np.linspace(0.6, 1.0, 160)[None, :, None] * np.array([1.0, 0.8, 0.55])
        rgb = np.clip(np.broadcast_to(ramp, (120, 160, 3)) * level, 0, 255).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(library / rel, quality=95)
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    # The light each frame recorded, and how bright its picture is (stops): one
    # frame metered a third of a stop under in the same light, one where the
    # photographer opened up for a darker stage.
    light = {"IMG_0001": 10.0, "IMG_0002": 9.67, "IMG_0003": 9.33, "IMG_0100": 12.0}
    seen = {"IMG_0001": 1.0, "IMG_0002": 2 ** -0.31, "IMG_0003": 1.0, "IMG_0100": 4.0}
    monkeypatch.setattr(paste_match, "_light", lambda source: light[Path(source).stem])
    # Flat frames: nothing for the anchors to find, so each frame is decided by its
    # capture settings and its picture's brightness alone.
    monkeypatch.setattr(paste_match, "_frame",
                        lambda source: np.full((60, 80, 3), 0.18 * seen[Path(source).stem], np.float32))
    import main

    with TestClient(main.app) as c:
        for folder in (FOLDER, OTHER):
            assert c.post(f"/api/catalog/index/{folder}").status_code == 200
        yield c


def _edit(c, rel):
    return c.get("/api/photos/edit", params={"filepath": rel}).json()["settings"]


COPIED = {"exposure": -0.5, "contrast": 40, "vibrance": 20, "temperature": 10}


class TestThroughTheGrid:
    def test_as_copied_is_what_paste_always_did(self, env):
        out = env.post("/api/photos/edit/batch", json={**COPIED, "filepaths": [DRIFTED], "source": HERO}).json()
        assert out["notes"] == {}
        assert _edit(env, DRIFTED)["exposure"] == -0.5

    def test_exposure_is_matched_where_only_the_meter_moved(self, env):
        out = env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED, RELIT], "source": HERO, "paste_as": "match_exposure"}).json()
        assert _edit(env, DRIFTED)["exposure"] == pytest.approx(-0.5 + 0.33)
        assert "from capture settings" in out["notes"][DRIFTED]
        # re-metered for changed light: left as pasted, and said so
        assert _edit(env, RELIT)["exposure"] == -0.5
        assert "light changed" in out["notes"][RELIT] and out["notes"][RELIT].startswith("light left as pasted")
        assert "nothing shared with the copied frame" in out["notes"][DRIFTED]
        # the rest of the edit is the copied one
        assert _edit(env, DRIFTED)["contrast"] == 40

    def test_keep_light_keeps_each_frame_s_own_exposure_and_white_balance(self, env):
        env.put("/api/photos/edit", json={"filepath": DRIFTED, "exposure": 0.7, "tint": 5})
        env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED], "source": HERO, "paste_as": "keep_light"})
        got = _edit(env, DRIFTED)
        assert got["exposure"] == 0.7 and got["tint"] == 5 and "temperature" not in got
        assert got["contrast"] == 40

    def test_a_match_can_be_taken_back(self, env):
        out = env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED], "source": HERO, "paste_as": "match_exposure"}).json()
        env.post("/api/photos/edit/restore", json={"edits": out["previous"]})
        assert _edit(env, DRIFTED) == {}

    def test_matching_the_look_says_when_the_frame_is_from_another_shoot(self, env):
        out = env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED, ELSEWHERE], "source": HERO, "paste_as": "match_look"}).json()
        assert "another shoot" in out["notes"][ELSEWHERE]
        assert "another shoot" not in out["notes"][DRIFTED]
        assert any(k.startswith("curve_") for k in _edit(env, ELSEWHERE))

    def test_an_unknown_way_is_refused(self, env):
        r = env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED], "source": HERO, "paste_as": "guess"})
        assert r.status_code == 400

    def test_matching_without_the_source_is_refused(self, env):
        r = env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED], "paste_as": "match_exposure"})
        assert r.status_code == 400


class TestInTheDevelopView:
    def test_one_frame_is_adjusted_and_nothing_saved(self, env):
        r = env.post("/api/photos/edit/paste-adjust", json={
            "filepath": DRIFTED, "source": HERO, "copied": COPIED, "edit": COPIED, "own": {},
            "paste_as": "match_exposure"})
        assert r.status_code == 200
        assert r.json()["settings"]["exposure"] == pytest.approx(-0.17)
        assert _edit(env, DRIFTED) == {}

    def test_as_copied_comes_back_unchanged(self, env):
        r = env.post("/api/photos/edit/paste-adjust", json={
            "filepath": DRIFTED, "source": HERO, "copied": COPIED, "edit": COPIED, "paste_as": "copied"})
        assert r.json() == {"settings": COPIED, "note": None}


class TestThroughTheCommand:
    def test_develop_paste_matches_exposure(self, env):
        r = env.post("/api/commands/develop.paste", json={
            "filepaths": [DRIFTED], "settings": COPIED, "source": HERO, "paste_as": "match_exposure"})
        assert r.status_code == 200, r.text
        assert _edit(env, DRIFTED)["exposure"] == pytest.approx(-0.17)

    def test_develop_paste_needs_the_source_to_match(self, env):
        r = env.post("/api/commands/develop.paste", json={
            "filepaths": [DRIFTED], "settings": COPIED, "paste_as": "match_exposure"})
        assert r.status_code >= 400


# ── Measuring the light from what two frames share ───────────────────

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _scene(seed=3, size=(300, 400)):
    """A textured scene in linear light: blobs of colour at several sizes, with
    the fine detail SIFT finds points in."""
    import cv2

    rng = np.random.default_rng(seed)
    stops = np.zeros(size + (3,), np.float32)
    for sigma in (3, 8, 20):
        layer = cv2.GaussianBlur(rng.random(size + (3,)).astype(np.float32), (0, 0), sigma)
        stops += (layer - layer.mean()) / layer.std()
    # a couple of stops either side of a mid-tone, as a lit scene has
    return np.clip(0.12 * np.exp2(stops * 0.6), 0.005, 0.8).astype(np.float32)


def _zoomed(rgb, z=0.8):
    """The middle ``z`` of a frame, at the frame's size: a longer lens."""
    import cv2

    h, w = rgb.shape[:2]
    y0, x0 = int(h * (1 - z) / 2), int(w * (1 - z) / 2)
    crop = rgb[y0:y0 + int(h * z), x0:x0 + int(w * z)]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _with_face(rgb, box, colour):
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    out = rgb.copy()
    out[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)] = colour
    return out


# What the target lacks against the hero: half a stop, and more red than blue,
# so the target is darker and cooler and its correction warms it.
SHIFT = np.array([0.6, 0.45, 0.3])


class TestAnchors:
    def test_scene_points_measure_a_zoomed_frame_s_light(self):
        hero = _scene()
        target = _zoomed(hero) / np.exp2(SHIFT).astype(np.float32)
        pairs = anchors.scene_pairs(hero, anchors.features(hero), target, anchors.features(target))
        found = anchors.solve([], pairs)
        assert found is not None and found["points"] >= anchors.MIN_SCENE_POINTS
        # within what real frames zoomed this way gave (about 0.05 stop a channel)
        assert np.allclose(found["stops"], SHIFT, atol=0.08)

    def test_the_same_person_is_compared_with_themselves(self):
        skin_colour = np.array([0.30, 0.20, 0.14])
        hero = _with_face(_scene(), (0.4, 0.3, 0.1, 0.15), skin_colour)
        target = _with_face(_scene(5), (0.2, 0.5, 0.12, 0.18), skin_colour / np.exp2(SHIFT))
        pairs = anchors.face_pairs(hero, [(7, 0.4, 0.3, 0.1, 0.15)], target, [(7, 0.2, 0.5, 0.12, 0.18)])
        assert len(pairs) == 1
        assert np.allclose(np.log2(pairs[0][0] / pairs[0][1]), SHIFT, atol=0.01)
        # someone else, or a face nobody is known by, is not an anchor
        assert anchors.face_pairs(hero, [(7, 0.4, 0.3, 0.1, 0.15)], target, [(8, 0.2, 0.5, 0.12, 0.18)]) == []
        assert anchors.face_pairs(hero, [(0, 0.4, 0.3, 0.1, 0.15)], target, [(0, 0.2, 0.5, 0.12, 0.18)]) == []

    def test_too_little_in_common_says_nothing(self):
        assert anchors.solve([], [(np.ones(3), np.ones(3))] * (anchors.MIN_SCENE_POINTS - 1)) is None


def _anchors(exposure, wb=(0.0, 0.0, 0.0), faces=2, points=120):
    stops = np.array(wb) + exposure
    return {"stops": stops, "exposure": float(stops @ LUMA), "faces": faces, "points": points, "spread": 0.05}


class TestRoutingTheLight:
    def test_capture_settings_confirmed_by_the_anchors(self):
        found = paste_match.route_light(0.33, 0.1, _anchors(0.30))
        assert found["used"] == "capture" and found["exposure"] == 0.33 and not found["changed"]
        assert "confirmed by 2 shared faces" in found["reason"]

    def test_anchors_that_disagree_mean_the_light_changed(self):
        # re-metered for a darker stage: the settings say -0.68, the same people say +0.05
        found = paste_match.route_light(-0.68, None, _anchors(0.05, wb=(0.1, 0.0, -0.1)))
        assert found["used"] == "anchors" and found["changed"]
        assert found["exposure"] == pytest.approx(0.05, abs=0.01)
        assert "the light changed" in found["reason"] and "warmer" in found["reason"]

    def test_a_small_white_balance_difference_is_left_alone(self):
        found = paste_match.route_light(0.0, None, _anchors(0.0, wb=(0.03, 0.0, -0.02)))
        assert not found["wb"].any()

    def test_without_anchors_the_picture_checks_the_settings(self):
        assert paste_match.route_light(0.33, 0.30, None)["used"] == "capture"
        assert paste_match.route_light(-0.68, 0.05, None)["used"] is None


class TestSliders:
    def test_no_colour_change_moves_the_exposure_only(self):
        out = paste_match.to_sliders({"exposure": -0.5, "temperature": 10}, 0.33, np.zeros(3))
        assert out == {"exposure": -0.17, "temperature": 10}

    def test_a_warm_frame_is_made_cooler(self):
        # the frame is warmer than the copied one, so the correction takes red down, blue up
        out = paste_match.to_sliders({}, 0.0, np.array([-0.1, 0.0, 0.1]))
        assert out["temperature"] < 0

    def test_the_sliders_give_the_gains_asked_for(self):
        from adjust.develop import adjustment

        wb = np.array([0.12, -0.03, -0.15])
        copied = {"exposure": 0.2, "temperature": 15, "tint": -5}
        out = paste_match.to_sliders(copied, 0.4, wb)
        before = adjustment(**{k: copied.get(k, 0) for k in ("exposure", "temperature", "tint")})
        after = adjustment(**out)
        got = np.log2([after[f"wb_{c}"] / before[f"wb_{c}"] for c in "rgb"]) + (after["ev"] - before["ev"])
        wanted = wb + 0.4
        assert np.allclose(got - got @ LUMA, wanted - wanted @ LUMA, atol=0.02)
        assert got @ LUMA == pytest.approx(wanted @ LUMA, abs=0.02)


class TestFaces:
    def test_a_raw_finds_the_faces_of_the_jpeg_beside_it(self):
        class Db:
            def list_faces(self, rel):
                if not rel.endswith(".JPG"):
                    return []
                return [{"cluster_id": 4, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                        {"cluster_id": 0, "x": 0.5, "y": 0.2, "w": 0.1, "h": 0.1}]
        assert paste_match.faces_from(Db())("2026/x/MK2A0001.CR3") == [(4, 0.1, 0.2, 0.1, 0.1)]


class TestLightThroughTheGrid:
    def test_anchors_correct_a_frame_whose_light_changed(self, env, monkeypatch):
        # Same capture settings, but the target is half a stop darker and cooler:
        # the light changed. The scene is shared, so the anchors measure it.
        hero = _scene()
        target = _zoomed(hero) / np.exp2(SHIFT).astype(np.float32)
        monkeypatch.setattr(paste_match, "_light", lambda source: 10.0)
        monkeypatch.setattr(paste_match, "_frame",
                            lambda source: hero if Path(source).stem == "IMG_0001" else target)
        out = env.post("/api/photos/edit/batch", json={
            **COPIED, "filepaths": [DRIFTED], "source": HERO, "paste_as": "match_exposure"}).json()
        note = out["notes"][DRIFTED]
        assert "the light changed" in note and "matched points" in note
        got = _edit(env, DRIFTED)
        assert got["exposure"] > -0.5 + 0.3                  # brought up by about half a stop
        assert got["temperature"] > COPIED["temperature"]   # and warmed
