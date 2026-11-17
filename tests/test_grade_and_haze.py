"""
Colour grading, the vignette's shape, dehaze and texture (brief 3).

The first thing held still is the old picture: an edit that carries none of the
new keys renders bit for bit as it did before they were added. The hashes below
were taken on the code before the change, from a fixed synthetic frame.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.render import LinearFrame, develop_frame  # noqa: E402


def scene(w=240, h=180, seed=7) -> np.ndarray:
    """A fixed linear frame: a sky-to-ground ramp, saturated patches, and fine noise."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = 0.02 + 0.9 * (1 - y) * (0.6 + 0.4 * x)
    rgb[..., 1] = 0.03 + 0.7 * (1 - y) * (0.8 - 0.3 * x)
    rgb[..., 2] = 0.05 + 0.5 * (1 - y) + 0.1 * x
    rgb[60:110, 30:90] = (0.6, 0.08, 0.04)
    rgb[60:110, 140:200] = (0.05, 0.3, 0.6)
    rgb[130:170, 80:160] = (1.6, 1.4, 1.1)
    rgb *= 1.0 + 0.08 * rng.standard_normal((h, w, 1)).astype(np.float32)
    return np.clip(rgb, 0.0, None).astype(np.float32)


def digest(edit: dict) -> str:
    out = develop_frame(LinearFrame(scene()), edit)
    return hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16]


BAND = {"kind": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.6}

# The edits as they were before this change, and the picture each made.
OLD_EDITS = {
    "split toning": {"shadow_hue": 210, "shadow_sat": 40, "highlight_hue": 40, "highlight_sat": 30},
    "vignette": {"vignette": -45},
    "vignette light": {"vignette": 30, "grain": 20},
    "clarity": {"clarity": 50, "contrast": 20, "shadows": 30},
    "masks": {"exposure": 0.3, "masks": [dict(BAND, exposure=-0.8, clarity=40, saturation=-30)]},
    "everything": {"exposure": -0.2, "clarity": -30, "fade": 20, "shadow_hue": 30, "shadow_sat": 25,
                   "vignette": -20, "sharpen": 40,
                   "masks": [dict(BAND, contrast=30, temperature=20)]},
}
OLD_HASHES = {
    "clarity": "57dd14c7811964f2",
    "everything": "4706c966e264c2d3",
    "masks": "2e6184f87d92ca51",
    "split toning": "36cad66dd62a70db",
    "vignette": "6ed57ff798baed86",
    "vignette light": "1b62f136456f5868",
}


@pytest.mark.parametrize("name", sorted(OLD_EDITS))
def test_an_old_edit_renders_as_before(name):
    assert digest(OLD_EDITS[name]) == OLD_HASHES[name]


# ── Colour grading ───────────────────────────────────────────────────

from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import _look_of, _vignette  # noqa: E402

GREYS = np.linspace(0.0, 1.0, 101, dtype=np.float32)[None, :, None].repeat(3, axis=2)


def graded(edit):
    return _look_of(GREYS.copy(), edit)[0]


class TestGrading:
    SPLIT = {"shadow_hue": 220, "shadow_sat": 50, "highlight_hue": 40, "highlight_sat": 50}

    def test_balance_zero_and_no_midtone_is_the_old_picture(self):
        assert np.array_equal(graded(dict(self.SPLIT, tone_balance=0, midtone_hue=90)), graded(self.SPLIT))
        assert digest(dict(OLD_EDITS["split toning"], tone_balance=0, midtone_hue=120)) \
            == OLD_HASHES["split toning"]

    def test_a_midtone_tint_moves_mid_grey_and_leaves_black_and_white(self):
        out = graded({"midtone_hue": 0, "midtone_sat": 60})
        mid = out[50]
        assert mid[0] - mid[2] > 0.08                 # mid grey turns red
        assert np.array_equal(out[0], GREYS[0, 0]) and np.array_equal(out[100], GREYS[0, 100])
        # The tint fades toward the ends, and is a colour, not a brightness.
        spread = out.max(axis=1) - out.min(axis=1)
        assert spread[5] < spread[25] < spread[50] and spread[95] < spread[75] < spread[50]
        assert abs(float(mid.mean()) - 0.5) < 1e-3

    @staticmethod
    def crossover(edit):
        """The grey at which the shadows' blue hands over to the highlights' orange."""
        out = graded(edit)
        warm = out[:, 0] - out[:, 2]
        return int(np.argmax(warm > 0))

    def test_balance_moves_the_crossover(self):
        plain = self.crossover(self.SPLIT)
        left = self.crossover(dict(self.SPLIT, tone_balance=-100))
        right = self.crossover(dict(self.SPLIT, tone_balance=100))
        assert 45 <= plain <= 55
        assert left > plain + 15                      # the shadows' tint has more of the range
        assert right < plain - 15

    def test_the_balance_needs_a_tint_and_the_midtone_hue_its_strength(self):
        assert normalize({"tone_balance": 40}) == {}
        assert normalize({"tone_balance": 40, "highlight_sat": 10})["tone_balance"] == 40
        assert normalize({"midtone_hue": 40}) == {}


# ── The vignette's shape ─────────────────────────────────────────────

FLAT = np.full((181, 241, 3), 0.5, np.float32)
WHOLE = (0.0, 0.0, 1.0, 1.0)


class TestVignette:
    def test_size_and_feather_at_zero_are_the_old_picture(self):
        assert np.array_equal(_vignette(FLAT, -60, WHOLE, 0, 0), _vignette(FLAT, -60, WHOLE))
        for name in ("vignette", "vignette light"):
            edit = dict(OLD_EDITS[name], vignette_size=0, vignette_feather=0)
            assert digest(edit) == OLD_HASHES[name]

    @pytest.mark.parametrize("size,feather", [(-100, 0), (100, 0), (0, -100), (0, 100), (-100, 100), (-100, -100)])
    def test_the_centre_is_never_changed(self, size, feather):
        for amount in (-100, 100):
            out = _vignette(FLAT, amount, WHOLE, size, feather)
            assert np.array_equal(out[89:92, 119:122], FLAT[89:92, 119:122])

    def test_size_reaches_further_in_or_keeps_to_the_corners(self):
        def darkening(size):
            return 0.5 - _vignette(FLAT, -80, WHOLE, size, 0)[90, 30, 0]    # part way out, on the middle row
        assert darkening(-100) > darkening(0) + 0.05 > darkening(100) + 0.05
        corner = [_vignette(FLAT, -80, WHOLE, s, 0)[0, 0, 0] for s in (-100, 0, 100)]
        assert max(corner) - min(corner) < 0.02          # the corners darken as much

    def test_feather_makes_the_edge_harder_or_softer(self):
        def steepest(feather):
            out = _vignette(FLAT, -100, WHOLE, 0, feather)
            diagonal = out[np.arange(181), np.round(np.arange(181) * 240 / 180).astype(int), 0]
            return float(np.abs(np.diff(diagonal)).max())
        assert steepest(-100) > 2 * steepest(0) > 2 * steepest(100)


def test_the_sidecar_keeps_the_new_keys():
    edit = {"shadow_hue": 200, "shadow_sat": 20, "midtone_hue": 35, "midtone_sat": 30, "tone_balance": -40,
            "vignette": -30, "vignette_size": -50, "vignette_feather": 60, "dehaze": 35, "texture": -20}
    crs = to_crs(edit)
    assert crs["crs:ColorGradeMidtoneHue"] == "35" and crs["crs:SplitToningBalance"] == "-40"
    back = from_crs(crs)
    assert {k: back.get(k) for k in edit} == edit


def test_controls_say_what_each_new_key_means():
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from commands.develop import controls

    listed = {c["key"]: c for c in controls()}
    for key in NEW_KEYS:
        assert key in listed and listed[key]["meaning"], key
        assert "\n" not in listed[key]["meaning"]
    assert (listed["midtone_hue"]["min"], listed["midtone_hue"]["max"]) == (0.0, 360.0)
    assert (listed["tone_balance"]["min"], listed["tone_balance"]["max"]) == (-100.0, 100.0)


NEW_KEYS = ("midtone_hue", "midtone_sat", "tone_balance", "vignette_size", "vignette_feather", "dehaze", "texture")


def test_a_lut_holds_the_grading():
    from adjust.lut import split

    point, left_out = split({"midtone_hue": 40, "midtone_sat": 30, "shadow_sat": 10, "tone_balance": 20,
                             "vignette": -20, "vignette_size": 30})
    assert set(point) == {"midtone_hue", "midtone_sat", "shadow_sat", "tone_balance"}
    assert left_out == ["Vignette"]
    _, left_out = split({"dehaze": 30, "texture": 20, "contrast": 10})
    assert left_out == ["Texture", "Dehaze"]


# ── Dehaze ───────────────────────────────────────────────────────────

from adjust import dehaze  # noqa: E402
from adjust.render import encode, region_frame, resize_linear  # noqa: E402

AIR = np.array((0.85, 0.9, 0.95), np.float32)


def clear_scene(w=480, h=320, seed=3) -> np.ndarray:
    """A clear, contrasty scene: cells of colour, each with one channel nearly black
    (what the dark channel prior expects of a clear day), and a little noise."""
    rng = np.random.default_rng(seed)
    cells = 16
    colours = rng.uniform(0.0, 0.8, (h // cells + 1, w // cells + 1, 3)).astype(np.float32)
    darkest = rng.integers(0, 3, colours.shape[:2])
    for c in range(3):
        colours[..., c][darkest == c] *= 0.03
    rgb = np.repeat(np.repeat(colours, cells, 0), cells, 1)[:h, :w]
    rgb = rgb * (1 + 0.15 * rng.standard_normal((h, w, 1))).astype(np.float32)
    return np.clip(rgb, 0, None).astype(np.float32)


def with_haze(clear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``clear`` seen through air that thickens toward the top, by the haze model
    I = J t + A (1 - t), under a band of sky that is all air."""
    h, w = clear.shape[:2]
    distance = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None]
    t = np.exp(-1.0 * distance) * np.ones((1, w, 1), np.float32)
    t[: h // 8] = 0.0
    return (clear * t + AIR * (1 - t)).astype(np.float32), t


class TestDehaze:
    def test_haze_added_by_the_model_is_taken_back_out(self):
        clear = clear_scene()
        hazy, t = with_haze(clear)
        frame = LinearFrame(hazy)
        assert np.abs(dehaze._kept(frame)["airlight"] - AIR).max() < 0.01
        haze, _ = dehaze.haze_map(frame, hazy, WHOLE)
        land = slice(clear.shape[0] // 8 + 8, None)      # below the sky, where there is a scene
        assert np.abs(haze[land] - (1 - t[land, :, 0])).mean() < 0.04

        def error(rgb):
            return float(np.abs(encode(np.clip(rgb[land], 0, 1)) - encode(np.clip(clear[land], 0, 1))).mean())

        before = error(hazy)
        after = error(dehaze.pixels(frame, hazy, WHOLE, 100))
        # 100 takes out 85% of the haze, so 15% is meant to stay: the error is what a
        # perfect estimate would leave, to within 0.02 of the range.
        taken = 0.85 * (1 - t)
        ideal = error((hazy - AIR * taken) / (1 - taken))
        assert after < ideal + 0.02 and after < 0.6 * before
        # Less is less: 30 is between none and 100.
        assert after < error(dehaze.pixels(frame, hazy, WHOLE, 30)) < before

    def test_a_clear_contrasty_frame_changes_little(self):
        frame = LinearFrame(clear_scene())
        a = np.asarray(develop_frame(frame, {}), np.float32)
        b = np.asarray(develop_frame(frame, {"dehaze": 30}), np.float32)
        assert np.abs(a - b).mean() < 2.0

    def test_negative_lays_an_even_haze_of_the_airlight(self):
        hazy, _ = with_haze(clear_scene())
        frame = LinearFrame(hazy)
        out = dehaze.pixels(frame, hazy, WHOLE, -100)
        f = dehaze.ADD
        assert np.allclose(out, hazy * (1 - f) + AIR * f, atol=0.01)

    def test_zero_is_no_edit_and_the_estimate_is_kept(self):
        hazy, _ = with_haze(clear_scene())
        frame = LinearFrame(hazy)
        assert normalize({"dehaze": 0}) == {}
        assert dehaze.dehazed(frame, 0) is frame
        once = dehaze.dehazed(frame, 40)
        assert dehaze.dehazed(frame, 40) is once and frame._haze is not None

    def test_a_window_and_a_preview_match_the_whole_frame(self):
        """A loupe's patch (a window of the full-size frame, toned by the preview) and
        the preview itself give what the full-size export gives there."""
        hazy, _ = with_haze(clear_scene(960, 640, 5))
        full = LinearFrame(hazy)
        preview = LinearFrame(resize_linear(hazy, 480), scale=0.5)
        box, (x0, y0, x1, y1) = (0.5, 0.25, 0.8, 0.65), (480, 160, 768, 416)
        for amount in (60, -40):
            edit = {"dehaze": amount, "contrast": 10}
            whole = np.asarray(develop_frame(full, edit), np.float32)
            patch = region_frame(preview, hazy[y0:y1, x0:x1], box)
            window = np.asarray(develop_frame(patch, edit), np.float32)
            plain = np.asarray(develop_frame(region_frame(preview, hazy[y0:y1, x0:x1], box), {"contrast": 10}),
                               np.float32)
            # What the window shows is what the export shows there, as closely as the
            # window without dehaze already does.
            base = np.abs(plain - np.asarray(develop_frame(full, {"contrast": 10}), np.float32)[y0:y1, x0:x1]).mean()
            assert np.abs(window - whole[y0:y1, x0:x1]).mean() < base + 0.5
            # The preview's change is the export's change, shrunk: within a level on average.
            shrunk = np.asarray(develop_frame(full, edit).resize((480, 320), Image.BOX), np.float32) \
                - np.asarray(develop_frame(full, {"contrast": 10}).resize((480, 320), Image.BOX), np.float32)
            small = np.asarray(develop_frame(preview, edit), np.float32) \
                - np.asarray(develop_frame(preview, {"contrast": 10}), np.float32)
            assert np.abs(shrunk - small).mean() < 1.0

    def test_the_loupe_takes_its_haze_from_the_whole_frame(self):
        hazy, _ = with_haze(clear_scene(960, 640, 5))
        preview = LinearFrame(resize_linear(hazy, 480), scale=0.5)
        box, (x0, y0, x1, y1) = (0.5, 0.25, 0.8, 0.65), (480, 160, 768, 416)
        mine = dehaze.pixels(preview, hazy[y0:y1, x0:x1], box, 70)
        whole = dehaze.pixels(LinearFrame(hazy), hazy, WHOLE, 70)[y0:y1, x0:x1]
        a = encode(np.clip(mine, 0, 1))
        b = encode(np.clip(whole, 0, 1))
        assert np.abs(a - b).mean() < 0.005


# ── Texture ──────────────────────────────────────────────────────────

def lines(period: float, w=400, h=120, level=0.18, depth=0.15) -> np.ndarray:
    x = np.arange(w, dtype=np.float32)[None, :]
    row = level * (1 + depth * np.sin(2 * np.pi * x / period)) * np.ones((h, 1), np.float32)
    return np.repeat(row[..., None], 3, axis=2).astype(np.float32)


# Every picture here goes through the tone stage, with Texture and without: an
# empty edit is drawn without the dither the tone stage ends with.
TONED = {"contrast": 1}


def shown(rgb, edit, scale=1.0):
    return np.asarray(develop_frame(LinearFrame(rgb, scale=scale), {**TONED, **edit}), np.float32)


class TestTexture:
    def test_a_mid_scale_pattern_gains_contrast(self):
        def swing(period, amount):
            return float(shown(lines(period), {"texture": amount})[60, 100:300, 1].std())
        assert swing(8, 100) > 1.4 * swing(8, 0)
        assert swing(8, 50) > 1.15 * swing(8, 0)
        assert swing(8, -100) < 0.6 * swing(8, 0)
        # Finer than the band (noise, sharpening's) and coarser (clarity's) are left nearly alone.
        # (Measured: period 2.5 px gains 1.04, 4 px 1.25, 8 px 1.60, 16 px 1.67, 40 px
        # 1.17, 80 px 1.02 at Texture 100.)
        assert swing(16, 100) > 1.4 * swing(16, 0)
        assert swing(2.5, 100) < 1.1 * swing(2.5, 0)
        assert swing(80, 100) < 1.1 * swing(80, 0)

    def test_a_hard_edge_has_no_halo_beyond_eight_pixels(self):
        rgb = np.full((120, 400, 3), 0.03, np.float32)
        rgb[:, 200:] = 0.6
        d = np.abs(shown(rgb, {"texture": 100}) - shown(rgb, {}))[60, :, 1]
        assert d[:192].max() <= 1 and d[208:].max() <= 1
        assert d.max() <= 12

    def test_a_flat_patch_is_unchanged(self):
        flat = np.full((120, 160, 3), 0.18, np.float32)
        assert np.array_equal(shown(flat, {"texture": 100}), shown(flat, {}))

    def test_the_deepest_shadows_are_left_alone(self):
        dark = lines(8, level=0.18 * 2 ** -7)
        assert np.array_equal(shown(dark, {"texture": 100}), shown(dark, {}))

    def test_a_preview_shows_what_the_export_gives(self):
        full = lines(10, w=800, h=200)
        preview = resize_linear(full, 400)
        for amount in (80, -80):
            export = Image.fromarray(shown(full, {"texture": amount}).astype(np.uint8)).resize((400, 100), Image.BOX)
            plain = Image.fromarray(shown(full, {}).astype(np.uint8)).resize((400, 100), Image.BOX)
            change_export = np.asarray(export, np.float32) - np.asarray(plain, np.float32)
            change_preview = shown(preview, {"texture": amount}, 0.5) - shown(preview, {}, 0.5)
            inner = (slice(20, 80), slice(20, 380))
            assert np.abs(change_export[inner]).mean() > 2
            assert np.abs(change_export[inner] - change_preview[inner]).mean() \
                < 0.35 * np.abs(change_export[inner]).mean()

    def test_a_window_matches_the_whole_frame(self):
        rng = np.random.default_rng(2)
        full = np.clip(0.18 * (1 + 0.3 * rng.standard_normal((300, 400, 1))), 0.01, None).astype(np.float32)
        full = np.repeat(full, 3, axis=2)
        import cv2
        full = cv2.GaussianBlur(full, (0, 0), 2.0)
        preview = LinearFrame(resize_linear(full, 200), scale=0.5)
        box, (x0, y0, x1, y1) = (0.25, 0.2, 0.75, 0.8), (100, 60, 300, 240)

        def change(frame):
            return np.asarray(develop_frame(frame, {**TONED, "texture": 100}), np.float32) - \
                np.asarray(develop_frame(frame, TONED), np.float32)
        whole = change(LinearFrame(full))[y0:y1, x0:x1]
        window = change(region_frame(preview, full[y0:y1, x0:x1], box))
        inner = (slice(20, -20), slice(20, -20))
        assert np.abs(whole[inner]).mean() > 1
        assert np.abs(window[inner] - whole[inner]).mean() < 0.25 * np.abs(whole[inner]).mean()


def test_the_loupe_shows_what_the_export_gives(tmp_path):
    """Dehaze and texture through export.engine.loupe_image: a 1:1 patch cut from the
    full decode and toned by the preview, against the full-size picture there."""
    from export import engine

    hazy, _ = with_haze(clear_scene(960, 640, 5))
    path = tmp_path / "IMG_0001.png"
    Image.fromarray(np.clip(encode(np.clip(hazy, 0, 1)) * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)
    engine.clear_preview_cache()
    try:
        full = engine._full_linear(path)
        side, u, v = 200, 0.65, 0.45
        fh, fw = full.shape[:2]
        left, top = int(round(u * fw - side / 2)), int(round(v * fh - side / 2))
        for edit in ({"dehaze": 60}, {"texture": 80}):
            patch = np.asarray(engine.loupe_image(path, edit, u, v, side=side, preview_size=480), np.float32)
            whole = np.asarray(develop_frame(LinearFrame(np.ascontiguousarray(full)), edit), np.float32)
            plain = np.asarray(engine.loupe_image(path, TONED, u, v, side=side, preview_size=480), np.float32)
            there = whole[top:top + side, left:left + side]
            plain_there = np.asarray(develop_frame(LinearFrame(np.ascontiguousarray(full)), TONED),
                                     np.float32)[top:top + side, left:left + side]
            # As close as the loupe without the control is to the export: the preview's
            # base tones the patch, so they are never quite the same.
            assert np.abs(patch - there).mean() < np.abs(plain - plain_there).mean() + 1.0, edit
    finally:
        engine.clear_preview_cache()
