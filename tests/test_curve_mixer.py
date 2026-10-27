"""
The tone curve and the colour mixer: how an edit stores them, what they do to
pixels, and that they survive the sidecar and the API.
"""

import colorsys
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import from_crs, normalize, normalize_curve, to_crs  # noqa: E402
from adjust.render import LinearFrame, curve_table, develop_frame, linear_from_image  # noqa: E402


def frame_of(rgb8) -> LinearFrame:
    """A frame whose display rendering is ``rgb8`` (an 8-bit array)."""
    return LinearFrame(linear_from_image(Image.fromarray(np.asarray(rgb8, np.uint8), "RGB")))


def patches(*colours, size=24):
    """Side-by-side flat patches of the given 8-bit colours."""
    return np.concatenate([np.full((size, size, 3), c, np.uint8) for c in colours], axis=1)


def patch_means(img, count, size=24):
    a = np.asarray(img, dtype=np.float32)
    return [a[4:-4, i * size + 4:(i + 1) * size - 4].reshape(-1, 3).mean(axis=0) for i in range(count)]


# Each at its band's centre: red 0°, blue 240°, green 120°.
GREY, RED, BLUE, GREEN = (128, 128, 128), (200, 40, 40), (40, 40, 200), (60, 170, 60)


class TestStoringACurve:
    def test_points_are_sorted_clamped_and_rounded(self):
        assert normalize_curve([[255, 260], [0, -5], [127.6, 140.2]]) == [[0, 0], [128, 140], [255, 255]]

    def test_points_may_be_objects(self):
        assert normalize_curve([{"x": 0, "y": 10}, {"x": 255, "y": 255}]) == [[0, 10], [255, 255]]

    def test_a_repeated_x_keeps_the_last(self):
        assert normalize_curve([[0, 0], [100, 90], [100, 120], [255, 255]]) == [[0, 0], [100, 120], [255, 255]]

    @pytest.mark.parametrize("points", [None, [], [[0, 0]], [[0, 0], [255, 255]],
                                        [[0, 0], [100, 100], [255, 255]], "nope", [["a", 1], [2]]])
    def test_a_curve_that_changes_nothing_is_no_curve(self, points):
        assert normalize_curve(points) is None

    def test_at_most_sixteen_points(self):
        assert len(normalize_curve([[x, min(255, x + 10)] for x in range(0, 255, 5)])) == 16

    def test_normalize_keeps_curves_and_mixer_and_drops_their_zeros(self):
        clean = normalize({"curve": [[0, 0], [128, 160], [255, 255]], "curve_r": [[0, 0], [255, 255]],
                           "sat_blue": -40, "hue_red": 0, "lum_aqua": 250})
        assert clean == {"curve": [[0, 0], [128, 160], [255, 255]], "sat_blue": -40, "lum_aqua": 100}


class TestTheCurve:
    def test_through_every_point(self):
        points = [[0, 0], [64, 40], [128, 150], [255, 255]]
        table = curve_table(points)
        for x, y in points:
            assert table[round(x / 255 * 4096)] * 255 == pytest.approx(y, abs=0.6)

    def test_never_overshoots_between_rising_points(self):
        # A steep step: an ordinary spline would dip below 0.2 and rise above 0.8.
        table = curve_table([[0, 0], [100, 50], [110, 200], [255, 255]])
        assert np.all(np.diff(table) >= -1e-6)
        assert table.min() >= 0.0 and table.max() <= 1.0

    def test_flat_beyond_the_ends(self):
        table = curve_table([[30, 20], [220, 240]])
        assert table[0] == pytest.approx(20 / 255, abs=1e-3)
        assert table[-1] == pytest.approx(240 / 255, abs=1e-3)


class TestRenderingTheCurve:
    def test_lifting_the_midtones_brightens_grey(self):
        frame = frame_of(patches(GREY))
        plain = patch_means(develop_frame(frame, {}), 1)[0]
        lifted = patch_means(develop_frame(frame, {"curve": [[0, 0], [128, 170], [255, 255]]}), 1)[0]
        assert lifted.mean() == pytest.approx(170, abs=4)
        assert lifted.mean() > plain.mean() + 30

    def test_a_channel_curve_moves_its_channel_only(self):
        frame = frame_of(patches(GREY))
        r, g, b = patch_means(develop_frame(frame, {"curve_r": [[0, 0], [128, 170], [255, 255]]}), 1)[0]
        assert r > 160 and abs(g - 128) < 4 and abs(b - 128) < 4


class TestTheMixer:
    def test_desaturating_blue_greys_the_blue_and_leaves_red(self):
        frame = frame_of(patches(RED, BLUE, GREY))
        red, blue, grey = patch_means(develop_frame(frame, {"sat_blue": -100}), 3)
        assert blue.max() - blue.min() < 6
        assert red[0] - red[2] > 120
        assert grey.max() - grey.min() < 3

    def test_a_desaturated_colour_keeps_its_brightness(self):
        frame = frame_of(patches(BLUE))
        luma = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]  # noqa: E731
        before = patch_means(develop_frame(frame, {}), 1)[0]
        grey = patch_means(develop_frame(frame, {"sat_blue": -100}), 1)[0]
        # Luma is kept in display values, so a mean of the channels is the
        # check: a blue taken to grey must stay a dark grey.
        assert grey.mean() < 90
        assert abs(luma(grey) - luma(before)) < 8

    def test_darkening_blue_spares_grey_and_green(self):
        frame = frame_of(patches(BLUE, GREY, GREEN))
        before = patch_means(develop_frame(frame, {}), 3)
        blue, grey, green = patch_means(develop_frame(frame, {"lum_blue": -100}), 3)
        assert blue.mean() < before[0].mean() * 0.7
        assert abs(grey.mean() - before[1].mean()) < 2
        assert abs(green.mean() - before[2].mean()) < 4

    def test_a_hue_shift_moves_red_toward_orange(self):
        frame = frame_of(patches(RED, GREY))
        before = patch_means(develop_frame(frame, {}), 2)
        red, grey = patch_means(develop_frame(frame, {"hue_red": 100}), 2)
        hue = colorsys.rgb_to_hsv(*(red / 255.0))[0] * 360.0
        assert hue == pytest.approx(30.0, abs=4.0)   # red turned orange…
        luma = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]  # noqa: E731
        assert abs(luma(red) - luma(before[0])) < 6  # …without turning lighter
        assert abs(grey - before[1]).max() < 2

    def test_bands_blend_so_a_hue_between_two_gets_some_of_each(self):
        between = (220, 85, 40)                      # 15°: halfway from red to orange
        frame = frame_of(patches(between))
        only_red = patch_means(develop_frame(frame, {"sat_red": -100}), 1)[0]
        both = patch_means(develop_frame(frame, {"sat_red": -100, "sat_orange": -100}), 1)[0]
        chroma = lambda c: c.max() - c.min()       # noqa: E731
        assert chroma(both) < 10 < chroma(only_red) < 150


class TestKeepingIt:
    def test_the_sidecar_round_trips_both(self):
        edit = {"curve": [[0, 10], [128, 150], [255, 245]], "curve_b": [[0, 0], [255, 230]],
                "sat_blue": -30, "lum_orange": 20, "hue_green": -15}
        written = to_crs(edit)
        assert written["crs:SaturationAdjustmentBlue"] == "-30"
        assert written["crs:PhotoolToneCurve"] == "0,10 128,150 255,245"
        assert from_crs(written) == normalize(edit)

    def test_through_the_api(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        folder = "2025/2025-03-06"
        library = tmp_path / "lib"
        (library / folder).mkdir(parents=True)
        Image.fromarray(patches(RED, BLUE, GREY)).save(library / folder / "IMG_0001.JPG", quality=95)
        (tmp_path / "config.yaml").write_text(
            # Single-quoted: YAML takes the backslashes of a Windows path literally there.
            f"library_root: '{library}'\n"
            f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
            f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
            "sd_mount_paths: []\nanalysis_auto_enabled: false\n"
        )
        monkeypatch.chdir(tmp_path)
        import main

        with TestClient(main.app) as c:
            c.post(f"/api/catalog/index/{folder}")
            edit = {"curve": [[0, 0], [128, 160], [255, 255]], "sat_blue": -50}
            res = c.put("/api/photos/edit", json={"filepath": f"{folder}/IMG_0001.JPG", **edit})
            assert res.json()["settings"] == edit
            got = c.get("/api/photos/edit", params={"filepath": f"{folder}/IMG_0001.JPG"}).json()
            assert got["settings"] == edit
            preview = c.post("/api/export/preview", json={
                "filepath": f"{folder}/IMG_0001.JPG", "size": 128,
                "adjustment": edit,
            })
            assert preview.status_code == 200


class TestTheToneIsKeptBetweenRenders:
    """Dragging the curve or the mixer reuses the light-and-tone result; it must
    never show a picture that a fresh render would not."""

    def test_a_cached_render_is_the_same_picture(self):
        frame = frame_of(patches(RED, BLUE, GREY))
        edit = {"shadows": 30, "curve": [[0, 0], [128, 160], [255, 255]], "sat_blue": -40}
        develop_frame(frame, {"shadows": 30})                  # fills the cache
        cached = np.asarray(develop_frame(frame, edit), dtype=np.int16)
        fresh = np.asarray(develop_frame(frame_of(patches(RED, BLUE, GREY)), edit), dtype=np.int16)
        assert np.abs(cached - fresh).max() <= 1               # dither only

    def test_moving_a_tone_control_is_not_served_from_the_cache(self):
        frame = frame_of(patches(GREY))
        low = patch_means(develop_frame(frame, {"exposure": 0.5, "curve": [[0, 0], [128, 150], [255, 255]]}), 1)[0]
        high = patch_means(develop_frame(frame, {"exposure": 1.0, "curve": [[0, 0], [128, 150], [255, 255]]}), 1)[0]
        assert high.mean() > low.mean() + 10

    def test_the_cache_does_not_leak_into_the_next_edit(self):
        frame = frame_of(patches(GREY))
        develop_frame(frame, {"exposure": 1.0, "mono": 1})
        plain = patch_means(develop_frame(frame, {}), 1)[0]
        assert plain.mean() == pytest.approx(128, abs=3)
