"""
Black and white: the treatment, the B&W mix, toning on top of it, and the two
effects a black-and-white print leans on, grain and the vignette.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame, linear_from_image, region_frame  # noqa: E402
from catalog.database import PhotoRecord  # noqa: E402
from catalog.search import gather, search  # noqa: E402

SKY, BRICK, GRASS, GREY = (70, 120, 210), (170, 60, 45), (70, 150, 60), (128, 128, 128)


def frame_of(rgb8) -> LinearFrame:
    return LinearFrame(linear_from_image(Image.fromarray(np.asarray(rgb8, np.uint8), "RGB")))


def bands(*colours, size=40):
    """Horizontal bands of flat colour, top to bottom."""
    return np.concatenate([np.full((size, size * 3, 3), c, np.uint8) for c in colours], axis=0)


def band_means(img, count, size=40):
    a = np.asarray(img, dtype=np.float32)
    return [a[i * size + 6:(i + 1) * size - 6, 6:-6].reshape(-1, 3).mean(axis=0) for i in range(count)]


class TestTheTreatment:
    def test_mono_is_grey(self):
        out = np.asarray(develop_frame(frame_of(bands(SKY, BRICK, GRASS)), {"mono": 1}), dtype=np.int16)
        assert np.abs(out[..., 0] - out[..., 1]).max() <= 1
        assert np.abs(out[..., 1] - out[..., 2]).max() <= 1

    def test_a_neutral_conversion_keeps_brightness(self):
        frame = frame_of(bands(SKY, BRICK, GRASS, GREY))
        luma = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]  # noqa: E731
        colour = band_means(develop_frame(frame, {}), 4)
        grey = band_means(develop_frame(frame, {"mono": 1}), 4)
        for before, after in zip(colour, grey):
            assert after.mean() == pytest.approx(luma(before), abs=6)

    def test_the_colour_mixer_has_no_say_in_black_and_white(self):
        frame = frame_of(bands(SKY, BRICK))
        plain = np.asarray(develop_frame(frame, {"mono": 1}), dtype=np.int16)
        mixed = np.asarray(develop_frame(frame, {"mono": 1, "lum_blue": -100, "sat_red": 100}), dtype=np.int16)
        assert np.abs(plain - mixed).max() <= 1

    def test_switched_off_the_mix_is_kept_and_does_nothing(self):
        assert normalize({"bw_blue": -40}) == {"bw_blue": -40}
        frame = frame_of(bands(SKY))
        plain = np.asarray(develop_frame(frame, {}), dtype=np.int16)
        kept = np.asarray(develop_frame(frame, {"bw_blue": -40, "exposure": 0.0001}), dtype=np.int16)
        assert np.abs(plain - kept).max() <= 1


class TestTheMix:
    def test_a_red_filter_darkens_the_sky_and_lightens_the_brick(self):
        frame = frame_of(bands(SKY, BRICK, GREY))
        sky, brick, grey = band_means(develop_frame(frame, {"mono": 1}), 3)
        f_sky, f_brick, f_grey = band_means(
            develop_frame(frame, {"mono": 1, "bw_blue": -60, "bw_aqua": -30, "bw_red": 40, "bw_orange": 30}), 3)
        assert f_sky.mean() < sky.mean() - 25
        assert f_brick.mean() > brick.mean() + 20
        assert abs(f_grey.mean() - grey.mean()) < 1.5

    def test_a_green_filter_lightens_foliage(self):
        frame = frame_of(bands(GRASS))
        plain = band_means(develop_frame(frame, {"mono": 1}), 1)[0]
        green = band_means(develop_frame(frame, {"mono": 1, "bw_green": 50}), 1)[0]
        assert green.mean() > plain.mean() + 20


class TestToning:
    def test_split_toning_tints_the_grey(self):
        frame = frame_of(bands(GREY, (60, 60, 60)))
        out = np.asarray(develop_frame(frame, {"mono": 1, "shadow_hue": 35, "shadow_sat": 40}), dtype=np.float32)
        assert (out[..., 0] - out[..., 2]).mean() > 8          # warm: sepia

    def test_a_curve_shapes_the_grey_as_drawn(self):
        frame = frame_of(bands(GREY))
        out = band_means(develop_frame(frame, {"mono": 1, "curve": [[0, 0], [128, 170], [255, 255]]}), 1)[0]
        assert out.mean() == pytest.approx(170, abs=4)


class TestGrain:
    def test_a_thumbnail_shows_grain_as_texture_not_noise(self):
        small = frame_of(np.full((150, 200, 3), 128, np.uint8))
        big = frame_of(np.full((800, 2000, 3), 128, np.uint8))
        on_small = np.asarray(develop_frame(small, {"grain": 80}), dtype=np.float32).std()
        on_big = np.asarray(develop_frame(big, {"grain": 80}), dtype=np.float32).std()
        assert on_small < on_big / 2

    def test_the_same_grains_every_time(self):
        frame = frame_of(bands(GREY, GREY))
        a = np.asarray(develop_frame(frame, {"grain": 50}))
        b = np.asarray(develop_frame(frame, {"grain": 50}))
        assert np.array_equal(a, b)

    # Grain is measured on frames of a real size: on a thumbnail it is averaged
    # out on purpose, as it would be on a small print.
    def test_more_grain_is_more_visible(self):
        frame = frame_of(np.full((800, 2000, 3), 128, np.uint8))
        light = np.asarray(develop_frame(frame, {"grain": 20}), dtype=np.float32).std()
        heavy = np.asarray(develop_frame(frame, {"grain": 80}), dtype=np.float32).std()
        assert heavy > light * 3 > 0

    def test_grain_is_strongest_in_the_midtones(self):
        frame = frame_of(np.concatenate([np.full((300, 2000, 3), v, np.uint8) for v in (128, 12)]))
        out = np.asarray(develop_frame(frame, {"grain": 80}), dtype=np.float32)
        mid, dark = out[20:280], out[320:580]
        assert mid.std() > dark.std() * 1.5

    def test_grain_has_the_same_strength_at_any_size(self):
        """A coarse grain, enlarged from the same field, must not fade."""
        frame = frame_of(np.full((800, 2400, 3), 128, np.uint8))
        fine = np.asarray(develop_frame(frame, {"grain": 60, "grain_size": -50}), dtype=np.float32).std()
        coarse = np.asarray(develop_frame(frame, {"grain": 60, "grain_size": 100}), dtype=np.float32).std()
        assert coarse == pytest.approx(fine, rel=0.35)

    def test_a_loupe_patch_has_the_grain_size_of_the_whole_at_full_size(self):
        """A 1:1 patch cut from a big frame shows grains as big as the big frame
        has them, not as big as a picture the patch's size would."""
        rgb = np.full((1200, 1800, 3), 128, np.uint8)
        whole = frame_of(rgb)
        big = np.asarray(develop_frame(whole, {"grain": 60}), dtype=np.float32)[..., 0]
        patch = region_frame(whole, whole.rgb[400:640, 600:840].copy(),
                             (600 / 1800, 400 / 1200, 840 / 1800, 640 / 1200))
        small = np.asarray(develop_frame(patch, {"grain": 60}), dtype=np.float32)[..., 0]
        # The neighbour-to-neighbour correlation says how many pixels a grain spans.
        corr = lambda a: np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]  # noqa: E731
        assert corr(small) == pytest.approx(corr(big[400:640, 600:840]), abs=0.12)
        # …and it is the same grain, in the same place.
        assert np.corrcoef(small.ravel(), big[400:640, 600:840].ravel())[0, 1] > 0.8


class TestVignette:
    def test_negative_darkens_the_corners_not_the_centre(self):
        frame = frame_of(np.full((200, 300, 3), 160, np.uint8))
        out = np.asarray(develop_frame(frame, {"vignette": -70}), dtype=np.float32)
        assert out[:15, :15].mean() < 115
        assert out[90:110, 140:160].mean() == pytest.approx(160, abs=2)

    def test_positive_lightens_the_corners(self):
        frame = frame_of(np.full((200, 300, 3), 100, np.uint8))
        out = np.asarray(develop_frame(frame, {"vignette": 70}), dtype=np.float32)
        assert out[:15, :15].mean() > 150

    def test_it_follows_the_crop(self):
        frame = frame_of(np.full((200, 300, 3), 160, np.uint8))
        out = np.asarray(develop_frame(frame, {"vignette": -70, "crop_x": 0.25, "crop_y": 0.25,
                                               "crop_w": 0.5, "crop_h": 0.5}), dtype=np.float32)
        h, w = out.shape[:2]
        assert out[:6, :6].mean() < 120                        # the crop's own corner
        assert out[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5].mean() == pytest.approx(160, abs=2)


class TestKeepingIt:
    def test_the_sidecar_carries_lightrooms_names(self):
        edit = {"mono": 1, "bw_blue": -60, "bw_red": 40, "grain": 30, "grain_size": -20,
                "vignette": -25, "shadow_hue": 35, "shadow_sat": 20}
        written = to_crs(edit)
        assert written["crs:ConvertToGrayscale"] == "True"
        assert written["crs:GrayMixerBlue"] == "-60"
        assert written["crs:GrainAmount"] == "30"
        assert written["crs:PostCropVignetteAmount"] == "-25"
        assert from_crs(written) == normalize(edit)

    def test_a_grain_size_without_grain_is_no_edit(self):
        assert normalize({"grain_size": 50}) == {}

    def test_search_finds_black_and_white_edits(self):
        def rec(name):
            return PhotoRecord(filepath=f"2026/2026-01-01/{name}", filename=name, folder="2026/2026-01-01",
                               extension=".jpg", base_name=name[:-4])
        photos = gather([rec("A.JPG"), rec("B.JPG"), rec("C.JPG")], {},
                        {"2026/2026-01-01/A.JPG", "2026/2026-01-01/B.JPG"}, set(), {"2026/2026-01-01/A.JPG"})
        assert [p.record.filename for p in search(photos, "is:bw")] == ["A.JPG"]
        assert sorted(p.record.filename for p in search(photos, "is:edited -is:bw")) == ["B.JPG"]

    def test_search_finds_denoised_edits(self):
        def rec(name):
            return PhotoRecord(filepath=f"2026/2026-01-01/{name}", filename=name, folder="2026/2026-01-01",
                               extension=".cr3", base_name=name[:-4])
        photos = gather([rec("A.CR3"), rec("B.CR3")], {}, {"2026/2026-01-01/A.CR3", "2026/2026-01-01/B.CR3"},
                        set(), set(), {"2026/2026-01-01/B.CR3"})
        assert [p.record.filename for p in search(photos, "is:denoised")] == ["B.CR3"]
        assert [p.record.filename for p in search(photos, "is:edited -is:nr")] == ["A.CR3"]
