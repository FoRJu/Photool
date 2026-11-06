"""A second exposure: how it is stored, placed and mixed, and what happens when its file goes."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import double  # noqa: E402
from adjust.develop import OWN_KEYS, from_crs, normalize, to_crs  # noqa: E402
from adjust.render import LinearFrame, develop_frame  # noqa: E402

WHOLE = (0.0, 0.0, 1.0, 1.0)


def second(**over):
    return double.normalize_second({"path": "a/two.jpg", **over})


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A library with one bright photograph at a/two.jpg."""
    (tmp_path / "a").mkdir()
    Image.new("RGB", (120, 80), (200, 200, 200)).save(tmp_path / "a" / "two.jpg")
    monkeypatch.setattr(double, "_library_root", None)
    monkeypatch.setattr(double, "_pictures", type(double._pictures)())
    monkeypatch.setattr(double, "_warned", set())
    double.configure(str(tmp_path))
    return tmp_path


class TestStored:
    def test_values_are_brought_into_range(self):
        s = second(mode="nonsense", amount=999, exposure=-9, x=500, scale=1, reach="nonsense", invert=True)
        assert s["mode"] == "add" and s["amount"] == 200 and s["exposure"] == -3.0
        assert s["x"] == 100 and s["scale"] == 100 and s["reach"] == "" and s["invert"] is False

    @pytest.mark.parametrize("path", ["", "   ", "/etc/passwd", "../outside.jpg", "a/../../outside.jpg",
                                      "C:/Windows/win.ini", "..\\outside.jpg", None, 7])
    def test_a_path_must_stay_under_the_library(self, path):
        assert double.normalize_second({"path": path}) is None

    def test_it_survives_the_sidecar(self):
        edit = normalize({"second": second(mode="lighten", amount=70, exposure=-1.5, x=-20, y=10, scale=150,
                                           flip=True, reach="people", invert=True)})
        assert from_crs(to_crs(edit)) == edit

    def test_it_stays_with_its_own_frame(self):
        assert "second" in OWN_KEYS

    def test_bad_text_is_no_second_exposure(self):
        assert double.normalize_second("{not json") is None
        assert "second" not in normalize({"second": 5})


class TestPlaced:
    def test_it_covers_a_frame_of_another_shape(self):
        picture = np.ones((40, 60, 3), np.float32)
        rgb, cover = double.place(picture, (200, 100), WHOLE, second())
        assert rgb.shape == (200, 100, 3) and cover.min() == 1.0

    def test_moved_it_leaves_part_of_the_frame_uncovered(self):
        rgb, cover = double.place(np.ones((100, 100, 3), np.float32), (100, 100), WHOLE, second(x=50))
        assert cover[:, :40].max() == 0.0 and cover[:, 60:].min() == 1.0

    def test_a_mirror_turns_it_round(self):
        picture = np.zeros((100, 100, 3), np.float32)
        picture[:, :50] = 1.0
        rgb, _ = double.place(picture, (100, 100), WHOLE, second(flip=True))
        assert rgb[50, 90, 0] == 1.0 and rgb[50, 10, 0] == 0.0

    def test_scale_enlarges_about_the_centre(self):
        picture = np.zeros((100, 100, 3), np.float32)
        picture[40:60, 40:60] = 1.0
        rgb, _ = double.place(picture, (100, 100), WHOLE, second(scale=200))
        assert rgb[50, 50, 0] == 1.0 and rgb[50, 75, 0] == 0.0 and rgb[50, 65, 0] == 1.0

    def test_a_part_of_the_frame_gets_its_part_of_the_second(self):
        """The loupe's patch has to line up with the preview it is a part of."""
        ramp = np.tile(np.linspace(0, 1, 400, dtype=np.float32)[None, :, None], (400, 1, 3))
        whole, _ = double.place(ramp, (400, 400), WHOLE, second(x=10, scale=130))
        part, _ = double.place(ramp, (100, 100), (0.5, 0.25, 0.75, 0.5), second(x=10, scale=130))
        assert np.abs(part - whole[100:200, 200:300]).max() < 0.01


class TestMixed:
    A = np.full((4, 4, 3), 0.2, np.float32)
    B = np.full((4, 4, 3), 0.4, np.float32)
    ALL = np.ones((4, 4), np.float32)

    @pytest.mark.parametrize("mode,want", [("add", 0.6), ("average", 0.3), ("lighten", 0.4), ("darken", 0.2)])
    def test_the_four_modes(self, mode, want):
        assert double.mix(self.A, self.B, self.ALL, second(mode=mode))[0, 0, 0] == pytest.approx(want)

    def test_amount_and_exposure_scale_the_second(self):
        out = double.mix(self.A, self.B, self.ALL, second(amount=50, exposure=1.0))
        assert out[0, 0, 0] == pytest.approx(0.2 + 0.4 * 2 * 0.5)

    @pytest.mark.parametrize("mode", double.MODES)
    def test_where_it_does_not_reach_the_first_is_untouched(self, mode):
        out = double.mix(self.A, self.B, np.zeros((4, 4), np.float32), second(mode=mode))
        assert np.array_equal(out, self.A)

    def test_lighten_and_darken_make_no_new_colours(self):
        red = np.full((1, 1, 3), [0.6, 0.0, 0.0], np.float32)
        green = np.full((1, 1, 3), [0.0, 0.6, 0.0], np.float32)
        one = np.ones((1, 1), np.float32)
        assert np.array_equal(double.mix(red, green, one, second(mode="lighten"))[0, 0], green[0, 0])
        assert np.array_equal(double.mix(red, green, one, second(mode="darken"))[0, 0], red[0, 0])


class TestThroughThePipeline:
    def test_it_brightens_the_frame(self, library):
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        before = np.asarray(develop_frame(frame, {"exposure": 0.01}), np.float32).mean()
        after = np.asarray(develop_frame(frame, {"exposure": 0.01, "second": second()}), np.float32).mean()
        assert after > before + 20

    def test_with_nothing_else_in_the_edit(self, library):
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        assert np.asarray(develop_frame(frame, {"second": second()})).mean() > \
            np.asarray(develop_frame(frame, {})).mean() + 20

    def test_a_file_gone_draws_the_frame_without_it(self, library):
        (library / "a" / "two.jpg").unlink()
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        s = second()
        assert double.missing(s)
        assert np.array_equal(np.asarray(develop_frame(frame, {"exposure": 0.2, "second": s})),
                              np.asarray(develop_frame(frame, {"exposure": 0.2})))

    def test_a_path_leading_out_of_the_library_is_not_read(self, library):
        (library.parent / "outside.jpg").write_bytes(b"x")
        assert double._file_of("a/../../outside.jpg") is None

    def test_kept_for_the_same_values(self, library):
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        assert double.exposed(frame, second()) is double.exposed(frame, second())
        assert double.exposed(frame, second()) is not double.exposed(frame, second(amount=50))

    def test_a_kept_picture_is_used_again(self, library, monkeypatch):
        import export.engine as engine

        reads = []
        real = engine._load_linear
        monkeypatch.setattr(engine, "_load_linear", lambda *a: reads.append(a) or real(*a))
        for amount in (100, 50):
            double.exposed(LinearFrame(np.full((60, 90, 3), 0.05, np.float32)), second(amount=amount))
        assert len(reads) == 1

    def test_a_model_that_cannot_be_had_leaves_the_frame_alone(self, library, monkeypatch):
        import adjust.render as render

        monkeypatch.setattr(render, "mask_reach", lambda frame, m: None)
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        assert double.exposed(frame, second(reach="people")) is frame


class TestAsEdited:
    """A second that names one of its versions comes finished by that version's own edit."""

    @pytest.fixture
    def red(self, library):
        Image.new("RGB", (120, 80), (200, 40, 40)).save(library / "a" / "red.jpg")
        return library

    @staticmethod
    def colour(picture) -> float:
        rgb = np.asarray(picture, np.float32)
        return float((rgb[..., 0] - rgb[..., 1]).mean())

    def test_it_keeps_its_version_and_a_clean_copy_of_its_edit(self):
        s = second(version=2, edit={"exposure": 1.5, "bogus": 3, "second": {"path": "a/three.jpg"}})
        assert s["version"] == 2 and s["edit"] == {"exposure": 1.5}
        assert double.developed(s) and not double.developed(second())

    @pytest.mark.parametrize("version", [-1, "2", True, float("nan")])
    def test_a_bad_version_is_as_shot(self, version):
        s = second(version=version, edit={"exposure": 1.0})
        assert "version" not in s and "edit" not in s

    def test_it_survives_the_sidecar(self):
        edit = normalize({"second": second(version=1, edit={"mono": True, "contrast": 20}, mode="average")})
        assert from_crs(to_crs(edit)) == edit

    def test_a_black_and_white_copy_stays_black_and_white(self, red):
        frame = LinearFrame(np.full((60, 90, 3), 0.02, np.float32))
        shot = develop_frame(frame, {"second": second(path="a/red.jpg")})
        edited = develop_frame(frame, {"second": second(path="a/red.jpg", version=1, edit={"mono": True})})
        assert self.colour(shot) > 40 and abs(self.colour(edited)) < 3

    def test_the_first_photo_edit_does_not_reach_it(self, red):
        """A black-and-white first frame greys a second as shot, not one as edited."""
        frame = LinearFrame(np.full((60, 90, 3), 0.02, np.float32))
        shot = develop_frame(frame, {"mono": True, "second": second(path="a/red.jpg")})
        edited = develop_frame(frame, {"mono": True, "second": second(path="a/red.jpg", version=0, edit={})})
        assert abs(self.colour(shot)) < 3 and self.colour(edited) > 40

    def test_with_nothing_else_in_the_edit_it_brightens_the_frame(self, library):
        frame = LinearFrame(np.full((60, 90, 3), 0.02, np.float32))
        assert np.asarray(develop_frame(frame, {"second": second(version=0)}), np.float32).mean() > \
            np.asarray(develop_frame(frame, {}), np.float32).mean() + 20

    def test_its_edit_changes_it(self, library):
        frame = LinearFrame(np.full((60, 90, 3), 0.02, np.float32))
        dark = develop_frame(frame, {"second": second(version=1, edit={"exposure": -3.0})})
        light = develop_frame(frame, {"second": second(version=1, edit={})})
        assert np.asarray(light, np.float32).mean() > np.asarray(dark, np.float32).mean() + 20

    def test_a_file_gone_draws_the_frame_without_it(self, library):
        (library / "a" / "two.jpg").unlink()
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        assert np.array_equal(np.asarray(develop_frame(frame, {"exposure": 0.2, "second": second(version=0)})),
                              np.asarray(develop_frame(frame, {"exposure": 0.2})))

    def test_kept_once_finished(self, library, monkeypatch):
        import adjust.render as render

        runs = []
        real = render.develop_frame
        monkeypatch.setattr(render, "develop_frame", lambda f, s, *a, **k: runs.append(s) or real(f, s, *a, **k))
        frame = LinearFrame(np.full((60, 90, 3), 0.05, np.float32))
        for amount in (100, 50):
            double.laid_on(frame, np.full((60, 90, 3), 0.2, np.float32), second(version=0, amount=amount))
        assert len(runs) == 1
