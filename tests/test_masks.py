"""
Local adjustments (adjust.masks, adjust.render): a gradient changes the part of
the picture it covers, by the same measure as the global slider.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import from_crs, normalize, to_crs  # noqa: E402
from adjust.masks import normalize_mask, normalize_masks, weight  # noqa: E402
from adjust.render import LUMA, LinearFrame, develop_frame, region_frame  # noqa: E402

TOP_DOWN = {"kind": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.5}


def _grey(h=60, w=90, level=0.09):
    return LinearFrame(np.full((h, w, 3), level, np.float32))


def _luma(img):
    return np.asarray(img, np.float32) @ LUMA


class TestWeight:
    def test_a_linear_gradient_is_whole_at_its_start_and_gone_at_its_end(self):
        w = weight(TOP_DOWN, (100, 100))
        assert w[0, 50] > 0.99 and w[49, 50] < 0.01 and w[99, 50] == 0.0
        assert 0.4 < w[25, 50] < 0.6
        assert np.allclose(w[:, 0], w[:, 99])                  # the same across

    def test_a_radial_is_whole_inside_and_fades_over_its_feather(self):
        mask = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.4, "angle": 0,
                "feather": 50, "invert": False}
        w = weight(mask, (101, 101))
        assert w[50, 50] == 1.0 and w[50, 50 + 15] == 1.0      # inside 1 - feather
        assert w[50, 50 + 41] == 0.0                           # past the edge
        assert 0.0 < w[50, 50 + 30] < 1.0
        inverted = weight({**mask, "invert": True}, (101, 101))
        assert np.allclose(inverted, 1.0 - w)

    def test_an_inverted_linear_gradient_is_turned_round(self):
        w = weight({**TOP_DOWN, "invert": True}, (100, 100))
        assert np.allclose(w, 1.0 - weight(TOP_DOWN, (100, 100)), atol=1e-6)
        assert w[0, 50] < 0.01 and w[99, 50] == 1.0

    def test_a_round_ellipse_is_round_on_a_wide_frame(self):
        # rx is a share of the width and ry of the height: on a 2:1 frame a
        # circle has ry twice rx.
        mask = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.1, "ry": 0.2, "feather": 0}
        w = weight(normalize_masks([mask])[0], (100, 200))
        across = int(w[50].sum())
        down = int(w[:, 100].sum())
        assert abs(across - down) <= 2

    def test_a_patch_gets_its_own_part_of_the_gradient(self):
        whole = weight(TOP_DOWN, (100, 100))
        patch = weight(TOP_DOWN, (50, 50), window=(0.5, 0.0, 1.0, 0.5))
        assert np.allclose(patch, whole[:50, 50:], atol=1e-5)


ACROSS = {"points": [[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]], "size": 0.05, "feather": 0}


class TestBrush:
    def test_a_stroke_covers_its_path_out_to_its_radius(self):
        w = weight({"kind": "brush", "strokes": [ACROSS]}, (100, 100))
        assert w[50, 50] == 1.0 and w[50, 12] == 1.0 and w[50, 88] == 1.0
        assert w[50 + 4, 50] == 1.0 and w[50 + 6, 50] == 0.0   # radius 5 px
        assert w[10, 50] == 0.0 and w[50, 2] == 0.0
        assert weight({"kind": "brush"}, (40, 40)).max() == 0.0  # nothing painted

    def test_a_soft_stroke_fades_and_flow_lays_down_less(self):
        soft = {**ACROSS, "size": 0.2, "feather": 100}
        w = weight({"kind": "brush", "strokes": [soft]}, (100, 100))
        assert w[50, 50] > 0.99 and 0.2 < w[60, 50] < 0.8 and w[71, 50] == 0.0
        light = weight({"kind": "brush", "strokes": [{**ACROSS, "flow": 40}]}, (100, 100))
        assert abs(light[50, 50] - 0.4) < 1e-5
        twice = weight({"kind": "brush", "strokes": [{**ACROSS, "flow": 40}] * 2}, (100, 100))
        assert abs(twice[50, 50] - 0.64) < 1e-5                # paint over paint

    def test_erasing_takes_away_from_a_brush_and_from_a_radial(self):
        dot = {"points": [[0.5, 0.5]], "size": 0.1, "feather": 0}
        brush = {"kind": "brush", "strokes": [ACROSS, {**dot, "erase": True}]}
        w = weight(brush, (100, 100))
        assert w[50, 50] == 0.0 and w[50, 20] == 1.0
        radial = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.4, "angle": 0,
                  "feather": 0, "invert": False, "strokes": [{**dot, "erase": True}]}
        w = weight(radial, (100, 100))
        assert w[50, 50] == 0.0 and w[50, 75] == 1.0

    def test_a_patch_gets_its_own_part_of_the_stroke(self):
        mask = {"kind": "brush", "strokes": [{**ACROSS, "feather": 60}]}
        whole = weight(mask, (100, 100))
        patch = weight(mask, (50, 50), window=(0.5, 0.25, 1.0, 0.75))
        assert np.allclose(patch, whole[25:75, 50:], atol=1e-4)

    def test_a_brush_brightens_only_where_it_was_painted(self):
        frame = _grey()
        out = _luma(develop_frame(frame, {"masks": [{"kind": "brush", "exposure": 1.0,
                                                     "strokes": [ACROSS]}]}))
        plain = _luma(develop_frame(frame, {}))
        assert out[30, 45] > plain[30, 45] + 10
        assert abs(out[5, 45] - plain[5, 45]) < 1.5             # dither aside

    def test_a_brush_with_nothing_painted_changes_nothing(self):
        frame = _grey()
        assert np.array_equal(np.asarray(develop_frame(frame, {"masks": [{"kind": "brush", "exposure": 2}]})),
                              np.asarray(develop_frame(frame, {})))

    def test_an_inverted_brush_is_everything_but_its_paint(self):
        dot = {"points": [[0.5, 0.5]], "size": 0.1, "feather": 0}
        mask = {"kind": "brush", "invert": True, "strokes": [ACROSS, {**dot, "erase": True}]}
        w = weight(mask, (100, 100))
        assert np.allclose(w, 1.0 - weight({**mask, "invert": False}, (100, 100)))
        assert w[10, 50] == 1.0 and w[50, 20] == 0.0 and w[50, 50] == 1.0
        assert weight({"kind": "brush", "invert": True}, (40, 40)).min() == 1.0

    def test_an_inverted_brush_changes_all_but_where_it_was_painted(self):
        frame = _grey()
        plain = _luma(develop_frame(frame, {}))
        out = _luma(develop_frame(frame, {"masks": [{"kind": "brush", "invert": True, "exposure": 1.0,
                                                     "strokes": [ACROSS]}]}))
        assert out[5, 45] > plain[5, 45] + 10
        assert abs(out[30, 45] - plain[30, 45]) < 1.5
        # With nothing painted it is the whole frame.
        whole = _luma(develop_frame(frame, {"masks": [{"kind": "brush", "invert": True, "exposure": 1.0}]}))
        assert whole.min() > plain.max() + 10

    def test_invert_round_trips_on_a_brush_and_a_linear(self):
        clean = normalize({"masks": [{"kind": "brush", "invert": 1, "exposure": 1},
                                     {**TOP_DOWN, "invert": True, "exposure": 1},
                                     {"kind": "brush", "invert": False, "exposure": 1}]})
        assert [m.get("invert") for m in clean["masks"]] == [True, True, None]
        assert from_crs(to_crs(clean)) == clean

    def test_strokes_are_cleaned_thinned_and_round_trip(self):
        long = {"points": [[i / 2000, 0.5] for i in range(2000)] + ["x", [1, "y"]],
                "size": 9, "feather": 140, "flow": 0, "erase": 1}
        clean = normalize({"masks": [{"kind": "brush", "shadows": 30, "strokes": [long, {"points": []}]}]})
        (mask,) = clean["masks"]
        (stroke,) = mask["strokes"]
        assert len(stroke["points"]) <= 601 and stroke["points"][-1] == [0.9995, 0.5]
        assert (stroke["size"], stroke["feather"], stroke["flow"], stroke["erase"]) == (0.5, 100, 1, True)
        assert from_crs(to_crs(clean)) == clean

    def test_the_brush_goes_down_to_a_hair(self):
        # The smallest brush is 0.05% of the longer side: a few pixels at full
        # size, enough to correct one edge by hand.
        dot = {"points": [[0.5, 0.5]], "size": 0.0001, "feather": 0}
        clean = normalize({"masks": [{"kind": "brush", "strokes": [dot]}]})
        assert clean["masks"][0]["strokes"][0]["size"] == 0.0005
        assert normalize_mask({"kind": "brush", "strokes": [{**dot, "size": 0.0007}]}
                              )["strokes"][0]["size"] == 0.0007
        assert normalize_mask({"kind": "brush", "strokes": [{**dot, "size": 9}]}
                              )["strokes"][0]["size"] == 0.5          # the widest is unchanged


class TestSection:
    """A mask's section (crop): the rectangle it reaches inside, and nowhere
    else, whatever its shape, strokes or inversion said."""

    TOP_HALF = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5, "feather": 0}

    def test_a_section_crops_the_mask_hard(self):
        base = weight(TOP_DOWN, (100, 100))
        w = weight(normalize_mask({**TOP_DOWN, "section": self.TOP_HALF}), (100, 100))
        assert w[50:].max() == 0.0
        assert np.allclose(w[:50], base[:50], atol=1e-6)       # whole inside, unchanged

    def test_a_feathered_section_fades_inwards(self):
        # feather 50 fades over the nearer quarter of the shorter side from each
        # edge, so the middle half of the section is a plateau.
        sec = {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0, "feather": 50}
        # Turned round, the gradient is whole along the middle row; the section's
        # vertical edges fade there no more, so the row shows its horizontal fade.
        w = weight(normalize_mask({**TOP_DOWN, "invert": True, "section": sec}), (100, 100))
        row = w[50]
        assert row[0] < 0.01 and row[12] > 0.99
        assert np.all(np.diff(row[:12]) >= 0)                  # fading in from the left edge
        assert np.allclose(row[12:38], 1.0, atol=1e-4)         # the plateau in the middle
        assert np.all(np.diff(row[37:50]) <= 0)                # and out towards the right one
        assert row[50:].max() == 0.0

    def test_the_softest_section_still_reaches_whole_in_the_middle(self):
        # feather 100 fades from each edge to the middle, as a radial's does:
        # its centre is whole, not dimmed.
        sec = {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0, "feather": 100}
        row = weight(normalize_mask({**TOP_DOWN, "invert": True, "section": sec}), (100, 100))[50]
        assert row[0] < 0.01 and abs(row[25] - 1.0) < 0.01
        assert row[50:].max() == 0.0

    def test_a_section_is_clamped_swapped_and_round_trips(self):
        clean = normalize({"masks": [{**TOP_DOWN, "section": {
            "x1": 1.4, "y1": -0.2, "x2": 0.3, "y2": 0.7, "feather": 140}}]})
        assert clean["masks"][0]["section"] == {"x1": 0.3, "y1": 0.0, "x2": 1.0, "y2": 0.7,
                                                "feather": 100}
        assert from_crs(to_crs(clean)) == clean
        tiny = normalize_mask({**TOP_DOWN, "section": {
            "x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5}})
        assert tiny["section"] == {"x1": 0.5, "y1": 0.5, "x2": 0.51, "y2": 0.51, "feather": 0}
        assert from_crs(to_crs(normalize({"masks": [tiny]}))) == normalize({"masks": [tiny]})

    def test_a_whole_frame_section_is_kept_switched_on(self):
        # Checking "Section" starts as the whole frame; that is a section turned
        # on and waiting to be dragged, so it must survive the save.
        clean = normalize({"masks": [{**TOP_DOWN, "section": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}]})
        assert clean["masks"][0]["section"] == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
                                                "feather": 0}
        assert from_crs(to_crs(clean)) == clean
        # No section given at all is no section key.
        assert "section" not in normalize_mask(TOP_DOWN)
        assert "section" not in normalize_mask({**TOP_DOWN, "section": None})
        assert "section" not in normalize_mask({**TOP_DOWN, "section": "whole"})

    def test_a_section_crops_every_kind(self):
        radial = {"kind": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.4, "angle": 30,
                  "feather": 0, "invert": False}
        right = {"x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0, "feather": 0}
        for shape in (TOP_DOWN, radial):
            for invert in (False, True):
                w = weight(normalize_mask({**shape, "invert": invert, "section": right}), (100, 100))
                assert w[:, :50].max() == 0.0
                assert w[:, 50:].max() > 0.5
        # An inverted brush: outside its section nothing, even where painted.
        dot = {"points": [[0.5, 0.5]], "size": 0.1, "feather": 0}
        brush = normalize_mask({"kind": "brush", "invert": True, "strokes": [dot],
                                "section": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5}})
        w = weight(brush, (100, 100))
        assert w[50:, :].max() == 0.0 and w[:, 50:].max() == 0.0
        assert w[5, 5] == 1.0 and w[50, 50] == 0.0

    def test_a_patch_gets_its_own_part_of_the_section(self):
        mask = normalize_mask({**TOP_DOWN, "section": {
            "x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 0.9, "feather": 40}})
        whole = weight(mask, (100, 100))
        patch = weight(mask, (50, 50), window=(0.5, 0.0, 1.0, 0.5))
        assert np.allclose(patch, whole[:50, 50:], atol=1e-5)

    def test_a_section_leaves_the_rest_of_the_picture_alone(self):
        frame = _grey()
        plain = _luma(develop_frame(frame, {}))
        got = _luma(develop_frame(frame, {"masks": [{**TOP_DOWN, "exposure": 1.0,
                                                     "section": dict(self.TOP_HALF)}]}))
        assert got[:5].mean() > plain[:5].mean() + 10
        assert np.all(np.abs(got[40:] - plain[40:]) < 1.5)     # dither aside

    def test_moving_the_section_moves_what_the_render_reaches(self):
        frame = _grey()
        top = _luma(develop_frame(frame, {"masks": [{**TOP_DOWN, "exposure": 1.0,
                                                     "section": dict(self.TOP_HALF)}]}))
        bottom = _luma(develop_frame(frame, {"masks": [{**TOP_DOWN, "exposure": 1.0,
                                                       "section": {"x1": 0.0, "y1": 0.5,
                                                                   "x2": 1.0, "y2": 1.0}}]}))
        assert top[:5].mean() > bottom[:5].mean() + 10


class TestRender:
    def test_plus_one_stop_where_the_gradient_is_whole_and_nothing_past_it(self):
        frame = _grey()
        plain = _luma(develop_frame(frame, {"exposure": 0.0, "masks": [dict(TOP_DOWN)]}))
        brighter = develop_frame(frame, {"masks": [{**TOP_DOWN, "exposure": 1.0}]})
        stop = _luma(develop_frame(frame, {"exposure": 1.0}))
        got = _luma(brighter)
        assert abs(got[0].mean() - stop[0].mean()) < 1.5          # the same stop as the slider
        assert abs(got[-1].mean() - plain[-1].mean()) < 0.5      # untouched below the band
        assert got[0].mean() > got[20].mean() > got[-1].mean()   # and a fade between

    def test_a_warmer_gradient_warms_only_its_side(self):
        frame = _grey()
        out = np.asarray(develop_frame(frame, {"masks": [{**TOP_DOWN, "temperature": 80}]}), np.float32)
        top, bottom = out[0].mean(axis=0), out[-1].mean(axis=0)
        assert top[0] - top[2] > 10 and abs(bottom[0] - bottom[2]) < 2

    def test_tone_and_colour_controls_act_on_their_side(self):
        rng = np.random.default_rng(1)
        rgb = (0.02 + 0.6 * rng.random((60, 90, 3))).astype(np.float32)
        frame = LinearFrame(rgb)
        plain = np.asarray(develop_frame(frame, {"masks": [dict(TOP_DOWN)]}), np.float32)
        grey = np.asarray(develop_frame(frame, {"masks": [{**TOP_DOWN, "saturation": -100}]}), np.float32)
        spread = lambda a: (a.max(axis=2) - a.min(axis=2)).mean()  # noqa: E731
        assert spread(grey[:3]) < 0.2 * spread(plain[:3])
        assert abs(spread(grey[-10:]) - spread(plain[-10:])) < 1.0
        punchy = np.asarray(develop_frame(frame, {"masks": [{**TOP_DOWN, "contrast": 100}]}), np.float32)
        assert punchy[:3].std() > plain[:3].std() * 1.05
        assert abs(punchy[-10:].std() - plain[-10:].std()) < 0.5

    def test_a_gradient_saturates_on_top_of_the_global_slider(self):
        # The whole picture grey, the gradient's side still in colour: the two
        # add, rather than the global slider's zero wiping out the gradient's.
        rng = np.random.default_rng(3)
        frame = LinearFrame((0.02 + 0.6 * rng.random((60, 90, 3))).astype(np.float32))
        spread = lambda a: (np.asarray(a, np.float32).max(axis=2)  # noqa: E731
                            - np.asarray(a, np.float32).min(axis=2)).mean()
        plain = np.asarray(develop_frame(frame, {}), np.float32)
        pop = np.asarray(develop_frame(frame, {"saturation": -100,
                                               "masks": [{**TOP_DOWN, "saturation": 100}]}), np.float32)
        assert spread(pop[-10:]) < 0.05 * spread(plain[-10:])          # grey below
        assert abs(spread(pop[:3]) - spread(plain[:3])) < 0.05 * spread(plain[:3])
        half = np.asarray(develop_frame(frame, {"saturation": -50,
                                                "masks": [{**TOP_DOWN, "saturation": 50}]}), np.float32)
        assert np.abs(half[:2] - plain[:2]).max() <= 2                 # they cancel
        # Past +100: a net gain over a grey picture.
        net = np.asarray(develop_frame(frame, {"saturation": -100,
                                               "masks": [{**TOP_DOWN, "saturation": 200}]}), np.float32)
        assert spread(net[:3]) > 1.3 * spread(plain[:3])
        assert normalize({"masks": [{**TOP_DOWN, "saturation": 500}]})["masks"][0]["saturation"] == 200

    def test_a_gradient_with_nothing_set_changes_nothing(self):
        frame = _grey()
        assert np.array_equal(np.asarray(develop_frame(frame, {"masks": [dict(TOP_DOWN)]})),
                              np.asarray(develop_frame(frame, {})))

    def test_the_loupe_patch_agrees_with_the_preview(self):
        rng = np.random.default_rng(2)
        rgb = (0.05 + 0.3 * rng.random((80, 80, 3))).astype(np.float32)
        frame = LinearFrame(rgb)
        edit = {"masks": [{**TOP_DOWN, "exposure": 1.5}]}
        whole = _luma(develop_frame(frame, edit))
        patch = region_frame(frame, rgb[:40, 40:].copy(), (0.5, 0.0, 1.0, 0.5))
        part = _luma(develop_frame(patch, edit))
        assert abs(part.mean() - whole[:40, 40:].mean()) < 2.0


class TestEditModel:
    def test_unknown_fields_and_kinds_are_dropped_and_values_clamped(self):
        clean = normalize({"masks": [
            {**TOP_DOWN, "exposure": 9, "shadows": 250, "colour": "red"},
            {"kind": "lasso"}, "nonsense",
        ]})
        assert clean["masks"] == [{"kind": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.5,
                                   "exposure": 4.0, "shadows": 100}]

    def test_gradients_round_trip_through_the_sidecar(self):
        edit = normalize({"exposure": 0.5, "masks": [
            {**TOP_DOWN, "exposure": -1.2, "temperature": -30},
            {"kind": "radial", "cx": 0.3, "cy": 0.6, "rx": 0.2, "ry": 0.25, "angle": 20,
             "feather": 70, "invert": True, "shadows": 40, "clarity": 25},
        ]})
        crs = to_crs(edit)
        assert "crs:PhotoolMasks" in crs
        assert from_crs(crs) == edit

    def test_no_gradients_is_no_key(self):
        assert "masks" not in normalize({"masks": []})
        assert "masks" not in normalize({"exposure": 1.0})


class TestChosen:
    """Subject and sky masks: the selection is the model's; what is done with it
    is tested here with a made-up one."""

    LEFT_HALF = np.zeros((40, 60), np.float32)
    LEFT_HALF[:, :30] = 1.0

    def test_a_matte_is_read_over_the_pixels_rendered(self):
        w = weight({"kind": "subject"}, (80, 120), matte=self.LEFT_HALF)
        assert w[:, :55].min() > 0.99 and w[:, 65:].max() < 0.01
        patch = weight({"kind": "subject"}, (40, 60), window=(0.5, 0.0, 1.0, 0.5), matte=self.LEFT_HALF)
        assert np.allclose(patch, w[:40, 60:], atol=0.02)

    def test_invert_is_everything_else_and_strokes_refine_it(self):
        sky = {"kind": "sky", "invert": True,
               "strokes": [{"points": [[0.75, 0.5]], "size": 0.05, "feather": 0, "erase": True}]}
        w = weight(sky, (80, 120), matte=self.LEFT_HALF)
        assert w[40, 10] < 0.01 and w[10, 100] > 0.99 and w[40, 90] == 0.0

    def test_without_its_model_it_reaches_nothing(self):
        assert weight({"kind": "sky", "invert": True}, (20, 30)).max() == 0.0

    def test_kept_in_the_edit_as_its_kind_alone(self):
        clean = normalize({"masks": [{"kind": "subject", "invert": 1, "exposure": 0.5, "rx": 3}]})
        assert clean["masks"] == [{"kind": "subject", "invert": True, "exposure": 0.5}]
        assert from_crs(to_crs(clean)) == clean


class TestSegment:
    def test_the_guided_filter_brings_a_rough_edge_to_the_picture(self):
        from adjust.segment import _guided

        guide = np.zeros((64, 64), np.float32)
        guide[:, 40:] = 1.0                         # the picture's edge is at 40
        rough = np.zeros((64, 64), np.float32)
        rough[:, 34:] = 1.0                         # the model's at 34
        rough = np.clip(rough + 0.0, 0, 1)
        import cv2
        rough = cv2.GaussianBlur(rough, (0, 0), 4)
        fine = _guided(guide, rough, 6, 1e-3)
        # Sharper at the picture's edge than the model's guess was.
        assert fine[32, 41] - fine[32, 38] > rough[32, 41] - rough[32, 38]

    def test_a_real_sky_is_found_when_the_model_is_here(self):
        from adjust import segment
        from config import load_config

        segment.configure(load_config().models_dir)
        if not segment.available("sky"):
            import pytest
            pytest.skip("the sky model is not downloaded here")
        h, w = 120, 180
        rgb = np.zeros((h, w, 3), np.float32)
        rgb[:60] = (0.25, 0.45, 0.9)                # blue above
        rgb[60:] = (0.08, 0.12, 0.04)               # dark land below
        rng = np.random.default_rng(3)
        rgb[60:] += 0.05 * rng.random((60, w, 3)).astype(np.float32)
        m = segment.matte(LinearFrame(rgb), "sky")
        top, bottom = m[: m.shape[0] // 3].mean(), m[-m.shape[0] // 3:].mean()
        assert top > 0.8 and top - bottom > 0.3      # a made-up land is not a real one
