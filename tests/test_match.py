"""
Match a look (adjust.match): a frame given the tone and colour of a reference,
as ordinary controls.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.develop import normalize  # noqa: E402
from adjust.match import MAX_SHIFT, _through, match, scaled  # noqa: E402


def _picture(seed, low=0, high=255, tint=(1.0, 1.0, 1.0), chroma=40.0, floor=0):
    rng = np.random.default_rng(seed)
    luma = rng.uniform(low, high, (80, 120, 1)).astype(np.float32)
    colour = rng.normal(0, chroma, (80, 120, 3)).astype(np.float32)
    return np.clip((luma + colour) * np.array(tint, np.float32), floor, 255).astype(np.uint8)


def test_a_warm_faded_reference_lifts_the_blacks_and_warms_the_frame():
    frame = _picture(1)
    reference = _picture(2, low=40, high=235, tint=(1.08, 1.0, 0.85), floor=40)
    found = match(frame, reference)
    r, g, b = (dict(map(tuple, found[k])) for k in ("curve_r", "curve_g", "curve_b"))
    assert r[0] > 15 and g[0] > 15                       # a matte black
    after = _through(frame, [found["curve_r"], found["curve_g"], found["curve_b"]])
    assert after[..., 0].mean() > after[..., 2].mean() + 15   # and warm
    for c in range(3):                                    # levels near the reference's
        assert abs(np.median(after[..., c]) - np.median(reference[..., c])) < 8


def test_a_black_and_white_reference_gives_black_and_white():
    reference = np.repeat(_picture(3, chroma=0)[..., :1], 3, axis=2)
    found = match(_picture(4), reference)
    assert found["mono"] == 1 and "curve" in found and "curve_r" not in found


def test_a_dull_reference_takes_colour_away():
    found = match(_picture(5, chroma=50), _picture(6, chroma=8))
    assert found["saturation"] < -20


def test_a_different_scene_is_not_dragged_all_the_way():
    # A dark frame against a bright snowfield: the look, not the snow.
    found = match(_picture(7, 0, 120), _picture(8, 150, 255))
    for key in ("curve_r", "curve_g", "curve_b"):
        assert all(abs(y - x) <= MAX_SHIFT for x, y in found[key])
        ys = [y for _, y in found[key]]
        assert ys == sorted(ys)                           # never turning back


def test_the_amount_scales_toward_no_change_and_the_edit_takes_it():
    found = match(_picture(1), _picture(2, 40, 235, (1.08, 1.0, 0.85), chroma=10))
    assert normalize(scaled(found, 0.0)).keys() <= {"saturation"} and not normalize(scaled(found, 0.0))
    half = scaled(found, 0.5)
    x, y = found["curve_r"][0]
    assert half["curve_r"][0] == [x, round(x + 0.5 * (y - x))]
    clean = normalize(scaled(found, 1.0))
    assert {"curve_r", "curve_g", "curve_b"} <= clean.keys()
