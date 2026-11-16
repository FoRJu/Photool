"""
The maths of a scanned negative (adjust.negative): a negative made from a known scene
by the film's own rule, and turned back.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import negative

BASE = np.array([0.75, 0.45, 0.25], dtype=np.float32)
BLACK = 0.01
STEPS = 12
PATCHES = [(0.6, 0.2, 0.1), (0.1, 0.5, 0.15), (0.1, 0.15, 0.6),
           (0.7, 0.6, 0.1), (0.4, 0.1, 0.5), (0.05, 0.4, 0.5)]


MARGIN = 0.1


def _step(i, w, n=STEPS):
    left = int(MARGIN * w)
    width = (w - 2 * left) / n
    return int(left + i * width), int(left + (i + 1) * width)


def scene(w=600, h=400, low=BLACK, high=1.0) -> np.ndarray:
    """A grey ramp of twelve steps across the top half, six coloured patches below, on
    a mid-grey ground (so that nothing at the frame's edge looks like film base)."""
    out = np.full((h, w, 3), 0.18, np.float32)
    levels = np.geomspace(low, high, STEPS)
    for i, level in enumerate(levels):
        x0, x1 = _step(i, w)
        out[int(MARGIN * h): h // 2, x0:x1] = level
    for i, colour in enumerate(PATCHES):
        x0, x1 = _step(i, w, 6)
        out[h // 2:h - int(MARGIN * h), x0:x1] = colour
    # Texture, the same in all three channels: a photograph is not flat, and a flat
    # patch at the frame's edge is what the border is told apart by.
    y, x = np.mgrid[0:h, 0:w]
    out *= (1.0 + 0.25 * np.sin(2 * np.pi * x / 7) * np.sin(2 * np.pi * y / 7))[..., None]
    return out.astype(np.float32)


def ramp_values(img: np.ndarray, w=600, h=400) -> np.ndarray:
    """(12, 3): the middle of each step of the ramp."""
    rows = slice(int(MARGIN * h) + h // 16, h // 2 - h // 16)
    out = []
    for i in range(STEPS):
        x0, x1 = _step(i, w)
        pad = (x1 - x0) // 4
        out.append(np.median(img[rows, x0 + pad:x1 - pad].reshape(-1, 3), axis=0))
    return np.array(out)


def film(s: np.ndarray, gamma=(0.6, 0.6, 0.6), base=BASE) -> np.ndarray:
    """The film's rule. Film is never thinner than its base: a scene darker than its
    black leaves the base as it is."""
    g = np.asarray(gamma, np.float32)
    return (base * (np.maximum(s, BLACK) / BLACK) ** (-g)).astype(np.float32)


def slopes(out: np.ndarray, s: np.ndarray, first=1, last=STEPS - 1) -> np.ndarray:
    """The slope of log(out) on log(scene) over the ramp's steps, per channel (the black
    step, half of it clipped at the base, and the top one left out)."""
    o = ramp_values(out)[first:last]
    x = np.log(ramp_values(s)[first:last, 0])
    return np.array([np.polyfit(x, np.log(o[:, c]), 1)[0] for c in range(3)])


def neutral(out: np.ndarray, first=1, last=STEPS - 1) -> float:
    """The largest spread of the three channels over a step, as a fraction."""
    o = ramp_values(out)[first:last]
    return float(np.max((o.max(axis=1) - o.min(axis=1)) / o.mean(axis=1)))


class TestRoundTrip:
    def test_the_scene_comes_back(self):
        s = scene()
        v = film(s)
        d = negative.measure(v)
        out = negative.positive(v, d)
        assert np.allclose(slopes(out, s), 1.0, atol=0.05), slopes(out, s)
        assert neutral(out) < 0.03

    def test_unequal_layers_are_evened_out(self):
        s = scene()
        v = film(s, gamma=(0.66, 0.60, 0.54))
        d = negative.measure(v, kind="colour")
        top = lambda out: ramp_values(out)[STEPS - 2]
        even = top(negative.positive(v, d))
        assert (even.max() - even.min()) / even.mean() < 0.06, even
        flat = top(negative.positive(v, {**d, "balance": 0}))
        assert (flat.max() - flat.min()) / flat.mean() > 0.06, flat


def framed(v: np.ndarray, base=BASE, band=24, holder=12, holes=True) -> np.ndarray:
    """``v`` in a frame: a band of pure base round it, the holder outside that, and four
    sprocket holes (clipped white) in the band."""
    h, w = v.shape[:2]
    out = np.full((h + 2 * (band + holder), w + 2 * (band + holder), 3), 0.002, np.float32)
    inner = out[holder:-holder, holder:-holder]
    inner[...] = base
    inner[band:-band, band:-band] = v
    if holes:
        for x in (60, 200, 400, 560):
            inner[4:16, x:x + 14] = 1.0
    return out


class TestTheBase:
    def test_from_the_border(self):
        v = framed(film(scene()))
        d = negative.measure(v)
        assert d["base_from"] == "border"
        assert np.allclose(d["base"], BASE, rtol=0.02), d["base"]

    def test_from_the_picture(self):
        d = negative.measure(film(scene()))
        assert d["base_from"] == "picture"
        assert np.allclose(d["base"], BASE, rtol=0.10), d["base"]

    def test_no_black_in_the_scene_and_the_roll_puts_it_right(self):
        s = scene(low=0.08)
        v = film(s)
        d = negative.measure(v)
        assert d["base_from"] == "picture"
        assert all(f < t for f, t in zip(d["base"], BASE))
        given = negative.measure(v, base=list(BASE), base_from="roll")
        assert given["base_from"] == "roll"
        out = negative.positive(v, given)
        assert np.allclose(slopes(out, s), 1.0, atol=0.05), slopes(out, s)

    def test_base_at(self):
        v = framed(film(scene()))
        h, w = v.shape[:2]
        found = negative.base_at(v, 30 / w, 200 / h)
        assert np.allclose(found, BASE, rtol=0.01)
        with pytest.raises(ValueError, match="bare light"):
            negative.base_at(v, (12 + 60 + 7) / w, (12 + 10) / h)


class TestTheRange:
    def test_a_steep_read_is_held(self):
        s = scene()
        v = film(s) ** 2           # a scan read with too steep a curve
        d = negative.measure(v)
        plain = negative.measure(film(s))
        assert np.allclose(d["range"], 2 * np.asarray(plain["range"]), rtol=0.03)
        out = negative.positive(v, d)
        o = ramp_values(out)[:, 1]
        log_range = np.log10(o[-1] / o[0])
        # (MAX_RANGE / FILM_GAMMA) = 3.0 decades, against the 2.0 of the scene; 4.0 unheld
        assert abs(log_range - 3.0) < 0.15, log_range
        assert neutral(out) < 0.03

    @pytest.mark.parametrize("contrast, factor", [(100, 2.0), (-100, 0.5)])
    def test_contrast(self, contrast, factor):
        s = scene()
        v = film(s)
        d = negative.measure(v)
        plain = ramp_values(negative.positive(v, d))[:, 1]
        moved = ramp_values(negative.positive(v, {**d, "contrast": contrast}))[:, 1]
        ratio = np.log(moved[-1] / moved[0]) / np.log(plain[-1] / plain[0])
        assert abs(ratio - factor) < 0.02 * factor

    def test_dust(self):
        v = film(scene())
        clean_d = negative.measure(v)
        rng = np.random.default_rng(3)
        dusty = v.copy()
        idx = rng.choice(dusty.shape[0] * dusty.shape[1], int(0.002 * dusty.shape[0] * dusty.shape[1]),
                         replace=False)
        dusty.reshape(-1, 3)[idx] = 0.0005
        d = negative.measure(dusty)
        assert np.allclose(d["range"], clean_d["range"], rtol=0.02)
        out = negative.positive(dusty, d)
        assert np.isfinite(out).all() and out.max() <= negative.CEILING


class TestKindAndArea:
    def test_bw(self):
        v = film(scene())
        out = negative.positive(v, {**negative.measure(v), "kind": "bw"})
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
        grey = film(scene(), base=np.array([0.6, 0.6, 0.6], np.float32))
        grey[...] = grey.mean(axis=2, keepdims=True)
        assert negative.measure(grey)["kind"] == "bw"
        assert negative.measure(v)["kind"] == "colour"

    def test_the_area(self):
        v = film(scene())
        plain = negative.measure(v)
        h, w = v.shape[:2]
        side = w // 4
        held = np.concatenate([np.full((h, side, 3), 0.002, np.float32), v], axis=1)
        width = w + side
        i = negative.INSET
        area = ((side + i * w) / width, i, (side + (1 - i) * w) / width, 1 - i)
        cut = negative.measure(held, area=area)
        assert cut["base_from"] == plain["base_from"]
        assert np.allclose(cut["base"], plain["base"], rtol=0.02), (cut, plain)
        assert np.allclose(cut["range"], plain["range"], rtol=0.02), (cut, plain)

    def test_any_size(self):
        import cv2

        v = film(scene())
        big = cv2.resize(v, (2400, 1600), interpolation=cv2.INTER_NEAREST)
        a, b = negative.measure(v), negative.measure(big)
        assert np.allclose(a["base"], b["base"], rtol=0.02)
        assert np.allclose(a["range"], b["range"], rtol=0.02)


class TestClean:
    def test_clean(self):
        assert negative.clean({"kind": "colour"}) is None
        assert negative.clean({"kind": "sepia", "base": [0.5, 0.5, 0.5]}) is None
        assert negative.clean({"kind": "colour", "base": [0.5, 0, 0.5]}) is None
        d = negative.clean({"kind": "bw", "base": [0.5, 0.5, 0.5], "contrast": 400,
                            "balance": -3, "range": [9, 0.1, 1]})
        assert d["contrast"] == 100 and d["balance"] == 0
        assert d["range"] == [3.5, 0.3, 1.0]


def test_a_full_frame_is_quick():
    import time

    v = np.random.default_rng(0).uniform(0.05, 0.7, (4000, 6000, 3)).astype(np.float32)
    d = negative.measure(v)
    t = time.perf_counter()
    negative.positive(v, d)
    assert time.perf_counter() - t < 3.0
