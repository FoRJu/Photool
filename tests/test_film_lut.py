"""Film looks from .cube LUTs (adjust.film_lut), kept and rendered as styles."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import film_lut, styles  # noqa: E402
from adjust.render import LinearFrame, develop_frame, encode  # noqa: E402


def _cube(fn=lambda r, g, b: (r, g, b), size=9, header="") -> str:
    steps = np.linspace(0, 1, size)
    lines = [header, f"LUT_3D_SIZE {size}"]
    for b in steps:
        for g in steps:
            for r in steps:
                lines.append("%.6f %.6f %.6f" % fn(r, g, b))
    return "\n".join(lines) + "\n"


@pytest.fixture()
def style_dir(tmp_path):
    styles.set_dir(tmp_path / "styles")
    yield tmp_path / "styles"
    styles.set_dir(None)


def test_a_cube_is_read_red_fastest():
    table, title = film_lut.parse_cube('TITLE "Warm"\n' + _cube(lambda r, g, b: (r, g, b), size=3))
    assert title == "Warm" and table.shape == (3, 3, 3, 3)
    assert np.allclose(table[0, 0, 1], [0.5, 0, 0]) and np.allclose(table[1, 0, 0], [0, 0, 0.5])


@pytest.mark.parametrize("text, words", [
    ("LUT_1D_SIZE 4\n0 0 0\n", "1D"),
    ("0 0 0\n", "size"),
    ("LUT_3D_SIZE 3\n0 0 0\n", "27"),
    ("DOMAIN_MAX 2 2 2\n" + _cube(size=2), "0..1"),
])
def test_a_file_it_cannot_use_says_why(text, words):
    with pytest.raises(ValueError, match=words):
        film_lut.parse_cube(text)


def test_a_name_from_the_file():
    assert styles._lut_name("classic_chrome_DisplayP3.cube") == "Classic Chrome"
    assert styles._lut_name("pro neg hi_sRGB.cube") == "Pro Neg Hi"
    assert styles._lut_name("Kodak 2383.cube") == "Kodak 2383"


def test_the_identity_gives_the_picture_back():
    lut = film_lut.FilmLut(film_lut.parse_cube(_cube(size=17))[0])
    rgb = np.random.default_rng(1).random((20, 20, 3)).astype(np.float32) * 0.9
    assert np.abs(film_lut.apply_lut(rgb, lut) - encode(rgb)).max() < 0.01


def test_display_p3_round_trips_through_the_identity():
    lut = film_lut.FilmLut(film_lut.parse_cube(_cube(size=17))[0], "p3")
    rgb = np.full((4, 4, 3), [0.2, 0.4, 0.1], np.float32)
    assert np.abs(film_lut.apply_lut(rgb, lut) - encode(rgb)).max() < 0.01
    assert film_lut.guess_space("# DisplayP3\n", "") == "p3"
    assert film_lut.guess_space("", "provia_DisplayP3.cube") == "p3"
    assert film_lut.guess_space("# sRGB\n", "provia_sRGB.cube") == "srgb"


def test_an_imported_lut_is_a_style(style_dir):
    darker = _cube(lambda r, g, b: (r * 0.8, g * 0.8, b * 0.8), header="# sRGB")
    style = styles.import_lut("", darker, "dim_sRGB.cube")
    assert style.name == "Dim" and style.lut.space == "srgb"
    listed = styles.list_styles()
    assert listed[0]["kind"] == "lut" and listed[0]["improvable"] is False
    again = styles.load(style.id)
    assert again.lut is not None and np.allclose(again.lut.table, style.lut.table)
    assert styles.rename(style.id, "Dim")
    assert styles.load(style.id).name == "Dim" and styles.load(style.id).lut is not None


def test_the_frame_keeps_its_brightness_through_the_lut(style_dir):
    """Anchored: a LUT that darkens everything does not darken the frame's middle;
    a frame with no camera look is anchored to the typical brightness."""
    style = styles.import_lut("Dim", _cube(lambda r, g, b: (r * 0.6, g * 0.6, b * 0.6)), "dim.cube")
    rgb = np.random.default_rng(2).random((64, 64, 3)).astype(np.float32) * 0.05   # a dark decode
    frame = LinearFrame(rgb)
    out = np.asarray(develop_frame(frame, {"style": style.id}), np.float32) / 255.0
    mid = float(np.median(out.reshape(-1, 3) @ np.array([0.2126, 0.7152, 0.0722])))
    assert abs(mid - styles.LUT_TYPICAL) < 0.06, mid


def test_a_missing_style_draws_the_frame(style_dir):
    rgb = np.random.default_rng(3).random((16, 16, 3)).astype(np.float32)
    frame = LinearFrame(rgb)
    assert develop_frame(frame, {"style": "not-here", "contrast": 10}) is not None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{tmp_path / 'lib'}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


def test_the_import_route(client):
    res = client.post("/api/styles/import-lut", json={"text": _cube(size=5), "filename": "Provia_DisplayP3.cube"})
    assert res.status_code == 200, res.text
    style = res.json()["style"]
    assert style["kind"] == "lut" and style["name"] == "Provia (P3)" and style["space"] == "p3"
    assert any(s["id"] == style["id"] for s in client.get("/api/styles").json()["styles"])
    assert client.post(f"/api/styles/{style['id']}/improve").status_code == 400
    bad = client.post("/api/styles/import-lut", json={"text": "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n"})
    assert bad.status_code == 400 and "1D" in bad.json()["detail"]
    assert client.post("/api/styles/import-lut", json={"text": _cube(size=2), "space": "rec2020"}).status_code == 400
