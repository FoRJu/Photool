"""A look as a .cube LUT (adjust.lut): its format, what it holds and leaves out,
and that looking a picture up in it gives what the renderer gives."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.lut import cube_of, lattice, per_pixel_frame  # noqa: E402
from adjust.render import develop_frame  # noqa: E402


def _table(text: str) -> tuple[int, np.ndarray]:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    size = int(next(ln for ln in lines if ln.startswith("LUT_3D_SIZE")).split()[1])
    rows = [ln for ln in lines if ln[0].isdigit() or ln[0] == "-"]
    return size, np.array([[float(v) for v in ln.split()] for ln in rows], dtype=np.float64)


def _look_up(picture: np.ndarray, size: int, table: np.ndarray) -> np.ndarray:
    """Trilinear lookup of display-encoded 0..1 colours, red fastest."""
    cube = table.reshape(size, size, size, 3)          # [b, g, r]
    p = np.clip(picture, 0.0, 1.0) * (size - 1)
    i = np.minimum(np.floor(p).astype(int), size - 2)
    f = p - i
    r0, g0, b0 = i[..., 0], i[..., 1], i[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0.0
    for db, wb in ((0, 1 - fb), (1, fb)):
        for dg, wg in ((0, 1 - fg), (1, fg)):
            for dr, wr in ((0, 1 - fr), (1, fr)):
                out = out + wb * wg * wr * cube[b0 + db, g0 + dg, r0 + dr]
    return out


def test_no_edit_is_an_error():
    with pytest.raises(ValueError):
        cube_of({})
    with pytest.raises(ValueError):
        cube_of({"clarity": 30, "grain": 20})


def test_the_format():
    text, _ = cube_of({"contrast": 10}, 33, "Night")
    assert "LUT_3D_SIZE 33" in text and 'TITLE "Night"' in text
    assert text.startswith("#")
    size, table = _table(text)
    assert table.shape == (33 ** 3, 3)
    assert np.allclose(table[0], 0.0)
    assert table[1][0] > 0 and table[1][1] == 0 and table[1][2] == 0


def test_a_small_contrast_is_near_the_identity_and_white_stays_white():
    size, table = _table(cube_of({"contrast": 1}, 33)[0])
    assert np.abs(table - lattice(size).reshape(-1, 3)).max() < 0.01
    assert np.allclose(table[-1], 1.0, atol=1e-4)


def test_black_and_white_is_grey():
    _, table = _table(cube_of({"mono": 1, "contrast": 10}, 17)[0])
    assert np.allclose(table[:, 0], table[:, 1]) and np.allclose(table[:, 1], table[:, 2])


def test_the_lut_gives_what_the_renderer_gives():
    edit = {"contrast": 20, "saturation": -15, "fade": 20,
            "curve": [[0, 0], [100, 120], [255, 255]], "sat_blue": -40}
    rng = np.random.default_rng(7)
    picture = rng.integers(0, 256, size=(64, 64, 3)).astype(np.float32) / 255.0
    a = np.asarray(develop_frame(per_pixel_frame(picture), edit), dtype=np.float64)
    size, table = _table(cube_of(edit, 33)[0])
    b = _look_up(picture.astype(np.float64), size, table) * 255.0
    diff = np.abs(a - b)
    assert diff.mean() < 1.5, diff.mean()
    assert diff.max() < 6, diff.max()


def test_what_a_lut_cannot_hold_is_named_and_changes_nothing():
    plain, none = cube_of({"contrast": 20, "vibrance": 10})
    text, left_out = cube_of({"contrast": 20, "vibrance": 10, "clarity": 30, "grain": 25, "style": "gone"})
    assert none == []
    assert {"Clarity", "Grain", "Style"} <= set(left_out)
    numbers = lambda t: [ln for ln in t.splitlines() if not ln.startswith("#")]  # noqa: E731
    assert numbers(text) == numbers(plain)


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


def test_the_route(client):
    assert client.post("/api/export/develop/presets", json={
        "name": "Night: blue", "sections": ["light", "presence", "effects"],
        "settings": {"contrast": 20, "clarity": 10, "grain": 20}}).status_code == 200
    res = client.get("/api/export/develop/presets/Night%3A%20blue/lut?size=17")
    assert res.status_code == 200, res.text
    assert res.headers["x-lut-left-out"] == "Clarity, Grain"
    assert "attachment" in res.headers["content-disposition"] and "Night_ blue.cube" in res.headers["content-disposition"]
    assert "LUT_3D_SIZE 17" in res.text
    assert client.get("/api/export/develop/presets/Nowhere/lut").status_code == 404
    assert client.get("/api/export/develop/presets/Night%3A%20blue/lut?size=20").status_code == 400
    assert client.post("/api/export/develop/presets", json={
        "name": "Grain", "settings": {"grain": 20}}).status_code == 200
    res = client.get("/api/export/develop/presets/Grain/lut")
    assert res.status_code == 400 and "Grain" in res.json()["detail"]
