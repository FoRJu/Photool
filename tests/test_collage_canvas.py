"""
Collages on a fixed canvas — a phone story, a square, a print: the layouts that
fill it with least cut off the photos, each photo cut around its people, and the
API that offers the layouts and saves the picture.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from export.collage import FORMATS, _cover, layouts, render_canvas  # noqa: E402


def _covered(cells, w, h):
    mask = np.zeros((h, w), np.int32)
    for x, y, cw, ch in cells:
        mask[round(y):round(y + ch), round(x):round(x + cw)] += 1
    return mask


def test_four_squares_on_a_square_are_two_by_two_uncut():
    best = layouts([1.0] * 4, 1000, 1000)[0]
    assert best["crop"] == 0
    assert sorted(best["lines"]) == [2, 2]


def test_every_layout_fills_the_canvas_once_with_every_photo():
    aspects = [1.5, 0.67, 1.5, 1.33, 0.75]
    for lay in layouts(aspects, 1080, 1920, gap=0, limit=6):
        assert len(lay["cells"]) == 5
        mask = _covered(lay["cells"], 1080, 1920)
        assert (mask == 1).mean() > 0.995      # no gaps and no overlaps, to rounding

def test_a_little_more_cut_is_better_than_stamps():
    """Six landscapes on a story: the least cut is two big frames over a row of
    four small ones; the first offered has cells nearer one size."""
    first = layouts([1.5] * 6, 1080, 1920)[0]
    areas = [c[2] * c[3] for c in first["cells"]]
    assert max(areas) / min(areas) < 5


def test_two_landscapes_on_a_story_stack_and_on_a_wide_page_sit_side_by_side():
    story = layouts([1.5, 1.5], 1080, 1920)[0]
    (_, y0, _, h0), (_, y1, _, _) = sorted(story["cells"], key=lambda c: c[1])
    assert y1 >= y0 + h0 - 0.5                     # one above the other
    wide = layouts([1.5, 1.5], 3000, 1000)[0]
    assert wide["lines"] == [2] and wide["by"] == "rows"


def test_gaps_leave_space_between_cells():
    lay = layouts([1.0, 1.0], 1000, 500, gap=20)[0]
    (x0, _, w0, _), (x1, _, _, _) = sorted(lay["cells"])
    assert abs(x1 - (x0 + w0) - 20) < 0.5


def test_a_cut_keeps_its_focus():
    """A bright spot near the left edge of a wide picture stays in a square cut
    of it when the cut is kept around that spot, and falls out of a middle cut."""
    img = np.full((200, 600, 3), 40, np.uint8)
    img[90:110, 40:60] = 250
    im = Image.fromarray(img)
    kept = np.asarray(_cover(im, 100, 100, (50 / 600, 0.5)))
    middle = np.asarray(_cover(im, 100, 100))
    assert kept.max() > 200 and middle.max() < 100


def test_a_print_is_its_size_at_300_dpi():
    images = [Image.new("RGB", (300, 200), (200, 50, 50)), Image.new("RGB", (200, 300), (50, 200, 50))]
    w, h = FORMATS["4x6"]
    jpeg = render_canvas(images, w, h, gap=10, margin=20, background=(0, 0, 0))
    out = Image.open(io.BytesIO(jpeg))
    assert out.size == (1800, 1200)
    assert round(out.info["dpi"][0]) == 300
    assert max(out.getpixel((5, 5))) < 20          # the black border


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    folder = "2025/2025-05-01"
    library = tmp_path / "lib"
    (library / folder).mkdir(parents=True)
    for i, size in enumerate([(600, 400), (400, 600), (600, 400)]):
        Image.new("RGB", size, (60 * i, 120, 200)).save(library / folder / f"IMG_{i}.JPG", quality=90)
    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)
    import main

    with TestClient(main.app) as c:
        assert c.post(f"/api/catalog/index/{folder}").status_code == 200
        yield c, [f"{folder}/IMG_{i}.JPG" for i in range(3)]


def test_through_the_api_layouts_then_the_picture(client):
    c, paths = client
    res = c.post("/api/library/collage/layouts", json={"filepaths": paths, "format": "story"})
    assert res.status_code == 200
    data = res.json()
    assert data["filepaths"] == paths and data["height"] > data["width"]
    first = data["layouts"][0]
    assert len(first["cells"]) == 3 and all(0 <= v <= 1.0001 for cell in first["cells"] for v in cell)

    jpeg = c.post("/api/library/collage", json={"filepaths": paths, "format": "story", "layout": 1, "gap": 8})
    assert jpeg.status_code == 200 and jpeg.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(jpeg.content)).size == (1080, 1920)

    small = c.post("/api/library/collage", json={"filepaths": paths, "format": "5x7", "portrait": True, "preview": True})
    w, h = Image.open(io.BytesIO(small.content)).size
    assert h == 900 and w < h

    assert c.post("/api/library/collage", json={"filepaths": paths, "format": "poster"}).status_code == 400


def test_a_cell_the_photos_own_shape_is_not_cut_past_its_edge():
    """Rounding can ask for a window a hair wider than the photo; it is cut to it."""
    im = Image.new("RGB", (1999, 1333), (90, 90, 90))
    for w, h in ((1200, 800), (1201, 800), (899, 600), (300, 200)):
        assert _cover(im, w, h, (0.0, 1.0)).size == (w, h)


def test_the_evenest_layout_is_always_offered():
    """Nine photos on a square: a three-by-three grid is among the choices,
    though other layouts cut less."""
    offered = layouts([1.5] * 9, 2048, 2048)
    assert [3, 3, 3] in [lay["lines"] for lay in offered]
    assert offered[0]["lines"] != [3, 3, 3]


# ── Hero layouts and styles ──────────────────────────────────────────

def test_a_hero_layout_makes_the_first_photo_the_big_one_and_fills_the_page():
    aspects = [1.5, 1.5, 0.67, 1.5, 1.0]
    heroes = [lay for lay in layouts(aspects, 2048, 2048, gap=0) if lay["by"].startswith("hero")]
    assert heroes, "a hero layout is offered for three photos or more"
    for lay in heroes:
        areas = [c[2] * c[3] for c in lay["cells"]]
        assert areas[0] == max(areas) and areas[0] > 2 * sorted(areas)[-2]
        mask = _covered(lay["cells"], 2048, 2048)
        assert (mask == 1).mean() > 0.995


def test_two_photos_get_no_hero():
    assert all(not lay["by"].startswith("hero") for lay in layouts([1.5, 1.5], 1080, 1920))


def test_the_first_photo_large_comes_after_the_grids_so_the_first_is_still_the_least_cut():
    found = layouts([1.0] * 4, 1000, 1000)
    special = [lay["by"].startswith("hero") or lay["by"] in ("centre", "mosaic") for lay in found]
    assert special == sorted(special) and not special[0]
    assert [lay["by"] for lay in found][-2:] == ["centre", "mosaic"]


@pytest.mark.parametrize("style", ["clean", "rounded", "polaroid"])
def test_every_style_draws_the_canvas(style):
    images = [Image.new("RGB", (300, 200), (200, 50, 50)), Image.new("RGB", (200, 300), (50, 50, 200)),
              Image.new("RGB", (300, 300), (50, 200, 50))]
    jpeg = render_canvas(images, 900, 900, layout=0, gap=20, margin=30, background=(30, 41, 59), style=style)
    out = Image.open(io.BytesIO(jpeg)).convert("RGB")
    assert out.size == (900, 900)
    corner = out.getpixel((2, 2))
    assert max(abs(a - b) for a, b in zip(corner, (30, 41, 59))) < 12     # the background shows at the edge


def test_a_polaroid_is_a_white_card_around_the_photo():
    jpeg = render_canvas([Image.new("RGB", (400, 400), (200, 30, 30))] * 2, 1000, 500, gap=40, margin=40,
                         background=(0, 0, 0), style="polaroid")
    out = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).astype(int)
    white = ((out > 235).all(axis=2)).mean()
    red = ((out[..., 0] > 150) & (out[..., 1] < 80)).mean()
    assert white > 0.05 and red > 0.2


def test_through_the_api_a_style_and_a_new_background(client):
    c, paths = client
    res = c.post("/api/library/collage", json={"filepaths": paths, "format": "square", "style": "polaroid",
                                               "background": "kraft", "preview": True})
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert c.post("/api/library/collage", json={"filepaths": paths, "format": "square",
                                                "style": "glitter"}).status_code == 400


def test_prints_have_their_own_arrangements():
    from export.collage import polaroid_layouts

    aspects = [1.5, 0.67, 1.5, 1.33, 1.0, 1.5, 0.75, 1.5, 1.5]
    found = polaroid_layouts(aspects, 2000, 2000)
    assert [f["by"] for f in found] == ["scatter", "feature", "pile", "fan", "cascade"]
    scatter, feature = found[0], found[1]
    # Scatter: one size, each in its photo's shape kept near square.
    areas = [w * h for _x, _y, w, h in scatter["cells"]]
    assert max(areas) / min(areas) < 1.01
    assert all(0.79 <= w / h <= 1.26 for _x, _y, w, h in scatter["cells"])
    # Feature: the first the biggest, and laid last, on top.
    first = feature["cells"][0][2] * feature["cells"][0][3]
    assert first > 2 * max(w * h for _x, _y, w, h in feature["cells"][1:])
    assert feature["order"][-1] == 0
    # The same photographs fall the same way every time.
    assert polaroid_layouts(aspects, 2000, 2000) == found
    assert [f["by"] for f in polaroid_layouts([1.5, 1.5], 2000, 2000)] == ["scatter", "pile", "cascade"]


@pytest.mark.parametrize("layout", [0, 1, 2, 3, 4])
def test_every_print_stays_on_the_page(layout):
    """Turned and nudged, no print runs off the edge: the page's border stays the table."""
    images = [Image.new("RGB", (600, 400) if k % 2 else (400, 600), (220, 30, 30)) for k in range(9)]
    jpeg = render_canvas(images, 800, 800, layout=layout, margin=16, background=(0, 0, 255), style="polaroid")
    out = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).astype(int)
    rim = np.concatenate([out[:6].reshape(-1, 3), out[-6:].reshape(-1, 3),
                          out[:, :6].reshape(-1, 3), out[:, -6:].reshape(-1, 3)])
    print_pixels = ((rim[:, 0] > 200) & (rim[:, 1] > 200)).mean() + (rim[:, 0] > 150).mean()
    assert print_pixels < 0.01


def test_the_api_offers_print_arrangements_for_polaroid(client):
    c, paths = client
    data = c.post("/api/library/collage/layouts",
                  json={"filepaths": paths, "format": "square", "style": "polaroid"}).json()
    assert [l["by"] for l in data["layouts"]] == ["scatter", "feature", "pile", "fan", "cascade"]
    assert all(len(l["angles"]) == 3 and sorted(l["order"]) == [0, 1, 2] for l in data["layouts"])
    grids = c.post("/api/library/collage/layouts", json={"filepaths": paths, "format": "square"}).json()
    assert all(l["angles"] is None for l in grids["layouts"])


# ── More styles, and what they lie on ────────────────────────────────

def _photos(n=5):
    return [Image.new("RGB", (600, 400) if k % 2 else (400, 600), (40 * k, 120, 200 - 30 * k)) for k in range(n)]


def test_colours_are_read_however_they_are_written():
    from export.collage_looks import parse_colour

    assert parse_colour("#ff8000") == (255, 128, 0)
    assert parse_colour("#fff") == (255, 255, 255)
    assert parse_colour("nonsense", (1, 2, 3)) == (1, 2, 3)


@pytest.mark.parametrize("name", ["dots", "stripes", "grid", "linen", "wood", "cork", "paper"])
def test_every_pattern_draws_something_in_its_colour(name):
    from export.collage_looks import pattern

    plain = np.asarray(pattern(300, 200, (180, 140, 100), ""), dtype=int)
    drawn = np.asarray(pattern(300, 200, (180, 140, 100), name), dtype=int)
    assert drawn.shape == plain.shape
    assert np.abs(drawn - plain).mean() > 0.5                 # something is there
    assert np.abs(drawn.mean(axis=(0, 1)) - (180, 140, 100)).max() < 40   # and it is still that colour


def test_a_photo_behind_is_blurred_and_dimmed():
    from PIL import ImageDraw

    from export.collage_looks import backdrop

    photo = Image.new("RGB", (400, 400), (250, 250, 250))
    ImageDraw.Draw(photo).rectangle((0, 0, 199, 399), fill=(0, 0, 0))
    sharp = np.asarray(backdrop(400, 400, (255, 255, 255), photo=photo, blur=False), dtype=int)
    soft = np.asarray(backdrop(400, 400, (255, 255, 255), photo=photo, blur=True), dtype=int)
    assert sharp.max() < 230                                   # dimmed
    assert np.abs(np.diff(soft[200, :, 0])).max() < np.abs(np.diff(sharp[200, :, 0])).max()


@pytest.mark.parametrize("style", ["clean", "rounded", "gallery", "polaroid", "taped", "pinned", "stamps", "film"])
def test_every_style_draws_every_one_of_its_layouts(style):
    from export.collage import layouts_for

    images = _photos(5)
    count = len(layouts_for(style, [im.width / im.height for im in images], 700, 500, 8))
    assert count >= 2
    for layout in range(count):
        jpeg = render_canvas(images, 760, 560, layout=layout, gap=8, margin=30, background="#204060",
                             style=style, pattern="dots")
        assert Image.open(io.BytesIO(jpeg)).size == (760, 560)


def test_film_puts_every_photo_on_a_strip_with_holes_through_it():
    from export.collage import film_layouts
    from export.collage_looks import film_strip

    found = film_layouts([1.5] * 5, 1000, 1000)
    assert [f["by"] for f in found] == ["film-across", "film-down"]
    for f in found:
        assert sorted(k for strip in f["strips"] for k in strip[5]) == [0, 1, 2, 3, 4]
    strip = film_strip([Image.new("RGB", (300, 200), "red")], [(10, 12, 150, 76)], [(0.5, 0.5)], 170, 100)
    alpha = np.asarray(strip.getchannel("A"))
    assert (alpha[:12] == 0).any() and (alpha[30:70, 20:150] == 255).all()   # holes in the band, the frame whole


def test_through_the_api_any_colour_a_pattern_and_a_photo_behind(client):
    c, paths = client
    base = {"filepaths": paths[:2], "format": "square", "preview": True}
    for extra in ({"background": "#3366cc", "pattern": "wood"},
                  {"style": "pinned", "pattern": "cork", "background": "#b98f62"},
                  {"style": "film", "background_photo": paths[2], "background_blur": False}):
        res = c.post("/api/library/collage", json={**base, **extra})
        assert res.status_code == 200, (extra, res.text)
    assert c.post("/api/library/collage", json={**base, "pattern": "plaid"}).status_code == 400
    assert c.post("/api/library/collage", json={**base, "background_photo": "nowhere/IMG.JPG"}).status_code == 404
    film = c.post("/api/library/collage/layouts", json={**base, "style": "film"}).json()
    assert [l["by"] for l in film["layouts"]] == ["film-across", "film-down"]


def test_a_centre_and_a_mosaic_put_the_first_photo_largest():
    from export.collage import centre_layout, mosaic_layout

    aspects = [1.5, 1.5, 0.67, 1.5, 1.33, 1.5, 0.75]
    for lay in (centre_layout(aspects, 2000, 1500, 10), mosaic_layout(aspects, 2000, 1500, 10)):
        areas = [c[2] * c[3] for c in lay["cells"]]
        assert areas[0] == max(areas) and len(lay["cells"]) == 7
        assert (_covered(lay["cells"], 2000, 1500) == 1).mean() > 0.97   # tiled, but for the gaps
    assert mosaic_layout([1.5] * 3, 1000, 1000) is None and centre_layout([1.5] * 2, 1000, 1000) is None


def test_a_fan_lays_the_first_photo_in_the_middle_on_top():
    from export.collage import polaroid_layouts

    fan = next(f for f in polaroid_layouts([1.5] * 7, 2000, 2000) if f["by"] == "fan")
    xs = [x + w / 2 for x, _y, w, _h in fan["cells"]]
    assert sorted(xs)[3] == xs[0] and fan["order"][-1] == 0
    assert fan["angles"][xs.index(min(xs))] > 0 > fan["angles"][xs.index(max(xs))]   # leaning out


def test_asked_for_one_layout_the_answer_is_the_best_not_the_evenest():
    """A hero's strip and each side of the centre layout ask for one layout: it
    used to be swapped for the evenest, which cut a portrait into a sliver."""
    from export.collage import _grid_layouts

    aspects = [0.67, 1.5, 1.5]
    best = _grid_layouts(aspects, 414, 952, 12, 6)[0]
    assert _grid_layouts(aspects, 414, 952, 12, 1)[0]["cells"] == best["cells"]
    assert best["crop"] < 0.25
