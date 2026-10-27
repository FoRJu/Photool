"""
Several photographs as one picture: rows of frames, each row the full width and
every frame whole (no cropping), the rows' heights chosen together so that none
comes out much taller or shorter than the rest. A title, when there is one, sits
above them.

The layout is a partition of the frames, in order, into rows: each row is scaled
to fill the width exactly, which fixes its height, and the partition picked is
the one whose row heights stray least from the target. With a few dozen frames
at most, every partition point is tried (a small dynamic programme).
"""

import io
import math
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 3000
GAP = 12
MARGIN = 48
BACKGROUND = (250, 249, 246)
INK = (40, 40, 40)


def rows(aspects: list[float], width: int = WIDTH, gap: int = GAP, target: Optional[float] = None) -> list[list[int]]:
    """Which frames go in which row: ``[[0, 1], [2, 3, 4], ...]``. ``aspects``
    are width / height; ``target`` is the row height aimed for (by default the
    one that makes about three frames a row, or the square root of their count)."""
    n = len(aspects)
    if n == 0:
        return []
    if target is None:
        per_row = max(1.0, min(4.0, round(n ** 0.5)))
        mean = sum(aspects) / n
        target = (width - gap * (per_row - 1)) / (per_row * mean)

    def height(i: int, j: int) -> float:
        return (width - gap * (j - i - 1)) / sum(aspects[i:j])

    cost = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - 6), j):
            h = height(i, j)
            # Straying short (a row of many) is as bad as straying tall, by ratio.
            stray = max(h / target, target / h) - 1.0
            c = cost[i] + stray * stray
            if c < cost[j]:
                cost[j], back[j] = c, i
    out, j = [], n
    while j > 0:
        out.append(list(range(back[j], j)))
        j = back[j]
    return out[::-1]


def _font(size: int):
    for name in ("segoeuil.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render(images: list[Image.Image], title: str = "", subtitle: str = "", width: int = WIDTH,
           gap: int = GAP, margin: int = MARGIN, background=BACKGROUND, quality: int = 90) -> bytes:
    """The collage as a JPEG."""
    images = [im.convert("RGB") for im in images if im.width > 0 and im.height > 0]
    if not images:
        raise ValueError("no pictures to lay out")
    inner = width - 2 * margin
    aspects = [im.width / im.height for im in images]
    layout = rows(aspects, inner, gap)
    heights = [round((inner - gap * (len(r) - 1)) / sum(aspects[k] for k in r)) for r in layout]

    head = 0
    if title:
        title_size = max(24, width // 22)
        sub_size = max(16, width // 60)
        head = round(title_size * 1.25) + (round(sub_size * 1.6) if subtitle else 0) + margin // 2
    total = margin + head + sum(heights) + gap * (len(layout) - 1) + margin
    canvas = Image.new("RGB", (width, total), background)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((margin, margin), title, font=_font(title_size), fill=INK)
        if subtitle:
            draw.text((margin, margin + round(title_size * 1.25)), subtitle, font=_font(sub_size), fill=(110, 110, 110))

    y = margin + head
    for row, h in zip(layout, heights):
        x = margin
        widths = [aspects[k] * h for k in row]
        # The last frame takes up the rounding, so every row ends at the margin.
        for n, k in enumerate(row):
            w = round(widths[n]) if n < len(row) - 1 else inner + margin - x
            canvas.paste(images[k].resize((max(1, w), h), Image.LANCZOS), (x, y))
            x += w + gap
        y += h + gap
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


# ── A fixed canvas: a phone story, a square, a print ─────────────────
#
# A print or a story has its size before any photograph is chosen, so the frames
# are cut to fit it rather than the other way round. A layout is rows (or
# columns) of the photographs in order. Each row first takes the height that
# fits its frames whole across the width; then the rows are stretched together to
# fill the height. That stretch is the same factor for every frame, so it is the
# whole of the crop, and the layouts offered are the ones that need least of it —
# weighed against how unequal the cells come out, since the least cut is often
# two big frames over a strip of four stamps.
# Each frame is cut around its people where its faces are known, else its middle.

FORMATS = {
    "story": (1080, 1920),        # 9:16, a phone's story
    "square": (2048, 2048),
    "4x6": (1800, 1200),          # prints at 300 dpi, landscape; swapped for portrait
    "5x7": (2100, 1500),
    "8x10": (3000, 2400),
    "wide": (3000, 2000),         # 3:2, a screen
}
MAX_PER_LINE = 4
LAYOUTS = 4
# A hero layout: the first photograph large on one side, the rest in the space
# left, laid out as the best grid of their own. The first is the one the person
# put first, which is the one they mean.
HEROES = 2
HERO_SHARES = (0.55, 0.62, 0.7)
# What a tenfold difference between the biggest and the smallest cell counts
# for, against the share of each frame cut away.
UNEVEN = 0.1


def _compositions(n: int, parts: int):
    """Every way to split ``n`` frames, in order, into ``parts`` lines of 1..MAX_PER_LINE."""
    if parts == 1:
        if 1 <= n <= MAX_PER_LINE:
            yield [n]
        return
    for first in range(1, min(MAX_PER_LINE, n - parts + 1) + 1):
        for rest in _compositions(n - first, parts - 1):
            yield [first] + rest


def _lines(aspects: list[float], counts: list[int], width: float, height: float, gap: float):
    """Cells ``[(x, y, w, h)]`` for rows of ``counts`` frames, and the stretch they need."""
    groups, k = [], 0
    for c in counts:
        groups.append(list(range(k, k + c)))
        k += c
    natural = [(width - gap * (len(g) - 1)) / sum(aspects[i] for i in g) for g in groups]
    stretch = (height - gap * (len(groups) - 1)) / sum(natural)
    cells, y = [None] * len(aspects), 0.0
    for g, h0 in zip(groups, natural):
        h = h0 * stretch
        total = sum(aspects[i] for i in g)
        x = 0.0
        for i in g:
            w = (width - gap * (len(g) - 1)) * aspects[i] / total
            cells[i] = (x, y, w, h)
            x += w + gap
        y += h + gap
    return cells, stretch


def layouts(aspects: list[float], width: int, height: int, gap: int = 0, limit: int = LAYOUTS,
            heroes: int = HEROES) -> list[dict]:
    """The best ``limit`` layouts of frames of these ``aspects`` (width / height) on
    a ``width`` × ``height`` canvas, best first (least cut, cells not too unequal),
    then up to ``heroes`` with the first frame large (:func:`hero_layouts`), then
    the first in the middle and as a mosaic (:func:`centre_layout`,
    :func:`mosaic_layout`), when the photographs allow them:
    ``[{cells, crop, lines, by}]`` — cells in pixels, crop the share of a frame
    cut away (0 is none)."""
    grids = _grid_layouts(aspects, width, height, gap, limit)
    extra = [f for f in (centre_layout(aspects, width, height, gap),
                         mosaic_layout(aspects, width, height, gap)) if f]
    return grids + hero_layouts(aspects, width, height, gap, heroes) + extra


def _grid_layouts(aspects: list[float], width: int, height: int, gap: int, limit: int) -> list[dict]:
    n = len(aspects)
    if n == 0:
        return []
    found = []
    for by in ("rows", "columns"):
        a = aspects if by == "rows" else [1.0 / x for x in aspects]
        w, h = (width, height) if by == "rows" else (height, width)
        for parts in range(1, n + 1):
            for counts in _compositions(n, parts):
                cells, stretch = _lines(a, counts, w, h, gap)
                if any(c[2] < 8 or c[3] < 8 for c in cells):
                    continue
                if by == "columns":
                    cells = [(y, x, ch, cw) for x, y, cw, ch in cells]
                crop = 1.0 - min(stretch, 1.0 / stretch)
                areas = [c[2] * c[3] for c in cells]
                uneven = UNEVEN * math.log10(max(areas) / min(areas))
                found.append({"cells": [tuple(round(v, 2) for v in c) for c in cells],
                              "crop": round(crop, 4), "lines": counts, "by": by,
                              "_score": crop + uneven})
    found.sort(key=lambda f: (f["_score"], len(f["lines"]), f["by"]))
    # The evenest layout is always offered, whatever it cuts: nine photos as a
    # grid of three by three is what a print of someone growing up wants.
    def unevenness(f):
        areas = [c[2] * c[3] for c in f["cells"]]
        return round(max(areas) / min(areas), 2)
    evenest = min(found, key=lambda f: (unevenness(f), f["crop"])) if found else None

    # Choices worth choosing between: the same rows in another order is not
    # much of one, so no more than two orders of one set of rows.
    out, seen, kinds = [], set(), {}
    for f in found:
        key = tuple(tuple(round(v) for v in c) for c in f["cells"])
        kind = (f["by"], tuple(sorted(f["lines"])))
        if key not in seen and kinds.get(kind, 0) < 2:
            seen.add(key)
            kinds[kind] = kinds.get(kind, 0) + 1
            out.append(f)
        if len(out) >= limit:
            break
    # Only among several: asked for the one best layout (a hero's strip, a side
    # of the centre layout) the answer is the best, not the evenest.
    if evenest is not None and evenest not in out and len(out) > 1:
        out[-1] = evenest
    for f in out:
        f.pop("_score", None)
    return out


def _cell_crop(cell_aspect: float, frame_aspect: float) -> float:
    r = cell_aspect / frame_aspect
    return 1.0 - min(r, 1.0 / r)


def hero_layouts(aspects: list[float], width: int, height: int, gap: int = 0,
                 limit: int = HEROES) -> list[dict]:
    """Up to ``limit`` layouts with the first frame filling a share of one side of
    the canvas and the others in the rest, least cut first, one per side."""
    if len(aspects) < 3 or limit <= 0:
        return []
    found = []
    for side in ("left", "top", "right", "bottom"):
        across = side in ("left", "right")
        best = None
        for share in HERO_SHARES:
            if across:
                hw = (width - gap) * share
                hero = (0.0 if side == "left" else width - hw, 0.0, hw, float(height))
                rest_w, rest_h = width - gap - hw, float(height)
                ox = hw + gap if side == "left" else 0.0
                oy = 0.0
            else:
                hh = (height - gap) * share
                hero = (0.0, 0.0 if side == "top" else height - hh, float(width), hh)
                rest_w, rest_h = float(width), height - gap - hh
                ox = 0.0
                oy = hh + gap if side == "top" else 0.0
            rest = _grid_layouts(aspects[1:], rest_w, rest_h, gap, 1)
            if not rest or hero[2] < 8 or hero[3] < 8:
                continue
            cells = [hero] + [(x + ox, y + oy, w, h) for x, y, w, h in rest[0]["cells"]]
            crops = [_cell_crop(c[2] / c[3], a) for c, a in zip(cells, aspects)]
            # The hero's own cut counts double: it is the picture people look at.
            score = (2 * crops[0] + sum(crops[1:])) / (len(crops) + 1)
            option = {"cells": [tuple(round(v, 2) for v in c) for c in cells],
                      "crop": round(sum(crops) / len(crops), 4), "lines": [1] + rest[0]["lines"],
                      "by": f"hero-{side}", "_score": score}
            if best is None or score < best["_score"]:
                best = option
        if best:
            found.append(best)
    found.sort(key=lambda f: f["_score"])
    for f in found:
        f.pop("_score", None)
    return found[:limit]


def _option(cells: list, aspects: list[float], by: str, lines: list) -> dict:
    crops = [_cell_crop(c[2] / c[3], a) for c, a in zip(cells, aspects)]
    return {"cells": [tuple(round(v, 2) for v in c) for c in cells],
            "crop": round(sum(crops) / len(crops), 4), "lines": lines, "by": by,
            "_score": (2 * crops[0] + sum(crops[1:])) / (len(crops) + 1)}


def centre_layout(aspects: list[float], width: int, height: int, gap: int = 0) -> Optional[dict]:
    """The first photograph large in the middle, the others in two groups either
    side of it — left and right on a wide page, above and below on a tall one.
    None for fewer than three."""
    n = len(aspects)
    if n < 3:
        return None
    wide = width >= height
    rest = aspects[1:]
    first_half = (len(rest) + 1) // 2
    groups = (rest[:first_half], rest[first_half:])
    best = None
    for share in (0.42, 0.5, 0.58):
        if wide:
            hero_w = (width - 2 * gap) * share
            side = (width - 2 * gap - hero_w) / 2
            hero = (side + gap, 0.0, hero_w, float(height))
            regions = ((0.0, 0.0, side, float(height)), (side + 2 * gap + hero_w, 0.0, side, float(height)))
        else:
            hero_h = (height - 2 * gap) * share
            side = (height - 2 * gap - hero_h) / 2
            hero = (0.0, side + gap, float(width), hero_h)
            regions = ((0.0, 0.0, float(width), side), (0.0, side + 2 * gap + hero_h, float(width), side))
        cells, lines, ok = [hero], [1], True
        for group, (rx, ry, rw, rh) in zip(groups, regions):
            if not group:
                continue
            found = _grid_layouts(list(group), rw, rh, gap, 1)
            if not found:
                ok = False
                break
            cells += [(x + rx, y + ry, w, h) for x, y, w, h in found[0]["cells"]]
            lines += found[0]["lines"]
        if not ok or hero[2] < 8 or hero[3] < 8:
            continue
        option = _option(cells, aspects, "centre", lines)
        if best is None or option["_score"] < best["_score"]:
            best = option
    if best:
        best.pop("_score")
    return best


def mosaic_layout(aspects: list[float], width: int, height: int, gap: int = 0) -> Optional[dict]:
    """A magazine page: the first photograph across a 2 × 2 block in the top left
    corner of a grid, the others filling the cells around it in order, the last
    of a row widened to its end when the photographs run out. None for fewer
    than four."""
    n = len(aspects)
    if n < 4:
        return None
    best = None
    for cols in range(3, 6):
        for rows in range(2, 6):
            free = rows * cols - 4
            if free < n - 1:
                continue
            # The cells taken, in order: no row left empty.
            slots = [(r, c) for r in range(rows) for c in range(cols) if not (r < 2 and c < 2)][:n - 1]
            if any(all(sr != r for sr, _ in slots) for r in range(2, rows)):
                continue
            uw = (width - gap * (cols - 1)) / cols
            uh = (height - gap * (rows - 1)) / rows
            if uw < 8 or uh < 8:
                continue
            cells = [(0.0, 0.0, 2 * uw + gap, 2 * uh + gap)]
            for k, (r, c) in enumerate(slots):
                x, y = c * (uw + gap), r * (uh + gap)
                w = uw
                last_in_row = k == len(slots) - 1 or slots[k + 1][0] != r
                if last_in_row:
                    w = width - x                       # the row runs to the edge
                cells.append((x, y, w, uh))
            option = _option(cells, aspects, "mosaic", [cols] * rows)
            if best is None or option["_score"] < best["_score"]:
                best = option
    if best:
        best.pop("_score")
    return best


# Styles: how each photograph sits in its cell.
STYLES = ("clean", "rounded", "gallery", "polaroid", "taped", "pinned", "stamps", "film")
# Styles whose pictures lie loose on the surface, each its own shape: they are
# arranged as prints (polaroid_layouts), not cut into a grid.
PRINT_STYLES = ("polaroid", "taped", "pinned", "stamps")


def _rounded(image: Image.Image, radius: int) -> Image.Image:
    """An RGBA copy with its corners rounded."""
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius, fill=255)
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


# Instant prints dropped on a table. A print keeps its own shape, near square as
# prints are, so the grids above — chosen for what they cut off photographs laid
# edge to edge — are no guide; the prints have arrangements of their own
# (:func:`polaroid_layouts`). The page is drawn at SUPERSAMPLE times its size and
# brought down at the end, which is what keeps the edges of a turned print, and
# its shadow, smooth.
SUPERSAMPLE = 2
PRINT_JITTER = 0.06          # how far off its place a print lands, of its size
PRINT_TILT = 6.0             # most degrees a print is turned, either way
PRINT_ASPECT = (0.8, 1.25)   # the narrowest and widest a print comes, width / height
SHADOW_BLUR = 0.025          # of the print's shorter side
SHADOW_DROP = 0.012          # how far the light puts the shadow, down and right
SHADOW_STRENGTH = 0.42


def _lay_print(page: Image.Image, piece: Image.Image, centre: tuple, angle: float,
               bounds: tuple) -> None:
    """Put ``piece`` (RGBA) on ``page`` around ``centre``, turned by ``angle``,
    with a soft shadow, moved inside ``bounds`` (left, top, right, bottom) if it
    would cross them."""
    side = min(piece.size)
    blur = max(1.0, side * SHADOW_BLUR)
    drop = side * SHADOW_DROP
    pad = int(blur * 3 + drop) + 2
    # Room around the piece for the blur to fade out in, and a transparent rim
    # for the turn to soften the edge against.
    sheet = Image.new("RGBA", (piece.width + 2 * pad, piece.height + 2 * pad), (0, 0, 0, 0))
    sheet.alpha_composite(piece.convert("RGBA"), (pad, pad))
    if angle:
        sheet = sheet.rotate(angle, resample=Image.BICUBIC, expand=True)
    alpha = sheet.getchannel("A")
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(blur)).point(lambda a: int(a * SHADOW_STRENGTH))
    shadow = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)

    # Where it lands: centred, then kept on the page by its visible extent.
    x = centre[0] - sheet.width / 2
    y = centre[1] - sheet.height / 2
    box = alpha.getbbox() or (0, 0, sheet.width, sheet.height)
    left, top, right, bottom = bounds
    x += max(0.0, left - (x + box[0])) - max(0.0, (x + box[2]) - right)
    y += max(0.0, top - (y + box[1])) - max(0.0, (y + box[3]) - bottom)
    page.alpha_composite(shadow, (round(x + drop), round(y + drop)))
    page.alpha_composite(sheet, (round(x), round(y)))


def _print_size(aspect: float, side: float) -> tuple[float, float]:
    """A print of about ``side`` square in area, in its photograph's shape (kept
    within PRINT_ASPECT)."""
    a = min(PRINT_ASPECT[1], max(PRINT_ASPECT[0], aspect))
    return side * math.sqrt(a), side / math.sqrt(a)


def polaroid_layouts(aspects: list[float], width: int, height: int) -> list[dict]:
    """Ways to lay these prints on a ``width`` × ``height`` table:

    * **scatter** — all one size on a loose grid that fills the page, the last
      row centred;
    * **feature** — the first large in the middle, the rest around the edges
      tucked under it (three photographs or more);
    * **pile** — tossed towards the middle, more overlap, more turn.

    ``[{cells, angles, order, by, crop, lines}]``: each print's unturned box in
    pixels, its turn in degrees, and the order to lay them down (last on top).
    The same photographs always fall the same way, so a redraw does not jump.
    """
    import random

    n = len(aspects)
    if n == 0:
        return []
    seed = n * 7919 + sum(round(a * 100) * (k + 1) for k, a in enumerate(aspects))
    out = []

    def place(rng, centre, size, a, tilt=PRINT_TILT):
        w, h = _print_size(a, size)
        cx = centre[0] + rng.uniform(-PRINT_JITTER, PRINT_JITTER) * w
        cy = centre[1] + rng.uniform(-PRINT_JITTER, PRINT_JITTER) * h
        return (cx - w / 2, cy - h / 2, w, h), rng.uniform(-tilt, tilt)

    # Scatter: the grid that gives the biggest prints, judged by the prints'
    # own shape (wide prints fit three across a square page as two rows, or two
    # across as three rows, and the second is bigger), then the rows packed
    # together and centred on the page rather than spread over it.
    rng = random.Random(seed)
    shape = sum(min(PRINT_ASPECT[1], max(PRINT_ASPECT[0], a)) for a in aspects) / n
    best = None
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        side = min(width / (cols * math.sqrt(shape)), height / (rows / math.sqrt(shape)))
        if best is None or side > best[0] + 1e-6:
            best = (side, rows, cols)
    side, rows, cols = best
    side *= 0.92                                         # room to tilt
    pitch_w, pitch_h = side * math.sqrt(shape) * 1.04, side / math.sqrt(shape) * 1.04
    top = (height - rows * pitch_h) / 2
    cells, angles = [], []
    for k in range(n):
        r, c = divmod(k, cols)
        in_row = min(cols, n - r * cols)
        left = (width - in_row * pitch_w) / 2            # every row centred, a short last one too
        centre = (left + (c + 0.5) * pitch_w, top + (r + 0.5) * pitch_h)
        cell, angle = place(rng, centre, side, aspects[k])
        cells.append(cell)
        angles.append(angle)
    out.append({"by": "scatter", "cells": cells, "angles": angles, "order": list(range(n))})

    # Feature: the first large in the middle, the others around it.
    if n >= 3:
        rng = random.Random(seed + 1)
        short = min(width, height)
        hero_side = short * (0.62 if n <= 5 else 0.56)
        others = n - 1
        # Around an ellipse near the page's edge, starting at the top left.
        rx, ry = width * 0.36, height * 0.36
        spacing = 2 * math.pi * math.sqrt((rx * rx + ry * ry) / 2) / others
        small = min(short * 0.34, spacing * 0.95)
        cells, angles = [None] * n, [0.0] * n
        cells[0], angles[0] = place(rng, (width / 2, height / 2), hero_side, aspects[0], tilt=3.0)
        for j in range(others):
            t = -3 * math.pi / 4 + 2 * math.pi * j / others
            centre = (width / 2 + rx * math.cos(t), height / 2 + ry * math.sin(t))
            cells[j + 1], angles[j + 1] = place(rng, centre, small, aspects[j + 1], tilt=PRINT_TILT * 1.4)
        out.append({"by": "feature", "cells": cells, "angles": angles, "order": list(range(1, n)) + [0]})

    # Pile: tossed towards the middle.
    rng = random.Random(seed + 2)
    short = min(width, height)
    size = short * min(0.6, max(0.38, 0.72 / math.sqrt(max(1, n) / 2)))
    cells, angles = [], []
    for k in range(n):
        centre = (width * (0.5 + rng.uniform(-0.34, 0.34)), height * (0.5 + rng.uniform(-0.34, 0.34)))
        cell, angle = place(rng, centre, size, aspects[k], tilt=PRINT_TILT * 2)
        cells.append(cell)
        angles.append(angle)
    out.append({"by": "pile", "cells": cells, "angles": angles, "order": list(range(n - 1, -1, -1))})

    # Fan: spread like a hand of cards from below the page, the first in the
    # middle and on top, the others out to either side beneath it.
    if n >= 3:
        rng = random.Random(seed + 3)
        size = short * min(0.5, max(0.3, 0.9 / math.sqrt(n)))
        spread = min(70.0, 14.0 * (n - 1))                 # degrees, end to end
        radius = height * 0.62
        pivot = (width / 2, height * 0.5 + radius * 0.82)
        # Places from the left; the first photograph takes the middle one.
        places = list(range(n))
        middle = (n - 1) // 2
        slot_of = [middle] + [p for p in places if p != middle]
        cells, angles = [None] * n, [0.0] * n
        for k, slot in enumerate(slot_of):
            theta = math.radians(-spread / 2 + spread * slot / (n - 1))
            centre = (pivot[0] + radius * math.sin(theta), pivot[1] - radius * math.cos(theta))
            w, h = _print_size(aspects[k], size)
            cells[k] = (centre[0] - w / 2, centre[1] - h / 2, w, h)
            angles[k] = -math.degrees(theta) + rng.uniform(-1.5, 1.5)
        # Outside in, so the middle lies on top.
        order = sorted(range(n), key=lambda k: -abs(slot_of[k] - middle))
        out.append({"by": "fan", "cells": cells, "angles": angles, "order": order})

    # Cascade: overlapping down the page from the top left, the first on top.
    if n >= 2:
        rng = random.Random(seed + 4)
        size = short * min(0.55, max(0.32, 1.1 / math.sqrt(n + 1)))
        cells, angles = [], []
        for k in range(n):
            t = k / (n - 1)
            centre = (width * (0.27 + 0.46 * t), height * (0.27 + 0.46 * t))
            w, h = _print_size(aspects[k], size)
            cells.append((centre[0] - w / 2, centre[1] - h / 2, w, h))
            angles.append((4.0 if k % 2 else -4.0) + rng.uniform(-2, 2))
        out.append({"by": "cascade", "cells": cells, "angles": angles, "order": list(range(n - 1, -1, -1))})

    for f in out:
        f["cells"] = [tuple(round(v, 2) for v in c) for c in f["cells"]]
        f["angles"] = [round(a, 2) for a in f["angles"]]
        f["crop"], f["lines"] = 0.0, []
    return out


def _prints(page: Image.Image, images: list, arrangement: dict, style: str, margin: int,
            focus: Optional[list]) -> None:
    """Lay the pictures on ``page`` (already SUPERSAMPLE times the canvas) as
    ``arrangement`` (one of :func:`polaroid_layouts`), each made by ``style``'s card."""
    import random

    from export import collage_looks

    ss = SUPERSAMPLE
    make = collage_looks.CARDS[style]
    width, height = page.size
    edge = max(margin * ss, round(min(width, height) * 0.02))
    bounds = (edge, edge, width - edge, height - edge)
    room_w, room_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    rng = random.Random(len(images) * 104729 + len(arrangement["cells"]))
    for k in arrangement["order"]:
        x, y, w, h = arrangement["cells"][k]
        angle = arrangement["angles"][k]
        if style == "pinned":
            angle *= 0.5            # a pin holds a card straighter
        cw, ch = w * ss, h * ss
        # Turned, it must still fit on the page, with a little to spare.
        turn = math.radians(abs(angle))
        span_w = cw * math.cos(turn) + ch * math.sin(turn)
        span_h = cw * math.sin(turn) + ch * math.cos(turn)
        fit = min(1.0, 0.96 * room_w / span_w, 0.96 * room_h / span_h)
        centre = ((margin + x + w / 2) * ss, (margin + y + h / 2) * ss)
        point = (focus[k] if focus and k < len(focus) and focus[k] else None) or (0.5, 0.5)
        piece = make(images[k], max(8, round(cw * fit)), max(8, round(ch * fit)), point, rng)
        _lay_print(page, piece, centre, angle, bounds)


# Film: the photographs in frames along strips of film, the strips across the
# page (or down it), the last one shorter when the frames do not divide evenly.
FILM_FRAME = 1.5             # a frame's width / height, as 35 mm
FILM_BAND = 0.12             # the sprocket band on each edge, of the strip's width


def film_layouts(aspects: list[float], width: int, height: int) -> list[dict]:
    """Strips across the page and strips down it: ``[{by, cells, strips}]`` with
    ``strips`` as ``[(x, y, length, thick, across, [frame indices])]``.

    A frame is FILM_FRAME times longer along the film than across it, as on 35 mm
    film, so strips across the page carry landscape frames and strips down it
    upright ones. The number of strips is the one that makes the frames largest.
    """
    n = len(aspects)
    if n == 0:
        return []
    out = []
    for across in (True, False):
        along_page, across_page = (width, height) if across else (height, width)
        best = None
        for rows in range(1, n + 1):
            per = math.ceil(n / rows)
            # Along: per frames and per + 1 gaps of 6% of a frame.
            by_length = 0.94 * along_page / (per + 0.06 * (per + 1))
            # Across: rows strips, each a frame and two sprocket bands, 18% apart.
            thick_per_frame = 1 / FILM_FRAME / (1 - 2 * FILM_BAND)
            by_width = 0.94 * across_page / (thick_per_frame * (rows + 0.18 * (rows - 1)))
            frame = min(by_length, by_width)
            if best is None or frame > best[0] + 1e-6:
                best = (frame, rows, per)
        frame, rows, per = best
        f_along, f_across = frame, frame / FILM_FRAME
        thick = f_across / (1 - 2 * FILM_BAND)
        band, gap, spacing = thick * FILM_BAND, frame * 0.06, thick * 0.18
        start = (across_page - (rows * thick + (rows - 1) * spacing)) / 2
        cells, strips = [None] * n, []
        for r in range(rows):
            members = list(range(r * per, min(n, (r + 1) * per)))
            if not members:
                continue
            length = len(members) * f_along + (len(members) + 1) * gap
            s0 = (along_page - length) / 2             # a shorter last strip, centred
            t0 = start + r * (thick + spacing)
            for j, k in enumerate(members):
                a = s0 + gap + j * (f_along + gap)
                cells[k] = ((a, t0 + band, f_along, f_across) if across
                            else (t0 + band, a, f_across, f_along))
            strips.append((s0, t0, length, thick, True, members) if across
                          else (t0, s0, length, thick, False, members))
        out.append({"by": "film-across" if across else "film-down",
                    "cells": [tuple(round(v, 2) for v in c) for c in cells], "strips": strips,
                    "crop": 0.0, "lines": [len(st[5]) for st in strips]})
    return out


def layouts_for(style: str, aspects: list[float], width: int, height: int, gap: int = 0) -> list[dict]:
    """The layouts a style offers: arrangements of loose prints, strips of film,
    or the grids (which cut least off photographs laid edge to edge)."""
    if style in PRINT_STYLES:
        return polaroid_layouts(aspects, width, height)
    if style == "film":
        return film_layouts(aspects, width, height)
    return layouts(aspects, width, height, gap)


def _cover(image: Image.Image, w: int, h: int, focus: tuple = (0.5, 0.5)) -> Image.Image:
    """The image cut to ``w`` × ``h``, filling it, the cut kept around ``focus``."""
    iw, ih = image.size
    scale = max(w / iw, h / ih)
    # A cell nearly the photo's own shape asks for a window a hair wider than it.
    cw, ch = min(iw, w / scale), min(ih, h / scale)
    left = max(0.0, min(focus[0] * iw - cw / 2, iw - cw))
    top = max(0.0, min(focus[1] * ih - ch / 2, ih - ch))
    return image.resize((w, h), Image.LANCZOS, box=(left, top, left + cw, top + ch))


def render_canvas(images: list[Image.Image], width: int, height: int, layout: int = 0, gap: int = 0,
                  margin: int = 0, background=(255, 255, 255), focus: Optional[list] = None,
                  quality: int = 92, style: str = "clean", pattern: str = "",
                  backdrop_photo: Optional[Image.Image] = None, backdrop_blur: bool = True) -> bytes:
    """``images`` laid out as layout number ``layout`` of :func:`layouts_for`
    ``style`` on a ``width`` × ``height`` canvas; ``focus`` is each image's point
    to keep, as fractions of it (None for its middle). The surface is
    ``background`` (a colour) with ``pattern`` drawn on it, or ``backdrop_photo``
    (see :mod:`export.collage_looks`)."""
    from export import collage_looks

    images = [im.convert("RGB") for im in images]
    if not images:
        raise ValueError("no pictures to lay out")
    if style not in STYLES:
        raise ValueError(f"no style {style!r}")
    inner_w, inner_h = width - 2 * margin, height - 2 * margin
    options = layouts_for(style, [im.width / im.height for im in images], inner_w, inner_h, gap)
    chosen = options[min(max(0, layout), len(options) - 1)]
    points = [(focus[k] if focus and k < len(focus) and focus[k] else None) or (0.5, 0.5)
              for k in range(len(images))]

    loose = style in PRINT_STYLES or style in ("gallery", "film")
    ss = SUPERSAMPLE if loose else 1
    surface = collage_looks.backdrop(width * ss, height * ss, collage_looks.parse_colour(background),
                                     pattern, backdrop_photo, backdrop_blur)
    if loose:
        # Things lying on a table: the light falls off towards its edges, and
        # everything is drawn large and brought down, for smooth turned edges.
        page = collage_looks.vignette(surface).convert("RGBA")
        if style in PRINT_STYLES:
            _prints(page, images, chosen, style, margin, focus)
        elif style == "gallery":
            bounds = (0, 0, page.width, page.height)
            rng = None
            for k, (x, y, w, h) in enumerate(chosen["cells"]):
                inset = min(w, h) * 0.05
                cw, ch = (w - 2 * inset) * ss, (h - 2 * inset) * ss
                piece = collage_looks.framed(images[k], max(8, round(cw)), max(8, round(ch)), points[k], rng)
                _lay_print(page, piece, ((margin + x + w / 2) * ss, (margin + y + h / 2) * ss), 0.0, bounds)
        else:
            import random

            rng = random.Random(len(images) * 31)
            bounds = (0, 0, page.width, page.height)
            for (sx, sy, length, thick, across, members) in chosen["strips"]:
                frames = []
                for k in members:
                    x, y, w, h = chosen["cells"][k]
                    frames.append(((x - sx) * ss, (y - sy) * ss, w * ss, h * ss))
                strip = collage_looks.film_strip([images[k] for k in members], frames,
                                                 [points[k] for k in members], length * ss, thick * ss, across)
                w_, h_ = (length, thick) if across else (thick, length)
                centre = ((margin + sx + w_ / 2) * ss, (margin + sy + h_ / 2) * ss)
                _lay_print(page, strip, centre, rng.uniform(-1.5, 1.5), bounds)
        canvas = page.convert("RGB").resize((width, height), Image.LANCZOS)
    else:
        canvas = surface
        for k, (image, (x, y, w, h)) in enumerate(zip(images, chosen["cells"])):
            # Rounded from the edges, so neighbouring cells meet without a seam.
            x0, y0 = round(margin + x), round(margin + y)
            x1, y1 = round(margin + x + w), round(margin + y + h)
            cw, ch = max(1, x1 - x0), max(1, y1 - y0)
            if style == "rounded":
                piece = _rounded(_cover(image, cw, ch, points[k]), max(2, round(min(cw, ch) * 0.06)))
                canvas.paste(piece, (x0, y0), piece)
            else:
                canvas.paste(_cover(image, cw, ch, points[k]), (x0, y0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality, dpi=(300, 300))
    return buffer.getvalue()
