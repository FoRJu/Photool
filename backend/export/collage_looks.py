"""
How a collage looks, apart from where its photographs go (export/collage.py):
the surface they lie on, and what each one is — a print, a taped snapshot, a
pinned card, a stamp, a framed picture, a strip of film.

The surface is a colour, a pattern drawn in that colour, or a photograph of the
person's own, blurred and dimmed so the pictures on it still stand out. Patterns
are drawn, not loaded: their sizes follow the canvas's shorter side, so a preview
and the saved picture match.
"""

import math
import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

PATTERNS = ("", "dots", "stripes", "grid", "linen", "wood", "cork", "paper")


def parse_colour(value, default=(255, 255, 255)) -> tuple:
    """``#rrggbb`` (or ``#rgb``) as an RGB tuple; a tuple passes through."""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(int(v) for v in value)
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        if len(text) == 6:
            return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        pass
    return default


def _ink(colour: tuple, amount: float) -> tuple:
    """A tone of ``colour`` that shows on it: lighter on a dark one, darker on a light one."""
    light = sum(colour) / 3 > 140
    target = 0 if light else 255
    return tuple(round(c + (target - c) * amount) for c in colour)


def _noise(rng: np.random.Generator, w: int, h: int, blur: float) -> np.ndarray:
    """Smooth noise, about -1..1."""
    field = Image.fromarray(((rng.standard_normal((h, w)) * 40) + 128).clip(0, 255).astype(np.uint8))
    if blur:
        field = field.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(field, dtype=np.float32) - 128
    return a / (np.abs(a).max() or 1)


def _shade(colour: tuple, delta: np.ndarray) -> Image.Image:
    """``colour`` lightened or darkened per pixel by ``delta`` (in levels)."""
    base = np.array(colour, dtype=np.float32)[None, None, :]
    return Image.fromarray(np.clip(base + delta[..., None], 0, 255).astype(np.uint8))


def pattern(width: int, height: int, colour: tuple, name: str) -> Image.Image:
    """``colour`` with pattern ``name`` drawn over it (see PATTERNS)."""
    s = min(width, height)
    page = Image.new("RGB", (width, height), colour)
    if not name:
        return page
    draw = ImageDraw.Draw(page)
    rng = np.random.default_rng(7)
    if name == "dots":
        step, r = s / 22, s / 110
        ink = _ink(colour, 0.18)
        for row, y in enumerate(np.arange(step / 2, height + step, step * 0.866)):
            for x in np.arange(step / 2 + (step / 2 if row % 2 else 0), width + step, step):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=ink)
    elif name == "stripes":
        band = s / 36
        ink = _ink(colour, 0.08)
        for k in range(-int(height / band) - 2, int(width / band) + 2, 2):
            x = k * band
            draw.polygon([(x, 0), (x + band, 0), (x + band + height, height), (x + height, height)], fill=ink)
    elif name == "grid":
        fine, bold = _ink(colour, 0.12), _ink(colour, 0.25)
        step = s / 40
        for k, x in enumerate(np.arange(0, width + 1, step)):
            draw.line((x, 0, x, height), fill=bold if k % 5 == 0 else fine, width=max(1, round(s / (900 if k % 5 else 500))))
        for k, y in enumerate(np.arange(0, height + 1, step)):
            draw.line((0, y, width, y), fill=bold if k % 5 == 0 else fine, width=max(1, round(s / (900 if k % 5 else 500))))
    elif name == "linen":
        # Threads both ways: noise stretched along x, and along y.
        across = np.asarray(Image.fromarray(((rng.standard_normal((height, max(1, width // 8))) * 50) + 128)
                                            .clip(0, 255).astype(np.uint8)).resize((width, height), Image.BILINEAR),
                            dtype=np.float32) - 128
        down = np.asarray(Image.fromarray(((rng.standard_normal((max(1, height // 8), width)) * 50) + 128)
                                          .clip(0, 255).astype(np.uint8)).resize((width, height), Image.BILINEAR),
                          dtype=np.float32) - 128
        page = _shade(colour, (across + down) * 0.12)
    elif name == "wood":
        # Planks across the page: grain as streaks stretched along them (noise
        # far longer than it is tall), a few broad bands of tone, and a seam
        # between planks.
        def streaks(length: int, rows: int, strength: float) -> np.ndarray:
            field = (rng.standard_normal((rows, length)) * 50 + 128).clip(0, 255).astype(np.uint8)
            return (np.asarray(Image.fromarray(field).resize((width, height), Image.BICUBIC),
                               dtype=np.float32) - 128) / 50 * strength

        grain = streaks(max(4, width // 90), max(8, height // 3), 9)
        fine = streaks(max(8, width // 20), max(8, height), 4)
        tone = streaks(3, max(4, height // 60), 10)
        delta = grain + fine + tone
        plank = s / 4.5
        for y in np.arange(plank, height, plank):
            y0 = int(y)
            delta[max(0, y0 - 1):y0 + max(1, round(s / 700)), :] -= 38
        page = _shade(colour, delta)
    elif name == "cork":
        grains = _noise(rng, width, height, max(0.6, s / 900))
        speck = (rng.random((height, width)) > 0.985) * -40.0
        page = _shade(colour, grains * 22 + speck)
        page = page.filter(ImageFilter.GaussianBlur(max(0.3, s / 3000)))
    elif name == "paper":
        fine = _noise(rng, width, height, 0.5)
        blotch = _noise(rng, max(8, width // 24), max(8, height // 24), 1.2)
        blotch = np.asarray(Image.fromarray(((blotch + 1) * 127.5).astype(np.uint8)).resize((width, height), Image.BICUBIC),
                            dtype=np.float32) / 127.5 - 1
        page = _shade(colour, fine * 5 + blotch * 6)
    return page


def backdrop(width: int, height: int, colour: tuple, name: str = "",
             photo: Optional[Image.Image] = None, blur: bool = True, focus=(0.5, 0.5)) -> Image.Image:
    """The surface: a photograph (blurred and dimmed, so what lies on it stands
    out) when one is given, else ``colour`` with pattern ``name``."""
    if photo is None:
        return pattern(width, height, colour, name)
    from export.collage import _cover

    page = _cover(photo.convert("RGB"), width, height, focus)
    if blur:
        page = page.filter(ImageFilter.GaussianBlur(min(width, height) * 0.018))
    return ImageEnhance.Brightness(page).enhance(0.78)


def vignette(page: Image.Image) -> Image.Image:
    """A touch darker towards the edges, as light falls on a table."""
    width, height = page.size
    shade = Image.radial_gradient("L").resize((width, height), Image.BILINEAR)
    dark = ImageEnhance.Brightness(page).enhance(0.88)
    return Image.composite(dark, page, shade.point(lambda v: min(255, int(v * 0.75))))


# ── What each picture is ─────────────────────────────────────────────
# Each returns an RGBA piece, unturned, with any part that reaches past the card
# (tape) already on it; the caller turns it and casts its shadow.

PAPER = (250, 250, 246)
TAPES = [(246, 191, 204), (184, 228, 214), (250, 226, 150), (180, 208, 240), (222, 200, 240)]
PINS = [(214, 52, 52), (40, 110, 200), (240, 190, 40), (40, 160, 90)]


def _bordered(image: Image.Image, w: int, h: int, focus, edge: float, foot: float) -> Image.Image:
    from export.collage import _cover

    side = min(w, h)
    border = max(2, round(side * edge))
    bottom = max(border, round(side * foot))
    card = Image.new("RGBA", (w, h), PAPER + (255,))
    card.paste(_cover(image, max(1, w - 2 * border), max(1, h - border - bottom), focus), (border, border))
    ImageDraw.Draw(card).rectangle((0, 0, w - 1, h - 1), outline=(226, 226, 220, 255), width=max(1, side // 400))
    return card


def polaroid(image, w, h, focus, rng) -> Image.Image:
    return _bordered(image, w, h, focus, 0.055, 0.19)


def taped(image, w, h, focus, rng) -> Image.Image:
    """A snapshot with a thin white edge, held down by washi tape: one strip
    across the top, or one over each top corner."""
    card = _bordered(image, w, h, focus, 0.035, 0.035)
    side = min(w, h)
    over = round(side * 0.09)
    sheet = Image.new("RGBA", (w + 2 * over, h + 2 * over), (0, 0, 0, 0))
    sheet.paste(card, (over, over))
    colour = TAPES[rng.randrange(len(TAPES))]
    strips = ([(over + w / 2, over, rng.uniform(-6, 6))] if rng.random() < 0.5 else
              [(over + side * 0.05, over + side * 0.05, -42 + rng.uniform(-5, 5)),
               (over + w - side * 0.05, over + side * 0.05, 42 + rng.uniform(-5, 5))])
    for cx, cy, angle in strips:
        tw, th = round(side * 0.32), round(side * 0.085)
        tape = Image.new("RGBA", (tw, th), colour + (200,))
        # Torn ends: a zigzag cut from each short side.
        mask = Image.new("L", (tw, th), 0)
        teeth = max(3, th // 6)
        points = [(0, 0)] + [(tw * 0.0 + (teeth * 0.6 if k % 2 else 0), th * k / 6) for k in range(1, 7)]
        right = [(tw - (teeth * 0.6 if k % 2 else 0), th * (6 - k) / 6) for k in range(0, 7)]
        ImageDraw.Draw(mask).polygon(points + right, fill=200)
        tape.putalpha(mask)
        tape = tape.rotate(angle, resample=Image.BICUBIC, expand=True)
        sheet.alpha_composite(tape, (round(cx - tape.width / 2), round(cy - tape.height / 2)))
    return sheet


def pinned(image, w, h, focus, rng) -> Image.Image:
    """A card with a push-pin through its top."""
    card = _bordered(image, w, h, focus, 0.04, 0.04)
    side = min(w, h)
    r = max(3, round(side * 0.042))
    cx, cy = w / 2 + rng.uniform(-0.08, 0.08) * w, max(r * 1.4, side * 0.045)
    colour = PINS[rng.randrange(len(PINS))]
    shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((cx - r + r * 0.5, cy - r + r * 0.7, cx + r + r * 0.5, cy + r + r * 0.7),
                                   fill=(0, 0, 0, 110))
    card.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(r * 0.35)))
    draw = ImageDraw.Draw(card)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colour + (255,),
                 outline=tuple(int(c * 0.6) for c in colour) + (255,), width=max(1, r // 6))
    hr = r * 0.38
    draw.ellipse((cx - r * 0.45 - hr, cy - r * 0.45 - hr, cx - r * 0.45 + hr, cy - r * 0.45 + hr),
                 fill=tuple(min(255, int(c + (255 - c) * 0.6)) for c in colour) + (255,))
    return card


def stamp(image, w, h, focus, rng) -> Image.Image:
    """A postage stamp: perforated edges, a white margin, the picture inside."""
    card = _bordered(image, w, h, focus, 0.085, 0.085)
    side = min(w, h)
    r = max(2, side * 0.022)
    step = r * 2.7
    alpha = card.getchannel("A")
    draw = ImageDraw.Draw(alpha)
    for x in np.arange(step / 2, w, step):
        for y in (0, h):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
    for y in np.arange(step / 2, h, step):
        for x in (0, w):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
    card.putalpha(alpha)
    return card


def framed(image, w, h, focus, rng, caption: Optional[dict] = None) -> Image.Image:
    """A picture in a white mat and a dark frame, as on a gallery wall; with a
    ``caption`` (see :func:`caption_of`) the mat is weighted at the bottom, as
    framers do, and the line written there."""
    from export.collage import _cover

    side = min(w, h)
    frame = max(2, round(side * 0.03))
    mat = max(2, round(side * 0.085))
    foot = round(mat * 1.7) if caption else mat
    piece = Image.new("RGBA", (w, h), (34, 32, 30, 255))
    draw = ImageDraw.Draw(piece)
    draw.rectangle((frame, frame, w - frame - 1, h - frame - 1), fill=(248, 247, 243, 255))
    inner, under = frame + mat, frame + foot
    piece.paste(_cover(image, max(1, w - 2 * inner), max(1, h - inner - under), focus), (inner, inner))
    # The mat's bevel: a fine line where it is cut.
    draw.rectangle((inner - 2, inner - 2, w - inner + 1, h - under + 1), outline=(220, 218, 210, 255), width=2)
    if caption:
        _mat_caption(piece, caption, (inner, h - under + 2, w - inner, h - frame), side)
    return piece


CARDS = {"polaroid": polaroid, "taped": taped, "pinned": pinned, "stamps": stamp, "gallery": framed}


def film_strip(images: list, frames: list, focus: list, length: float, thick: float,
               across: bool = True) -> Image.Image:
    """A strip of film ``length`` long and ``thick`` wide carrying ``images`` in
    ``frames`` (x, y, w, h, relative to the strip), sprocket holes along both
    edges, the holes cut through so the surface shows."""
    from export.collage import _cover

    w, h = (round(length), round(thick)) if across else (round(thick), round(length))
    strip = Image.new("RGBA", (w, h), (24, 22, 20, 255))
    for image, (x, y, fw, fh), point in zip(images, frames, focus):
        strip.paste(_cover(image, max(1, round(fw)), max(1, round(fh)), point), (round(x), round(y)))
    alpha = strip.getchannel("A")
    draw = ImageDraw.Draw(alpha)
    band = thick * 0.12
    hole_w, hole_h = band * 0.55, band * 0.75
    step = band * 1.25
    for t in np.arange(step / 2, length, step):
        for edge in (band / 2, thick - band / 2):
            if across:
                box = (t - hole_w / 2, edge - hole_h / 2, t + hole_w / 2, edge + hole_h / 2)
            else:
                box = (edge - hole_h / 2, t - hole_w / 2, edge + hole_h / 2, t + hole_w / 2)
            draw.rounded_rectangle(box, radius=max(1, hole_w * 0.25), fill=0)
    strip.putalpha(alpha)
    return strip


# ── What the camera saw, written on the card ─────────────────────────
# On a print's white chin or a frame's mat: the body and the exposure in small,
# quiet type, and the day in the orange of a film camera's date back. Only on
# the paper, never over the picture.

_SANS = {
    "medium": [("Avenir Next.ttc", 5), ("HelveticaNeue.ttc", 10), ("segoeuisb.ttf", 0),
               ("bahnschrift.ttf", 0), ("DejaVuSans.ttf", 0)],
    "regular": [("Avenir Next.ttc", 7), ("HelveticaNeue.ttc", 0), ("segoeui.ttf", 0),
                ("bahnschrift.ttf", 0), ("DejaVuSans.ttf", 0)],
}
_MONO = [("SFNSMono.ttf", 0), ("Menlo.ttc", 0), ("consola.ttf", 0), ("DejaVuSansMono.ttf", 0)]
_FONT_DIRS = ("", "/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/",
              "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/")
_found: dict = {}

INK = (58, 56, 52)            # the body, in a soft near-black
PENCIL = (128, 125, 118)      # the exposure, lighter
DATE_BACK = (222, 104, 34)    # a date back's orange
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _face(chain: list, size: int):
    size = max(4, int(size))
    key = (id(chain), size)
    if key in _found:
        return _found[key]
    from PIL import ImageFont

    font = None
    for name, index in chain:
        for folder in _FONT_DIRS:
            try:
                font = ImageFont.truetype(folder + name, size, index=index)
                break
            except OSError:
                continue
        if font:
            break
    _found[key] = font or ImageFont.load_default(size=size)
    return _found[key]


def _number(text: str, unit: str = "") -> str:
    """``"4.0"`` as ``"4"``, ``"45 mm"`` as ``"45mm"``."""
    text = str(text or "").strip().replace(" ", "")
    if unit and text.lower().endswith(unit):
        text = text[: -len(unit)]
    try:
        value = float(text)
        text = f"{value:g}"
    except ValueError:
        pass
    return text + unit if text else ""


def caption_of(camera: str = "", focal_length: str = "", aperture: str = "", shutter_speed: str = "",
               iso: str = "", capture_date: str = "", **_) -> Optional[dict]:
    """A catalog record's EXIF as what a card shows: ``{body, exposure, date,
    stamp}``, or None when there is nothing to say."""
    exposure = []
    if focal_length:
        exposure.append(_number(focal_length, "mm"))
    if aperture:
        f = _number(str(aperture).lower().removeprefix("f/").removeprefix("f"))
        if f:
            exposure.append(f"f/{f}")
    if shutter_speed:
        s = str(shutter_speed).strip().removesuffix("s").strip()
        if "/" not in s:
            s = _number(s)
        if s:
            exposure.append(f"{s}s")
    if iso:
        exposure.append(f"ISO {str(iso).strip()}")
    date, stamp = "", ""
    text = str(capture_date or "").strip()
    if len(text) >= 10:
        try:
            y, m, d = int(text[0:4]), int(text[5:7]), int(text[8:10])
            if 1 <= m <= 12 and 1 <= d <= 31:
                date = f"{d} {_MONTHS[m - 1]} {y}"
                stamp = f"'{y % 100:02d} {m:02d} {d:02d}"
        except ValueError:
            pass
    body = " ".join(str(camera or "").split())
    if not (body or exposure or date):
        return None
    return {"body": body, "exposure": "  ".join(exposure), "date": date, "stamp": stamp}


def _width(draw, text: str, font, tracking: float = 0.0) -> float:
    if not text:
        return 0.0
    return draw.textlength(text, font=font) + tracking * (len(text) - 1)


def _write(draw, xy: tuple, text: str, font, fill, tracking: float = 0.0) -> None:
    """``text`` with its left edge at x and its baseline at y, letters ``tracking`` further apart."""
    x, y = xy
    if not tracking:
        draw.text((x, y), text, font=font, fill=fill, anchor="ls")
        return
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += draw.textlength(ch, font=font) + tracking


def _type_size(caption: dict, side: float) -> float:
    """What type sizes are fractions of, for a card ``side`` across: its side,
    pulled towards a fixed share of the page, so a small card's line can still
    be read and a large card's does not shout."""
    page = caption.get("page") or side / 0.4
    return math.sqrt(side * page * 0.4)


def _print_caption(card: Image.Image, caption: dict, box: tuple, side: int) -> None:
    """On a print's chin ``box`` (left, top, right, bottom): the body over the
    exposure on the left, the date back's stamp on the right."""
    draw = ImageDraw.Draw(card)
    left, top, right, bottom = box
    room = right - left
    unit = _type_size(caption, side)
    for scale in (1.0, 0.9, 0.8, 0.7, 0.62, 0.55, 0.48):
        big, small = unit * 0.042 * scale, unit * 0.031 * scale
        body_font = _face(_SANS["medium"], big)
        exp_font = _face(_SANS["regular"], small)
        stamp_font = _face(_MONO, unit * 0.036 * scale)
        track = small * 0.06
        text_w = max(_width(draw, caption["body"], body_font), _width(draw, caption["exposure"], exp_font, track))
        stamp_w = _width(draw, caption["stamp"], stamp_font)
        tall = (big if caption["body"] else 0) + (small * 1.45 if caption["exposure"] else 0)
        if text_w + (stamp_w + unit * 0.05 if stamp_w else 0) <= room and tall <= (bottom - top) * 0.75:
            break
    lines = [(caption["body"], body_font, INK, 0.0, big), (caption["exposure"], exp_font, PENCIL, track, small)]
    lines = [line for line in lines if line[0]]
    # The block sits a little above the chin's middle, where a hand would write.
    lead = [h * (0.8 if i == 0 else 1.45) for i, (*_, h) in enumerate(lines)]
    baseline = top + (bottom - top - sum(lead)) * 0.44
    for (text, font, fill, tr, _h), step in zip(lines, lead):
        baseline += step
        _write(draw, (left, baseline), text, font, fill, tr)
    if caption["stamp"]:
        sw = _width(draw, caption["stamp"], stamp_font)
        # Level with the last line, as a date back prints along the frame's edge.
        last = baseline if lines else top + (bottom - top) * 0.55
        layer = Image.new("RGBA", card.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((right - sw, last), caption["stamp"], font=stamp_font,
                                   fill=DATE_BACK + (245,), anchor="ls")
        # Printed light into paper, the stamp's edge is a touch soft.
        card.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(0.3, side * 0.0009))))


def _mat_caption(piece: Image.Image, caption: dict, box: tuple, side: int) -> None:
    """One quiet line, centred in a frame's bottom mat: as a gallery labels a print."""
    draw = ImageDraw.Draw(piece)
    left, top, right, bottom = box
    unit = min(_type_size(caption, side) * 0.03, (bottom - top) * 0.3)
    # Below this a line is a grey smudge, not words: say less, or nothing.
    least = (caption.get("page") or side / 0.4) * 0.011
    # Smaller first; then, with no room, without the body, then without the date.
    tries = [(caption["body"], caption["exposure"], caption["date"]),
             (caption["exposure"], caption["date"]), (caption["exposure"] or caption["date"],)]
    for parts in tries:
        text = "   ·   ".join(p for p in parts if p)
        for scale in (1.0, 0.88, 0.76, 0.66):
            size = unit * scale
            if size < least:
                break
            font = _face(_SANS["regular"], size)
            track = size * 0.08
            width = _width(draw, text, font, track)
            if width <= (right - left) * 0.94:
                _write(draw, ((left + right - width) / 2, (top + bottom) / 2 + size * 0.36),
                       text, font, PENCIL, track)
                return


def captioned(make, caption: Optional[dict]):
    """``make`` (one of CARDS) writing ``caption`` on its paper, when it has paper for it."""
    if not caption:
        return make
    if make is polaroid:
        def card(image, w, h, focus, rng):
            piece = polaroid(image, w, h, focus, rng)
            side = min(w, h)
            border = max(2, round(side * 0.055))
            bottom = max(border, round(side * 0.19))
            _print_caption(piece, caption, (border, h - bottom, w - border, h - border * 0.35), side)
            return piece
        return card
    if make is framed:
        return lambda image, w, h, focus, rng: framed(image, w, h, focus, rng, caption)
    return make
