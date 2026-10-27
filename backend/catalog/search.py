"""
Searching the whole library with a line of text: ``rating:>=3 iso:>1600 lens:35``.

The grid's filters work on one folder and one control at a time. This is the
same question asked of every photo at once, in words: a query is terms separated
by spaces, and a photo is found when every term holds.

    rating:>=3        stars, compared: = > >= < <=, or a range 2..4
    flag:pick         pick | reject | none
    label:red,green   colour label; a comma means either
    tag:"school play" a tag, exactly (quotes for spaces)
    camera:r6         camera model, any part of it
    lens:35           lens name, any part of it
    iso:>1600         numbers compare: iso, f (aperture), focal (mm), shutter
    f:<=2.8           aperture
    focal:24..35      focal length in mm
    shutter:<1/100    exposure time, as 1/200 or 0.5
    date:2026-04      a year, month or day; compared and ranged like numbers
    folder:hawaii     any part of the folder path
    type:raw          raw | jpg | cr3 | cr2 | dng | … (what the pair holds)
    is:edited         edited | unedited | unreviewed | reviewed | pick | reject
                      | raw | jpg | basket | bw (developed in black and white)
                      | colour
    shows:"kids at the beach"
                      what the picture shows, in words (analysis.clip): the
                      photographs nearest the description, best first
    -term             any term turned around: -tag:family, -is:edited
    plain words       anywhere in the filename, folder, camera, lens or tags;
                      or, searched with ``describe``, what the picture shows

A RAW+JPEG pair is one photo here, as it is in the grid: its rating, flag,
label, tags and edit are the pair's, whichever file they were put on.

Parsing and matching are kept apart from the catalog so the language can be
tested on its own; :meth:`CatalogService.search` gathers the photos.
"""

import re
import shlex
from dataclasses import dataclass, field
from typing import Callable, Optional

RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
JPG_EXTENSIONS = {".jpg", ".jpeg"}

_FLAGS = {"pick": 1, "picked": 1, "reject": -1, "rejected": -1, "none": 0, "unflagged": 0}
_LABELS = {"none": 0, "red": 1, "yellow": 2, "green": 3, "blue": 4, "purple": 5}
_IS = ("edited", "unedited", "unreviewed", "reviewed", "pick", "reject", "raw", "jpg", "basket",
       "bw", "colour", "denoised")

# Names people reach for, to the field they mean.
_ALIASES = {
    "stars": "rating", "star": "rating",
    "color": "label", "colour": "label",
    "aperture": "f", "fnumber": "f",
    "focal_length": "focal", "mm": "focal",
    "speed": "shutter", "exposure": "shutter",
    "model": "camera", "body": "camera",
    "ext": "type", "format": "type",
    "tags": "tag",
    "show": "shows", "showing": "shows", "looks": "shows",
}
FIELDS = ("rating", "flag", "label", "tag", "camera", "lens", "iso", "f", "focal", "shutter",
          "date", "folder", "type", "is", "shows")

# A shows: term finds the photographs at least SHOWS_FLOOR relevant (see
# analysis.clip.scores; set by looking down ranked results of a 29,000-photo
# library for dogs, a piano, kids at a beach, cakes, until they stopped being
# that). When even the best is under SHOWS_CLEAR, the library likely holds no
# such thing and what is found is only the closest it has.
SHOWS_FLOOR = 0.025
SHOWS_CLEAR = 0.04

_COMPARE = re.compile(r"^(>=|<=|>|<|=)?(.*)$")


class SearchError(ValueError):
    """The query could not be read; the message says which term and why."""


@dataclass
class Photo:
    """One photo as search sees it: a single file, or a RAW+JPEG pair merged."""
    record: object                    # the PhotoRecord shown for it
    extensions: set = field(default_factory=set)
    filepaths: list = field(default_factory=list)
    rating: int = 0
    flag: int = 0
    label: int = 0
    tags: set = field(default_factory=set)     # as written; matched without case
    edited: bool = False
    mono: bool = False                # developed in black and white
    denoised: bool = False            # developed with noise reduction
    in_basket: bool = False
    date: str = ""                    # YYYY-MM-DD, from EXIF or else the folder name
    # How near each shows: description this photo is, by its text; filled in by
    # search() for the terms of the query in hand.
    shows: dict = field(default_factory=dict)


@dataclass
class Term:
    field: str                        # "" for a plain word
    value: str
    test: Callable[[Photo], bool]
    negate: bool = False

    def __call__(self, photo: Photo) -> bool:
        return self.test(photo) != self.negate


# ── Values ───────────────────────────────────────────────────────────

def _number(text: str) -> Optional[float]:
    """The number in a stored EXIF string: '78.0 mm' → 78, 'f/5.6' → 5.6, '1/200' → 0.005."""
    text = (text or "").strip().lower().removeprefix("f/").removeprefix("f")
    match = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if match:
        denominator = float(match.group(2))
        return float(match.group(1)) / denominator if denominator else None
    match = re.match(r"^(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


_DATE_IN_PATH = re.compile(r"(?<!\d)((?:19|20)\d\d)[-_/.]?(0[1-9]|1[0-2])[-_/.]?(0[1-9]|[12]\d|3[01])(?!\d)")


def photo_date(capture_date: str, folder: str) -> str:
    """YYYY-MM-DD for a photo: its EXIF date, or a date in its folder's path.

    Many libraries have frames whose EXIF was never read, but a library laid out
    by date already says when they were taken in the folder name.
    """
    match = re.match(r"^(\d{4})[:-](\d\d)[:-](\d\d)", capture_date or "")
    if match and match.group(1) != "0000":
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = _DATE_IN_PATH.search(folder or "")
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return ""


def _period(text: str) -> tuple[str, str]:
    """A year, month or day as the first and last day it covers."""
    match = re.match(r"^(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?$", text.strip())
    if not match:
        raise SearchError(f"'{text}' is not a date; write 2026, 2026-04 or 2026-04-05")
    year, month, day = match.group(1), match.group(2), match.group(3)
    if month is None:
        return f"{year}-01-01", f"{year}-12-31"
    month = f"{int(month):02d}"
    if day is None:
        return f"{year}-{month}-01", f"{year}-{month}-31"
    day = f"{int(day):02d}"
    return f"{year}-{month}-{day}", f"{year}-{month}-{day}"


# ── Comparisons ──────────────────────────────────────────────────────

def _numeric_test(value: str, read: Callable[[Photo], Optional[float]], parse=_number, name="") -> Callable:
    """A test for one numeric field: `800`, `>800`, `<=2.8`, `24..35`, `400,800`."""
    if ".." in value:
        low_text, high_text = value.split("..", 1)
        low = parse(low_text) if low_text else None
        high = parse(high_text) if high_text else None
        if (low_text and low is None) or (high_text and high is None):
            raise SearchError(f"'{value}' is not a range of numbers for {name}")

        def in_range(photo):
            v = read(photo)
            return v is not None and (low is None or v >= low) and (high is None or v <= high)
        return in_range
    if "," in value:
        tests = [_numeric_test(part, read, parse, name) for part in value.split(",") if part]
        return lambda photo: any(t(photo) for t in tests)
    op, text = _COMPARE.match(value).groups()
    target = parse(text)
    if target is None:
        raise SearchError(f"'{value}' is not a number for {name}")
    compare = {
        None: lambda v: abs(v - target) < 1e-6 * max(1.0, abs(target)),
        "=": lambda v: abs(v - target) < 1e-6 * max(1.0, abs(target)),
        ">": lambda v: v > target, ">=": lambda v: v >= target,
        "<": lambda v: v < target, "<=": lambda v: v <= target,
    }[op]

    def test(photo):
        v = read(photo)
        return v is not None and compare(v)
    return test


def _date_test(value: str) -> Callable:
    if ".." in value:
        low_text, high_text = value.split("..", 1)
        start = _period(low_text)[0] if low_text else ""
        end = _period(high_text)[1] if high_text else "9999"
        return lambda photo: bool(photo.date) and start <= photo.date <= end
    if "," in value:
        tests = [_date_test(part) for part in value.split(",") if part]
        return lambda photo: any(t(photo) for t in tests)
    op, text = _COMPARE.match(value).groups()
    start, end = _period(text)
    compare = {
        None: lambda d: start <= d <= end, "=": lambda d: start <= d <= end,
        ">": lambda d: d > end, ">=": lambda d: d >= start,
        "<": lambda d: d < start, "<=": lambda d: d <= end,
    }[op]
    return lambda photo: bool(photo.date) and compare(photo.date)


def _choice(value: str, choices: dict, name: str) -> list:
    picked = []
    for part in value.lower().split(","):
        if part in choices:
            picked.append(choices[part])
        elif part.lstrip("-").isdigit() and int(part) in choices.values():
            picked.append(int(part))
        else:
            raise SearchError(f"{name}:{part} — {name} is one of {', '.join(choices)}")
    return picked


def _contains(read: Callable[[Photo], str], value: str) -> Callable:
    wanted = [part.strip().lower() for part in value.split(",") if part.strip()]
    return lambda photo: any(w in (read(photo) or "").lower() for w in wanted)


def _is(value: str) -> Callable:
    tests = {
        "edited": lambda p: p.edited,
        "unedited": lambda p: not p.edited,
        "unreviewed": lambda p: p.rating == 0 and p.flag == 0 and p.label == 0,
        "reviewed": lambda p: p.rating > 0 or p.flag != 0 or p.label != 0,
        "pick": lambda p: p.flag == 1,
        "reject": lambda p: p.flag == -1,
        "raw": lambda p: bool(p.extensions & RAW_EXTENSIONS),
        "jpg": lambda p: bool(p.extensions & JPG_EXTENSIONS),
        "basket": lambda p: p.in_basket,
        "bw": lambda p: p.mono,
        "mono": lambda p: p.mono,
        "denoised": lambda p: p.denoised,
        "nr": lambda p: p.denoised,
        "colour": lambda p: not p.mono,
        "color": lambda p: not p.mono,
    }
    chosen = []
    for part in value.lower().split(","):
        if part not in tests:
            raise SearchError(f"is:{part} — is: is one of {', '.join(_IS)}")
        chosen.append(tests[part])
    return lambda photo: any(t(photo) for t in chosen)


def _type(value: str) -> Callable:
    wanted = set()
    for part in value.lower().split(","):
        if part == "raw":
            wanted |= RAW_EXTENSIONS
        elif part in ("jpg", "jpeg"):
            wanted |= JPG_EXTENSIONS
        elif part:
            wanted.add("." + part.lstrip("."))
    return lambda photo: bool(photo.extensions & wanted)


def _term(name: str, value: str) -> Callable:
    if not value:
        raise SearchError(f"{name}: needs a value, e.g. {name}:{_EXAMPLES.get(name, '…')}")
    if name == "rating":
        return _numeric_test(value, lambda p: p.rating, name=name)
    if name == "flag":
        flags = _choice(value, _FLAGS, "flag")
        return lambda photo: photo.flag in flags
    if name == "label":
        labels = _choice(value, _LABELS, "label")
        return lambda photo: photo.label in labels
    if name == "tag":
        tags = {part.strip().lower() for part in value.split(",") if part.strip()}
        return lambda photo: any(t.lower() in tags for t in photo.tags)
    if name == "camera":
        return _contains(lambda p: p.record.camera, value)
    if name == "lens":
        return _contains(lambda p: p.record.lens, value)
    if name == "folder":
        return _contains(lambda p: p.record.folder, value)
    if name == "iso":
        return _numeric_test(value, lambda p: _number(p.record.iso), name=name)
    if name == "f":
        return _numeric_test(value, lambda p: _number(p.record.aperture) or None, name="f")
    if name == "focal":
        return _numeric_test(value, lambda p: _number(p.record.focal_length) or None, name=name)
    if name == "shutter":
        return _numeric_test(value, lambda p: _number(p.record.shutter_speed), name=name)
    if name == "date":
        return _date_test(value)
    if name == "type":
        return _type(value)
    if name == "is":
        return _is(value)
    if name == "shows":
        # Scored against the library when the search runs (search()).
        return lambda photo: photo.shows.get(value, -1.0) >= 1.0
    raise AssertionError(name)


_EXAMPLES = {
    "rating": ">=3", "flag": "pick", "label": "red", "tag": "family", "camera": "r6", "lens": "35",
    "iso": ">1600", "f": "<=2.8", "focal": "24..35", "shutter": "<1/100", "date": "2026-04",
    "folder": "hawaii", "type": "raw", "is": "edited", "shows": '"a dog on a beach"',
}


def _words(query: str) -> list[str]:
    lexer = shlex.shlex(query, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.quotes = '"'
    try:
        return list(lexer)
    except ValueError:
        raise SearchError("A quote is not closed")


def parse(query: str, describe: bool = False) -> list[Term]:
    """The terms of a query. Raises SearchError naming the term it could not read.

    With ``describe``, the plain words together are one shows: term — "kids on
    a beach rating:>=3" — rather than words to find in names."""
    terms = []
    plain = []
    for word in _words(query or ""):
        negate = word.startswith("-") and len(word) > 1
        body = word[1:] if negate else word
        name, sep, value = body.partition(":")
        key = _ALIASES.get(name.lower(), name.lower())
        if sep and key in FIELDS:
            terms.append(Term(key, value.strip(), _term(key, value.strip()), negate))
        elif sep and re.fullmatch(r"[a-z_]+", name.lower()):
            raise SearchError(f"Unknown field '{name}:'. Fields: {', '.join(FIELDS)}")
        elif describe and not negate:
            plain.append(body)
        else:
            text = body.lower()
            terms.append(Term("", body, lambda p, t=text: _anywhere(p, t), negate))
    if plain:
        text = " ".join(plain)
        terms.append(Term("shows", text, _term("shows", text)))
    return terms


def _anywhere(photo: Photo, text: str) -> bool:
    r = photo.record
    return (
        text in r.filename.lower() or text in r.folder.lower()
        or text in (r.camera or "").lower() or text in (r.lens or "").lower()
        or any(text in tag.lower() for tag in photo.tags)
    )


def matches(terms: list[Term], photo: Photo) -> bool:
    return all(term(photo) for term in terms)


# ── From catalog rows to search photos ───────────────────────────────

def gather(records, tags_by_path: dict, edited: set, basket: set, mono: set = frozenset(),
           denoised: set = frozenset()) -> list[Photo]:
    """Merge each RAW+JPEG pair into one Photo, shown as the grid shows it (the JPEG)."""
    groups: dict[tuple, list] = {}
    for record in records:
        groups.setdefault((record.folder, record.base_name), []).append(record)

    photos = []
    for group in groups.values():
        group.sort(key=lambda r: (r.extension not in JPG_EXTENSIONS, r.filename))
        shown = group[0]
        # A camera's JPEG may have lost its EXIF while the raw kept it: the pair
        # is that camera's photo either way.
        if not (shown.camera or "").strip():
            shown.camera = next((r.camera for r in group if (r.camera or "").strip()), shown.camera)
        flags = [r.flag for r in group if r.flag]
        photo = Photo(
            record=shown,
            extensions={r.extension.lower() for r in group},
            filepaths=[r.filepath for r in group],
            rating=max(r.rating for r in group),
            flag=flags[0] if flags else 0,
            label=next((r.color_label for r in group if r.color_label), 0),
            tags={t for r in group for t in tags_by_path.get(r.filepath, ())},
            edited=any(r.filepath in edited for r in group),
            in_basket=any(r.filepath in basket for r in group),
            mono=any(r.filepath in mono for r in group),
            denoised=any(r.filepath in denoised for r in group),
            date=photo_date(next((r.capture_date for r in group if r.capture_date), ""), shown.folder),
        )
        photos.append(photo)
    return photos


SORTS = {
    "relevance": lambda p: (sum(p.shows.values()), p.date or "0000"),
    "date": lambda p: (p.date or "0000", p.record.filepath),
    "rating": lambda p: (p.rating, p.date, p.record.filepath),
    "filename": lambda p: (p.record.filename.lower(), p.record.folder),
    "folder": lambda p: (p.record.folder, p.record.filename.lower()),
}


def search(photos: list[Photo], query: str, sort: str = "", desc: bool = True,
           scorer: Optional[Callable[[str], dict]] = None, describe: bool = False) -> list[Photo]:
    """The photos a query finds, sorted. ``sort`` "" is by relevance when the
    query says what the pictures show, else by date. ``scorer(text)`` gives
    ``{filepath: similarity}`` for a shows: term; without one, such a term is
    an error saying smart search is not set up."""
    terms = parse(query, describe)
    shown = [t for t in terms if t.field == "shows"]
    for term in shown:
        if scorer is None:
            raise SearchError("shows: needs smart search, which is not set up yet "
                              "(the Describe button by the search box sets it up)")
        found_scores = scorer(term.value)
        for photo in photos:
            values = [found_scores[f] for f in photo.filepaths if f in found_scores]
            if values:
                photo.shows[term.value] = max(values)
        term.test = lambda p, v=term.value: p.shows.get(v, -1.0) >= SHOWS_FLOOR
    found = [p for p in photos if matches(terms, p)]
    if not sort:
        sort = "relevance" if shown else "date"
    if sort == "relevance" and not shown:
        sort = "date"
    found.sort(key=SORTS.get(sort, SORTS["date"]), reverse=desc)
    return found


def unclear(found: list[Photo]) -> list[str]:
    """The shows: descriptions nothing found matches clearly: the library
    likely holds no such thing, and these are only the closest."""
    texts = {text for p in found for text in p.shows}
    return sorted(t for t in texts if max(p.shows.get(t, -1.0) for p in found) < SHOWS_CLEAR)


def describe(query: str, describe_words: bool = False) -> list[str]:
    """The query back as the terms it was read as, for showing what was understood."""
    def shown(t):
        value = f'"{t.value}"' if t.field == "shows" and " " in t.value else t.value
        return ("-" if t.negate else "") + (f"{t.field}:" if t.field else "") + value
    return [shown(t) for t in parse(query, describe_words)]
