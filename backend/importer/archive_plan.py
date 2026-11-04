"""
Bringing an old collection of photographs into the library: the rules.

A collection that grew over twenty years is not one shape. Part of it is
already laid out by day (``2010/2010-02-21 Picnic``); part is camera cards
copied off whole (``Card backup/DCIM/101CANON``); part is film scans, whose
files were made years after the pictures were taken and whose only date is in
the folder's name (``200210 Lakeside``, ``Coast trip Aug 2006/10234``); and
much of it is there twice or three times, from one backup copied into another.

So each folder of the source is given one of:

``merge``   already laid out by day: it joins the library's own year folders,
            into the day folder that is already there when there is one.
``archive`` everything else, under ``Archive/<year>/<date> <name>``: one folder
            per event, its date from the folder's name (a scan) or the camera
            (a digital photograph).
``split``   a camera's dump with no event to it (``100CANON``, ``SD400``): its
            photographs go to a folder per day, as a card import files them.
``star``    a set of favourites that are copies of photographs found elsewhere
            (Picasa's starred photos): the originals are given stars instead.
``skip``    not family photographs (an application's caches, or what a local
            rules file names), or nothing in it that is not already somewhere
            else.

What one collection holds besides photographs, and what its own folders are
called, is its own business: those go in a rules file beside the plan
(:class:`Rules`, ``archive-rules.txt``), never in this code.

Nothing here touches a file. :func:`plan_folders` turns what was read of the
source into rows a person reviews in a spreadsheet before anything is copied
(tools/archive_import.py).
"""

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from config import RASTER_EXTENSIONS, VIDEO_EXTENSIONS

ARCHIVE = "Archive"

# ── Dates in folder names ─────────────────────────────────────────────

_MONTH_WORDS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = re.compile(
    r"(?<![a-z])(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])\s*'?(\d{4}|\d{2}(?!\d))?")
_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
              "十一": 11, "十二": 12}
_CN_MONTH_RE = re.compile(r"(十[一二]?|[一二三四五六七八九])月[底初中]?")
_Y = r"((?:19|20)\d\d)"
_M = r"(0[1-9]|1[0-2])"
_D = r"(0[1-9]|[12]\d|3[01])"
# (pattern, precision, order of the groups as y/m/d)
_DAY_PATTERNS = [
    (re.compile(rf"(?<!\d){_Y}[-_.]{_M}[-_.]{_D}(?!\d)"), "ymd"),          # 2010-01-31
    (re.compile(rf"(?<!\d){_Y}{_M}{_D}(?!\d)"), "ymd"),                     # 20061105
    (re.compile(rf"(?<!\d){_Y}\s+{_M}{_D}(?!\d)"), "ymd"),                  # 2010 0612
    (re.compile(rf"(?<!\d){_M}{_D}{_Y}(?!\d)"), "mdy"),                     # 09182004
    (re.compile(rf"(?<!\d){_M}[-_.]{_D}[-_.](\d\d)(?!\d)"), "mdyy"),        # 11-03-08
]
_MONTH_PATTERNS = [
    (re.compile(rf"(?<!\d){_Y}[-_.]?{_M}(?!\d)"), "ym"),                    # 200210, 2002-10
    (re.compile(rf"(?<!\d){_M}{_Y}(?!\d)"), "my"),                          # 052005
]
_YEAR_RE = re.compile(rf"(?<!\d){_Y}(?!\d)")
_MMYY_RE = re.compile(r"(?:(?<=\s)|^)(0[1-9]|1[0-2])(\d\d)(?=\s|$)")      # "Canyon 0603"


@dataclass
class FoundDate:
    value: str          # 2003-09-01, 2003-09 or 2003
    precision: str      # day | month | year
    rest: str           # the name with the date taken out

    @property
    def first_day(self) -> date:
        parts = [int(p) for p in self.value.split("-")] + [1, 1]
        return date(parts[0], parts[1], parts[2])


def _ok(y: int, m: int = 1, d: int = 1, not_after: Optional[date] = None) -> bool:
    try:
        when = date(y, m, d)
    except ValueError:
        return False
    return 1900 <= y and (not_after is None or when <= not_after)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -_.,&~#")


def find_date(text: str, not_after: Optional[date] = None) -> Optional[FoundDate]:
    """The date a folder's name gives, to the day, the month or the year.

    ``not_after`` is the oldest file's date: a picture is not taken after it is
    saved, so a reading that would put it later is the wrong reading.
    """
    low = text.casefold()

    def cut(spans):
        out = text
        for a, b in sorted(spans, reverse=True):
            out = out[:a] + " " + out[b:]
        return _clean(out)

    for pattern, order in _DAY_PATTERNS:
        for m in pattern.finditer(low):
            g = m.groups()
            if order == "ymd":
                y, mo, d = int(g[0]), int(g[1]), int(g[2])
            elif order == "mdy":
                mo, d, y = int(g[0]), int(g[1]), int(g[2])
            else:
                yy = int(g[2])
                mo, d, y = int(g[0]), int(g[1]), (2000 + yy if yy <= 30 else 1900 + yy)
            if _ok(y, mo, d, not_after):
                # A range (20080305-20080512) names its first day; both go from the name.
                spans = [m.span()] + [x.span() for x in pattern.finditer(low) if x.span() != m.span()]
                return FoundDate(f"{y:04d}-{mo:02d}-{d:02d}", "day", cut(spans))

    for pattern, order in _MONTH_PATTERNS:
        for m in pattern.finditer(low):
            g = m.groups()
            y, mo = (int(g[0]), int(g[1])) if order == "ym" else (int(g[1]), int(g[0]))
            if _ok(y, mo, 1, not_after):
                return FoundDate(f"{y:04d}-{mo:02d}", "month", cut([m.span()]))

    word = _MONTH_RE.search(low)
    cn = _CN_MONTH_RE.search(text)
    if word or cn:
        if word:
            mo = _MONTH_WORDS[word.group(1)[:3]]
            span, yy = word.span(), word.group(2)
        else:
            mo, span, yy = _CN_DIGITS[cn.group(1)], cn.span(), None
        spans = [span]
        y = None
        if yy:
            y = int(yy) if len(yy) == 4 else (2000 + int(yy) if int(yy) <= 30 else 1900 + int(yy))
        else:
            ym = _YEAR_RE.search(low)
            if ym:
                y = int(ym.group(1))
                spans.append(ym.span())
        if y and _ok(y, mo, 1, not_after):
            return FoundDate(f"{y:04d}-{mo:02d}", "month", cut(spans))

    m = _YEAR_RE.search(low)
    if m and _ok(int(m.group(1)), 1, 1, not_after):
        return FoundDate(m.group(1), "year", cut([m.span()]))

    m = _MMYY_RE.search(low)
    if m and not_after is not None:
        mo, yy = int(m.group(1)), int(m.group(2))
        y = 2000 + yy if yy <= 30 else 1900 + yy
        if _ok(y, mo, 1, not_after):
            return FoundDate(f"{y:04d}-{mo:02d}", "month", cut([m.span()]))
    return None


# ── Names ─────────────────────────────────────────────────────────────

# Folder names that say nothing about what is in them: a camera's own, a
# backup's, a stage of work (Working, Converted, DNG), a roll or a set number.
_GENERIC = re.compile(
    r"^(?:dcim|\d{3}[a-z0-9_]{5}|new folder(?: \d+)?|originals?|pictures|my pictures|photos?|"
    r"untitled(?:-\d+)?|done|working|converted|raws?|dng|compressed dng|set \d+(?: jpeg)?|"
    r"rol+ ?\d+|roll|#?\d+|inbox|desktop|albums?|.*backup.*|old photos?|film scans|raw scans|"
    r"scans?|vuescan(?: ?temp)?|picasa edits|camera|images?|misc|other|new|"
    r"s\d{3}a?|sd\d{3}|e-p\d|eos)$")

# What a photograph library is not: the caches and catalogs applications keep
# beside the pictures. Anything else to leave out is named in the rules file.
_SKIP = re.compile(r"^(?:lightroom catalog|ipod photo cache|picasa web exports|thumbnails|"
                   r"\.?thumbs|@eadir|\$recycle\.bin|system volume information)$")

_STARRED = re.compile(r"starred|favou?rites?", re.IGNORECASE)


class Rules:
    """What one collection says about itself, from a text file kept beside the
    plan (and out of any repository), one rule a line::

        # folders that are not family photographs
        skip: lab data
        skip: music*
        # folder names that say nothing about what is in them
        generic: back from *
        # folders of favourites whose originals are elsewhere
        star: picks for the album

    A pattern is a folder name, with ``*`` and ``?`` as wildcards, case not minded.
    """

    def __init__(self, skip=(), generic=(), star=()):
        self.skip = [p.casefold() for p in skip]
        self.generic = [p.casefold() for p in generic]
        self.star = [p.casefold() for p in star]

    @classmethod
    def read(cls, path) -> "Rules":
        found = {"skip": [], "generic": [], "star": []}
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                kind, pattern = line.split(":", 1)
                kind = kind.strip().lower()
                if kind in found and pattern.strip():
                    found[kind].append(pattern.strip())
        return cls(**found)

    @staticmethod
    def _any(name: str, patterns: list) -> bool:
        low = name.casefold()
        return any(fnmatch.fnmatchcase(low, p) for p in patterns)

    def skips(self, name: str) -> bool:
        return bool(_SKIP.match(name.casefold())) or self._any(name, self.skip)

    def is_generic(self, text: str) -> bool:
        return bool(_GENERIC.match(text)) or self._any(text, self.generic)

    def stars(self, name: str) -> bool:
        return bool(_STARRED.search(name)) or self._any(name, self.star)


NO_RULES = Rules()
_SCAN_HINT = re.compile(r"scan|film|slide|vuescan|negative|\broll\b", re.IGNORECASE)
_SCANNER_SOFTWARE = re.compile(r"vuescan|silverfast|epson scan|nikon scan|coolscan|canoscan|scan",
                               re.IGNORECASE)
_DAY_FOLDER = re.compile(rf"^{_Y}-{_M}-{_D}(?:\s+(.*))?$")
_YEAR_FOLDER = re.compile(r"^(?:19|20)\d\d$")
_MONTH_FOLDER = re.compile(r"^(?:0[1-9]|1[0-2])$")
_DAY_IN_MONTH = re.compile(rf"^{_D}(?:\s+(.*))?$")
_CAMERA_FOLDER = re.compile(r"^\d{3}[A-Za-z0-9_]{5}$")
_RANGE = re.compile(r"\d{6,8}\s*[-~]\s*\d{6,8}")


def is_generic(name: str, not_after: Optional[date] = None, rules: Rules = NO_RULES) -> bool:
    stripped = find_date(name, not_after)
    text = (stripped.rest if stripped else name).casefold().strip()
    return not text or rules.is_generic(text)


def safe_name(text: str) -> str:
    """A folder name Windows accepts."""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    return _clean(text).rstrip(". ")[:80]


# ── What was read of the source ───────────────────────────────────────

@dataclass
class SourceFile:
    name: str
    size: int
    modified: date                     # the file's own time: when it was saved
    fingerprint: str = ""              # size and a hash of its first and last 64 KB
    taken: Optional[datetime] = None   # DateTimeOriginal / CreateDate
    make: str = ""
    software: str = ""

    @property
    def ext(self) -> str:
        i = self.name.rfind(".")
        return self.name[i:].lower() if i >= 0 else ""

    @property
    def stem(self) -> str:
        i = self.name.rfind(".")
        return self.name[:i] if i >= 0 else self.name


@dataclass
class SourceFolder:
    rel: str                           # relative to the source, with "/"
    media: list = field(default_factory=list)   # SourceFile
    other: int = 0                     # files that are not pictures (.ini, .db, .doc)

    @property
    def parts(self) -> list:
        return [p for p in self.rel.split("/") if p and p != "."]


@dataclass
class Row:
    action: str
    source: str
    dest: str = ""
    date: str = ""
    precision: str = ""
    kind: str = ""                     # digital | scan | mixed
    set_date: str = "no"
    files: int = 0
    new: int = 0
    duplicates: int = 0
    in_library: int = 0
    other_files: int = 0
    review: str = ""
    notes: str = ""

    FIELDS = ["action", "source", "dest", "date", "precision", "kind", "set_date", "files", "new",
              "duplicates", "in_library", "other_files", "review", "notes"]

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}


# ── Deciding one folder ───────────────────────────────────────────────

def _kind(folder: SourceFolder) -> str:
    pictures = [f for f in folder.media if f.ext not in VIDEO_EXTENSIONS]
    if not pictures:
        return "digital" if any(f.taken for f in folder.media) else "scan"
    if _SCAN_HINT.search(folder.rel):
        return "scan"
    camera = sum(1 for f in pictures
                 if f.taken and f.make and not _SCANNER_SOFTWARE.search(f.software or ""))
    raster = sum(1 for f in pictures if f.ext in RASTER_EXTENSIONS)
    if camera >= 0.5 * len(pictures):
        return "digital"
    if camera and raster < 0.5 * len(pictures):
        return "mixed"
    return "scan"


def _taken_days(folder: SourceFolder) -> list:
    return sorted(f.taken.date() for f in folder.media if f.taken and f.taken.year >= 1990)


def _meaningful_name(parts: list, not_after: Optional[date] = None, rules: Rules = NO_RULES) -> tuple:
    """(name, index of the part it came from) for the nearest folder that says
    what it is, the folder itself first; ("", -1) when none does."""
    for i in range(len(parts) - 1, -1, -1):
        if not is_generic(parts[i], not_after, rules):
            found = find_date(parts[i], not_after)
            return safe_name(found.rest if found else parts[i]), i
    return "", -1


def _date_from_names(parts: list, not_after: Optional[date]) -> tuple:
    """(FoundDate, index) from the folder's name, else its parents'."""
    for i in range(len(parts) - 1, -1, -1):
        found = find_date(parts[i], not_after)
        if found:
            return found, i
    return None, -1


def _merge_dest(parts: list, pattern: str) -> Optional[tuple]:
    """(date, name) when the folder is already laid out by day, else None:
    ``2010/2010-02-21 Picnic``, ``2012/07/16 party``, ``2011/Camera 2011-06-12``,
    or a camera folder inside one of these."""
    leaf = parts[-1]
    m = _DAY_FOLDER.match(leaf)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", (m.group(4) or "")
    if len(parts) >= 3 and _YEAR_FOLDER.match(parts[-3]) and _MONTH_FOLDER.match(parts[-2]):
        m = _DAY_IN_MONTH.match(leaf)
        if m and _ok(int(parts[-3]), int(parts[-2]), int(m.group(1))):
            return f"{parts[-3]}-{parts[-2]}-{m.group(1)}", (m.group(2) or "")
    if len(parts) >= 2 and _YEAR_FOLDER.match(parts[-2]):
        found = find_date(leaf)
        if found and found.precision == "day" and found.value[:4] == parts[-2]:
            return found.value, found.rest
    if len(parts) >= 2 and _CAMERA_FOLDER.match(leaf):
        return _merge_dest(parts[:-1], pattern)
    return None


def day_folder(pattern: str, day: str, name: str = "") -> str:
    """The library's own folder for a day, as the card import makes it, with a name."""
    y, m, d = day.split("-")
    path = pattern.format(year=y, month=m, day=d)
    return f"{path} {name}".strip() if name else path


def decide(folder: SourceFolder, pattern: str = "{year}/{year}-{month}-{day}",
           archive: str = ARCHIVE, rules: Rules = NO_RULES) -> Row:
    """The plan for one folder, before duplicates are counted."""
    parts = folder.parts or ["Unsorted"]          # pictures loose in the source itself
    row = Row(action="archive", source=folder.rel, files=len(folder.media), other_files=folder.other)
    for p in parts:
        if rules.skips(p):
            row.action, row.notes = "skip", f"left out by the rules ({p})"
            return row
    total = len(folder.media) + folder.other
    if folder.other > 0.7 * total:
        row.action, row.notes = "skip", "mostly documents and other files"
        return row
    if any(rules.stars(p) for p in parts):
        row.action, row.kind = "star", "copies"
        row.dest = f"{archive}/Starred"
        row.notes = "copies of favourites: the originals get stars; ones not found elsewhere are copied here"
        return row

    oldest_saved = min((f.modified for f in folder.media), default=None)
    row.kind = _kind(folder)
    days = _taken_days(folder)

    # A dump of several weeks in the day-folder tree (2008/20080305-20080512):
    # its photographs go to the library's own day folders, a folder per day.
    if (len(parts) >= 2 and _YEAR_FOLDER.match(parts[-2]) and _RANGE.search(parts[-1])
            and row.kind == "digital" and days):
        row.action, row.precision = "split", "day"
        row.dest = pattern
        row.notes = f"photographs from {days[0].isoformat()} to {days[-1].isoformat()}, a folder per day"
        return row

    merged = _merge_dest(parts, pattern)
    if merged:
        day, name = merged
        row.action, row.date, row.precision = "merge", day, "day"
        row.dest = day_folder(pattern, day, safe_name(name))
        if days and row.kind == "digital":
            common = Counter(days).most_common(1)[0][0]
            if abs((common - date.fromisoformat(day)).days) > 3:
                row.review = "yes"
                row.notes = f"the camera says {common.isoformat()}"
        return row

    name, name_at = _meaningful_name(parts, oldest_saved, rules)
    found, found_at = _date_from_names(parts, oldest_saved)

    if row.kind in ("digital", "mixed") and days:
        first, last = days[0], days[-1]
        span = (last - first).days
        row.precision = "day"
        if found and found.precision == "day" and abs((date.fromisoformat(found.value) - first).days) <= 3:
            row.date = found.value
        else:
            row.date = first.isoformat()
        if not name or (span > 14 and len(folder.media) > 300):
            row.action = "split"
            row.date = ""
            row.dest = f"{archive}/{{year}}/{{date}} {name}".strip()
            row.precision = "day"
            row.notes = f"a camera's photographs over {span + 1} days, a folder per day"
            return row
        row.dest = f"{archive}/{row.date[:4]}/{row.date} {name}"
        if span > 14:
            row.review = "yes"
            row.notes = f"photographs from {first.isoformat()} to {last.isoformat()}"
        if first.year < 2000:
            row.review = "yes"
            row.notes = (row.notes + "; " if row.notes else "") + "the camera's clock may not have been set"
        return row

    # A scan, or pictures with no date of their own: the folder's name says when.
    row.kind = "scan" if row.kind != "mixed" else row.kind
    label = name or "Scans"
    if found:
        row.date, row.precision = found.value, found.precision
        row.set_date = "yes"
        row.dest = f"{archive}/{found.value[:4]}/{found.value} {label}"
        if found.precision == "year":
            row.notes = "only the year is known"
    else:
        row.review = "yes"
        row.dest = f"{archive}/Undated/{label}"
        hint = f"saved {oldest_saved.isoformat()[:7]}" if oldest_saved else ""
        row.notes = ("no date in the name: put one at the start of dest's last part, "
                     f"as in {archive}/2004/2004-06 {label} (2004, 2004-06 or 2004-06-12)") + (
            f"; the files were {hint}, which is when they were scanned or copied, not taken" if hint else "")
    return row


# ── Duplicates, across the whole plan ─────────────────────────────────

_ORDER = {"merge": 0, "archive": 1, "split": 1, "star": 3, "skip": 4}
_PRECISION = {"day": 0, "month": 1, "year": 2, "": 3}


def row_order(row: Row) -> tuple:
    """Which copy of a photograph found twice is kept: one going into the day
    folders first, then one with the best known date, then the shallowest."""
    return (_ORDER.get(row.action, 5), _PRECISION.get(row.precision, 3),
            row.source.count("/"), row.source.casefold())


def plan_folders(folders: list, library: dict, pattern: str = "{year}/{year}-{month}-{day}",
                 archive: str = ARCHIVE, rules: Rules = NO_RULES) -> list:
    """Rows for every source folder with pictures in it, duplicates counted.

    ``library`` is ``{lowercase name: [(size, path)]}`` of what the library
    already holds (importer.copy.build_library_index): a file there by the same
    name and size is not copied again.
    """
    by_rel = {f.rel: f for f in folders}
    rows = [decide(f, pattern, archive, rules) for f in folders if f.media]
    seen: dict = {}
    for row in sorted(rows, key=row_order):
        folder = by_rel[row.source]
        if row.action == "skip":
            continue
        dup_from = Counter()
        for f in folder.media:
            if f.fingerprint and f.fingerprint in seen:
                row.duplicates += 1
                dup_from[seen[f.fingerprint]] += 1
            elif any(size == f.size for size, _ in library.get(f.name.lower(), ())):
                row.in_library += 1
            else:
                row.new += 1
                if f.fingerprint:
                    seen[f.fingerprint] = row.source
        if row.action == "star":
            continue
        if dup_from:
            where = ", ".join(f"{n} in {src}" for src, n in dup_from.most_common(2))
            row.notes = (row.notes + "; " if row.notes else "") + f"copies of photographs also {where}"
        if row.new == 0:
            row.action = "skip"
            row.notes = "nothing new: " + ("every photograph is also elsewhere"
                                           if row.duplicates else "all already in the library")
    rows.sort(key=lambda r: r.source.casefold())
    return rows
