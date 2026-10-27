"""
Perspectives: a year seen as Family, as School, as Friends — whichever part of
the library one person cares about.

A perspective is a name and the people it is about. People are matched by name,
exactly, or by a prefix ending in ``*``: a school's band named "BAND Alex",
"BAND Jo"… is ``BAND*``, and a new member named later joins it by itself.

A perspective picks *shoots* (folders), not single photos: the food, the stage
and the venue of a family dinner have no faces in them and are still the
dinner. A folder belongs to a perspective when that perspective's people are in
at least MIN_PHOTOS of its photos and in at least SHARE of as many as the
perspective most present there — so a band review with three hundred frames of
the band and a hundred of the family's own child in it is School, and a dinner
with that child is Family. A folder no perspective is present in belongs to none.

SHARE was set on a real year of a family whose child is in a school band
(one perspective for the band's prefix, one for the family's names): on the
school days the family's presence ran 0.04-0.46 of the band's, on the family
days 1.25 and up (most with no band member at all). A trip is mostly scenery,
a couple of photos with the family in them out of a few hundred, so the floor
is low.
"""

import json
import re
from collections import defaultdict
from typing import Optional

MIN_PHOTOS = 2
SHARE = 0.6
# A name prefix shared by at least this many people is offered as a perspective.
SUGGEST_PREFIX = 2


def matches(name: str, patterns: list[str]) -> bool:
    """Whether a person's name is one of ``patterns`` (exact, or ``Prefix*``), without case."""
    low = (name or "").casefold()
    for p in patterns or []:
        p = (p or "").strip().casefold()
        if not p:
            continue
        if p.endswith("*"):
            if low.startswith(p[:-1]):
                return True
        elif low == p:
            return True
    return False


def folder_people(rows) -> dict:
    """``{folder: {photo key: {names}}}`` from ``(folder, base_name, name)`` rows
    (faces of named people and person tags)."""
    out: dict = defaultdict(lambda: defaultdict(set))
    for folder, base, name in rows:
        out[folder][base].add(name)
    return out


def presence(photos: dict, patterns: list[str]) -> int:
    """How many of one folder's photos have someone of the perspective in them."""
    return sum(1 for names in photos.values() if any(matches(n, patterns) for n in names))


def folders_of(perspective: dict, all_perspectives: list[dict], people: dict) -> set:
    """The folders that belong to ``perspective``."""
    mine = perspective.get("people") or []
    out = set()
    for folder, photos in people.items():
        here = presence(photos, mine)
        if here < MIN_PHOTOS:
            continue
        most = max((presence(photos, p.get("people") or []) for p in all_perspectives), default=0)
        if here >= SHARE * most:
            out.add(folder)
    return out


def suggestions(names: list[str], existing: list[dict]) -> list[dict]:
    """Perspectives worth offering from the named people (most photographed
    first): one per name prefix several people share ("BAND Alex", "BAND-Pat
    L." → ``BAND*``), and one of the most photographed people with no prefix.
    Not offered again once a perspective has those people."""
    taken = {p.casefold() for e in existing for p in e.get("people") or []}
    prefix_of = {}
    for n in names:
        m = re.match(r"^([A-Za-z0-9#]+)[ \-]", n)
        if m:
            prefix_of[n] = m.group(1)
    counts: dict = defaultdict(int)
    for p in prefix_of.values():
        counts[p] += 1
    out = []
    for prefix, n in sorted(counts.items(), key=lambda t: -t[1]):
        if n >= SUGGEST_PREFIX and f"{prefix}*".casefold() not in taken:
            out.append({"name": prefix, "people": [f"{prefix}*"]})
    grouped = {p for p, n in counts.items() if n >= SUGGEST_PREFIX}
    plain = [n for n in names if prefix_of.get(n) not in grouped and n.casefold() not in taken][:6]
    if plain:
        out.append({"name": "Family", "people": plain})
    return out


def load(value: str) -> list[str]:
    try:
        people = json.loads(value or "[]")
    except ValueError:
        return []
    return [str(p) for p in people if str(p).strip()] if isinstance(people, list) else []


def find(perspectives: list[dict], pid: Optional[int]) -> Optional[dict]:
    return next((p for p in perspectives if p["id"] == pid), None)
