"""
An agent's culling suggestions, kept apart from the person's own decisions.

An agent (a vision model over MCP) can look at a folder and say "pick this, two
stars, reject that one: her eyes are closed". Written straight into the rating
and flag, those answers would be indistinguishable from the person's own, and a
wrong one would sit in the library looking like a decision somebody made. So
they are *proposals*: one row per photograph, with the agent's reason, shown in
the grid beside the person's marks until the person accepts or dismisses them.

Three rules keep the person in charge:

* **Accepting never overwrites a decision the person made**, unless asked to
  (accepting one photograph by hand in the grid is asking). A field the person
  has already set (stars above 0, a pick or reject, a label) is kept, and the
  answer says how many were.
* **Every accepted pass can be undone.** Accepting keeps what each field held
  before, and undoing puts it back, but only where the field still holds what
  the proposal set. A photograph the person has re-rated since is theirs.
* **Nothing here is written to the sidecar** until it is accepted, at which
  point it is an ordinary rating, written as every rating is.

A proposal is also how an agent is checked before it is trusted: propose on a
folder the person has already culled, then :func:`evaluate` compares the two.
"""

from datetime import datetime
from typing import Iterable, Optional

FIELDS = ("rating", "flag", "color_label")
# What an unset field holds in the catalog: a field at this value is the
# person's to fill, not a decision to protect.
UNSET = {"rating": 0, "flag": 0, "color_label": 0}
STATUSES = ("pending", "accepted", "dismissed")

_COLUMNS = ("filepath", "folder", "rating", "flag", "color_label", "reason", "run", "agent",
            "status", "prev_rating", "prev_flag", "prev_color_label", "created_at", "decided_at")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _rows(conn, where: str, args: list) -> list[dict]:
    return [dict(r) for r in conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM cull_proposals WHERE {where} ORDER BY filepath", args)]


def select(db, *, folder: Optional[str] = None, run: Optional[str] = None,
           filepaths: Optional[Iterable[str]] = None, status: Optional[str] = None) -> list[dict]:
    """The proposals matching every condition given (all of them when none is)."""
    where, args = ["1 = 1"], []
    if folder is not None:
        where.append("folder = ?")
        args.append(folder)
    if run:
        where.append("run = ?")
        args.append(run)
    if status:
        where.append("status = ?")
        args.append(status)
    with db._connect() as conn:
        if filepaths is None:
            return _rows(conn, " AND ".join(where), args)
        paths = list(dict.fromkeys(filepaths))
        out = []
        for start in range(0, len(paths), 500):
            chunk = paths[start:start + 500]
            out += _rows(conn, " AND ".join(where + [f"filepath IN ({','.join('?' * len(chunk))})"]),
                         args + chunk)
        return out


def propose(db, items: list[dict], run: str = "", agent: str = "") -> int:
    """Keep each ``{filepath, folder, rating?, flag?, color_label?, reason}`` as the
    photograph's pending proposal, replacing any earlier one, whatever its state."""
    now = _now()
    rows = [(it["filepath"], it["folder"], it.get("rating"), it.get("flag"), it.get("color_label"),
             (it.get("reason") or "").strip(), run, agent, now) for it in items]
    with db._connect() as conn:
        conn.executemany(
            """INSERT INTO cull_proposals
                 (filepath, folder, rating, flag, color_label, reason, run, agent, status,
                  prev_rating, prev_flag, prev_color_label, created_at, decided_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, '')
               ON CONFLICT(filepath) DO UPDATE SET
                 folder = excluded.folder, rating = excluded.rating, flag = excluded.flag,
                 color_label = excluded.color_label, reason = excluded.reason, run = excluded.run,
                 agent = excluded.agent, status = 'pending', prev_rating = NULL, prev_flag = NULL,
                 prev_color_label = NULL, created_at = excluded.created_at, decided_at = ''""",
            rows,
        )
    return len(rows)


def _current(record) -> dict:
    return {"rating": int(record.rating or 0), "flag": int(record.flag or 0),
            "color_label": int(record.color_label or 0)}


_SETTERS = {"rating": "set_rating", "flag": "set_flag", "color_label": "set_color_label"}


def accept(catalog, proposals: list[dict], overwrite: bool = False) -> dict:
    """Make pending proposals the photographs' ratings, flags and labels.

    A field the person has set is kept unless ``overwrite``. Returns how many
    proposals were accepted, how many fields were set, and how many were kept
    because the person had already decided them."""
    accepted = changed = kept = 0
    now = _now()
    for p in proposals:
        if p["status"] != "pending":
            continue
        record = catalog.get_photo(p["filepath"])
        if record is None:
            continue
        current = _current(record)
        prev = {}
        for field in FIELDS:
            want = p[field]
            if want is None or want == current[field]:
                continue
            if current[field] != UNSET[field] and not overwrite:
                kept += 1
                continue
            prev[field] = current[field]
            getattr(catalog, _SETTERS[field])(p["filepath"], int(want))
            changed += 1
        with catalog.db._connect() as conn:
            conn.execute(
                "UPDATE cull_proposals SET status = 'accepted', decided_at = ?, prev_rating = ?, "
                "prev_flag = ?, prev_color_label = ? WHERE filepath = ?",
                (now, prev.get("rating"), prev.get("flag"), prev.get("color_label"), p["filepath"]),
            )
        accepted += 1
    return {"accepted": accepted, "changed": changed, "kept": kept}


def dismiss(db, proposals: list[dict]) -> int:
    """Set pending proposals aside; they stay on record for :func:`evaluate`."""
    paths = [p["filepath"] for p in proposals if p["status"] == "pending"]
    with db._connect() as conn:
        conn.executemany("UPDATE cull_proposals SET status = 'dismissed', decided_at = ? WHERE filepath = ?",
                         [(_now(), fp) for fp in paths])
    return len(paths)


def undo(catalog, proposals: list[dict]) -> dict:
    """Put back what accepting replaced, and make the proposals pending again.

    A field is restored only while it still holds what the proposal set: a
    photograph the person has changed since keeps the person's answer."""
    undone = restored = kept = 0
    for p in proposals:
        if p["status"] != "accepted":
            continue
        record = catalog.get_photo(p["filepath"])
        if record is not None:
            current = _current(record)
            for field in FIELDS:
                prev = p[f"prev_{field}"]
                if prev is None:
                    continue
                if current[field] != p[field]:
                    kept += 1
                    continue
                getattr(catalog, _SETTERS[field])(p["filepath"], int(prev))
                restored += 1
        with catalog.db._connect() as conn:
            conn.execute(
                "UPDATE cull_proposals SET status = 'pending', decided_at = '', prev_rating = NULL, "
                "prev_flag = NULL, prev_color_label = NULL WHERE filepath = ?", (p["filepath"],))
        undone += 1
    return {"undone": undone, "restored": restored, "kept": kept}


def _kept(rating: int, flag: int, keep_rating: int) -> bool:
    """A photograph somebody would keep: picked, or rated at least ``keep_rating``,
    and not rejected."""
    if flag == -1:
        return False
    return flag == 1 or rating >= keep_rating


# Two keeps are the same moment when they are this close: the person's choice
# between near-identical twins is close to arbitrary (calibration/culling.example.md),
# so keeping 0151 where the person kept 0150 is agreement about the picture.
MOMENT_FRAMES = 2
MOMENT_SECONDS = 3.0


def _moment(record) -> Optional[tuple]:
    """``(camera prefix, frame number, capture time)`` for matching moments, or
    None when the name carries no frame number or the date cannot be read."""
    import re

    m = re.match(r"^(.*?)(\d+)$", record.base_name or "")
    try:
        when = datetime.strptime((record.capture_date or "")[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    return (m.group(1), int(m.group(2)), when) if m else None


def _same_moment(a: Optional[tuple], b: Optional[tuple]) -> bool:
    return (a is not None and b is not None and a[0] == b[0]
            and abs(a[1] - b[1]) <= MOMENT_FRAMES
            and abs((a[2] - b[2]).total_seconds()) <= MOMENT_SECONDS)


def _matched(keeps: list, others: list) -> int:
    """How many of ``keeps`` have a keep of the other side at the same moment."""
    return sum(1 for k in keeps if any(_same_moment(k, o) for o in others))


def evaluate(catalog, proposals: list[dict], keep_rating: int = 1, limit: int = 50) -> dict:
    """How far an agent's proposals agree with the person's own cull.

    Only pending and dismissed proposals are compared: an accepted one has
    become the person's rating and would agree with itself. A photograph the
    person never marked counts as not kept, so this is meant for folders the
    person has finished culling."""
    flags = {a: {b: 0 for b in ("pick", "none", "reject")} for a in ("pick", "none", "reject")}
    name = {1: "pick", 0: "none", -1: "reject"}
    both = agent_only = person_only = neither = 0
    rating_n = rating_exact = rating_near = 0
    rating_err = 0.0
    disagreements = []
    compared = 0
    agent_moments, person_moments = [], []
    for p in proposals:
        if p["status"] == "accepted":
            continue
        record = catalog.get_photo(p["filepath"])
        if record is None:
            continue
        compared += 1
        person = _current(record)
        if p["flag"] is not None:
            flags[name[int(p["flag"])]][name[person["flag"]]] += 1
        if p["rating"] is not None:
            rating_n += 1
            diff = abs(int(p["rating"]) - person["rating"])
            rating_err += diff
            rating_exact += diff == 0
            rating_near += diff <= 1
        agent_keep = _kept(int(p["rating"] or 0), int(p["flag"] or 0), keep_rating)
        person_keep = _kept(person["rating"], person["flag"], keep_rating)
        if agent_keep:
            agent_moments.append(_moment(record))
        if person_keep:
            person_moments.append(_moment(record))
        if agent_keep and person_keep:
            both += 1
        elif agent_keep:
            agent_only += 1
        elif person_keep:
            person_only += 1
        else:
            neither += 1
        if agent_keep != person_keep and len(disagreements) < limit:
            disagreements.append({
                "filepath": p["filepath"], "agent": "keep" if agent_keep else "drop",
                "person": "keep" if person_keep else "drop", "reason": p["reason"],
                "agent_rating": p["rating"], "agent_flag": p["flag"],
                "person_rating": person["rating"], "person_flag": person["flag"],
            })

    def share(a, b):
        return round(a / b, 3) if b else None

    return {
        "compared": compared,
        "keep": {
            "agreement": share(both + neither, compared),
            # Of what the agent would keep, how much the person kept; of what the
            # person kept, how much the agent found.
            "precision": share(both, both + agent_only),
            "recall": share(both, both + person_only),
            "both_keep": both, "agent_only": agent_only, "person_only": person_only, "both_drop": neither,
            # The same, counting a keep as found when the other side kept a frame of
            # the same moment (MOMENT_FRAMES, MOMENT_SECONDS): which twin is kept
            # does not count against the agent.
            "by_moment": {
                "precision": share(_matched(agent_moments, person_moments), len(agent_moments)),
                "recall": share(_matched(person_moments, agent_moments), len(person_moments)),
            },
        },
        "flags": {"agent_by_person": flags,
                  "agreement": share(sum(flags[k][k] for k in flags), sum(sum(r.values()) for r in flags.values()))},
        "rating": {"compared": rating_n, "mean_abs_error": share(rating_err, rating_n),
                   "exact": share(rating_exact, rating_n), "within_one": share(rating_near, rating_n)},
        "disagreements": disagreements,
    }
