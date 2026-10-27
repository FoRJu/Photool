"""
Grouping near-identical frames — the burst problem.

A school concert produces forty frames of the same three seconds. Two signals
together say which frames belong to one moment: the difference hash says two
frames look alike, and the file's modification time says they were written
seconds apart. Neither is enough on its own. The hash alone merges a static
stage across a whole minute of shooting; the timestamps alone merge every frame
of a burst whether the subject moved or not.

Clustering is seeded rather than transitive on purpose. Walking the frames in
time order, each unclaimed frame gathers what looks the same and is within
``MAX_GAP_SECONDS`` of *it*. A chain of frames each 9 seconds apart therefore
does not collapse into one group spanning ten minutes, which is what a
transitive closure would do and what makes burst tools suddenly hide half a
folder.

The keeper suggestion is the sharpest frame in the group. Sharpness is the
variance of the Laplacian measured from the cached thumbnail, so it is only
comparable inside one folder — which is exactly the scope a burst lives in.
"""

from typing import Iterable, Optional

from analysis.image_stats import hamming_distance

# How many of the 64 hash bits may differ and still count as the same frame.
# Frames of one burst differ by a few bits; a recomposed frame differs by many
# more. 12 leaves room for a subject moving between frames without reaching
# into the range where two different shots start looking alike.
MAX_DISTANCE = 12

# Frames further apart in time than this are treated as separate moments even
# when they look identical — the same stage between two songs, say.
#
# A default, not a rule: it is configurable because the right number depends on
# how the camera was being held. A concert shot in three-second bursts of three
# frames wants a tight window; a walk-and-shoot afternoon where the same view is
# framed twice a minute apart wants a wider one.
MAX_GAP_SECONDS = 10

# A single frame is not a burst.
MIN_GROUP = 2


def _usable(row: dict) -> bool:
    """A member needs a hash of the right shape and a timestamp to sit in."""
    digest = row.get("dhash")
    if not isinstance(digest, str) or len(digest) < 8:
        return False
    try:
        int(row.get("mtime") or 0)
    except (TypeError, ValueError):
        return False
    return True


def find_groups(
    members: Iterable[dict], max_gap_seconds: int = MAX_GAP_SECONDS
) -> list[list[dict]]:
    """Cluster frames that are the same shot, oldest first.

    Each member needs ``dhash`` and ``mtime`` (unix seconds); ``filepath`` is
    what the caller uses to identify it. Anything without a usable hash is left
    out rather than guessed at, and is never silently treated as a match.

    ``max_gap_seconds`` is the time window described above. It is a parameter
    rather than a constant so the effect of widening it can be measured against
    a real folder before anything is made the default.

    Returns groups of two or more members, each sorted by capture time. A photo
    lands in at most one group: the first frame to claim it keeps it.
    """
    rows = [row for row in members if _usable(row)]
    rows.sort(key=lambda row: (int(row.get("mtime") or 0), str(row.get("filepath", ""))))

    groups: list[list[dict]] = []
    claimed = [False] * len(rows)

    for i, seed in enumerate(rows):
        if claimed[i]:
            continue
        claimed[i] = True
        seed_time = int(seed.get("mtime") or 0)
        group = [seed]

        for j in range(i + 1, len(rows)):
            other = rows[j]
            # Rows are in time order, so once the window is passed nothing
            # further along can join this group.
            if int(other.get("mtime") or 0) - seed_time > max_gap_seconds:
                break
            if claimed[j]:
                continue
            if hamming_distance(seed["dhash"], other["dhash"]) <= MAX_DISTANCE:
                claimed[j] = True
                group.append(other)

        if len(group) >= MIN_GROUP:
            groups.append(group)

    return groups


def suggest_keeper(group: Iterable[Optional[dict]]) -> Optional[dict]:
    """The frame to keep: the sharpest, earliest on a tie.

    Returns ``None`` for an empty group. A group with no sharpness figures
    keeps its first frame rather than pretending to have an opinion.
    """
    best = None
    best_key = None
    for row in group or []:
        if not row:
            continue
        try:
            sharpness = float(row.get("sharpness") or 0.0)
        except (TypeError, ValueError):
            sharpness = 0.0
        key = (sharpness, -int(row.get("mtime") or 0))
        if best_key is None or key > best_key:
            best, best_key = row, key
    return best
