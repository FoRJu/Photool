"""
Group faces by who they are, without ever knowing who that is.

The rule is one line: a face joins a group when its embedding points the same way
as the group's average, by at least ``threshold``. Three decisions turn that line
into something worth using.

**Big faces go first.** The order of a greedy pass is the algorithm's whole
character: whoever is seen first defines a group, and everything afterwards is
measured against that definition. A 300px face gives a far more trustworthy
vector than a 30px one, so the pass starts with the clearest faces and lets the
marginal ones fall in behind them.

**A group is its average, not its first member.** Comparing against every member
instead would let a chain of look-alikes join two people into one — A resembles
B, B resembles C, so A and C end up together. Comparing against the average of
what a group already contains cannot chain that way.

**Named groups are seeds.** A pass over new faces must not renumber or dissolve
a group somebody has already put a name to, so named groups arrive as seeds and
keep their identity. They can still grow.

What is left is fragmentation: the same person split across two groups because
the first few photographs of them were unalike. A closing pass merges groups
whose averages agree more closely than the same-person bar — closer, because a
merge is a name on the wrong photographs whereas a split is a click.

Three refinements on top, all aimed at the two ways the plain rule failed on a
family library:

**Good faces define a person; poor ones can only join one.** Every face carries
a quality (size, pose, sharpness, detector confidence — ``faces.detect.quality``).
The pass goes in order of quality rather than size, a face below
``min_seed_quality`` never starts a group of its own, and it joins an existing
one only at a slightly stricter bar. A dozen blurred profiles used to become a
dozen one-photo "people".

**A person has more than one face.** A frontal and a profile view of the same
child can sit further apart than the average allows, which is how one person
became two groups. A group therefore keeps a few *exemplars* — members that are
unlike each other — and a face may join through the exemplar it resembles,
provided it is still reasonably close to the average and not unlike *any*
exemplar. That last condition is what keeps the chain rule above: a face that
is a stranger to one of the group's own exemplars is a stranger to the group.

**"Not this person" is remembered.** A face carries the groups it was taken out
of (``not_people``); it is never put back into them, however close it sits.

**And it teaches.** The faces taken out of a person are kept as that person's
*counterexamples*. A new face that resembles one of them more than it resembles
the person does not join: the sibling taken out of Alice once keeps the next
photograph of that sibling out too. The same holds for an unnamed group's average
in the closing merge.
"""

import numpy as np

# How much more than the join threshold two group averages must agree before the
# groups are called one person.
MERGE_MARGIN = 0.10

# Exemplars. A group keeps up to this many members as reference faces, each one
# less alike than EXEMPLAR_DISTINCT to the ones it already has.
MAX_EXEMPLARS = 6
EXEMPLAR_DISTINCT = 0.85
# A face joining through an exemplar is scored a little below its similarity to
# it, must be within EXEMPLAR_REACH of the threshold on the average, and must not
# fall more than EXEMPLAR_FLOOR below the threshold against any exemplar.
EXEMPLAR_SLACK = 0.04
EXEMPLAR_REACH = 0.12
EXEMPLAR_FLOOR = 0.25
# A face below the seed quality joins only at this much above the threshold.
LOW_QUALITY_MARGIN = 0.05


def unit(vector: np.ndarray) -> np.ndarray:
    """The same direction at unit length, so a dot product is a cosine."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(vector)
    return vector / norm


def centroid(vectors) -> np.ndarray:
    """The average direction of a group of faces."""
    stack = np.stack([unit(v) for v in vectors])
    return unit(stack.mean(axis=0))


def pick_exemplars(vectors, qualities=None, limit: int = MAX_EXEMPLARS,
                   distinct: float = EXEMPLAR_DISTINCT) -> np.ndarray:
    """A few members of a group that are unlike each other, best quality first:
    the frontal face, then the profile, then the face from three years later."""
    vectors = [unit(v) for v in vectors]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    qualities = list(qualities) if qualities is not None else [1.0] * len(vectors)
    order = sorted(range(len(vectors)), key=lambda i: -float(qualities[i]))
    chosen: list[np.ndarray] = []
    for i in order:
        v = vectors[i]
        if not chosen or max(float(c @ v) for c in chosen) < distinct:
            chosen.append(v)
            if len(chosen) >= limit:
                break
    return np.stack(chosen).astype(np.float32)


def _score(vector, average, exemplars, threshold) -> float:
    """How strongly a face belongs to one group, on the threshold's scale.

    The average alone, or the exemplar it most resembles (a little discounted),
    whichever is higher — the exemplar route only when the face is also near the
    average and not a stranger to any exemplar.
    """
    on_average = float(average @ vector)
    if exemplars is None or len(exemplars) < 2:
        return on_average
    sims = exemplars @ vector
    if on_average < threshold - EXEMPLAR_REACH or float(sims.min()) < threshold - EXEMPLAR_FLOOR:
        return on_average
    return max(on_average, float(sims.max()) - EXEMPLAR_SLACK)


def _refuted(vector, score, counter) -> bool:
    """Whether a face is more like one of a person's counterexamples than like the person."""
    return counter is not None and len(counter) > 0 and float((counter @ vector).max()) > score


def group(
    faces: list,
    threshold: float,
    seeds: list[tuple] | None = None,
    min_px: float = 0.0,
    merge_margin: float = MERGE_MARGIN,
    min_seed_quality: float = 0.0,
) -> tuple[list[int], dict[int, np.ndarray], dict[int, int]]:
    """Assign every face to a group.

    ``faces`` is anything with ``embedding``, ``px`` and ``face_id``, and
    optionally ``quality`` (0..1) and ``not_people`` (group ids it must not
    join). ``seeds`` are ``(group_id, average, member_count)`` for groups that
    already have a name, optionally with a fourth item: an array of exemplar
    vectors, and a fifth: an array of counterexamples, faces taken out of it. Faces under ``min_px`` come back assigned to 0, meaning "found, not
    grouped"; so do faces under ``min_seed_quality`` that resemble no group.

    Returns ``(assignments, averages, counts)`` where ``assignments[i]`` is the
    group for ``faces[i]``, and the two dictionaries describe the resulting
    groups.
    """
    assignments = [0] * len(faces)
    counts: dict[int, int] = {}
    counter: dict[int, np.ndarray] = {}
    # The groups each group's members have refused, for the closing merge.
    refusing: dict[int, set[int]] = {}
    next_id = 1

    # Every group is a row: its average, and its exemplars padded to a fixed
    # width. A face is then scored against all groups in two matrix products
    # instead of one Python call per group — the difference, on a library of
    # 67,000 faces and a few thousand groups, between ten minutes and seconds.
    seeds = [s for s in seeds or [] if np.any(unit(s[1]))]
    width = max([MAX_EXEMPLARS] + [len(s[3]) for s in seeds
                                   if len(s) > 3 and s[3] is not None])
    dim = 0
    for vector in [s[1] for s in seeds] + [f.embedding for f in faces]:
        dim = np.asarray(vector).size
        if dim:
            break
    capacity = max(16, len(seeds))
    group_ids: list[int] = []
    row_of: dict[int, int] = {}
    avg_rows = np.zeros((capacity, dim), dtype=np.float32)
    ex_rows = np.zeros((capacity, width, dim), dtype=np.float32)
    ex_count = np.zeros(capacity, dtype=np.int64)

    def add_group(group_id: int, average: np.ndarray, members: list[np.ndarray]) -> None:
        nonlocal avg_rows, ex_rows, ex_count
        row = len(group_ids)
        if row == len(avg_rows):
            grow = len(avg_rows)
            avg_rows = np.concatenate([avg_rows, np.zeros_like(avg_rows[:grow])])
            ex_rows = np.concatenate([ex_rows, np.zeros_like(ex_rows[:grow])])
            ex_count = np.concatenate([ex_count, np.zeros_like(ex_count[:grow])])
        group_ids.append(group_id)
        row_of[group_id] = row
        avg_rows[row] = average
        ex_rows[row, :len(members)] = members
        ex_count[row] = len(members)

    for seed in seeds:
        group_id, average, count = seed[0], seed[1], seed[2]
        vector = unit(average)
        if len(seed) > 3 and seed[3] is not None and len(seed[3]):
            members = [unit(v) for v in seed[3]]
        else:
            members = [vector]
        add_group(group_id, vector, members)
        counts[group_id] = max(0, int(count))
        if len(seed) > 4 and seed[4] is not None and len(seed[4]):
            counter[group_id] = np.stack([unit(v) for v in seed[4]])
        next_id = max(next_id, group_id + 1)

    def quality_of(face) -> float:
        q = getattr(face, "quality", None)
        return 1.0 if q is None else float(q)

    order = sorted(
        range(len(faces)),
        key=lambda i: (
            -quality_of(faces[i]),
            -float(getattr(faces[i], "px", 0.0)),
            int(getattr(faces[i], "face_id", i)),
        ),
    )
    slots = np.arange(width)

    for index in order:
        face = faces[index]
        if float(getattr(face, "px", 0.0)) < min_px:
            continue
        vector = unit(face.embedding)
        if not np.any(vector):
            continue
        poor = quality_of(face) < min_seed_quality
        refused = set(getattr(face, "not_people", None) or ())

        rows = len(group_ids)
        if rows:
            # _score for every group at once: the average, or the exemplar the
            # face most resembles when the face is near the average and no
            # stranger to any exemplar. A lone exemplar is the average's route.
            on_average = (avg_rows[:rows] @ vector).astype(np.float64)
            scores = on_average.copy()
            # Only a group whose average is within reach can be joined through
            # an exemplar, and for a stranger that is almost none of them.
            near = np.flatnonzero((ex_count[:rows] >= 2)
                                  & (on_average >= threshold - EXEMPLAR_REACH))
            if near.size:
                sims = (ex_rows[near] @ vector).astype(np.float64)
                held = slots[None, :] < ex_count[near, None]
                nearest = np.where(held, sims, -np.inf).max(axis=1)
                farthest = np.where(held, sims, np.inf).min(axis=1)
                through = farthest >= threshold - EXEMPLAR_FLOOR
                scores[near] = np.where(through, np.maximum(on_average[near], nearest - EXEMPLAR_SLACK),
                                        on_average[near])
            for group_id in refused:
                if group_id in row_of:
                    scores[row_of[group_id]] = -np.inf
            for group_id, examples in counter.items():
                row = row_of[group_id]
                if scores[row] >= threshold and _refuted(vector, float(scores[row]), examples):
                    scores[row] = -np.inf
            # The first of equals wins, as it did when the groups were walked in turn.
            row = int(np.argmax(scores))
            best = float(scores[row])
            bar = threshold + (LOW_QUALITY_MARGIN if poor else 0.0)
            if best > -2.0 and best >= bar:
                best_id = group_ids[row]
                size = counts[best_id]
                avg_rows[row] = unit((avg_rows[row] * size + vector) / (size + 1))
                counts[best_id] = size + 1
                assignments[index] = best_id
                if refused:
                    refusing.setdefault(best_id, set()).update(refused)
                kept = int(ex_count[row])
                if (not poor and kept < MAX_EXEMPLARS
                        and float((ex_rows[row, :kept] @ vector).max()) < EXEMPLAR_DISTINCT):
                    ex_rows[row, kept] = vector
                    ex_count[row] = kept + 1
                continue

        if poor:
            continue
        add_group(next_id, vector, [vector])
        counts[next_id] = 1
        assignments[index] = next_id
        if refused:
            refusing[next_id] = set(refused)
        next_id += 1

    averages = {group_id: avg_rows[row].copy() for group_id, row in row_of.items()}

    if merge_margin is not None:
        assignments, averages, counts = _merge_close_groups(
            assignments,
            averages,
            counts,
            min(1.0, threshold + merge_margin),
            {seed[0] for seed in seeds or []},
            refusing,
            counter,
        )
    return assignments, averages, counts


def _merge_close_groups(
    assignments: list[int],
    averages: dict[int, np.ndarray],
    counts: dict[int, int],
    bar: float,
    protected: set[int],
    refusing: dict[int, set[int]] | None = None,
    counter: dict[int, np.ndarray] | None = None,
) -> tuple[list[int], dict[int, np.ndarray], dict[int, int]]:
    """Fold together groups whose averages agree more closely than ``bar``.

    ``protected`` groups — the named ones — are never absorbed. A name is a
    decision somebody made, and it may well have been the decision to *keep*
    two groups apart: a rebuild that quietly undid that would be worse than the
    fragmentation it was tidying up. A protected group can still take an
    unnamed one into itself, which is only the join rule applied to averages
    instead of single faces.

    Larger groups keep their identity, so a group that has already collected
    faces does not disappear into a smaller one.
    """
    ids = sorted(averages, key=lambda i: (i not in protected, -counts.get(i, 0), i))
    merged_into: dict[int, int] = {}
    kept: list[int] = []
    position: dict[int, int] = {}
    dim = next(iter(averages.values())).size if averages else 0
    kept_matrix = np.zeros((len(ids), dim), dtype=np.float32)
    # Kept groups whose average a merge has moved. Their rows catch up when the
    # next group is kept, which is when the matrix used to be stacked afresh.
    moved: set[int] = set()

    for group_id in ids:
        vector = averages[group_id]
        if group_id not in protected and kept:
            sims = kept_matrix[:len(kept)] @ vector
            # A group whose faces were told "not that person" is never folded
            # into that person, however alike the averages.
            for other in (refusing or {}).get(group_id, set()):
                if other in position:
                    sims[position[other]] = -2.0
            for other, examples in (counter or {}).items():
                k = position.get(other)
                if k is not None and sims[k] >= bar and _refuted(vector, float(sims[k]), examples):
                    sims[k] = -2.0
            best = int(np.argmax(sims))
            if float(sims[best]) >= bar:
                target = kept[best]
                total = counts.get(target, 0) + counts.get(group_id, 0)
                if total:
                    combined = (
                        averages[target] * counts.get(target, 0) + vector * counts.get(group_id, 0)
                    ) / total
                    averages[target] = unit(combined)
                    moved.add(best)
                counts[target] = total
                merged_into[group_id] = target
                continue
        for k in moved:
            kept_matrix[k] = averages[kept[k]]
        moved.clear()
        position[group_id] = len(kept)
        kept_matrix[len(kept)] = vector
        kept.append(group_id)

    if not merged_into:
        return assignments, averages, counts

    def resolve(group_id: int) -> int:
        while group_id in merged_into:
            group_id = merged_into[group_id]
        return group_id

    out = [0 if a == 0 else resolve(a) for a in assignments]
    return (
        out,
        {i: averages[i] for i in kept},
        {i: counts[i] for i in kept},
    )
