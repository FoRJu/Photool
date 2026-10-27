"""
The two passes that make people out of faces, and the pictures to look at.

``scan_folder`` walks a folder the way the measurement pass does: read the
cheapest rendering that is good enough, do the work, store it, skip anything
already done. What is different is the rendering — a face has to be a decent
number of pixels across before its vector means anything. The 2000px
``PREVIEW`` rendering is read by default (``face_scan_size: preview``), the one
the cull view already keeps: at the 1024px ``FACE`` rendering the faces of a
school concert or a family table are mostly 20-35px wide, under any floor a
vector can be trusted at, and doubling the resolution doubles every face.

``rebuild_people`` turns stored vectors into groups. It runs after every scan and
on request, and it is a full pass rather than an incremental one because the
whole library's worth of vectors is only a few megabytes and a group's meaning
shifts as its members change.
"""

import io
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from catalog.database import CLEAR_FACE_QUALITY, CatalogDB
from faces.clustering import _score, centroid, group, pick_exemplars, unit
from faces.detect import MIN_GROUP_PX

# A face this far under the seed quality joins only a person who is already clear.
MIN_SEED_QUALITY = CLEAR_FACE_QUALITY
# An unnamed group whose best match among the named people scores at least this
# far below the join threshold is offered as "is this Alice?".
SUGGEST_BELOW = 0.15

# Faces too blurred, small or turned away to say who they are do not scatter:
# the recogniser puts them all near one point, whoever they are — the
# "unrecognisable identity" its training leaves behind. Grouped like any other
# face they became one "person" of thousands (8,264 faces on a real library,
# median quality 0.06), while real people's faces, poor ones included, sit far
# from that point (median 0.05). The point is found afresh on every rebuild as
# the average of the poorest faces, which needs enough of them to be steady.
UNCLEAR_QUALITY = 0.05
UNCLEAR_MIN_FACES = 100
# How alike a face may be to that point and still be grouped. At 0.40 a real
# library lost 86% of the blurry group and under 1% of the faces of the others.
UNCLEAR_SIMILARITY = 0.40


def unclear_centre(faces: list) -> Optional[np.ndarray]:
    """Where the recogniser puts faces it cannot tell apart, or None when there
    are too few poor faces to say."""
    poor = [unit(f.embedding) for f in faces if f.quality < UNCLEAR_QUALITY]
    if len(poor) < UNCLEAR_MIN_FACES:
        return None
    return unit(np.stack(poor).mean(axis=0))


def split_unclear(faces: list, centre: Optional[np.ndarray]) -> tuple[list, list]:
    """``(clear, unclear)``: the faces that say who they are, and the rest."""
    if centre is None or not faces:
        return list(faces), []
    stack = np.stack([unit(f.embedding) for f in faces])
    near = (stack @ centre) >= UNCLEAR_SIMILARITY
    return ([f for f, n in zip(faces, near) if not n],
            [f for f, n in zip(faces, near) if n])

# Which rendering a scan reads: the name in config, the thumbnail size it means.
SCAN_SIZES = ("preview", "face")


def _scan_size(name: str):
    from thumbnails import ThumbnailSize

    return ThumbnailSize.FACE if (name or "").lower() == "face" else ThumbnailSize.PREVIEW

logger = logging.getLogger(__name__)

# How many frames to render before looking for faces in them. The rendering is
# what costs the time, and it happens in one batch per chunk: one exiftool
# process for a couple of dozen files, which is fast, against one for a whole
# folder, which is not. Handing six hundred frames to the batch at once writes
# several gigabytes of embedded JPEGs to a temporary directory and holds them all
# in memory, and the exiftool call that does it takes longer than its own
# timeout — so everything falls back to one process a frame, which is the
# slowest path there is. A chunk keeps the memory bounded and the bar moving.
RENDER_CHUNK = 24

# How many frames are looked at at once. This is what decides how many cores a
# scan uses: both models run on one thread inside OpenCV, so a serial pass held
# 0.99 of a core no matter that the machine has twenty. Measured over 96 frames
# of a real folder (65 ms of detection each): two threads 1.6x, four 1.9x, six
# 2.0x, and past eight it stops improving and starts costing — sixteen threads
# were slower than eight while burning ten cores. The ceiling is not the core
# count but the part of each frame that has to hold the GIL, so this is capped
# rather than scaled to the machine.
FACE_WORKERS = min(os.cpu_count() or 4, 8)


def _vector(row: dict) -> Optional[np.ndarray]:
    """One stored face's vector, or None when the row cannot be read."""
    blob = row.get("embedding")
    if not blob:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    return vector if vector.size else None


class _StoredFace:
    """A stored face in the shape the grouping wants."""

    __slots__ = ("face_id", "filepath", "px", "embedding", "cluster_id", "quality", "not_people")

    def __init__(self, row: dict, vector: np.ndarray):
        from catalog.database import parse_refused

        self.face_id = int(row["id"])
        self.filepath = row["filepath"]
        self.px = float(row.get("px") or 0.0)
        self.embedding = vector
        self.cluster_id = int(row.get("cluster_id") or 0)
        quality = row.get("quality")
        self.quality = 1.0 if quality is None else float(quality)
        self.not_people = parse_refused(row.get("not_people") or "")


def scan_folder(
    db: CatalogDB,
    cache,
    library_root: str,
    folder: str,
    models_dir: str,
    min_group_px: float = MIN_GROUP_PX,
    should_stop: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    recognizer: str = "arcface",
    scan_size: str = "preview",
) -> dict:
    """Find and describe the faces in every photograph of one folder.

    One photograph, not one file. A CR3 and the JPG beside it are two files of
    the same moment, so the scan follows the pairing the grid uses and looks at
    whichever half that is — otherwise every person in the library appears twice,
    once per half. Faces stored against the other half by an earlier run are
    dropped for the same reason.

    Never raises for a single file: a frame that cannot be read is counted and
    the pass continues. A missing model does raise, because that is a fact about
    the installation rather than about one photograph, and every frame after it
    would fail the same way.
    """
    from faces.detect import configure, find_faces, model_id
    from thumbnails import generate_thumbnails_batch, get_thumbnail

    configure(models_dir, recognizer)
    model = model_id(recognizer)
    rendering = _scan_size(scan_size)

    folder_dir = Path(library_root) / folder
    if not folder_dir.is_dir():
        return {
            "folder": folder,
            "total": 0,
            "scanned": 0,
            "skipped": 0,
            "failed": 0,
            "faces": 0,
            "error": "Folder not found",
        }

    photos = db.get_photos_in_folder(folder, pair_mode="paired")
    result = {
        "folder": folder,
        "total": len(photos),
        "scanned": 0,
        "skipped": 0,
        "failed": 0,
        "faces": 0,
        "dropped": 0,
    }

    # The other half of each pair, and anything smaller than a face is worth
    # describing: rows written before either rule existed are cleaned up here,
    # once per folder rather than once per frame.
    keep = {photo.filepath for photo in photos}
    for other in db.get_photos_in_folder(folder, pair_mode="all"):
        if other.filepath not in keep:
            result["dropped"] += db.forget_faces(other.filepath)
    result["dropped"] += db.prune_small_faces(min_group_px)

    todo: list[tuple[object, Path]] = []
    for photo in photos:
        source = folder_dir / photo.filename
        try:
            stat = source.stat()
        except OSError:
            result["failed"] += 1
            continue
        if db.faces_are_current(photo.filepath, int(stat.st_mtime), int(stat.st_size), model=model):
            result["skipped"] += 1
            continue
        todo.append((photo, source))

    def _describe(source: Path):
        """Read one frame's rendering and describe the faces in it.

        Runs on a worker thread and deliberately touches no database: the scan
        stores everything on its own thread, so many detectors never contend
        for the single SQLite write lock and every face row still lands from
        one place. Returns None when the frame has no usable rendering.
        """
        jpeg = get_thumbnail(source, cache, rendering)
        if not jpeg:
            return None
        image = Image.open(io.BytesIO(jpeg))
        image.load()
        return find_faces(image, min_px=min_group_px)

    with ThreadPoolExecutor(
        max_workers=FACE_WORKERS, thread_name_prefix="faces"
    ) as pool:
        for start in range(0, len(todo), RENDER_CHUNK):
            if should_stop and should_stop():
                break
            chunk = todo[start:start + RENDER_CHUNK]

            # The batch is only a head start: a frame it missed is rendered on
            # its own below, so a batch that fails must not end the folder.
            try:
                generate_thumbnails_batch(
                    [source for _, source in chunk], cache, rendering
                )
            except Exception as exc:
                logger.warning("Could not render a batch in %s: %s", folder, exc)
            if progress:
                progress("rendering", min(start + len(chunk), len(todo)), len(todo))

            # Look at the whole chunk at once, then store the results in this
            # thread. The frames are independent, and detection is essentially
            # all of a scan's cost, so leaving this serial is what pinned a scan
            # to one core.
            futures = [pool.submit(_describe, source) for _, source in chunk]

            for (photo, source), future in zip(chunk, futures):
                if should_stop and should_stop():
                    for pending in futures:
                        pending.cancel()
                    break

                try:
                    faces = future.result()
                except Exception as exc:
                    logger.warning("Could not look for faces in %s: %s", photo.filename, exc)
                    result["failed"] += 1
                    continue

                if faces is None:
                    result["failed"] += 1
                    continue

                try:
                    stat = source.stat()
                except OSError:
                    result["failed"] += 1
                    continue

                db.replace_faces(photo.filepath, faces, int(stat.st_mtime), int(stat.st_size),
                                 model=model)
                result["scanned"] += 1
                result["faces"] += len(faces)

                if progress:
                    done = result["scanned"] + result["failed"]
                    progress("scanning", done, len(todo))

    return result


def _weighted_centroid(faces: list) -> np.ndarray:
    """A group's average, the clear faces counting for more than the poor ones."""
    weights = np.array([max(0.05, f.quality) for f in faces], dtype=np.float32)
    stack = np.stack([unit(f.embedding) for f in faces])
    return unit((stack * weights[:, None]).sum(axis=0))


def rebuild_people(db: CatalogDB, threshold: float, min_group_px: float = MIN_GROUP_PX,
                   model: Optional[str] = None) -> dict:
    """Group every stored face, keeping the names already given out.

    A named group enters the pass as a seed carrying the quality-weighted average
    of its members and a few exemplars, so it keeps its identity and its faces,
    and can still grow — through a profile as well as a frontal face. It also
    carries the faces taken out of it, so a face more like one of those than
    like the person stays out. Everything
    else is grouped from scratch — which is why a rebuild can move a face between
    two unnamed groups but never out of a named one. A face that was taken out of
    a group (or whose group was told "that is not Alice") never goes back in.

    Only the faces ``model`` described take part; faces of another model keep
    the group they have until their frame is scanned again.

    Finally every unnamed group is compared with the named people, and the
    closest one within reach is stored as a question for the panel to ask.
    """
    rows = db.list_all_faces(model=model)
    faces: list[_StoredFace] = []
    for row in rows:
        vector = _vector(row)
        if vector is not None:
            faces.append(_StoredFace(row, vector))
    # Vectors of one length only: a stray row from another model is not compared.
    if faces:
        length = max({f.embedding.size for f in faces}, key=lambda n: sum(f.embedding.size == n for f in faces))
        faces = [f for f in faces if f.embedding.size == length]

    named = {c["id"] for c in db.list_clusters() if c.get("name")}
    seeds: list[tuple] = []
    pending: list[_StoredFace] = []
    by_named: dict[int, list[_StoredFace]] = {}
    for face in faces:
        if face.cluster_id in named:
            by_named.setdefault(face.cluster_id, []).append(face)
        else:
            pending.append(face)
    taken_out: dict[int, list[np.ndarray]] = {}
    for face in faces:
        for refused in face.not_people:
            if refused in named:
                taken_out.setdefault(refused, []).append(face.embedding)
    for cluster_id, members in by_named.items():
        seeds.append((
            cluster_id,
            _weighted_centroid(members),
            len(members),
            pick_exemplars([m.embedding for m in members], [m.quality for m in members]),
            taken_out.get(cluster_id, []),
        ))

    # A face that cannot say who it is stays found but joins no one: left in,
    # such faces gather into one enormous "person" (see UNCLEAR_SIMILARITY).
    # Named faces are not touched — a person decided by hand stays decided.
    pending, unclear = split_unclear(pending, unclear_centre(faces))

    assignments, averages, _counts = group(
        pending, threshold, seeds=seeds, min_px=min_group_px,
        min_seed_quality=MIN_SEED_QUALITY,
    )

    # Every group the pass produced needs a row of its own before any face points
    # at it. Existing unnamed groups are left to be dropped once they are empty.
    fresh = sorted({a for a in assignments if a and a not in named})
    reserved: dict[int, int] = dict(zip(fresh, db.create_clusters(len(fresh))))

    db.set_face_clusters(
        {
            **{face.face_id: 0 for face in unclear},
            **{face.face_id: reserved.get(cluster, cluster)
               for face, cluster in zip(pending, assignments)},
        }
    )
    dropped = db.drop_empty_clusters()

    # "Is this Alice?" — each unnamed group against each named person, by the
    # best of the person's average and exemplars. A group whose faces have been
    # told "not Alice" is not asked about her again.
    suggestions: dict[int, tuple[int, float]] = {}
    if seeds:
        members: dict[int, list[_StoredFace]] = {}
        for face, cluster in zip(pending, assignments):
            if cluster and cluster not in named:
                members.setdefault(cluster, []).append(face)
        floor = threshold - SUGGEST_BELOW
        for temp_id, group_faces in members.items():
            average = averages.get(temp_id)
            if average is None:
                continue
            refused = set().union(*(f.not_people for f in group_faces))
            best_id, best = 0, floor
            for seed_id, seed_average, _count, exemplars, _taken_out in seeds:
                if seed_id in refused:
                    continue
                score = float(seed_average @ average)
                if len(exemplars):
                    score = max(score, float((exemplars @ average).max()) - 0.04)
                if score >= best:
                    best_id, best = seed_id, score
            if best_id:
                suggestions[reserved[temp_id]] = (best_id, round(best, 3))
    db.set_cluster_suggestions(suggestions)
    clusters = db.list_clusters()

    return {
        "faces": len(faces),
        "grouped": sum(1 for f in faces if f.cluster_id in named)
        + sum(1 for a in assignments if a),
        "people": len(clusters),
        "dropped": dropped,
        "threshold": threshold,
        "suggestions": len(suggestions),
        "unclear": len(unclear),
    }


def scanned_half(db: CatalogDB, filepath: str) -> str:
    """The file of a photograph that holds its faces.

    A CR3 and its JPG are one photograph and the scan looks at one half, the one
    the grid shows; the viewer may be showing either. The half that was scanned
    wins, then the half a scan would pick, then the file itself.
    """
    folder, _, filename = filepath.rpartition("/")
    base = Path(filename).stem
    with db._connect() as conn:
        halves = [r[0] for r in conn.execute(
            "SELECT filepath FROM photos WHERE folder = ? AND base_name = ?", (folder, base))]
        if not halves:
            return filepath
        marks = ",".join("?" * len(halves))
        scanned = [r[0] for r in conn.execute(
            f"SELECT filepath FROM face_scans WHERE filepath IN ({marks})", halves)]
    if filepath in scanned:
        return filepath
    if scanned:
        return scanned[0]
    shown = {p.filepath for p in db.get_photos_in_folder(folder, pair_mode="paired")}
    return next((h for h in halves if h in shown), filepath)


def full_image(source: Path) -> Optional[Image.Image]:
    """A photograph at its full size, the right way up — the camera's own JPEG,
    or the one a RAW carries."""
    from thumbnails.extract import (RAW_EXTENSIONS, _extract_source_jpeg, _get_raw_orientation,
                                    _normalize_image)

    data = _extract_source_jpeg(source)
    if not data:
        return None
    raw = source.suffix.lower() in RAW_EXTENSIONS
    image, _ = _normalize_image(data, _get_raw_orientation(source) if raw else 1)
    return image.convert("RGB")


def look_closer(
    db: CatalogDB,
    library_root: str,
    filepaths: list[str],
    models_dir: str,
    min_group_px: float = MIN_GROUP_PX,
    recognizer: str = "arcface",
    should_stop: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Look for faces again in each photograph at its full size.

    The scan reads a 2000px preview, where a face at the back of a hall or on a
    stage is a dozen pixels and is never found; at full size about half the
    frames the scan found nobody in turn out to hold someone. That costs a few
    seconds a frame, so it is asked for, a photograph or a folder at a time,
    rather than done to the whole library.

    The faces replace the ones stored. A named face keeps its person where the
    new box lands on the old one (see ``CatalogDB.replace_faces``).
    """
    from faces.detect import configure, find_faces, model_id

    configure(models_dir, recognizer)
    model = model_id(recognizer)
    result = {"scanned": 0, "skipped": 0, "failed": 0, "faces": 0, "found": 0}
    halves = list(dict.fromkeys(scanned_half(db, f) for f in filepaths))
    for done, filepath in enumerate(halves):
        if should_stop and should_stop():
            break
        if progress:
            progress("scanning", done, len(halves))
        source = Path(library_root) / filepath
        try:
            before = len(db.list_faces(filepath))
            image = full_image(source)
            if image is None:
                result["failed"] += 1
                continue
            faces = find_faces(image, min_px=min_group_px)
            stat = source.stat()
        except Exception as exc:
            logger.warning("Could not look closer at %s: %s", filepath, exc)
            result["failed"] += 1
            continue
        db.replace_faces(filepath, faces, int(stat.st_mtime), int(stat.st_size), model=model)
        result["scanned"] += 1
        result["faces"] += len(faces)
        result["found"] += max(0, len(faces) - before)
    if progress:
        progress("scanning", len(halves), len(halves))
    return result


def crop(
    db: CatalogDB,
    cache,
    library_root: str,
    filepath: str,
    face_index: int,
    size: int = 112,
    margin: float = 0.35,
    scan_size: str = "preview",
) -> Optional[bytes]:
    """A small square JPEG of one face, for the row of thumbnails in the panel.

    Cropped from the same rendering the face was found in, so the picture shown
    is the picture the vector was taken from. A margin around the box
    keeps ears and chin, which is what makes a face recognisable at 112px.
    """
    from thumbnails import get_thumbnail

    row = next(
        (r for r in db.list_faces(filepath) if int(r["face_index"]) == int(face_index)),
        None,
    )
    if not row:
        return None

    jpeg = get_thumbnail(Path(library_root) / filepath, cache, _scan_size(scan_size))
    if not jpeg:
        return None
    try:
        image = Image.open(io.BytesIO(jpeg))
        image.load()
        image = image.convert("RGB")
    except OSError:
        return None

    width, height = image.size
    side = max(float(row["w"]) * width, float(row["h"]) * height) * (1.0 + margin)
    if side < size / 2:
        # A face found by looking closer can be a dozen pixels in the preview:
        # cut it from the full-size photograph instead of enlarging the blur.
        full = full_image(Path(library_root) / filepath)
        if full is not None:
            image = full
            width, height = image.size
            side = max(float(row["w"]) * width, float(row["h"]) * height) * (1.0 + margin)
    cx = (float(row["x"]) + float(row["w"]) / 2.0) * width
    cy = (float(row["y"]) + float(row["h"]) / 2.0) * height
    left = max(0, int(cx - side / 2))
    top = max(0, int(cy - side / 2))
    right = min(width, int(cx + side / 2))
    bottom = min(height, int(cy + side / 2))
    if right - left < 2 or bottom - top < 2:
        return None

    image = image.crop((left, top, right, bottom)).resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# A portrait for a timeline: 3:2, the face centred across and a little above the
# middle, at this share of the height — the same in every frame, so one after
# another the person stays put and only grows.
PORTRAIT_ASPECT = 1.5
PORTRAIT_FACE = 0.36
PORTRAIT_EYE_LINE = 0.42


def portrait(
    db: CatalogDB,
    cache,
    library_root: str,
    filepath: str,
    face_index: int,
    width: int = 1200,
    scan_size: str = "preview",
) -> Optional[bytes]:
    """One face's photograph cut so the face sits where every other portrait's
    does. Where the cut runs past the photograph's edge it is black."""
    from thumbnails import get_thumbnail

    row = next(
        (r for r in db.list_faces(filepath) if int(r["face_index"]) == int(face_index)),
        None,
    )
    if not row:
        return None
    jpeg = get_thumbnail(Path(library_root) / filepath, cache, _scan_size(scan_size))
    if not jpeg:
        return None
    try:
        image = Image.open(io.BytesIO(jpeg))
        image.load()
        image = image.convert("RGB")
    except OSError:
        return None

    iw, ih = image.size
    face = max(float(row["w"]) * iw, float(row["h"]) * ih)
    if face < 2:
        return None
    frame_h = face / PORTRAIT_FACE
    frame_w = frame_h * PORTRAIT_ASPECT
    cx = (float(row["x"]) + float(row["w"]) / 2.0) * iw
    cy = (float(row["y"]) + float(row["h"]) / 2.0) * ih
    left = cx - frame_w / 2.0
    top = cy - frame_h * PORTRAIT_EYE_LINE
    height = round(width / PORTRAIT_ASPECT)
    # One affine step crops, pads with black and scales together.
    scale = frame_w / width
    image = image.transform((width, height), Image.AFFINE, (scale, 0, left, 0, scale, top),
                            resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def person_faces(db: CatalogDB, cluster_id: int, threshold: float) -> list[dict]:
    """Every face in one group, the least alike first.

    "Alike" is the score that lets a face in: its cosine to the group's own
    quality-weighted average, or to the exemplar it resembles when it joined
    through one — so a profile that is plainly the person through the group's
    other profiles is not flagged for being unlike the frontal faces. A face is
    never its own exemplar. A face that joined by a whisker, or one
    carried across from an older model by where it sat in the frame, sinks to
    the front, which is where a person checking the group looks first. Faces the
    group's main model did not describe have no score and come last.
    """
    rows = db.faces_in_cluster(cluster_id)
    faces = []
    for row in rows:
        vector = _vector(row)
        faces.append((row, vector))
    sized = [v.size for _, v in faces if v is not None]
    length = max(set(sized), key=sized.count) if sized else 0
    usable = [_StoredFace(r, v) for r, v in faces if v is not None and v.size == length]
    average = _weighted_centroid(usable) if usable else None

    def exemplars_without(skip=None):
        rest = [f for f in usable if f is not skip]
        return pick_exemplars([f.embedding for f in rest], [f.quality for f in rest]) if rest else None

    exemplars = exemplars_without()
    stored = iter(usable)
    out = []
    for row, vector in faces:
        score = None
        if average is not None and vector is not None and vector.size == length:
            face = next(stored)
            v = unit(vector)
            ex = exemplars
            # One of the exemplars itself: the group's exemplars as they would
            # be without it (a handful of faces per group, so cheap enough).
            if len(ex) and bool(np.all(ex == v, axis=1).any()):
                ex = exemplars_without(face)
            score = round(_score(v, average, ex, threshold), 3)
        out.append({
            "filepath": row["filepath"],
            "face_index": int(row["face_index"]),
            "px": round(float(row.get("px") or 0.0), 1),
            "quality": round(float(row.get("quality") if row.get("quality") is not None else 1.0), 2),
            "similarity": score,
        })
    out.sort(key=lambda f: (f["similarity"] is None, f["similarity"] if f["similarity"] is not None else 0.0,
                            f["filepath"], f["face_index"]))
    return out


REPAIRED_KEY = "drawn_faces_on_the_frame"


def repair_drawn_on_edits(db: CatalogDB, catalog, cache, library_root: str) -> int:
    """Put faces drawn by hand on a cropped or levelled photograph back on the
    frame, once.

    Until issue #37 a box drawn in the viewer was kept as drawn, on the finished
    picture, where every face is a box on the whole frame: the viewer showed it
    in the right place, and the face cut out for the person's page and the year
    came from somewhere else. A face drawn after the photo's edit was last saved
    was drawn on that crop, so it can be walked back exactly. Returns how many
    were moved; done once a pass has looked at every one.
    """
    from adjust.geometry import normalize_geometry, source_box
    from thumbnails import ThumbnailSize, get_thumbnail

    if not getattr(db, "writable", True):
        return 0
    with db._connect() as conn:
        if conn.execute("SELECT 1 FROM library_settings WHERE key = ?", (REPAIRED_KEY,)).fetchone():
            return 0
        rows = conn.execute(
            "SELECT f.id, f.filepath, f.x, f.y, f.w, f.h, f.created_at, e.updated_at "
            "FROM photo_faces f JOIN photo_edits e ON e.filepath = f.filepath WHERE f.drawn = 1").fetchall()
    moved, unread = [], 0
    for r in rows:
        if (r["created_at"] or "") <= (r["updated_at"] or ""):
            continue                    # drawn before this crop: on the frame then
        settings = catalog.photo_edit(r["filepath"])
        if not normalize_geometry(settings):
            continue
        jpeg = get_thumbnail(Path(library_root) / r["filepath"], cache, ThumbnailSize.PREVIEW)
        if not jpeg:
            unread += 1                 # the drive is away: try again next start
            continue
        width, height = Image.open(io.BytesIO(jpeg)).size
        box = source_box(settings, (r["x"], r["y"], r["w"], r["h"]), width, height)
        moved.append((*box, max(box[2], box[3]) * ThumbnailSize.PREVIEW.value, r["id"]))
    with db._connect() as conn:
        conn.executemany("UPDATE photo_faces SET x = ?, y = ?, w = ?, h = ?, px = ? WHERE id = ?", moved)
        if not unread:
            conn.execute(
                "INSERT INTO library_settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO NOTHING",
                (REPAIRED_KEY, str(len(moved)), datetime.now().isoformat(timespec="seconds")))
    if moved:
        logger.info("Moved %d face(s) drawn on a cropped photo back onto the frame", len(moved))
    return len(moved)
