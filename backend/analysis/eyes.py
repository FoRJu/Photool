"""
Faces in a burst: whose eyes are open, and which frame has the people at their best.

A burst keeper chosen by the whole frame's sharpness keeps the frame where the
background is crispest, which is not the frame anyone wants when the subject
blinked in it. This looks at the faces instead.

**What is measured.** Each frame of a burst is looked at again with the face
detector (no recognition, which is the expensive half): every face is turned
upright on the same 112px template the face scan uses, and two numbers are kept
for it — its *quality* (size, how frontal, how sharp, how sure the detector was:
``faces.detect.quality``) and its *eye contrast*, the spread of brightness in the
two eye patches against that of the face. An open eye is dark iris on light
sclera under a lid; a closed one is a lid of skin with a line of lashes.

**Why only within a burst.** On its own the eye number is weak: shadow, glasses
and a downward glance all lower it (on real frames open eyes measured 1.0–1.9,
closed or downcast ones 0.3–0.7, but an open eye in a dark room 0.76). Across
the frames of one burst the person, the light and the pose are the same, so the
comparison that matters is with the same person a second earlier: a face is
called *eyes closed* when its contrast falls under ``CLOSED_RATIO`` of that
person's best in the burst, and only when that best is clearly open. A person is
followed through the burst by where the face is — a burst is seconds long and
people do not cross the frame in that time.

The result is a hint and a preference, never a decision: the keeper avoids a
frame where someone's eyes are closed, and the frame says why.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Faces smaller than this (pixels, on the PREVIEW rendering) or less sure than
# this are left out: at that size the eye patch is a few pixels, and weak
# detections at events are hands, phones and patterned shirts.
MIN_PX = 60
MIN_SCORE = 0.8
# A person's frame counts as eyes-closed under this share of their best frame's
# contrast, when that best is at least OPEN_MIN. Set from real bursts: closed
# eyes measured 0.69–0.74 of the same person's open ones a second earlier, and
# eyes open but looking down 0.80.
CLOSED_RATIO = 0.75
OPEN_MIN = 0.9
# What a frame's people are worth when someone in it has closed eyes.
CLOSED_WEIGHT = 0.3
# Two faces in two frames are one person when their centres are this many face
# widths apart or less.
SAME_PLACE = 0.6
# A face under this quality is not judged at all. The detector is sure of some
# things that are not faces worth judging — the back of a head (0.20), a
# blurred profile (0.10) — and letting them count chose a frame of someone's
# back over one of their face. A small, downcast blink is lost with them.
MIN_QUALITY = 0.25
# Faces of a frame that count: the largest few. Background faces at an event
# are many and say nothing about whether this is the frame to keep.
PEOPLE_PER_FRAME = 4
# Whose eyes can be judged at all. A head turned aside reads as closed eyes (a
# sideways glance shows lid and lashes, not iris on white), and so do eyes
# behind sunglasses, whose patch is the lens: both were called closed on frames
# the person kept (2026-05-25, 3957 and 3951). So a face is judged only when it
# is frontal (faces.detect.pose at least JUDGE_POSE) and its eyes can be seen
# (their patch at least SEEN_MIN as bright as the middle of the face), and only
# against the same person's other frames that can be judged too.
JUDGE_POSE = 0.70
SEEN_MIN = 0.42


def eye_contrast(gray: np.ndarray) -> float:
    """The eye patches' brightness spread against the face's, from the aligned
    112px grey crop: higher when the eyes are open."""
    from faces.detect import ALIGN_TEMPLATE

    face = gray[40:100, 30:82].astype(np.float32)
    spreads = []
    for cx, cy in ALIGN_TEMPLATE[:2]:
        x, y = int(round(float(cx))), int(round(float(cy)))
        spreads.append(float(gray[y - 6:y + 6, x - 11:x + 11].astype(np.float32).std()))
    return round(float(np.mean(spreads)) / (float(face.std()) + 1e-3), 4)


def eyes_seen(gray: np.ndarray) -> float:
    """How bright the eye patches are against the middle of the face, from the
    aligned 112px grey crop: about 0.7–1 for eyes that can be seen, low behind
    sunglasses."""
    from faces.detect import ALIGN_TEMPLATE

    middle = float(gray[58:80, 30:82].astype(np.float32).mean())
    patches = []
    for cx, cy in ALIGN_TEMPLATE[:2]:
        x, y = int(round(float(cx))), int(round(float(cy)))
        patches.append(float(gray[y - 6:y + 6, x - 11:x + 11].astype(np.float32).mean()))
    return round(float(np.mean(patches)) / (middle + 1e-3), 4)


def judged(face: dict) -> bool:
    """Whether this face's eyes can be judged: frontal, and not behind dark glasses.
    A face measured before these were kept is not judged."""
    return (face.get("pose", 0.0) >= JUDGE_POSE and face.get("seen", 0.0) >= SEEN_MIN
            and face.get("quality", 0.0) >= MIN_QUALITY)


def measure(image) -> list[dict]:
    """The faces worth judging in one rendered frame, largest first:
    ``[{x, y, w, px, score, quality, pose, eyes, seen}]`` with x/y the centre,
    as fractions."""
    import cv2

    from faces.detect import _engines, align, pose, quality

    detector, _recognizer = _engines()
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        return []
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    detector.setInputSize((width, height))
    _, detected = detector.detect(bgr)
    out = []
    for row in detected if detected is not None else []:
        fx, fy, fw, fh = (float(v) for v in row[:4])
        score = float(row[14])
        if fw < MIN_PX or score < MIN_SCORE:
            continue
        landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        try:
            aligned = align(bgr, landmarks)
        except cv2.error:
            aligned = None
        if aligned is None:
            continue
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        out.append({
            "x": round((fx + fw / 2) / width, 4), "y": round((fy + fh / 2) / height, 4),
            "w": round(fw / width, 4), "px": round(fw, 1), "score": round(score, 3),
            "quality": quality(fw, score, landmarks, aligned), "pose": round(pose(landmarks), 3),
            "eyes": eye_contrast(gray), "seen": eyes_seen(gray),
        })
    out.sort(key=lambda f: -f["px"])
    return out[:PEOPLE_PER_FRAME]


def assess(frames: dict) -> dict:
    """``{filepath: faces}`` for the frames of one burst → ``{filepath: {faces,
    closed, score}}``: how many people, how many of them with closed eyes, and
    what the frame's people are worth (higher is better). A frame with no faces
    scores 0 and is never called closed."""
    people: list[list] = []                      # [[(filepath, face), …], …]
    for filepath, faces in frames.items():
        for face in faces or []:
            if face["quality"] < MIN_QUALITY:
                continue
            home = None
            for person in people:
                ref = person[-1][1]
                reach = SAME_PLACE * max(ref["w"], face["w"])
                if (abs(ref["x"] - face["x"]) <= reach and abs(ref["y"] - face["y"]) <= reach
                        and all(fp != filepath for fp, _ in person)):
                    home = person
                    break
            if home is None:
                people.append([(filepath, face)])
            else:
                home.append((filepath, face))

    out = {fp: {"faces": sum(1 for f in faces or [] if f["quality"] >= MIN_QUALITY),
                "closed": 0, "score": 0.0} for fp, faces in frames.items()}
    for person in people:
        seen = [f["eyes"] for _, f in person if judged(f)]
        best = max(seen) if seen else 0.0
        for filepath, face in person:
            shut = (len(seen) > 1 and judged(face) and best >= OPEN_MIN
                    and face["eyes"] < CLOSED_RATIO * best)
            if shut:
                out[filepath]["closed"] += 1
            out[filepath]["score"] += face["quality"] * (CLOSED_WEIGHT if shut else 1.0)
    for value in out.values():
        value["score"] = round(value["score"], 4)
    return out


# ── Kept per file ────────────────────────────────────────────────────

def stored(db, filepaths) -> dict:
    """``{filepath: (source_size, faces)}`` for those measured."""
    filepaths = list(filepaths)
    out = {}
    with db._connect() as conn:
        for start in range(0, len(filepaths), 500):
            chunk = filepaths[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                f"SELECT filepath, source_size, faces FROM photo_eyes WHERE filepath IN ({marks})", chunk
            ):
                try:
                    out[row["filepath"]] = (int(row["source_size"]), json.loads(row["faces"]))
                except (TypeError, ValueError):
                    pass
    return out


def store(db, rows: list[tuple]) -> None:
    """Keep ``(filepath, source_size, faces)`` rows."""
    if not rows:
        return
    now = datetime.now().isoformat(timespec="seconds")
    with db._connect() as conn:
        conn.executemany(
            "INSERT INTO photo_eyes (filepath, source_size, faces, measured_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(filepath) DO UPDATE SET source_size = excluded.source_size, "
            "faces = excluded.faces, measured_at = excluded.measured_at",
            [(fp, int(size), json.dumps(faces), now) for fp, size, faces in rows],
        )


def measure_files(db, library_root, cache, filepaths, models_dir: str,
                  workers: int = 4, progress=None) -> int:
    """Measure the frames among ``filepaths`` not measured as they are now, from
    the PREVIEW rendering the face scan reads. Returns how many were measured."""
    import io
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from PIL import Image

    from faces.detect import configure
    from thumbnails import ThumbnailSize, get_thumbnail

    configure(models_dir)
    known = stored(db, filepaths)
    todo = []
    for filepath in filepaths:
        source = Path(library_root) / filepath
        try:
            size = source.stat().st_size
        except OSError:
            continue
        if (filepath in known and known[filepath][0] == size
                and all("pose" in face for face in known[filepath][1])):
            continue                # measured as it is now, and with every number
        todo.append((filepath, source, size))

    def one(item):
        filepath, source, size = item
        try:
            jpeg = get_thumbnail(source, cache, ThumbnailSize.PREVIEW)
            if not jpeg:
                return None
            image = Image.open(io.BytesIO(jpeg))
            image.load()
            return filepath, size, measure(image)
        except Exception as exc:
            logger.debug("Could not look at the faces in %s: %s", filepath, exc)
            return None

    done = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eyes") as pool:
        for start in range(0, len(todo), 24):
            found = [r for r in pool.map(one, todo[start:start + 24]) if r]
            store(db, found)
            done += len(found)
            if progress:
                progress(min(start + 24, len(todo)), len(todo))
    return done


def frames_of(db, filepaths) -> dict:
    """``{filepath: faces}`` for the frames measured, whatever their size now;
    the caller decides whether a stale one is worth measuring again."""
    return {fp: faces for fp, (_size, faces) in stored(db, filepaths).items()}


def best_frame(group: list[dict], assessed: Optional[dict]) -> Optional[dict]:
    """The frame of a burst to keep: the people at their best (a closed pair of
    eyes already counts against a frame there), then the sharpest, then the
    earliest. With nothing known of the faces it is the sharpest, as before."""
    best, best_key = None, None
    for row in group or []:
        if not row:
            continue
        a = (assessed or {}).get(row["filepath"]) or {}
        try:
            sharpness = float(row.get("sharpness") or 0.0)
        except (TypeError, ValueError):
            sharpness = 0.0
        key = (float(a.get("score", 0.0)), sharpness, -int(row.get("mtime") or 0))
        if best_key is None or key > best_key:
            best, best_key = row, key
    return best
