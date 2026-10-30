"""
SQLite catalog database.

Stores photo metadata, ratings, flags, and color labels.
Designed to be the fast query layer — XMP sidecars are the
portable truth, and the DB is rebuilt from them if needed.
"""

import json
import sqlite3
import logging
from array import array
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional

from storage_health import open_readonly, probe_writable

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────

class Flag(IntEnum):
    NONE = 0
    PICK = 1      # P key — keeper
    REJECT = -1   # X key — trash


class ColorLabel(IntEnum):
    NONE = 0
    RED = 1
    YELLOW = 2
    GREEN = 3
    BLUE = 4
    PURPLE = 5


def _chunks(items: list, size: int = 500):
    """Yield slices of a list so an IN (...) clause never outgrows SQLite."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ── Data model ───────────────────────────────────────────────────────

@dataclass
class PhotoRecord:
    """A photo entry in the catalog."""
    id: Optional[int] = None
    filepath: str = ""          # relative to library_root
    filename: str = ""
    folder: str = ""            # e.g. "2025/2025-03-06"
    extension: str = ""
    base_name: str = ""         # filename without ext, e.g. "IMG_0001"
    file_size: int = 0          # bytes
    rating: int = 0             # 0-5 stars
    flag: int = 0               # Flag enum: -1, 0, 1
    color_label: int = 0        # ColorLabel enum
    # EXIF cache (denormalized for fast queries)
    camera: str = ""
    lens: str = ""
    focal_length: str = ""
    aperture: str = ""
    shutter_speed: str = ""
    iso: str = ""
    capture_date: str = ""
    dimensions: str = ""
    # When it was taken in the time where it was taken (catalog/clock.py), when
    # that is not what the camera's clock said; "" when it is. A record read
    # from the catalog carries it in capture_date, and the clock's own reading
    # in camera_date.
    local_date: str = ""
    camera_date: str = ""
    # A video's length in seconds; 0 for a photograph.
    duration: float = 0.0
    # Timestamps
    imported_at: str = ""
    updated_at: str = ""


@dataclass
class ThumbnailJobRecord:
    """A background thumbnail generation job."""
    id: Optional[int] = None
    folder: str = ""
    status: str = "queued"      # queued / running / paused / completed / failed / cancelled
    priority: int = 100
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    last_file: str = ""
    error_message: str = ""
    created_at: str = ""
    started_at: str = ""
    updated_at: str = ""


@dataclass
class ExportBasketRecord:
    """A photo selected for export."""
    id: Optional[int] = None
    filepath: str = ""
    folder: str = ""
    filename: str = ""
    added_at: str = ""
    # Whose basket: "local" for this computer, "device:<id>" for a paired one
    # (access.owner_of). Two people choosing photos at once must not export
    # each other's choices.
    owner: str = "local"
    # JSON for one person's own nudge — see export.adjust. Empty means none.
    adjustment: str = ""


# ── Database ─────────────────────────────────────────────────────────

# Derived measurements, one row per photo. Defined once and created through the
# migration list, which runs on every startup whether the database is brand new
# or five years old — that way a new measurement can only be added in one place.
#
# This is deliberately a separate table from ``photos``: nothing in it was
# authored by the user, so losing it costs a re-measure, and a schema change
# here should never touch the table that holds ratings, flags, and tags.
_PHOTO_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS photo_stats (
    filepath TEXT PRIMARY KEY,

    -- What the row was measured from. A row whose mtime or size no longer
    -- matches the file is stale and gets recomputed.
    source_mtime INTEGER DEFAULT 0,
    source_size INTEGER DEFAULT 0,
    source_kind TEXT DEFAULT '',
    computed_at TEXT DEFAULT '',

    -- Bumped when a measurement changes shape, so old rows are recomputed
    -- instead of compared against numbers produced by different code.
    stats_version INTEGER DEFAULT 0,

    -- Exposure: Rec. 709 luma percentiles and the fraction of pixels that are
    -- blown or crushed. Used to match a set of frames, and to warn about one.
    width INTEGER DEFAULT 0,
    height INTEGER DEFAULT 0,
    mean_lum REAL DEFAULT 0,
    lum_p05 REAL DEFAULT 0,
    lum_p50 REAL DEFAULT 0,
    lum_p95 REAL DEFAULT 0,
    clip_high REAL DEFAULT 0,
    clip_low REAL DEFAULT 0,

    -- Grey-world white balance: channel means as ratios against green, i.e. the
    -- colour the camera was treating as white for this frame.
    wb_r REAL DEFAULT 1,
    wb_g REAL DEFAULT 1,
    wb_b REAL DEFAULT 1,

    -- Soft-frame hint (variance of the Laplacian) and the 64-bit difference
    -- hash that groups near-identical frames.
    sharpness REAL DEFAULT 0,
    dhash TEXT DEFAULT ''
);
"""

# One detected face. The box is a fraction of the frame, not pixels, so the same
# row can crop a face out of a 400px grid thumbnail or a 2000px preview without
# knowing either one's size. The embedding is 128 float32 values — unit length,
# so a dot product between two of them is the cosine similarity the grouping
# works in.
_PHOTO_FACES_TABLE = """
CREATE TABLE IF NOT EXISTS photo_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    face_index INTEGER NOT NULL DEFAULT 0,

    x REAL NOT NULL DEFAULT 0,
    y REAL NOT NULL DEFAULT 0,
    w REAL NOT NULL DEFAULT 0,
    h REAL NOT NULL DEFAULT 0,

    -- Detector confidence, and the face's width in the frame that was analysed.
    -- Size is kept because it is the only handle on how much the embedding can
    -- be trusted: a face a few dozen pixels across describes noise.
    score REAL NOT NULL DEFAULT 0,
    px REAL NOT NULL DEFAULT 0,

    embedding BLOB,
    cluster_id INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT '',
    UNIQUE (filepath, face_index)
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    folder TEXT NOT NULL,
    extension TEXT NOT NULL DEFAULT '',
    base_name TEXT NOT NULL DEFAULT '',
    file_size INTEGER DEFAULT 0,

    -- User ratings / cull state
    rating INTEGER DEFAULT 0,
    flag INTEGER DEFAULT 0,
    color_label INTEGER DEFAULT 0,

    -- EXIF cache
    camera TEXT DEFAULT '',
    lens TEXT DEFAULT '',
    focal_length TEXT DEFAULT '',
    aperture TEXT DEFAULT '',
    shutter_speed TEXT DEFAULT '',
    iso TEXT DEFAULT '',
    capture_date TEXT DEFAULT '',
    dimensions TEXT DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,

    -- Timestamps
    imported_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_photos_folder ON photos(folder);
CREATE INDEX IF NOT EXISTS idx_photos_rating ON photos(rating);
CREATE INDEX IF NOT EXISTS idx_photos_flag ON photos(flag);
CREATE INDEX IF NOT EXISTS idx_photos_capture_date ON photos(capture_date);
CREATE INDEX IF NOT EXISTS idx_photos_base_name ON photos(base_name);

CREATE TABLE IF NOT EXISTS thumbnail_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER DEFAULT 100,
    total_items INTEGER DEFAULT 0,
    completed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    last_file TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    created_at TEXT DEFAULT '',
    started_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_thumbnail_jobs_status ON thumbnail_jobs(status);
CREATE INDEX IF NOT EXISTS idx_thumbnail_jobs_folder ON thumbnail_jobs(folder);

CREATE TABLE IF NOT EXISTS export_basket (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL DEFAULT 'local',
    filepath TEXT NOT NULL,
    folder TEXT NOT NULL DEFAULT '',
    filename TEXT NOT NULL DEFAULT '',
    added_at TEXT DEFAULT '',
    adjustment TEXT NOT NULL DEFAULT '',
    UNIQUE (owner, filepath)
);

-- The export panel's old batch styles. Nothing reads it since they were removed
-- (develop presets and paste do that job); kept so the migrations stay in order.
CREATE TABLE IF NOT EXISTS style_presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    settings TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL DEFAULT 'event',
    created_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS photo_tags (
    filepath TEXT NOT NULL,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    added_at TEXT DEFAULT '',
    PRIMARY KEY (filepath, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_photo_tags_tag ON photo_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_photo_tags_file ON photo_tags(filepath);
-- photo_stats is not here: it is created by the migration list below, which
-- runs on every startup, so its DDL only has to exist in one place.
"""

def _pair_key(record: "PhotoRecord") -> tuple:
    """What pairs a RAW with its JPEG: the name. A video is never one of a pair,
    even when a camera numbers it in the same run as its stills."""
    from config import is_video

    return record.base_name, is_video(record.extension)


# A photo's date in SQL as a record reads it: where it was taken.
LOCAL_DATE = "COALESCE(NULLIF(local_date, ''), capture_date)"

# Migration: add base_name column if missing (for existing DBs)
_MIGRATIONS = [
    # The time where a photograph was taken, when its camera's clock was on
    # another (catalog/clock.py). capture_date stays what the clock said.
    "ALTER TABLE photos ADD COLUMN local_date TEXT NOT NULL DEFAULT ''",
    # The time zone a file says its clock was on ("-07:00", "+00:00"; "" when
    # it says none), read once per file at the size it had.
    """CREATE TABLE IF NOT EXISTS photo_clock (
        filepath TEXT PRIMARY KEY,
        clock_offset TEXT NOT NULL DEFAULT '',
        source_size INTEGER NOT NULL DEFAULT 0,
        read_at TEXT DEFAULT ''
    )""",
    # What a person knows about a camera's clock over some days (by what the
    # clock said): the zone it was on, and the zone it was in. Either may be
    # "" for "as the file says" and "where the folder was, or home".
    """CREATE TABLE IF NOT EXISTS camera_clocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera TEXT NOT NULL,
        date_from TEXT NOT NULL DEFAULT '',
        date_to TEXT NOT NULL DEFAULT '',
        clock_zone TEXT NOT NULL DEFAULT '',
        where_zone TEXT NOT NULL DEFAULT '',
        created_at TEXT DEFAULT ''
    )""",
    "ALTER TABLE photos ADD COLUMN base_name TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_photos_base_name ON photos(base_name)",
    # A video's length in seconds (config.VIDEO_EXTENSIONS); 0 for a photograph.
    "ALTER TABLE photos ADD COLUMN duration REAL NOT NULL DEFAULT 0",
    # style_presets table for existing DBs
    """CREATE TABLE IF NOT EXISTS style_presets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        settings TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )""",
    # Event/keyword tags for existing DBs. dc:subject in the XMP sidecar stays
    # the portable copy; these tables are the query index over it.
    """CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        kind TEXT NOT NULL DEFAULT 'event',
        created_at TEXT DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS photo_tags (
        filepath TEXT NOT NULL,
        tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
        added_at TEXT DEFAULT '',
        PRIMARY KEY (filepath, tag_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_photo_tags_tag ON photo_tags(tag_id)",
    "CREATE INDEX IF NOT EXISTS idx_photo_tags_file ON photo_tags(filepath)",
    # Derived measurements for existing databases.
    _PHOTO_STATS_TABLE,
    # Where the user stopped reviewing each folder, so reopening one can offer to
    # pick up where they left off. Not in _SCHEMA: the migration list runs on
    # every startup, so this DDL lives in one place.
    """CREATE TABLE IF NOT EXISTS folder_state (
        folder TEXT PRIMARY KEY,
        last_filepath TEXT,
        updated_at INTEGER NOT NULL DEFAULT 0
    )""",
    # A personal exposure and white-balance nudge, held on the basket row rather
    # than on the photo. It describes how someone wants this frame to come out of
    # this export, not what the library knows about it, and the photo table is
    # the one place that must stay free of anything the export pipeline does.
    "ALTER TABLE export_basket ADD COLUMN adjustment TEXT NOT NULL DEFAULT ''",
    # Saved edit sets: one photo's own eight controls, kept under a name so they
    # can be put back on another frame later. A table of their own rather than
    # sharing style_presets, because `style_presets.name` is UNIQUE in databases
    # that already exist, and a name shared between a batch look and a personal
    # edit would be a name that means two different things.
    """CREATE TABLE IF NOT EXISTS develop_presets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        settings TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )""",
    # Faces found in a frame, and where each one is. A separate table rather
    # than more columns on photo_stats because there is no useful upper bound on
    # how many faces a photograph holds, and a group shot would then need a
    # schema change.
    _PHOTO_FACES_TABLE,
    "CREATE INDEX IF NOT EXISTS idx_photo_faces_file ON photo_faces(filepath)",
    "CREATE INDEX IF NOT EXISTS idx_photo_faces_cluster ON photo_faces(cluster_id)",
    # Which frames have been looked at for faces, and whether they have changed
    # since. A frame with no faces still needs a row here, or every scan would
    # look at it again forever.
    """CREATE TABLE IF NOT EXISTS face_scans (
        filepath TEXT PRIMARY KEY,
        source_mtime INTEGER NOT NULL DEFAULT 0,
        source_size INTEGER NOT NULL DEFAULT 0,
        faces INTEGER NOT NULL DEFAULT 0,
        scanned_at TEXT DEFAULT ''
    )""",
    # A person, as far as this app is concerned: a name the user gave to a group
    # of faces. The photographs stay in the catalog; the group is only a way of
    # reading it.
    """CREATE TABLE IF NOT EXISTS person_clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL DEFAULT '',
        tag_id INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )""",
    # One photograph's own edit, keyed on the photograph rather than on a row in
    # the basket. Non-destructive means the edit outlives the export it was made
    # for: it belongs to the picture, it is there when the basket is cleared, and
    # a folder browsed months later still opens with the frame as it was left.
    #
    # The sidecar holds the same edit and is the durable copy — this table is the
    # index over it, so drawing a grid of two thousand frames does not mean
    # opening two thousand sidecar files. `index_folder` reconciles the two.
    """CREATE TABLE IF NOT EXISTS photo_edits (
        filepath TEXT PRIMARY KEY,
        settings TEXT NOT NULL DEFAULT '',
        source_mtime INTEGER NOT NULL DEFAULT 0,
        source_size INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS idx_photo_edits_updated ON photo_edits(updated_at)",
    # Move the edits that used to live on the basket row onto the photographs
    # they describe. OR IGNORE so a photo that already has its own edit keeps it,
    # and the UPDATE blanks the source afterwards: the migration is then
    # idempotent without a version marker, because the second run finds nothing
    # left to move.
    """
    INSERT OR IGNORE INTO photo_edits (filepath, settings, updated_at)
    SELECT filepath, adjustment, COALESCE(added_at, '')
    FROM export_basket WHERE adjustment IS NOT NULL AND adjustment != ''
    """,
    "UPDATE export_basket SET adjustment = '' WHERE adjustment != ''",
    # Virtual copies: more than one edit of the same file. Version 0 is the photo's
    # own edit and stays in photo_edits, exactly as before; a row here is a copy
    # (1, 2, ...) with its own settings, name and whether an export writes it. A
    # copy with no edit is legitimate (an untouched second start), so unlike
    # photo_edits an empty settings string is kept, not deleted.
    """CREATE TABLE IF NOT EXISTS photo_versions (
        filepath TEXT NOT NULL,
        version INTEGER NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        settings TEXT NOT NULL DEFAULT '',
        export INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT DEFAULT '',
        PRIMARY KEY (filepath, version)
    )""",
    # Faces, second version. What each vector was made with (vectors of two
    # models live in different spaces and are never compared), how far each face
    # can be trusted, and the people a face was taken out of so that a rebuild
    # does not put it straight back.
    "ALTER TABLE photo_faces ADD COLUMN model TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE photo_faces ADD COLUMN quality REAL NOT NULL DEFAULT 1",
    "ALTER TABLE photo_faces ADD COLUMN not_people TEXT NOT NULL DEFAULT ''",
    # A face drawn by hand where the detector found none: a box and a person,
    # with no vector, so it never shapes how anyone is recognised.
    "ALTER TABLE photo_faces ADD COLUMN drawn INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE face_scans ADD COLUMN model TEXT NOT NULL DEFAULT ''",
    # Which of the develop panel's sections a saved look carries (a JSON list);
    # '' for one saved before looks said, which carries what it holds.
    "ALTER TABLE develop_presets ADD COLUMN sections TEXT NOT NULL DEFAULT ''",
    # The named person an unnamed group most resembles, offered as a question.
    "ALTER TABLE person_clusters ADD COLUMN suggest_id INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE person_clusters ADD COLUMN suggest_score REAL NOT NULL DEFAULT 0",
    # Where a photograph was taken, read once from its EXIF (places/gps.py). A
    # row with no coordinates means "looked, none there", so a later pass does
    # not open the file again; the size says when the file is not the one read.
    """CREATE TABLE IF NOT EXISTS photo_gps (
        filepath TEXT PRIMARY KEY,
        lat REAL,
        lon REAL,
        source_size INTEGER NOT NULL DEFAULT 0,
        read_at TEXT DEFAULT ''
    )""",
    # The faces of a burst's frame as the culling reads them (analysis/eyes.py):
    # where each is, how good, how open its eyes; with the size it had.
    """CREATE TABLE IF NOT EXISTS photo_eyes (
        filepath TEXT PRIMARY KEY,
        source_size INTEGER NOT NULL DEFAULT 0,
        faces TEXT NOT NULL DEFAULT '[]',
        measured_at TEXT DEFAULT ''
    )""",
    # What a photograph shows, as CLIP describes it (analysis/clip.py): 512
    # float16 numbers, by the model that made them and the size the file had.
    """CREATE TABLE IF NOT EXISTS photo_clip (
        filepath TEXT PRIMARY KEY,
        model TEXT NOT NULL DEFAULT '',
        source_size INTEGER NOT NULL DEFAULT 0,
        vector BLOB NOT NULL,
        described_at TEXT DEFAULT ''
    )""",
    # A photograph looked at for its camera data (catalog/exif_fill.py), with the
    # size it had, so one that carries none is not opened again.
    """CREATE TABLE IF NOT EXISTS photo_exif_reads (
        filepath TEXT PRIMARY KEY,
        source_size INTEGER NOT NULL DEFAULT 0,
        read_at TEXT DEFAULT ''
    )""",
    # A folder in a person's own words: a title, a line about it, the frame
    # that stands for it, and where it was when the photographs cannot say.
    # Empty means "work it out" (places/store.py), never "blank".
    """CREATE TABLE IF NOT EXISTS folder_notes (
        folder TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        subtitle TEXT NOT NULL DEFAULT '',
        cover TEXT NOT NULL DEFAULT '',
        place TEXT NOT NULL DEFAULT '',
        lat REAL,
        lon REAL,
        updated_at TEXT DEFAULT ''
    )""",
    # The day a folder's photographs were taken when their clock gave none.
    "ALTER TABLE folder_notes ADD COLUMN taken_on TEXT NOT NULL DEFAULT ''",
    # What kind of shoot the folder is, "school" or "family" (places.store.SHOOT_KINDS).
    "ALTER TABLE folder_notes ADD COLUMN kind TEXT NOT NULL DEFAULT ''",
    # Library-wide choices that belong with the photographs rather than with one
    # machine's config file: today, where home is (places/store.py).
    """CREATE TABLE IF NOT EXISTS library_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL DEFAULT '',
        updated_at TEXT DEFAULT ''
    )""",
    # Shoots one device (one person) finds less important: a whole folder
    # (camera '*'), or one camera's photos in it ('' for those with none
    # recorded). Nothing is hidden; they sink, and stay out of what is picked
    # for that person (on this day, a year in review).
    """CREATE TABLE IF NOT EXISTS demoted_shoots (
        owner TEXT NOT NULL DEFAULT 'local',
        folder TEXT NOT NULL,
        camera TEXT NOT NULL DEFAULT '*',
        created_at TEXT DEFAULT '',
        PRIMARY KEY (owner, folder, camera)
    )""",
    # Perspectives (catalog/perspectives.py): a name and the people it is
    # about, shared by the household; which one a device opens on is its own.
    """CREATE TABLE IF NOT EXISTS perspectives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL DEFAULT '',
        people TEXT NOT NULL DEFAULT '[]',
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )""",
    # One device's (one person's) own choices, by key.
    """CREATE TABLE IF NOT EXISTS owner_settings (
        owner TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL DEFAULT '',
        updated_at TEXT DEFAULT '',
        PRIMARY KEY (owner, key)
    )""",
    # What an agent suggests for a photograph's cull (catalog/proposals.py), kept
    # apart from the rating and flag the person sets until the person accepts
    # it. NULL is "no opinion" on that field. prev_* hold what accepting
    # replaced, so a pass can be undone.
    """CREATE TABLE IF NOT EXISTS cull_proposals (
        filepath TEXT PRIMARY KEY,
        folder TEXT NOT NULL DEFAULT '',
        rating INTEGER,
        flag INTEGER,
        color_label INTEGER,
        reason TEXT NOT NULL DEFAULT '',
        run TEXT NOT NULL DEFAULT '',
        agent TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        prev_rating INTEGER,
        prev_flag INTEGER,
        prev_color_label INTEGER,
        created_at TEXT DEFAULT '',
        decided_at TEXT DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cull_proposals_folder ON cull_proposals(folder, status)",
]


# The measured columns, in insert order, with the value to use when a
# measurement is absent. The writer fills this in, so adding a measurement means
# adding it here and to the table above — nothing else.
_STAT_FIELDS = {
    "stats_version": 0,
    "width": 0,
    "height": 0,
    "mean_lum": 0.0,
    "lum_p05": 0.0,
    "lum_p50": 0.0,
    "lum_p95": 0.0,
    "clip_high": 0.0,
    "clip_low": 0.0,
    "wb_r": 1.0,
    "wb_g": 1.0,
    "wb_b": 1.0,
    "sharpness": 0.0,
    "dhash": "",
}


def _decode_settings(payload: str) -> dict:
    """A stored edit set, or ``{}`` when the row cannot be read.

    A row written by a newer build, or one that was truncated, is treated as "no
    edit" rather than as an error. The grid and the export both ask this
    question thousands of times, and a single bad row must not be able to fail
    either of them.
    """
    if not payload:
        return {}
    try:
        decoded = json.loads(payload)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


# A face found again over the box of one stored before keeps that face's person
# and its "not this person" list (see CatalogDB.replace_faces).
_CARRY_IOU = 0.4


def _box_iou(a: tuple, b: tuple) -> float:
    """Overlap of two (x, y, w, h) boxes, as intersection over union."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_refused(text: str) -> set[int]:
    """The ``not_people`` column: group ids, comma-separated."""
    out: set[int] = set()
    for part in (text or "").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


# A face at least this good counts as a clear one in a group's ``clear`` count:
# the quality a face needs to start a group of its own (faces.service).
CLEAR_FACE_QUALITY = 0.25


class CatalogDB:
    """SQLite-backed photo catalog."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.writable = True
        self.write_error = ""
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # probe_writable below reports why the path cannot be written

        writable, error = probe_writable(db_path)
        if not writable:
            # A read-only volume still serves reads. Refusing to boot would hide
            # the reason; start degraded and let /health and the UI say what is
            # wrong. Writes raise from SQLite and are reported per request.
            self.writable = False
            self.write_error = error
            print(f"[catalog] catalog database is not writable: {error}")
            return
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Run migrations for existing databases
            for sql in _MIGRATIONS:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column/index already exists
            self._basket_per_owner(conn)

    @staticmethod
    def _basket_per_owner(conn) -> None:
        """Give the basket an owner. The old table made ``filepath`` UNIQUE on its
        own, which SQLite cannot relax in place, so it is rebuilt once; what was
        in it becomes this computer's basket."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(export_basket)")}
        if "owner" in columns:
            return
        conn.executescript("""
            ALTER TABLE export_basket RENAME TO export_basket_shared;
            CREATE TABLE export_basket (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT NOT NULL DEFAULT 'local',
                filepath TEXT NOT NULL,
                folder TEXT NOT NULL DEFAULT '',
                filename TEXT NOT NULL DEFAULT '',
                added_at TEXT DEFAULT '',
                adjustment TEXT NOT NULL DEFAULT '',
                UNIQUE (owner, filepath)
            );
            INSERT INTO export_basket (owner, filepath, folder, filename, added_at, adjustment)
                SELECT 'local', filepath, folder, filename, added_at, COALESCE(adjustment, '')
                FROM export_basket_shared;
            DROP TABLE export_basket_shared;
        """)

    @contextmanager
    def _connect(self):
        if self.writable:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
        else:
            # A read-only volume refuses the plain open too, because a WAL-mode
            # database needs a writable -shm beside it. open_readonly falls back
            # to immutable mode so browsing keeps working.
            conn = open_readonly(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            if self.writable:
                conn.commit()
        except Exception:
            if self.writable:
                conn.rollback()
            raise
        finally:
            conn.close()

    # ── Insert / upsert ──────────────────────────────────────────

    def upsert_photo(self, photo: PhotoRecord) -> int:
        """Insert or update a photo record. Returns row id."""
        now = datetime.now().isoformat()
        photo.updated_at = now
        if not photo.imported_at:
            photo.imported_at = now
        if not photo.base_name:
            photo.base_name = Path(photo.filename).stem

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO photos (
                    filepath, filename, folder, extension, base_name, file_size,
                    rating, flag, color_label,
                    camera, lens, focal_length, aperture, shutter_speed,
                    iso, capture_date, dimensions, duration,
                    imported_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?
                )
                ON CONFLICT(filepath) DO UPDATE SET
                    base_name = excluded.base_name,
                    rating = excluded.rating,
                    flag = excluded.flag,
                    color_label = excluded.color_label,
                    camera = excluded.camera,
                    lens = excluded.lens,
                    focal_length = excluded.focal_length,
                    aperture = excluded.aperture,
                    shutter_speed = excluded.shutter_speed,
                    iso = excluded.iso,
                    capture_date = excluded.capture_date,
                    dimensions = excluded.dimensions,
                    duration = excluded.duration,
                    updated_at = excluded.updated_at
                """,
                (
                    photo.filepath, photo.filename, photo.folder,
                    photo.extension, photo.base_name, photo.file_size,
                    photo.rating, photo.flag, photo.color_label,
                    photo.camera, photo.lens, photo.focal_length,
                    photo.aperture, photo.shutter_speed,
                    photo.iso, photo.camera_date or photo.capture_date, photo.dimensions,
                    photo.duration, photo.imported_at, photo.updated_at,
                ),
            )
            return cursor.lastrowid

    def bulk_upsert(self, photos: list[PhotoRecord]):
        """Batch upsert for import operations."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            for photo in photos:
                photo.updated_at = now
                if not photo.imported_at:
                    photo.imported_at = now
                if not photo.base_name:
                    photo.base_name = Path(photo.filename).stem
                conn.execute(
                    """
                    INSERT INTO photos (
                        filepath, filename, folder, extension, base_name, file_size,
                        rating, flag, color_label,
                        camera, lens, focal_length, aperture, shutter_speed,
                        iso, capture_date, dimensions, duration,
                        imported_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(filepath) DO UPDATE SET
                        base_name = excluded.base_name,
                        duration = CASE WHEN excluded.duration > 0
                                        THEN excluded.duration ELSE duration END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        photo.filepath, photo.filename, photo.folder,
                        photo.extension, photo.base_name, photo.file_size,
                        photo.rating, photo.flag, photo.color_label,
                        photo.camera, photo.lens, photo.focal_length,
                        photo.aperture, photo.shutter_speed,
                        photo.iso, photo.camera_date or photo.capture_date, photo.dimensions,
                        photo.duration, photo.imported_at, photo.updated_at,
                    ),
                )

    # ── Rating / flagging ────────────────────────────────────────

    def set_rating(self, filepath: str, rating: int) -> bool:
        rating = max(0, min(5, rating))
        return self._update_field(filepath, "rating", rating)

    def set_flag(self, filepath: str, flag: int) -> bool:
        flag = max(-1, min(1, flag))
        return self._update_field(filepath, "flag", flag)

    def set_color_label(self, filepath: str, color: int) -> bool:
        color = max(0, min(5, color))
        return self._update_field(filepath, "color_label", color)

    def _update_field(self, filepath: str, field: str, value) -> bool:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE photos SET {field} = ?, updated_at = ? WHERE filepath = ?",
                (value, now, filepath),
            )
            return cursor.rowcount > 0

    # ── Tags (event labels / keywords) ─────────────────────────────

    def create_tag(self, name: str, kind: str = "event") -> Optional[int]:
        """Create a tag, or return the existing one with the same name."""
        name = (name or "").strip()
        if not name:
            return None
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tags (name, kind, created_at) VALUES (?, ?, ?)",
                (name, kind, now),
            )
            row = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
            return row["id"] if row else None

    def list_tags(self) -> list[dict]:
        """All tags with the number of photos carrying each."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.name, t.kind, t.created_at,
                          COUNT(pt.filepath) AS photo_count
                   FROM tags t
                   LEFT JOIN photo_tags pt ON pt.tag_id = t.id
                   GROUP BY t.id
                   ORDER BY photo_count DESC, t.name COLLATE NOCASE ASC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_tag(self, name: str) -> bool:
        """Delete a tag and detach it from every photo."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM photo_tags WHERE tag_id = ?", (row["id"],))
            conn.execute("DELETE FROM tags WHERE id = ?", (row["id"],))
            return True

    def rename_tag(self, old_name: str, new_name: str) -> bool:
        """Rename a tag, merging into an existing tag of the same name."""
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        now = datetime.now().isoformat()
        with self._connect() as conn:
            old = conn.execute("SELECT id FROM tags WHERE name = ?", (old_name,)).fetchone()
            if not old:
                return False
            existing = conn.execute(
                "SELECT id FROM tags WHERE name = ? AND id != ?", (new_name, old["id"])
            ).fetchone()
            if existing:
                # Merge: move the photos onto the target tag and drop this one.
                conn.execute(
                    """INSERT OR IGNORE INTO photo_tags (filepath, tag_id, added_at)
                       SELECT filepath, ?, ? FROM photo_tags WHERE tag_id = ?""",
                    (existing["id"], now, old["id"]),
                )
                conn.execute("DELETE FROM photo_tags WHERE tag_id = ?", (old["id"],))
                conn.execute("DELETE FROM tags WHERE id = ?", (old["id"],))
            else:
                conn.execute(
                    "UPDATE tags SET name = ? WHERE id = ?", (new_name, old["id"])
                )
            return True

    def get_photo_tags(self, filepath: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.name FROM photo_tags pt
                   JOIN tags t ON t.id = pt.tag_id
                   WHERE pt.filepath = ?
                   ORDER BY t.name COLLATE NOCASE ASC""",
                (filepath,),
            ).fetchall()
            return [r["name"] for r in rows]

    def get_tags_for_filepaths(self, filepaths: list[str]) -> dict[str, list[str]]:
        """Tags for many photos in one query — used to decorate a grid listing."""
        if not filepaths:
            return {}
        result: dict[str, list[str]] = {}
        with self._connect() as conn:
            # Chunked so a huge folder never exceeds SQLite's variable limit.
            for batch in _chunks(filepaths):
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"""SELECT pt.filepath, t.name FROM photo_tags pt
                        JOIN tags t ON t.id = pt.tag_id
                        WHERE pt.filepath IN ({placeholders})
                        ORDER BY t.name COLLATE NOCASE ASC""",
                    batch,
                ).fetchall()
                for r in rows:
                    result.setdefault(r["filepath"], []).append(r["name"])
        return result

    def search_snapshot(self, owner: str = "local") -> tuple[list[PhotoRecord], dict[str, list[str]], set[str], set[str], set[str], set[str]]:
        """Everything a library search reads, in one connection: every photo, the
        tags by photo, which photos are edited, which are in the basket, which
        are developed in black and white and which have noise reduction.

        Whole-library on purpose. A search compares numbers that are stored as
        EXIF text ('78.0 mm', '1/200'), which SQL cannot do, and thirty thousand
        rows read and tested in Python is well under a second.
        """
        with self._connect() as conn:
            records = [self._row_to_record(r) for r in conn.execute("SELECT * FROM photos")]
            tags: dict[str, list[str]] = {}
            for r in conn.execute(
                "SELECT pt.filepath, t.name FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            ):
                tags.setdefault(r["filepath"], []).append(r["name"])
            edited, mono, denoised = set(), set(), set()
            for r in conn.execute("SELECT filepath, settings FROM photo_edits"):
                edited.add(r["filepath"])
                try:
                    settings = json.loads(r["settings"] or "{}")
                    if settings.get("mono"):
                        mono.add(r["filepath"])
                    if any(float(settings.get(k) or 0) > 0 for k in ("nr_luma", "nr_color")):
                        denoised.add(r["filepath"])
                except (TypeError, ValueError, AttributeError):
                    pass
            basket = {r["filepath"] for r in conn.execute(
                "SELECT filepath FROM export_basket WHERE owner = ?", (owner,))}
        return records, tags, edited, basket, mono, denoised

    def set_photo_tags(self, filepath: str, names: list[str]) -> bool:
        """Replace a photo's tag set. Unknown tag names are created."""
        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM photos WHERE filepath = ?", (filepath,)
            ).fetchone() is None:
                return False
            conn.execute("DELETE FROM photo_tags WHERE filepath = ?", (filepath,))
            self._attach_tags(conn, [filepath], names)
            return True

    def add_tag_to_photos(self, filepaths: list[str], name: str) -> int:
        """Attach one tag to many photos. Returns how many newly got it."""
        name = (name or "").strip()
        if not name or not filepaths:
            return 0
        added = 0
        with self._connect() as conn:
            for batch in _chunks(filepaths):
                placeholders = ",".join("?" for _ in batch)
                have = {
                    r["filepath"]
                    for r in conn.execute(
                        f"""SELECT pt.filepath FROM photo_tags pt
                            JOIN tags t ON t.id = pt.tag_id
                            WHERE t.name = ? AND pt.filepath IN ({placeholders})""",
                        (name, *batch),
                    ).fetchall()
                }
                missing = [fp for fp in batch if fp not in have]
                self._attach_tags(conn, missing, [name])
                added += len(missing)
            return added

    def remove_tag_from_photos(self, filepaths: list[str], name: str) -> int:
        """Detach one tag from many photos. Returns how many changed."""
        if not filepaths:
            return 0
        removed = 0
        with self._connect() as conn:
            for batch in _chunks(filepaths):
                placeholders = ",".join("?" for _ in batch)
                cursor = conn.execute(
                    f"""DELETE FROM photo_tags
                        WHERE tag_id = (SELECT id FROM tags WHERE name = ?)
                        AND filepath IN ({placeholders})""",
                    (name, *batch),
                )
                removed += cursor.rowcount
            return removed

    @staticmethod
    def _attach_tags(conn, filepaths: list[str], names: list[str]) -> None:
        """Insert tag rows, creating any tags that do not exist yet."""
        now = datetime.now().isoformat()
        clean = []
        for n in names:
            n = (n or "").strip()
            if n and n not in clean:
                clean.append(n)
        if not clean:
            return
        for name in clean:
            conn.execute(
                "INSERT OR IGNORE INTO tags (name, kind, created_at) VALUES (?, 'event', ?)",
                (name, now),
            )
            tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()["id"]
            for fp in filepaths:
                conn.execute(
                    "INSERT OR IGNORE INTO photo_tags (filepath, tag_id, added_at) VALUES (?, ?, ?)",
                    (fp, tag_id, now),
                )


    # ── Queries ──────────────────────────────────────────────────

    def get_photo(self, filepath: str) -> Optional[PhotoRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE filepath = ?", (filepath,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def get_photos_in_folder(
        self,
        folder: str,
        min_rating: int = 0,
        flag_filter: Optional[int] = None,
        color_filter: Optional[int] = None,
        ext_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        unreviewed_only: bool = False,
        pair_mode: str = "all",
        camera_filter: Optional[str] = None,
        hide_cameras: Optional[list] = None,
        base_names: Optional[list] = None,
        sort_by: str = "filename",
        sort_desc: bool = False,
    ) -> list[PhotoRecord]:
        """
        Get photos in a folder with optional filters.

        pair_mode:
          "all"      — show every file
          "raw"      — only RAW files (CR3, CR2, etc.)
          "jpg"      — only JPEGs
          "paired"   — one entry per base_name, prefer JPG for display
          "raw_only" — RAW files that have no matching JPG
          "jpg_only" — JPGs that have no matching RAW

        tag_filter matches a photo that carries the tag, or the other half of
        its RAW+JPG pair — tagging one file of a pair must not hide the pair.

        unreviewed_only keeps only frames with no cull state at all: no rating,
        no flag, no colour label. Event tags are deliberately not part of that —
        labelling a photo as part of an event is not a decision about the photo.
        Like tag_filter this is pair-aware, so a frame whose twin has already
        been decided does not come back asking to be decided again.

        camera_filter keeps one camera's frames: a day shot on two cameras is
        often imported into one folder. Pair-aware as well, since a camera's
        JPEG may carry no EXIF while its raw twin does. An empty string means
        the frames no camera is known for.

        hide_cameras leaves those cameras' frames out, pair-aware the same way:
        the cameras demoted in this folder, which the grid does not show.

        base_names keeps only those pairs: the ones a search typed in the folder
        found, so an action on "these" acts on what the grid shows.
        """
        allowed_sorts = {
            "filename", "rating", "flag", "capture_date",
            "focal_length", "iso", "file_size", "base_name",
        }
        if sort_by not in allowed_sorts:
            sort_by = "filename"

        direction = "DESC" if sort_desc else "ASC"
        raw_exts = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef"}
        jpg_exts = {".jpg", ".jpeg"}

        conditions = ["folder = ?"]
        params: list = [folder]

        if min_rating > 0:
            conditions.append("rating >= ?")
            params.append(min_rating)
        if flag_filter is not None:
            conditions.append("flag = ?")
            params.append(flag_filter)
        if color_filter is not None:
            conditions.append("color_label = ?")
            params.append(color_filter)
        if ext_filter:
            conditions.append("extension = ?")
            params.append(ext_filter.lower())
        if tag_filter:
            conditions.append(
                """(
                    filepath IN (
                        SELECT pt.filepath FROM photo_tags pt
                        JOIN tags t ON t.id = pt.tag_id WHERE t.name = ?
                    )
                    OR (folder, base_name) IN (
                        SELECT p2.folder, p2.base_name FROM photo_tags pt2
                        JOIN tags t2 ON t2.id = pt2.tag_id
                        JOIN photos p2 ON p2.filepath = pt2.filepath
                        WHERE t2.name = ?
                    )
                )"""
            )
            params.extend([tag_filter, tag_filter])
        if camera_filter is not None:
            if camera_filter:
                conditions.append(
                    "(camera = ? OR (folder, base_name) IN "
                    "(SELECT p3.folder, p3.base_name FROM photos p3 WHERE p3.camera = ?))")
                params.extend([camera_filter, camera_filter])
            else:
                conditions.append(
                    "(camera = '' AND (folder, base_name) NOT IN "
                    "(SELECT p3.folder, p3.base_name FROM photos p3 WHERE p3.camera != ''))")
        for camera in hide_cameras or []:
            if camera:
                conditions.append(
                    "(folder, base_name) NOT IN "
                    "(SELECT p4.folder, p4.base_name FROM photos p4 WHERE p4.camera = ?)")
                params.append(camera)
            else:
                conditions.append(
                    "(folder, base_name) IN "
                    "(SELECT p4.folder, p4.base_name FROM photos p4 WHERE p4.camera != '')")
        if base_names is not None:
            conditions.append("base_name IN (SELECT value FROM json_each(?))")
            params.append(json.dumps([str(b) for b in base_names]))
        if unreviewed_only:
            # Decided means any cull state on this frame or on its pair twin.
            conditions.append(
                """(folder, base_name) NOT IN (
                    SELECT folder, base_name FROM photos
                    WHERE folder = ?
                      AND (rating > 0 OR flag <> 0 OR color_label <> 0)
                )"""
            )
            params.append(folder)

        # Simple extension-only filters
        if pair_mode == "raw":
            placeholders = ",".join("?" for _ in raw_exts)
            conditions.append(f"extension IN ({placeholders})")
            params.extend(raw_exts)
        elif pair_mode == "jpg":
            placeholders = ",".join("?" for _ in jpg_exts)
            conditions.append(f"extension IN ({placeholders})")
            params.extend(jpg_exts)

        where = " AND ".join(conditions)

        order = LOCAL_DATE if sort_by == "capture_date" else sort_by
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM photos WHERE {where} ORDER BY {order} {direction}",
                params,
            ).fetchall()
            records = [self._row_to_record(r) for r in rows]

        # Post-process for pair modes that need cross-reference
        if pair_mode == "paired":
            return self._dedupe_pairs(records, prefer_jpg=True)
        elif pair_mode == "raw_only":
            return self._filter_unpaired(records, keep_raw=True)
        elif pair_mode == "jpg_only":
            return self._filter_unpaired(records, keep_raw=False)

        return records

    def _dedupe_pairs(
        self, records: list[PhotoRecord], prefer_jpg: bool = True
    ) -> list[PhotoRecord]:
        """
        Deduplicate CR3+JPG pairs: one entry per base_name.
        If prefer_jpg, show the JPG (faster thumbnails). Otherwise show RAW.
        """
        by_base: dict[tuple, list[PhotoRecord]] = {}
        for r in records:
            by_base.setdefault(_pair_key(r), []).append(r)

        result = []
        jpg_exts = {".jpg", ".jpeg"}
        for base_name, group in by_base.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Pick preferred format
                jpgs = [r for r in group if r.extension in jpg_exts]
                raws = [r for r in group if r.extension not in jpg_exts]
                if prefer_jpg and jpgs:
                    chosen = jpgs[0]
                    # Carry over rating/flag from RAW if it's higher
                    for raw in raws:
                        if raw.rating > chosen.rating:
                            chosen.rating = raw.rating
                        if raw.flag != 0 and chosen.flag == 0:
                            chosen.flag = raw.flag
                    result.append(chosen)
                elif raws:
                    result.append(raws[0])
                else:
                    result.append(group[0])
        return result

    def _filter_unpaired(
        self, records: list[PhotoRecord], keep_raw: bool
    ) -> list[PhotoRecord]:
        """Return only files that don't have a matching pair."""
        by_base: dict[tuple, list[PhotoRecord]] = {}
        for r in records:
            by_base.setdefault(_pair_key(r), []).append(r)

        jpg_exts = {".jpg", ".jpeg"}
        result = []
        for base_name, group in by_base.items():
            has_jpg = any(r.extension in jpg_exts for r in group)
            has_raw = any(r.extension not in jpg_exts for r in group)

            if has_jpg and has_raw:
                continue  # it's paired, skip

            for r in group:
                is_raw = r.extension not in jpg_exts
                if keep_raw and is_raw:
                    result.append(r)
                elif not keep_raw and not is_raw:
                    result.append(r)

        return result

    def get_folder_stats(self, folder: str) -> dict:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ?", (folder,)
            ).fetchone()[0]

            picks = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ? AND flag = 1",
                (folder,),
            ).fetchone()[0]

            rejects = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ? AND flag = -1",
                (folder,),
            ).fetchone()[0]

            rated = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ? AND rating > 0",
                (folder,),
            ).fetchone()[0]

            # Pair stats
            raw_count = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ? AND extension IN ('.cr3','.cr2','.arw','.nef','.dng','.raf','.orf','.rw2','.pef')",
                (folder,),
            ).fetchone()[0]

            jpg_count = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ? AND extension IN ('.jpg','.jpeg')",
                (folder,),
            ).fetchone()[0]

            unique_bases = conn.execute(
                "SELECT COUNT(DISTINCT base_name) FROM photos WHERE folder = ?",
                (folder,),
            ).fetchone()[0]

            # "Decided" is the review-progress count: any cull state at all. It
            # is deliberately not the same as flagged or rated — a reject is a
            # decision, and so is a colour label. Counted per image rather than
            # per file, like unique_images above, because that is what the grid
            # shows and deciding one half of a RAW+JPG pair decides the pair.
            decided = conn.execute(
                """SELECT COUNT(DISTINCT base_name) FROM photos WHERE folder = ?
                   AND (rating > 0 OR flag <> 0 OR color_label <> 0)""",
                (folder,),
            ).fetchone()[0]

            return {
                "total": total,
                "picks": picks,
                "rejects": rejects,
                "unflagged": total - picks - rejects,
                "rated": rated,
                "unrated": total - rated,
                "decided": decided,
                "undecided": unique_bases - decided,
                "raw_count": raw_count,
                "jpg_count": jpg_count,
                "unique_images": unique_bases,
                "pairs": total - unique_bases,
            }

    def get_all_folders(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT folder, COUNT(*) as count,
                       COUNT(DISTINCT base_name) as unique_images,
                       SUM(CASE WHEN flag = 1 THEN 1 ELSE 0 END) as picks,
                       SUM(CASE WHEN flag = -1 THEN 1 ELSE 0 END) as rejects,
                       COUNT(DISTINCT CASE WHEN rating > 0 OR flag <> 0 OR color_label <> 0
                                           THEN base_name END) as decided
                FROM photos
                GROUP BY folder
                ORDER BY folder DESC
                """
            ).fetchall()
            return [
                {
                    "folder": row["folder"],
                    "count": row["count"],
                    "unique_images": row["unique_images"],
                    "picks": row["picks"],
                    "rejects": row["rejects"],
                    "decided": row["decided"],
                }
                for row in rows
            ]

    # ── Review position ─────────────────────────────────────────
    #
    # Where the user last stopped reviewing a folder. This is convenience state,
    # not cull data: it is never written to XMP, and losing it costs nothing
    # worse than starting a folder from its first frame.

    def get_folder_state(self, folder: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_filepath, updated_at FROM folder_state WHERE folder = ?",
                (folder,),
            ).fetchone()
        return {
            "folder": folder,
            "last_filepath": row["last_filepath"] if row else None,
            "updated_at": row["updated_at"] if row else 0,
        }

    def set_folder_state(self, folder: str, last_filepath: Optional[str]) -> dict:
        """Remember where the user stopped in a folder; an empty path clears it."""
        if not last_filepath:
            with self._connect() as conn:
                conn.execute("DELETE FROM folder_state WHERE folder = ?", (folder,))
            return {"folder": folder, "last_filepath": None, "updated_at": 0}
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO folder_state (folder, last_filepath, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(folder) DO UPDATE SET
                     last_filepath = excluded.last_filepath,
                     updated_at = excluded.updated_at""",
                (folder, last_filepath, int(datetime.now().timestamp())),
            )
        return self.get_folder_state(folder)

    # ── Thumbnail job management ──────────────────────────────

    def create_thumbnail_job(self, folder: str, priority: int = 100) -> ThumbnailJobRecord:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO thumbnail_jobs (folder, status, priority, created_at, updated_at)
                   VALUES (?, 'queued', ?, ?, ?)""",
                (folder, priority, now, now),
            )
            return self.get_thumbnail_job(cursor.lastrowid, _conn=conn)

    def list_thumbnail_jobs(self) -> list[ThumbnailJobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM thumbnail_jobs ORDER BY priority ASC, id ASC"
            ).fetchall()
            return [self._row_to_job_record(r) for r in rows]

    def get_thumbnail_job(self, job_id: int, *, _conn=None) -> Optional[ThumbnailJobRecord]:
        def _query(conn):
            row = conn.execute(
                "SELECT * FROM thumbnail_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return self._row_to_job_record(row) if row else None

        if _conn is not None:
            return _query(_conn)
        with self._connect() as conn:
            return _query(conn)

    def set_thumbnail_job_status(
        self, job_id: int, status: str, error_message: str = ""
    ) -> bool:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            updates = "status = ?, error_message = ?, updated_at = ?"
            params: list = [status, error_message, now]
            if status == "running":
                updates += ", started_at = ?"
                params.append(now)
            params.append(job_id)
            cursor = conn.execute(
                f"UPDATE thumbnail_jobs SET {updates} WHERE id = ?", params
            )
            return cursor.rowcount > 0

    def reset_running_thumbnail_jobs(self) -> int:
        if not self.writable:
            # Nothing to reset — the catalog cannot be written and the thumbnail
            # queue is paused for the same reason. Startup must not fail on this.
            return 0
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE thumbnail_jobs SET status = 'queued', updated_at = ? WHERE status = 'running'",
                (now,),
            )
            return cursor.rowcount

    def claim_next_thumbnail_job(self) -> Optional[ThumbnailJobRecord]:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id FROM thumbnail_jobs
                   WHERE status = 'queued'
                   ORDER BY priority ASC, id ASC
                   LIMIT 1"""
            ).fetchone()
            if not row:
                return None
            job_id = row["id"]
            conn.execute(
                "UPDATE thumbnail_jobs SET status = 'running', started_at = ?, updated_at = ? WHERE id = ?",
                (now, now, job_id),
            )
            return self.get_thumbnail_job(job_id, _conn=conn)

    def reorder_thumbnail_job(self, job_id: int, direction: str) -> bool:
        with self._connect() as conn:
            current = conn.execute(
                "SELECT id, priority FROM thumbnail_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not current:
                return False

            cur_priority = current["priority"]
            if direction == "up":
                neighbor = conn.execute(
                    """SELECT id, priority FROM thumbnail_jobs
                       WHERE priority < ? OR (priority = ? AND id < ?)
                       ORDER BY priority DESC, id DESC LIMIT 1""",
                    (cur_priority, cur_priority, job_id),
                ).fetchone()
            else:
                neighbor = conn.execute(
                    """SELECT id, priority FROM thumbnail_jobs
                       WHERE priority > ? OR (priority = ? AND id > ?)
                       ORDER BY priority ASC, id ASC LIMIT 1""",
                    (cur_priority, cur_priority, job_id),
                ).fetchone()

            if not neighbor:
                return False

            conn.execute(
                "UPDATE thumbnail_jobs SET priority = ? WHERE id = ?",
                (neighbor["priority"], job_id),
            )
            conn.execute(
                "UPDATE thumbnail_jobs SET priority = ? WHERE id = ?",
                (cur_priority, neighbor["id"]),
            )
            return True

    def get_active_thumbnail_job_for_folder(
        self, folder: str
    ) -> Optional[ThumbnailJobRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM thumbnail_jobs
                   WHERE folder = ? AND status IN ('queued', 'running', 'paused')
                   ORDER BY id DESC LIMIT 1""",
                (folder,),
            ).fetchone()
            return self._row_to_job_record(row) if row else None

    def update_thumbnail_job_progress(
        self,
        job_id: int,
        *,
        total_items: Optional[int] = None,
        completed_delta: int = 0,
        failed_delta: int = 0,
        last_file: str = "",
    ) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            parts = ["updated_at = ?"]
            params: list = [now]
            if total_items is not None:
                parts.append("total_items = ?")
                params.append(total_items)
            if completed_delta:
                parts.append("completed_items = completed_items + ?")
                params.append(completed_delta)
            if failed_delta:
                parts.append("failed_items = failed_items + ?")
                params.append(failed_delta)
            if last_file:
                parts.append("last_file = ?")
                params.append(last_file)
            params.append(job_id)
            conn.execute(
                f"UPDATE thumbnail_jobs SET {', '.join(parts)} WHERE id = ?",
                params,
            )

    # ── Helpers ──────────────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> PhotoRecord:
        d = dict(row)
        # Handle DBs before base_name migration
        if "base_name" not in d:
            d["base_name"] = Path(d.get("filename", "")).stem
        record = PhotoRecord(**d)
        # Everything that reads a date reads the one where the photo was taken.
        if record.local_date:
            record.camera_date, record.capture_date = record.capture_date, record.local_date
        return record

    def _row_to_job_record(self, row: sqlite3.Row) -> Optional[ThumbnailJobRecord]:
        if not row:
            return None
        return ThumbnailJobRecord(**dict(row))

    # ── Export basket ────────────────────────────────────────────

    # Every basket call names whose basket it is; "local" is this computer's.

    def add_to_basket(self, filepath: str, folder: str, filename: str, owner: str = "local") -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO export_basket (owner, filepath, folder, filename, added_at) VALUES (?, ?, ?, ?, ?)",
                (owner, filepath, folder, filename, now),
            )

    def remove_from_basket(self, filepath: str, owner: str = "local") -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM export_basket WHERE owner = ? AND filepath = ?", (owner, filepath))

    def clear_basket(self, owner: str = "local") -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM export_basket WHERE owner = ?", (owner,))

    # ── One photograph's own edit ─────────────────────────────────
    #
    # Keyed on the photograph, not on a basket row: the edit is a statement
    # about the picture, so clearing the basket must not discard it and adding a
    # frame to a different export must find it already there. The sidecar holds
    # the same numbers and is the copy that survives losing this database; these
    # methods are the index over it.

    def set_photo_edit(
        self,
        filepath: str,
        settings: Optional[dict],
        source_mtime: int = 0,
        source_size: int = 0,
    ) -> None:
        """Store an edit, or clear it when ``settings`` is empty.

        An empty set deletes the row rather than filing a row that means
        nothing: "no edit" and "an edit that happens to be neutral" have to be
        the same state, or a frame the user reset would still be counted as
        edited and still be rewritten on every save.
        """
        payload = json.dumps(settings, sort_keys=True) if settings else ""
        with self._connect() as conn:
            if not payload:
                conn.execute("DELETE FROM photo_edits WHERE filepath = ?", (filepath,))
                return
            conn.execute(
                """
                INSERT INTO photo_edits (filepath, settings, source_mtime, source_size, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(filepath) DO UPDATE SET
                    settings = excluded.settings,
                    source_mtime = excluded.source_mtime,
                    source_size = excluded.source_size,
                    updated_at = excluded.updated_at
                """,
                (filepath, payload, int(source_mtime), int(source_size),
                 datetime.now().isoformat()),
            )

    def get_photo_edit(self, filepath: str) -> dict:
        """The edit stored for one photograph, or ``{}`` when it has none."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT settings FROM photo_edits WHERE filepath = ?", (filepath,)
            ).fetchone()
        return _decode_settings(row["settings"]) if row else {}

    def list_photo_edits(self, folder: Optional[str] = None) -> dict[str, dict]:
        """Every stored edit, or the ones under one folder, by filepath.

        One query for a whole grid: asking per frame would be two thousand
        round trips to decide which thumbnails get a marker.
        """
        with self._connect() as conn:
            if folder:
                rows = conn.execute(
                    "SELECT filepath, settings FROM photo_edits WHERE filepath LIKE ?",
                    (f"{folder}/%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT filepath, settings FROM photo_edits"
                ).fetchall()
        edits = {}
        for row in rows:
            decoded = _decode_settings(row["settings"])
            if decoded:
                edits[row["filepath"]] = decoded
        return edits

    def clear_photo_edit(self, filepath: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM photo_edits WHERE filepath = ?", (filepath,)
            )
            return cursor.rowcount > 0

    def photo_edit_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM photo_edits").fetchone()[0]

    # ── Virtual copies ────────────────────────────────────────────

    def list_versions(self, filepath: str) -> list[dict]:
        """The copies of one photograph, oldest first. Version 0 is not listed."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, name, settings, export FROM photo_versions "
                "WHERE filepath = ? ORDER BY version", (filepath,)
            ).fetchall()
        return [
            {"version": r["version"], "name": r["name"],
             "settings": _decode_settings(r["settings"]), "export": bool(r["export"])}
            for r in rows
        ]

    def get_version(self, filepath: str, version: int) -> Optional[dict]:
        for found in self.list_versions(filepath):
            if found["version"] == version:
                return found
        return None

    def create_version(self, filepath: str, settings: Optional[dict], name: str = "") -> int:
        """Add a copy and return its number (1 for the first, never reused)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM photo_versions WHERE filepath = ?",
                (filepath,),
            ).fetchone()
            version = int(row[0]) + 1
            conn.execute(
                "INSERT INTO photo_versions (filepath, version, name, settings, export, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (filepath, version, name, json.dumps(settings, sort_keys=True) if settings else "",
                 datetime.now().isoformat()),
            )
        return version

    def restore_version(self, filepath: str, version: int, name: str,
                        settings: Optional[dict], export: bool) -> None:
        """Put a copy back under its own number (from the on-disk record)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO photo_versions "
                "(filepath, version, name, settings, export, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (filepath, version, name, json.dumps(settings, sort_keys=True) if settings else "",
                 int(export), datetime.now().isoformat()),
            )

    def update_version(self, filepath: str, version: int, settings: Optional[dict] = None,
                       name: Optional[str] = None, export: Optional[bool] = None,
                       replace_settings: bool = False) -> bool:
        sets, args = [], []
        if replace_settings:
            sets.append("settings = ?")
            args.append(json.dumps(settings, sort_keys=True) if settings else "")
        if name is not None:
            sets.append("name = ?")
            args.append(name)
        if export is not None:
            sets.append("export = ?")
            args.append(int(export))
        if not sets:
            return False
        sets.append("updated_at = ?")
        args.append(datetime.now().isoformat())
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE photo_versions SET {', '.join(sets)} WHERE filepath = ? AND version = ?",
                (*args, filepath, version),
            )
            return cursor.rowcount > 0

    def delete_version(self, filepath: str, version: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM photo_versions WHERE filepath = ? AND version = ?",
                (filepath, version),
            )
            return cursor.rowcount > 0

    def version_counts(self, folder: Optional[str] = None) -> dict[str, int]:
        """How many copies each photograph has, for the ones that have any."""
        with self._connect() as conn:
            if folder:
                rows = conn.execute(
                    "SELECT filepath, COUNT(*) AS n FROM photo_versions "
                    "WHERE filepath LIKE ? GROUP BY filepath", (f"{folder}/%",)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT filepath, COUNT(*) AS n FROM photo_versions GROUP BY filepath"
                ).fetchall()
        return {r["filepath"]: r["n"] for r in rows}

    def list_basket(self, owner: str = "local") -> list[ExportBasketRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM export_basket WHERE owner = ? ORDER BY added_at ASC", (owner,)
            ).fetchall()
            return [ExportBasketRecord(**dict(r)) for r in rows]

    def basket_count(self, owner: str = "local") -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM export_basket WHERE owner = ?", (owner,)
            ).fetchone()[0]

    def add_basket_from_photos(self, records, owner: str = "local") -> int:
        """
        Add a list of PhotoRecord (or anything with filepath/folder/filename)
        to the basket in one transaction. Returns how many were newly added.
        """
        now = datetime.now().isoformat()
        added = 0
        with self._connect() as conn:
            for r in records:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO export_basket (owner, filepath, folder, filename, added_at) VALUES (?, ?, ?, ?, ?)",
                    (owner, r.filepath, r.folder, r.filename, now),
                )
                added += cursor.rowcount
        return added

    # ── Photo measurements ───────────────────────────────────────
    #
    # Everything here is derived from the file and can be recomputed. A row that
    # is missing, or whose mtime/size no longer match, is not an error — it just
    # means the next pass measures that photo again.

    def upsert_photo_stats(
        self,
        filepath: str,
        stats: dict,
        *,
        source_mtime: int = 0,
        source_size: int = 0,
        source_kind: str = "",
    ) -> None:
        """Store the measurements for one photo, replacing any previous row."""
        values = {
            "computed_at": datetime.now().isoformat(),
            "source_mtime": int(source_mtime),
            "source_size": int(source_size),
            "source_kind": source_kind,
        }
        for key, default in _STAT_FIELDS.items():
            value = stats.get(key, default)
            values[key] = default if value is None else value

        columns = ["filepath"] + list(values)
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO photo_stats ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                [filepath] + list(values.values()),
            )

    def get_photo_stats(self, filepath: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM photo_stats WHERE filepath = ?", (filepath,)
            ).fetchone()
            return dict(row) if row else None

    def list_photo_stats_for_folder(self, folder: str) -> dict[str, dict]:
        """Measurements for the indexed photos of one folder, keyed by filepath.

        Joined through ``photos`` so a row whose photo has gone from the catalog
        stops being served, and a folder's statistics are exactly the set of
        photos its viewer can see.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.* FROM photo_stats s
                   JOIN photos p ON p.filepath = s.filepath
                   WHERE p.folder = ?""",
                (folder,),
            ).fetchall()
            return {r["filepath"]: dict(r) for r in rows}

    def photo_stats_status(self, folder: str) -> dict:
        """How much of a folder has been measured. Drives the "analyze" hint."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE folder = ?", (folder,)
            ).fetchone()[0]
            measured = conn.execute(
                """SELECT COUNT(*) FROM photo_stats s
                   JOIN photos p ON p.filepath = s.filepath
                   WHERE p.folder = ?""",
                (folder,),
            ).fetchone()[0]
        return {
            "folder": folder,
            "total": int(total),
            "measured": int(measured),
            "missing": int(max(total - measured, 0)),
        }

    def photo_stats_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM photo_stats").fetchone()[0])

    def next_unmeasured_folder(self, after: str = "") -> Optional[str]:
        """The next folder to measure, walking the library newest event first.

        Walks away from ``after`` rather than always returning the worst folder,
        so a background backfill sweeps the library once and a folder it cannot
        measure does not block the ones behind it.

        The walk goes backwards through names on purpose. Folder keys are
        ``YYYY/YYYY-MM-DD``, so descending name order is descending event date:
        the shoot someone is about to cull is the one they indexed this week.
        Walking forwards instead spends the first hour of idle time on 2016 and
        produces hints for nothing anyone is looking at.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT p.folder FROM photos p
                   LEFT JOIN photo_stats s ON s.filepath = p.filepath
                   WHERE s.filepath IS NULL AND (? = '' OR p.folder < ?)
                   GROUP BY p.folder ORDER BY p.folder DESC LIMIT 1""",
                (after, after),
            ).fetchone()
            return row[0] if row else None

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

    # ── Saved edit sets ──────────────────────────────────────────
    # The payload is stored verbatim. What the eight numbers mean is the edit
    # model's business, and it is the route that clamps them into range before
    # they arrive, so this layer never has to know the shape.

    def save_develop_preset(self, name: str, settings: dict, sections: Optional[list] = None) -> int:
        """Insert or update a saved edit set, with the panel sections it carries.
        Returns the row id."""
        import json as _json
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO develop_presets (name, settings, sections, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET settings = excluded.settings,
                       sections = excluded.sections, updated_at = excluded.updated_at""",
                (name, _json.dumps(settings, sort_keys=True), _json.dumps(sections) if sections else "", now, now),
            )
            return cursor.lastrowid

    def list_develop_presets(self) -> list[dict]:
        import json as _json
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, settings, sections FROM develop_presets ORDER BY name ASC"
            ).fetchall()
            out = []
            for row in rows:
                try:
                    settings = _json.loads(row["settings"])
                    sections = _json.loads(row["sections"]) if row["sections"] else None
                except ValueError:
                    # A row that cannot be read is a row this build cannot use;
                    # it should not take the whole preset list down with it.
                    continue
                if isinstance(settings, dict):
                    preset = {"name": row["name"], "settings": settings}
                    if isinstance(sections, list):
                        preset["sections"] = [s for s in sections if isinstance(s, str)]
                    out.append(preset)
            return out

    def delete_develop_preset(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM develop_presets WHERE name = ?", (name,))
            return cursor.rowcount > 0

    # ── Faces and people ─────────────────────────────────────────
    # Embeddings are stored as raw float32 bytes rather than a JSON list: 128
    # numbers per face over a whole library is a lot of text to parse, and the
    # only thing that ever reads them is numpy.

    def named_clusters(self) -> list[dict]:
        """Every named group, by name, with how many faces it has."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.id, c.name, COUNT(f.id) AS faces
                   FROM person_clusters c LEFT JOIN photo_faces f ON f.cluster_id = c.id
                   WHERE c.name != '' GROUP BY c.id ORDER BY c.name COLLATE NOCASE""").fetchall()
            return [{"id": int(r["id"]), "name": r["name"], "faces": int(r["faces"])} for r in rows]

    def cluster_names(self, cluster_ids) -> dict[int, str]:
        """``{group: name}`` for these groups ("" for an unnamed one)."""
        ids = sorted({int(i) for i in cluster_ids if i})
        if not ids:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, name FROM person_clusters WHERE id IN ({','.join('?' * len(ids))})", ids)
            return {int(r["id"]): r["name"] or "" for r in rows}

    def faces_are_scanned(self, filepath: str) -> bool:
        """Whether any scan has looked at this file, by any model."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM face_scans WHERE filepath = ?", (filepath,)).fetchone() is not None

    def faces_are_current(self, filepath: str, source_mtime: int, source_size: int,
                          model: Optional[str] = None) -> bool:
        """Whether this frame has already been looked at, unchanged — and, when
        ``model`` is given, described by that model. Rows written before the
        model was recorded were SFace's."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source_mtime, source_size, model FROM face_scans WHERE filepath = ?",
                (filepath,),
            ).fetchone()
        if not row:
            return False
        if model is not None and (row["model"] or "sface") != model:
            return False
        return int(row["source_mtime"]) == int(source_mtime) and int(row["source_size"]) == int(source_size)

    def replace_faces(
        self, filepath: str, faces: list, source_mtime: int, source_size: int,
        model: str = "",
    ) -> int:
        """Store the faces of one photo, replacing whatever was there.

        The rows are replaced rather than matched up, because a re-run of the
        detector has no correspondence with the last one's left to right order.
        An empty list still writes the ``face_scans`` row: "this frame has no
        faces" is a result, and without it every scan would look at it again.

        What a person decided about a face is carried across by where it is in
        the frame: a new face over the box of an old one (overlapping by more than
        ``_CARRY_IOU``) keeps the old one's *named* person and its "not this
        person" list. That is what lets a library be scanned again with a better
        model without losing the names already given out.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            old = conn.execute(
                """SELECT f.x, f.y, f.w, f.h, f.cluster_id, f.not_people,
                          COALESCE(c.name, '') AS name
                   FROM photo_faces f LEFT JOIN person_clusters c ON c.id = f.cluster_id
                   WHERE f.filepath = ? AND f.drawn = 0""",
                (filepath,),
            ).fetchall()
            # Faces drawn by hand stay: a scan found nothing there the first time.
            conn.execute("DELETE FROM photo_faces WHERE filepath = ? AND drawn = 0", (filepath,))
            taken: set[int] = set()
            for index, face in enumerate(faces):
                cluster_id, refused = 0, ""
                best, best_iou = None, _CARRY_IOU
                for k, row in enumerate(old):
                    if k in taken:
                        continue
                    iou = _box_iou(
                        (float(face.x), float(face.y), float(face.w), float(face.h)),
                        (row["x"], row["y"], row["w"], row["h"]),
                    )
                    if iou >= best_iou:
                        best, best_iou = k, iou
                if best is not None:
                    taken.add(best)
                    if old[best]["name"]:
                        cluster_id = int(old[best]["cluster_id"])
                    refused = old[best]["not_people"] or ""
                conn.execute(
                    """INSERT INTO photo_faces
                       (filepath, face_index, x, y, w, h, score, px, embedding,
                        cluster_id, created_at, model, quality, not_people)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        filepath,
                        index,
                        float(face.x),
                        float(face.y),
                        float(face.w),
                        float(face.h),
                        float(face.score),
                        float(face.px),
                        array("f", (float(v) for v in face.embedding)).tobytes(),
                        cluster_id,
                        now,
                        model,
                        float(getattr(face, "quality", 1.0)),
                        refused,
                    ),
                )
            conn.execute(
                """INSERT OR REPLACE INTO face_scans
                   (filepath, source_mtime, source_size, faces, scanned_at, model)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (filepath, int(source_mtime), int(source_size), len(faces), now, model),
            )
        return len(faces)

    def list_faces(self, filepath: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM photo_faces WHERE filepath = ? ORDER BY face_index ASC",
                (filepath,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_faces_for_folder(self, folder: str) -> list[dict]:
        """Every stored face in one folder, joined through the catalog.

        Joined so a face whose photo has left the catalog stops being served,
        and so a caller can ask for faces and their photos in one pass.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT f.* FROM photo_faces f
                   JOIN photos p ON p.filepath = f.filepath
                   WHERE p.folder = ?
                   ORDER BY f.filepath, f.face_index""",
                (folder,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_all_faces(self, grouped_only: bool = False, model: Optional[str] = None) -> list[dict]:
        """Every stored face whose photo is still catalogued. ``model`` keeps only
        the faces one model described (rows from before models were recorded
        count as SFace's)."""
        clauses, params = [], []
        if grouped_only:
            clauses.append("f.cluster_id > 0")
        if model is not None:
            clauses.append("(CASE WHEN f.model = '' THEN 'sface' ELSE f.model END) = ?")
            params.append(model)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT f.* FROM photo_faces f
                    JOIN photos p ON p.filepath = f.filepath
                    {where}
                    ORDER BY f.id ASC""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def face_scans_by_other_model(self, model: str) -> int:
        """How many frames were last described by a model other than ``model``."""
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM face_scans WHERE (CASE WHEN model = '' THEN 'sface' ELSE model END) != ?",
                (model,),
            ).fetchone()[0])

    def set_cluster_suggestions(self, suggestions: dict[int, tuple[int, float]]) -> None:
        """Replace every group's "looks like" with ``{group: (named group, score)}``."""
        with self._connect() as conn:
            conn.execute("UPDATE person_clusters SET suggest_id = 0, suggest_score = 0")
            conn.executemany(
                "UPDATE person_clusters SET suggest_id = ?, suggest_score = ? WHERE id = ?",
                [(int(other), float(score), int(cid)) for cid, (other, score) in suggestions.items()],
            )

    def refuse_person(self, cluster_id: int, other_id: int) -> int:
        """Record that the faces of one group are *not* the person ``other_id``:
        the answer "no" to "is this Alice?". The faces carry it from then on, so
        no rebuild puts them into that group."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, not_people FROM photo_faces WHERE cluster_id = ?", (int(cluster_id),)
            ).fetchall()
            updates = []
            for row in rows:
                refused = parse_refused(row["not_people"])
                refused.add(int(other_id))
                updates.append((",".join(str(i) for i in sorted(refused)), int(row["id"])))
            conn.executemany("UPDATE photo_faces SET not_people = ? WHERE id = ?", updates)
            conn.execute(
                "UPDATE person_clusters SET suggest_id = 0, suggest_score = 0 WHERE id = ?",
                (int(cluster_id),),
            )
        return len(updates)

    def set_face_clusters(self, assignments: dict[int, int]) -> int:
        """Point each face row at a person. ``{face_id: cluster_id}``."""
        if not assignments:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "UPDATE photo_faces SET cluster_id = ? WHERE id = ?",
                [(int(cluster), int(face_id)) for face_id, cluster in assignments.items()],
            )
        return len(assignments)

    def face_scan_status(self) -> dict:
        """How much of the library has been looked at for faces."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            scanned = conn.execute("SELECT COUNT(*) FROM face_scans").fetchone()[0]
            faces = conn.execute("SELECT COUNT(*) FROM photo_faces").fetchone()[0]
            grouped = conn.execute(
                "SELECT COUNT(*) FROM photo_faces WHERE cluster_id > 0"
            ).fetchone()[0]
            people = conn.execute(
                "SELECT COUNT(*) FROM person_clusters WHERE name != ''"
            ).fetchone()[0]
        return {
            "photos": total,
            "scanned": scanned,
            "faces": faces,
            "grouped": grouped,
            "named": people,
        }

    # ── People ───────────────────────────────────────────────────

    def list_clusters(self) -> list[dict]:
        """Every group of faces, with a face to show for it.

        Named groups first, then the rest by how many clear faces they have
        (``clear``): a group of a thousand blurred faces is a worse lead than one
        of forty sharp ones, and used to head the list.

        The sample is the largest face in the group, which is the one most
        likely to be recognisable to a person looking at a row of thumbnails.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.id, c.name, c.tag_id, c.suggest_id, c.suggest_score,
                          COALESCE(s.name, '') AS suggest_name,
                          COUNT(f.id) AS faces,
                          COUNT(DISTINCT f.filepath) AS photos,
                          SUM(COALESCE(f.quality, 1.0) >= ?) AS clear
                   FROM person_clusters c
                   LEFT JOIN photo_faces f ON f.cluster_id = c.id
                   LEFT JOIN person_clusters s ON s.id = c.suggest_id AND s.name != ''
                   GROUP BY c.id
                   HAVING faces > 0
                   ORDER BY (c.name = '') ASC, clear DESC, faces DESC, c.id ASC""",
                (CLEAR_FACE_QUALITY,),
            ).fetchall()
            out = []
            for row in rows:
                # The clearest face, not merely the largest: a big blurred
                # profile is a poor picture to put a name to.
                sample = conn.execute(
                    """SELECT filepath, face_index, x, y, w, h, px FROM photo_faces
                       WHERE cluster_id = ? ORDER BY quality * MIN(px, 200) DESC, px DESC, id ASC
                       LIMIT 1""",
                    (row["id"],),
                ).fetchone()
                entry = dict(row)
                if not entry.get("suggest_name"):
                    entry["suggest_id"], entry["suggest_score"] = 0, 0.0
                entry["sample"] = dict(sample) if sample else None
                out.append(entry)
            return out

    def get_cluster(self, cluster_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM person_clusters WHERE id = ?", (cluster_id,)
            ).fetchone()
            return dict(row) if row else None

    def create_cluster(self, name: str = "", tag_id: int = 0) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO person_clusters (name, tag_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (name, int(tag_id), now, now),
            )
            return int(cursor.lastrowid)

    def create_clusters(self, count: int) -> list[int]:
        """``count`` unnamed groups in one transaction, their ids in order."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            return [
                int(conn.execute(
                    "INSERT INTO person_clusters (name, tag_id, created_at, updated_at) "
                    "VALUES ('', 0, ?, ?)",
                    (now, now),
                ).lastrowid)
                for _ in range(count)
            ]

    def set_cluster_name(self, cluster_id: int, name: str, tag_id: int = 0) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE person_clusters SET name = ?, tag_id = ?, updated_at = ? WHERE id = ?",
                (name, int(tag_id), datetime.now().isoformat(), int(cluster_id)),
            )
            return cursor.rowcount > 0

    def cluster_filepaths(self, cluster_id: int) -> list[str]:
        """The photos one person appears in, each once."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT f.filepath FROM photo_faces f
                   JOIN photos p ON p.filepath = f.filepath
                   WHERE f.cluster_id = ? ORDER BY f.filepath ASC""",
                (int(cluster_id),),
            ).fetchall()
            return [r["filepath"] for r in rows]

    def merge_clusters(self, keep_id: int, absorb_ids: list[int]) -> int:
        """Move every face of the absorbed groups into the kept one."""
        ids = [int(i) for i in absorb_ids if int(i) != int(keep_id)]
        if not ids:
            return 0
        marks = ", ".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE photo_faces SET cluster_id = ? WHERE cluster_id IN ({marks})",
                [int(keep_id), *ids],
            )
            conn.execute(
                f"DELETE FROM person_clusters WHERE id IN ({marks})", ids
            )
        return cursor.rowcount

    def detach_cluster(self, cluster_id: int) -> int:
        """Take a group apart: its faces go back to being ungrouped."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE photo_faces SET cluster_id = 0 WHERE cluster_id = ?",
                (int(cluster_id),),
            )
            conn.execute("DELETE FROM person_clusters WHERE id = ?", (int(cluster_id),))
        return cursor.rowcount

    def faces_in_cluster(self, cluster_id: int) -> list[dict]:
        """One group's faces whose photos are still catalogued, vectors included."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT f.* FROM photo_faces f JOIN photos p ON p.filepath = f.filepath
                   WHERE f.cluster_id = ? ORDER BY f.filepath, f.face_index""",
                (int(cluster_id),),
            ).fetchall()
            return [dict(r) for r in rows]

    def demoted_shoots(self, owner: str = "local") -> set[tuple[str, str]]:
        """``{(folder, camera)}`` one owner has demoted; camera '*' is the whole folder."""
        with self._connect() as conn:
            rows = conn.execute("SELECT folder, camera FROM demoted_shoots WHERE owner = ?", (owner,)).fetchall()
            return {(r["folder"], r["camera"]) for r in rows}

    def set_demoted(self, owner: str, folder: str, camera: str = "*", demoted: bool = True) -> None:
        with self._connect() as conn:
            if demoted:
                conn.execute(
                    "INSERT OR IGNORE INTO demoted_shoots (owner, folder, camera, created_at) VALUES (?, ?, ?, ?)",
                    (owner, folder, camera, datetime.now().isoformat()))
            else:
                conn.execute("DELETE FROM demoted_shoots WHERE owner = ? AND folder = ? AND camera = ?",
                             (owner, folder, camera))

    def list_perspectives(self) -> list[dict]:
        from catalog.perspectives import load

        with self._connect() as conn:
            rows = conn.execute("SELECT id, name, people FROM perspectives ORDER BY id").fetchall()
            return [{"id": int(r["id"]), "name": r["name"], "people": load(r["people"])} for r in rows]

    def save_perspective(self, name: str, people: list[str], pid: Optional[int] = None) -> int:
        now = datetime.now().isoformat()
        value = json.dumps([str(p).strip() for p in people if str(p).strip()])
        with self._connect() as conn:
            if pid:
                conn.execute("UPDATE perspectives SET name = ?, people = ?, updated_at = ? WHERE id = ?",
                             (name, value, now, int(pid)))
                return int(pid)
            cursor = conn.execute("INSERT INTO perspectives (name, people, created_at, updated_at) VALUES (?, ?, ?, ?)",
                                  (name, value, now, now))
            return int(cursor.lastrowid)

    def delete_perspective(self, pid: int) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM perspectives WHERE id = ?", (int(pid),)).rowcount > 0

    def owner_setting(self, owner: str, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM owner_settings WHERE owner = ? AND key = ?", (owner, key)).fetchone()
            return row["value"] if row else default

    def set_owner_setting(self, owner: str, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO owner_settings (owner, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(owner, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (owner, key, value, datetime.now().isoformat()))

    def people_in_photos(self) -> list[tuple]:
        """``(folder, base_name, name)`` for every named person in a photo: by
        their face, or by a tag spelled as their name (named before the face
        was, or tagged where no face was found)."""
        with self._connect() as conn:
            faces = conn.execute(
                """SELECT DISTINCT p.folder, p.base_name, c.name FROM photo_faces f
                   JOIN person_clusters c ON c.id = f.cluster_id AND c.name != ''
                   JOIN photos p ON p.filepath = f.filepath""").fetchall()
            # Which tags are people, worked out here: matching names in the join
            # would compare every tag of every photo without an index.
            people = conn.execute("SELECT name, tag_id FROM person_clusters WHERE name != ''").fetchall()
            by_name = {r["name"].casefold(): r["name"] for r in people}
            person_of = {}
            for t in conn.execute("SELECT id, name FROM tags").fetchall():
                linked = next((r["name"] for r in people if r["tag_id"] == t["id"]), None)
                name = linked or by_name.get(t["name"].casefold())
                if name:
                    person_of[int(t["id"])] = name
            tags = conn.execute(
                """SELECT DISTINCT p.folder, p.base_name, pt.tag_id FROM photo_tags pt
                   JOIN photos p ON p.filepath = pt.filepath
                   WHERE pt.tag_id IN (SELECT value FROM json_each(?))""",
                (json.dumps(list(person_of)),)).fetchall()
            return ([(r[0], r[1], r[2]) for r in faces]
                    + [(r[0], r[1], person_of[int(r[2])]) for r in tags])

    def named_people(self) -> list[str]:
        """Every named person, the most photographed first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.name FROM person_clusters c JOIN photo_faces f ON f.cluster_id = c.id
                   WHERE c.name != '' GROUP BY c.name ORDER BY COUNT(DISTINCT f.filepath) DESC, c.name""").fetchall()
            return [r[0] for r in rows]

    def photos_with_unnamed_people(self, min_quality: float, min_px: float) -> set[tuple[str, str]]:
        """``{(folder, base_name)}`` of photographs holding someone nobody has
        named: a face the grouping put in a group, or one clear and large enough
        to have been. A faint face in the background does not count."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT p.folder, p.base_name FROM photo_faces f
                   JOIN photos p ON p.filepath = f.filepath
                   LEFT JOIN person_clusters c ON c.id = f.cluster_id
                   WHERE COALESCE(c.name, '') = ''
                     AND (f.cluster_id > 0 OR (COALESCE(f.quality, 1.0) >= ? AND f.px >= ?))""",
                (float(min_quality), float(min_px))).fetchall()
            return {(r["folder"], r["base_name"]) for r in rows}

    def named_faces(self) -> list[dict]:
        """Every face of a named person, with the name (no vectors)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT f.filepath, f.face_index, f.x, f.y, f.w, f.h, f.px, f.quality,
                          f.cluster_id, c.name
                   FROM photo_faces f JOIN person_clusters c ON c.id = f.cluster_id AND c.name != ''"""
            ).fetchall()
            return [dict(r) for r in rows]

    def clip_vectors(self, filepaths: list[str]) -> dict:
        """The smart-search descriptions of these photos that have one, as unit
        float32 vectors (any model: they are only compared with each other)."""
        import numpy as np

        out = {}
        paths = list(dict.fromkeys(filepaths))
        with self._connect() as conn:
            for start in range(0, len(paths), 500):
                chunk = paths[start:start + 500]
                marks = ", ".join("?" for _ in chunk)
                for r in conn.execute(f"SELECT filepath, vector FROM photo_clip WHERE filepath IN ({marks})", chunk):
                    v = np.frombuffer(r["vector"], np.float16).astype(np.float32)
                    n = float(np.linalg.norm(v))
                    if n > 0:
                        out[r["filepath"]] = v / n
        return out

    def photo_stats_brief(self) -> dict[str, dict]:
        """What picking a frame reads of every measured photo: sharpness, clipped
        highlights and the look's hash."""
        with self._connect() as conn:
            rows = conn.execute("SELECT filepath, sharpness, clip_high, dhash FROM photo_stats").fetchall()
            return {r["filepath"]: dict(r) for r in rows}

    def faces_with_dates(self, cluster_id: int) -> list[dict]:
        """One group's faces with what a timeline needs of their photo: its
        date, folder, rating and flag (no vectors)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT f.filepath, f.face_index, f.x, f.y, f.w, f.h, f.px, f.quality,
                          COALESCE(NULLIF(p.local_date, ''), p.capture_date) AS capture_date,
                          p.folder, p.rating, p.flag
                   FROM photo_faces f JOIN photos p ON p.filepath = f.filepath
                   WHERE f.cluster_id = ? ORDER BY f.filepath, f.face_index""",
                (int(cluster_id),),
            ).fetchall()
            return [dict(r) for r in rows]

    def face_cluster(self, filepath: str, face_index: int) -> int:
        """The group one face is in, 0 when none (or no such face)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cluster_id FROM photo_faces WHERE filepath = ? AND face_index = ?",
                (filepath, int(face_index)),
            ).fetchone()
            return int(row["cluster_id"]) if row else 0

    def photo_in_cluster(self, filepath: str, cluster_id: int) -> bool:
        """Whether any face of this photo is still in the group."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM photo_faces WHERE filepath = ? AND cluster_id = ? LIMIT 1",
                (filepath, int(cluster_id)),
            ).fetchone() is not None

    def clusters_named(self, name: str) -> list[int]:
        """The groups carrying a name, compared without regard to case."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM person_clusters WHERE name != '' AND lower(name) = lower(?)",
                ((name or "").strip(),),
            ).fetchall()
            return [int(r["id"]) for r in rows]

    def move_face(self, filepath: str, face_index: int, into: int) -> int:
        """Put one face in another group; returns the group it left (0 if none).

        The group it leaves goes on its "not this person" list, so no rebuild
        puts it back, and the group it joins comes off that list.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, cluster_id, not_people FROM photo_faces WHERE filepath = ? AND face_index = ?",
                (filepath, int(face_index)),
            ).fetchone()
            if row is None:
                return 0
            left = int(row["cluster_id"] or 0)
            refused = parse_refused(row["not_people"])
            if left and left != int(into):
                refused.add(left)
            refused.discard(int(into))
            conn.execute(
                "UPDATE photo_faces SET cluster_id = ?, not_people = ? WHERE id = ?",
                (int(into), ",".join(str(i) for i in sorted(refused)), int(row["id"])),
            )
            return left

    def detach_face(self, filepath: str, face_index: int) -> bool:
        """Take one face out of its group, leaving the rest where they are.

        This is the correction for the mistake the grouping is most likely to
        make: one photograph of a child joining their sibling's group.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, cluster_id, not_people FROM photo_faces
                   WHERE filepath = ? AND face_index = ? AND cluster_id > 0""",
                (filepath, int(face_index)),
            ).fetchone()
            changed = row is not None
            if changed:
                # Remembered on the face, or the next rebuild would put it
                # straight back into the group it was just taken out of.
                refused = parse_refused(row["not_people"])
                refused.add(int(row["cluster_id"]))
                conn.execute(
                    "UPDATE photo_faces SET cluster_id = 0, not_people = ? WHERE id = ?",
                    (",".join(str(i) for i in sorted(refused)), int(row["id"])),
                )
        if changed:
            self.drop_empty_clusters()
        return changed

    def forget_faces(self, filepath: str) -> int:
        """Drop what is known about the faces of one file.

        Used for the other half of a RAW+JPG pair: those two files are one
        photograph, and leaving a second set of faces against the half that is
        not being scanned makes every person in the library appear twice.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM photo_faces WHERE filepath = ?", (filepath,))
            conn.execute("DELETE FROM face_scans WHERE filepath = ?", (filepath,))
            return cursor.rowcount

    # Every table that holds a row per photograph, and every one per folder.
    _PER_PHOTO = ("photo_tags", "photo_stats", "photo_faces", "face_scans", "photo_edits",
                  "photo_versions", "photo_gps", "photo_eyes", "photo_exif_reads", "photo_clip", "photo_clock",
                  "export_basket",
                  "photos")
    _PER_FOLDER = ("folder_state", "folder_notes", "demoted_shoots")

    def forget_non_photos(self, extensions) -> int:
        """Forget rows that were never photographs: a dot-file (a Mac's
        "._IMG_0001.JPG" beside each file it touched) and any file of the given
        extensions (videos). The files are not touched. Returns how many."""
        exts = sorted({e.lower() for e in extensions})
        marks = ",".join("?" * len(exts)) or "''"
        where = f"(filename LIKE '.%' OR lower(extension) IN ({marks}))"
        with self._connect() as conn:
            paths = [r[0] for r in conn.execute(f"SELECT filepath FROM photos WHERE {where}", exts)]
            if not paths:
                return 0
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            for table in self._PER_PHOTO:
                if table in existing:
                    conn.executemany(f"DELETE FROM {table} WHERE filepath = ?", [(p,) for p in paths])
        self.drop_empty_clusters()
        return len(paths)

    def forget_folder_tree(self, folder: str) -> int:
        """Forget every photograph under ``folder`` (the folder and everything in
        it), in every table, matched without case as Windows matches paths. The
        files are not touched. Returns how many photographs were forgotten."""
        prefix = folder.strip("/").lower()
        if not prefix:
            raise ValueError("Refusing to forget the whole library")
        like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
        with self._connect() as conn:
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            count = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE lower(filepath) LIKE ? ESCAPE '\\'", (like,)
            ).fetchone()[0]
            for table in self._PER_PHOTO:
                if table in existing:
                    conn.execute(f"DELETE FROM {table} WHERE lower(filepath) LIKE ? ESCAPE '\\'", (like,))
            for table in self._PER_FOLDER:
                if table in existing:
                    conn.execute(
                        f"DELETE FROM {table} WHERE lower(folder) = ? OR lower(folder) LIKE ? ESCAPE '\\'",
                        (prefix, like))
        if count:
            self.drop_empty_clusters()
        return count

    # Drawn faces are numbered from here, clear of anything a detector finds (a
    # crowd looked at closely has given 129).
    DRAWN_FACE_INDEX = 1000

    def add_drawn_face(self, filepath: str, x: float, y: float, w: float, h: float,
                       px: float = 0.0) -> int:
        """Store a face drawn by hand, as fractions of the frame. Returns its index."""
        with self._connect() as conn:
            top = conn.execute(
                "SELECT MAX(face_index) FROM photo_faces WHERE filepath = ? AND drawn = 1",
                (filepath,)).fetchone()[0]
            index = max(self.DRAWN_FACE_INDEX, (top or 0) + 1)
            conn.execute(
                """INSERT INTO photo_faces
                   (filepath, face_index, x, y, w, h, score, px, embedding, cluster_id,
                    created_at, model, quality, not_people, drawn)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 0, ?, 'drawn', 0, '', 1)""",
                (filepath, index, float(x), float(y), float(w), float(h), float(px), b"",
                 datetime.now().isoformat()),
            )
            return index

    def delete_drawn_face(self, filepath: str, face_index: int) -> Optional[int]:
        """Take away a face drawn by hand. Returns the group it was in (0 for
        none), or None when there was no such drawn face."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cluster_id FROM photo_faces WHERE filepath = ? AND face_index = ? AND drawn = 1",
                (filepath, int(face_index))).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM photo_faces WHERE filepath = ? AND face_index = ? AND drawn = 1",
                         (filepath, int(face_index)))
            return int(row["cluster_id"] or 0)

    def prune_small_faces(self, min_px: float) -> int:
        """Drop faces smaller than the floor.

        Rows were stored before the floor existed, and there are a great many of
        them: on one real library 17,012 faces were held and 16,998 were under
        40px, with a single frame contributing 221 of them. Nothing reads a face
        that small — it cannot be grouped, and the frame's own face count is not
        shown anywhere — so they are weight rather than information.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM photo_faces WHERE px < ? AND drawn = 0", (float(min_px),))
            return cursor.rowcount

    def drop_empty_clusters(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """DELETE FROM person_clusters WHERE id NOT IN
                   (SELECT DISTINCT cluster_id FROM photo_faces WHERE cluster_id > 0)"""
            )
            return cursor.rowcount
