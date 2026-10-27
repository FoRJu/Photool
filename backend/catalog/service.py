"""
Catalog service — ties together the SQLite DB, XMP sidecars, and EXIF extraction.

This is the main interface for the rest of the app to interact with
photo metadata. All rating/flag operations go through here to ensure
the DB and XMP sidecars stay in sync.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from analysis.groups import MAX_GAP_SECONDS, find_groups, suggest_keeper
from analysis.hints import compute_hints, folder_baseline, peer_sharpness, peers
from analysis.image_stats import STATS_VERSION
from config import is_video

from .database import CatalogDB, PhotoRecord, Flag, ColorLabel
from .xmp import read_xmp, write_xmp

logger = logging.getLogger(__name__)


def is_export_folder(folder: str) -> bool:
    """Whether a library folder is the export folder or inside it (export.jobs)."""
    from export.jobs import EXPORT_DIR

    top = (folder or "").replace("\\", "/").strip("/").split("/", 1)[0]
    return top.lower() == EXPORT_DIR


class CatalogService:
    """
    High-level catalog operations.

    Ensures DB and XMP sidecars stay synchronized on every write.
    """

    def __init__(
        self,
        db: CatalogDB,
        library_root: str,
        burst_max_gap_seconds: int = MAX_GAP_SECONDS,
    ):
        self.db = db
        self.library_root = Path(library_root)
        # The grouping window is configured per library, so every caller —
        # browsing bursts and rejecting duplicates in one go — agrees on what a
        # burst is.
        self.burst_max_gap_seconds = int(burst_max_gap_seconds)
        # What was picked for one person lately: see _picked.
        self._picks: dict = {}
        self._picks_lock = threading.Lock()

    # ── Indexing ─────────────────────────────────────────────────

    def index_folder(
        self,
        folder: str,
        extensions: list[str],
        exif_extractor=None,
    ) -> dict:
        """
        Index all photos in a folder into the catalog.

        Reads any existing XMP sidecars to preserve ratings.
        Optionally extracts EXIF data via the provided function.

        Args:
            folder: relative path e.g. "2024/06/15"
            extensions: list of file extensions to index
            exif_extractor: optional callable(Path) -> dict with EXIF fields

        Returns:
            stats dict with counts
        """
        if is_export_folder(folder):
            # Copies of photographs already in the library (see is_export_folder).
            return {"indexed": 0, "folder": folder, "skipped": "export"}
        folder_path = self.library_root / folder
        if not folder_path.exists():
            return {"indexed": 0, "error": "folder not found"}

        ext_set = {ext.lower() for ext in extensions}

        # Collect all image files. A name that starts with a dot is never a
        # photograph, whatever it ends in: a Mac leaves "._IMG_0001.JPG" beside
        # every file it touches on a non-Mac drive, and .DS_Store in every folder.
        image_files = sorted(
            f for f in folder_path.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in ext_set
        )

        if not image_files:
            return {"indexed": 0, "folder": folder}

        # Fast path: skip re-indexing if file count matches and all are in DB
        existing_count = len(self.db.get_photos_in_folder(folder))
        if existing_count == len(image_files):
            return {"indexed": existing_count, "folder": folder, "cached": True}

        photos = []
        indexed = 0
        tags_from_xmp: dict[str, list[str]] = {}
        edits_from_xmp: dict[str, dict] = {}
        copies_from_file: dict[str, list] = {}
        known_copies = self.db.version_counts(folder)

        from adjust.develop import from_crs

        # One query for the folder's edits rather than one per frame: every file
        # in the loop below asks whether the sidecar and the catalog already
        # agree, and that question is the same for the whole folder. Normalised
        # on the way out, so it compares equal to what `from_crs` produces.
        stored_edits = self.photo_edits(folder)

        # Batch EXIF extraction: one exiftool call for all files
        exif_map = {}
        if exif_extractor:
            # Check which files need EXIF
            files_needing_exif = []
            for f in image_files:
                rel_path = f"{folder}/{f.name}"
                existing = self.db.get_photo(rel_path)
                if not existing or not existing.camera:
                    files_needing_exif.append(f)

            if files_needing_exif:
                try:
                    # Use batch extraction if available
                    from thumbnails.extract import get_exif_batch
                    exif_map = get_exif_batch(files_needing_exif)
                except (ImportError, Exception):
                    # Fallback to per-file extraction
                    for f in files_needing_exif:
                        try:
                            exif_map[str(f)] = exif_extractor(f)
                        except Exception:
                            pass

        for f in image_files:
            rel_path = f"{folder}/{f.name}"
            existing = self.db.get_photo(rel_path)
            xmp = read_xmp(f)

            # An edit the sidecar carries is adopted into the catalog index. The
            # sidecar is the durable copy — it travels with the photographs and
            # survives losing this database — so a rebuild from sidecars comes
            # back with every edit, and a change made in another editor arrives
            # with the next full index of that folder.
            #
            # Only a sidecar that *has* an edit is adopted. Treating the absence
            # of one as a deletion would mean a sidecar that failed to write, or
            # one that was never created, silently discarding work the catalog
            # is holding; an edit is only ever cleared by the code that clears
            # the row and the file together.
            edit = from_crs(xmp.develop) if xmp.develop else {}
            if edit and edit != stored_edits.get(rel_path, {}):
                edits_from_xmp[rel_path] = edit

            # Virtual copies are adopted from their file only when the catalog has
            # none for this photograph: the file is the durable copy, the catalog
            # the index, and a rebuild must not lose them or overwrite newer ones.
            if rel_path not in known_copies:
                found = self._read_copies_file(f)
                if found:
                    copies_from_file[rel_path] = found

            record = PhotoRecord(
                filepath=rel_path,
                filename=f.name,
                folder=folder,
                extension=f.suffix.lower(),
                base_name=f.stem,
                file_size=f.stat().st_size,
                rating=xmp.rating,
                flag=xmp.flag,
                color_label=xmp.color_label,
            )

            # Preserve existing DB ratings if no XMP sidecar
            if existing and xmp.rating == 0 and xmp.flag == 0:
                record.rating = existing.rating
                record.flag = existing.flag
                record.color_label = existing.color_label

            # Apply EXIF from batch extraction
            exif = exif_map.get(str(f), {})
            if exif:
                record.camera = exif.get("camera", "")
                record.lens = exif.get("lens", "")
                record.focal_length = exif.get("focal_length", "")
                record.aperture = exif.get("aperture", "")
                record.shutter_speed = exif.get("shutter_speed", "")
                record.iso = exif.get("iso", "")
                record.capture_date = exif.get("date", "")
                record.dimensions = exif.get("dimensions", "")
                record.duration = float(exif.get("duration") or 0.0)
            elif existing and existing.camera:
                # Keep cached EXIF from previous index
                record.camera = existing.camera
                record.lens = existing.lens
                record.focal_length = existing.focal_length
                record.aperture = existing.aperture
                record.shutter_speed = existing.shutter_speed
                record.iso = existing.iso
                record.capture_date = existing.capture_date
                record.dimensions = existing.dimensions
                record.duration = existing.duration

            photos.append(record)
            indexed += 1

            # Keywords travel in the sidecar; import them after the rows exist.
            if xmp.tags:
                tags_from_xmp[rel_path] = xmp.tags

        if photos:
            self.db.bulk_upsert(photos)

        for rel_path, tag_names in tags_from_xmp.items():
            self.db.set_photo_tags(rel_path, tag_names)

        for rel_path, edit in edits_from_xmp.items():
            self.db.set_photo_edit(rel_path, edit)

        for rel_path, copies in copies_from_file.items():
            for c in copies:
                self.db.restore_version(
                    rel_path, c["version"], c["name"], c["settings"], c["export"]
                )

        return {"indexed": indexed, "folder": folder}

    # ── Rating operations (DB + XMP dual-write) ──────────────────

    def set_rating(self, filepath: str, rating: int) -> bool:
        """Set star rating (0-5) and sync to XMP."""
        rating = max(0, min(5, rating))
        success = self.db.set_rating(filepath, rating)
        if success:
            self._sync_to_xmp(filepath)
        return success

    def set_flag(self, filepath: str, flag: int) -> bool:
        """Set flag (-1=reject, 0=none, 1=pick) and sync to XMP."""
        flag = max(-1, min(1, flag))
        success = self.db.set_flag(filepath, flag)
        if success:
            self._sync_to_xmp(filepath)
        return success

    def set_color_label(self, filepath: str, color: int) -> bool:
        """Set color label (0-5) and sync to XMP."""
        color = max(0, min(5, color))
        success = self.db.set_color_label(filepath, color)
        if success:
            self._sync_to_xmp(filepath)
        return success

    def toggle_flag(self, filepath: str, target_flag: int) -> int:
        """
        Toggle a flag: if current flag matches target, clear it; else set it.
        Returns the new flag value.
        """
        record = self.db.get_photo(filepath)
        if not record:
            return 0

        if record.flag == target_flag:
            new_flag = 0
        else:
            new_flag = target_flag

        self.set_flag(filepath, new_flag)
        return new_flag

    # ── Tag operations (DB + XMP dual-write) ─────────────────────

    def set_photo_tags(self, filepath: str, tags: list[str]) -> bool:
        """Replace one photo's event tags and sync them to the sidecar."""
        success = self.db.set_photo_tags(filepath, tags)
        if success:
            self._sync_to_xmp(filepath)
        return success

    def tag_matching_photos(self, folder: str, tag: str, **filters) -> dict:
        """
        Apply one tag to every photo in a folder that matches the filters
        currently shown in the grid. Also writes dc:subject to each sidecar.
        """
        records = self.db.get_photos_in_folder(folder, **filters)
        paths = [r.filepath for r in records]
        tagged = self.db.add_tag_to_photos(paths, tag)
        for r in records:
            self._sync_to_xmp(r.filepath)
        return {"matched": len(records), "tagged": tagged}

    def untag_matching_photos(self, folder: str, tag: str, **filters) -> dict:
        """Remove one tag from every photo in a folder matching the filters."""
        records = self.db.get_photos_in_folder(folder, **filters)
        paths = [r.filepath for r in records]
        removed = self.db.remove_tag_from_photos(paths, tag)
        for r in records:
            self._sync_to_xmp(r.filepath)
        return {"matched": len(records), "untagged": removed}

    # ── People ───────────────────────────────────────────────────
    # A group of faces becomes useful the moment it has a name, and the name is
    # written as an ordinary event tag. Nothing downstream — the filter, the
    # bulk actions, the basket, the sidecar writer — has to know that a person
    # exists; a person is a tag that happens to have been decided by arithmetic
    # rather than by hand.

    def name_person(self, cluster_id: int, name: str) -> dict:
        """Name a group of faces, and put that name on its photographs.

        Renaming takes the old name off and puts the new one on, so a corrected
        name does not leave the wrong one behind on the files.
        """
        cluster = self.db.get_cluster(cluster_id)
        if not cluster:
            return {"name": "", "photos": 0, "tagged": 0}
        paths = self.db.cluster_filepaths(cluster_id)
        previous = (cluster.get("name") or "").strip()
        if previous and previous != name:
            self.db.remove_tag_from_photos(paths, previous)
        tagged = self.db.add_tag_to_photos(paths, name) if name else 0
        self.db.set_cluster_name(cluster_id, name)
        for filepath in paths:
            self._sync_to_xmp(filepath)
        return {"name": name, "photos": len(paths), "tagged": tagged}

    def merge_people(self, cluster_id: int, into: int) -> dict:
        """Fold one group into another, keeping one name on the photographs.

        When both groups were named, the name that loses is taken off: the two
        groups were the same person, so the name that was wrong about them has no
        business staying on the files.
        """
        kept = self.db.get_cluster(into) or {}
        absorbed = self.db.get_cluster(cluster_id) or {}
        name = (kept.get("name") or "").strip()
        lost = (absorbed.get("name") or "").strip()
        moved = self.db.merge_clusters(into, [cluster_id])
        if not name and lost:
            name = lost
        if lost and lost != name:
            self.db.remove_tag_from_photos(self.db.cluster_filepaths(into), lost)
        summary = self.name_person(into, name)
        return {**summary, "moved": moved, "people": len(self.db.list_clusters())}

    def detach_person(self, cluster_id: int) -> dict:
        """Take a group apart, taking its name off the photographs with it."""
        cluster = self.db.get_cluster(cluster_id) or {}
        name = (cluster.get("name") or "").strip()
        paths = self.db.cluster_filepaths(cluster_id)
        faces = self.db.detach_cluster(cluster_id)
        if name:
            self.db.remove_tag_from_photos(paths, name)
            for filepath in paths:
                self._sync_to_xmp(filepath)
        return {"faces": faces, "people": len(self.db.list_clusters())}

    def detach_face(self, filepath: str, face_index: int) -> bool:
        """Take one face out of its group; the group keeps its name."""
        return self.remove_faces([(filepath, face_index)])["removed"] == 1

    def _untag_if_gone(self, filepaths, cluster_id: int) -> set:
        """Take a named group's name off photos none of its faces are in any more.

        A face taken out of "Alice" used to leave the tag "Alice" on its photo,
        so the photo went on being found as hers, and her card still counted it.
        """
        cluster = self.db.get_cluster(cluster_id) if cluster_id else None
        name = ((cluster or {}).get("name") or "").strip()
        if not name:
            return set()
        gone = [fp for fp in dict.fromkeys(filepaths) if not self.db.photo_in_cluster(fp, cluster_id)]
        if gone:
            self.db.remove_tag_from_photos(gone, name)
        return set(gone)

    def draw_face(self, filepath: str, box: tuple, into: int = 0, name: str = "",
                  px: float = 0.0) -> dict:
        """A face drawn by hand where none was found, put with a person.

        The box carries a person and nothing else: no vector is made from it, so
        a silhouette or a blur that was drawn because it could not be recognised
        never teaches the grouping what that person looks like. The name goes on
        the photo as for any other face.
        """
        index = self.db.add_drawn_face(filepath, *box, px=px)
        result = self.move_faces([(filepath, index)], into=into, name=name)
        if not result["into"]:
            self.db.delete_drawn_face(filepath, index)
            return {"face_index": None, **result}
        return {"face_index": index, **result}

    def remove_drawn_face(self, filepath: str, face_index: int) -> bool:
        """Take away a drawn face, and its person's name from the photo when
        no other face of theirs is in it."""
        cluster_id = self.db.delete_drawn_face(filepath, face_index)
        if cluster_id is None:
            return False
        if cluster_id:
            self._untag_if_gone([filepath], cluster_id)
            self.db.drop_empty_clusters()
        self._sync_to_xmp(filepath)
        return True

    def remove_faces(self, faces) -> dict:
        """Take faces out of their groups: "this is not them". Remembered per face."""
        left: dict[int, list[str]] = {}
        removed = 0
        for filepath, face_index in faces:
            cluster_id = self.db.face_cluster(filepath, face_index)
            if cluster_id and self.db.detach_face(filepath, face_index):
                removed += 1
                left.setdefault(cluster_id, []).append(filepath)
        touched: set = set()
        for cluster_id, paths in left.items():
            touched |= self._untag_if_gone(paths, cluster_id)
        for filepath in touched:
            self._sync_to_xmp(filepath)
        return {"removed": removed, "untagged": len(touched)}

    def move_faces(self, faces, into: int = 0, name: str = "") -> dict:
        """Put faces in another person's group, or a new group with a new name.

        Moving into a named group is the strongest answer there is: named groups
        keep their faces through every rebuild, and the group left behind is
        remembered as "not this person" on each face.
        """
        name = (name or "").strip()
        if not into:
            if not name:
                return {"moved": 0, "into": 0}
            existing = self.db.clusters_named(name)
            into = existing[0] if existing else self.db.create_cluster()
        target = self.db.get_cluster(into)
        if not target:
            return {"moved": 0, "into": 0}
        left: dict[int, list[str]] = {}
        moved_paths: list[str] = []
        for filepath, face_index in faces:
            was = self.db.move_face(filepath, face_index, into)
            if was == into:
                continue
            moved_paths.append(filepath)
            if was:
                left.setdefault(was, []).append(filepath)
        touched: set = set()
        for cluster_id, paths in left.items():
            touched |= self._untag_if_gone(paths, cluster_id)
        self.db.drop_empty_clusters()
        if name and not (target.get("name") or "").strip():
            self.name_person(into, name)            # tags every photo of the group
        else:
            label = (target.get("name") or "").strip()
            if label and moved_paths:
                self.db.add_tag_to_photos(moved_paths, label)
            touched |= set(moved_paths)
        for filepath in touched:
            self._sync_to_xmp(filepath)
        return {"moved": len(moved_paths), "into": into}

    def edit_photos(self, filepaths, rating=None, flag=None, color_label=None,
                    add_tags=(), remove_tags=()) -> dict:
        """One change to many photos: stars, flag, colour, tags.

        Taking a person's name off photos also takes that person's face out of
        them. A name on a photo and a face in a group are one fact, and leaving
        the face behind would put the name back the next time the group is
        renamed or merged.
        """
        filepaths = [fp for fp in dict.fromkeys(filepaths) if fp and self.db.get_photo(fp)]
        for filepath in filepaths:
            if rating is not None:
                self.db.set_rating(filepath, max(0, min(5, int(rating))))
            if flag is not None:
                self.db.set_flag(filepath, max(-1, min(1, int(flag))))
            if color_label is not None:
                self.db.set_color_label(filepath, max(0, min(5, int(color_label))))
        tagged = untagged = faces_removed = 0
        for tag in add_tags or ():
            if (tag or "").strip():
                tagged += self.db.add_tag_to_photos(filepaths, tag.strip())
        for tag in remove_tags or ():
            tag = (tag or "").strip()
            if not tag:
                continue
            # The tag as it is spelled in the catalog: "alice" removes "Alice".
            spellings = {t["name"] for t in self.db.list_tags()
                         if t["name"].casefold() == tag.casefold()} or {tag}
            for spelling in spellings:
                untagged += self.db.remove_tag_from_photos(filepaths, spelling)
            people = set(self.db.clusters_named(tag))
            if people:
                wanted = set(filepaths)
                faces = [
                    (row["filepath"], int(row["face_index"]))
                    for cluster_id in people
                    for row in self.db.faces_in_cluster(cluster_id)
                    if row["filepath"] in wanted
                ]
                faces_removed += self.remove_faces(faces)["removed"]
        for filepath in filepaths:
            self._sync_to_xmp(filepath)
        return {"photos": len(filepaths), "tagged": tagged, "untagged": untagged,
                "faces_removed": faces_removed}

    # ── Export basket ────────────────────────────────────────────

    def add_matching_to_basket(self, folder: str, owner: str = "local", **filters) -> dict:
        """
        One-shot: copy every photo matching the current filter into the
        basket. Uses the same query as the grid, so the basket gets exactly
        what the user was looking at.
        """
        records = self.db.get_photos_in_folder(folder, **filters)
        added = self.db.add_basket_from_photos(records, owner)
        return {"matched": len(records), "added": added, "basket": self.db.basket_count(owner)}

    # ── Burst grouping ───────────────────────────────────────────

    def _burst_members(self, folder: str, **filters) -> tuple[list[dict], int, int]:
        """The frames of a folder that can be grouped, plus the counts.

        Only current measurements are used. A photo that has not been measured
        yet is counted in ``unmeasured`` rather than guessed at, so a folder
        still being measured reports fewer bursts instead of wrong ones.
        """
        # A clip is never one frame of a burst.
        records = [r for r in self.db.get_photos_in_folder(folder, **filters)
                   if not is_video(r.extension)]
        measurements = self.db.list_photo_stats_for_folder(folder)

        members: list[dict] = []
        unmeasured = 0
        for record in records:
            row = measurements.get(record.filepath)
            if (
                not row
                or row.get("stats_version") != STATS_VERSION
                or not row.get("dhash")
            ):
                unmeasured += 1
                continue
            members.append(
                {
                    "filepath": record.filepath,
                    "filename": record.filename,
                    "flag": record.flag,
                    # The measurement records when the file was written, which
                    # for a card straight from a camera is the capture time.
                    "mtime": int(row.get("source_mtime") or 0),
                    "sharpness": float(row.get("sharpness") or 0.0),
                    "dhash": row["dhash"],
                }
            )
        return members, unmeasured, len(records)

    def burst_of(self, filepath: str) -> list[str]:
        """Every frame that shares a burst with ``filepath``, itself included.

        The grid's filters are deliberately not applied. A burst is a fact about
        the photographs, while a filter is a fact about what is on screen; a
        frame hidden by the filter is still in the burst, and leaving it out
        would make "the rest of the burst" quietly mean something narrower.

        A frame that is not in a measured burst comes back on its own, so a
        caller can tell there is nothing to copy to instead of guessing.
        """
        record = self.db.get_photo(filepath)
        if not record:
            return [filepath]
        members, _unmeasured, _total = self._burst_members(record.folder)
        for group in find_groups(members, self.burst_max_gap_seconds):
            paths = [m["filepath"] for m in group]
            if filepath in paths:
                return paths
        return [filepath]

    def find_bursts(self, folder: str, max_gap_seconds: Optional[int] = None, **filters) -> dict:
        """Near-identical frames in one folder, as groups.

        The filters are the ones the grid is showing, so a burst is a set of
        frames the user can actually see — and a CR3 never groups with its own
        JPG, because the grid has already collapsed that pair.

        ``max_gap_seconds`` overrides the time window the grouping uses; left
        alone it uses the library default.
        """
        members, unmeasured, total = self._burst_members(folder, **filters)

        # Resolved here rather than left to the grouping's default so the
        # payload can say which window produced these groups.
        gap = self.burst_max_gap_seconds if max_gap_seconds is None else int(max_gap_seconds)

        from analysis import eyes

        found_groups = [g for g in find_groups(members, gap)
                        # A burst whose every frame is already rejected has been
                        # culled; offering it again is noise.
                        if not all(m["flag"] == Flag.REJECT for m in g)]
        # The faces, where a burst's frames have been looked at (analysis/eyes.py):
        # a burst measured whole keeps the frame with its people at their best.
        faces = eyes.frames_of(self.db, [m["filepath"] for g in found_groups for m in g])
        checked = 0
        groups = []
        for group in found_groups:
            measured = {m["filepath"]: faces[m["filepath"]] for m in group if m["filepath"] in faces}
            checked += len(measured)
            assessed = eyes.assess(measured) if len(measured) == len(group) else None
            keeper = eyes.best_frame(group, assessed) if assessed else suggest_keeper(group)
            groups.append(
                {
                    "size": len(group),
                    "keep": keeper["filepath"] if keeper else None,
                    "faces_checked": assessed is not None,
                    "closed": sorted(fp for fp, a in (assessed or {}).items() if a["closed"]),
                    "members": [
                        {
                            "filepath": m["filepath"],
                            "filename": m["filename"],
                            "flag": m["flag"],
                        }
                        for m in group
                    ],
                }
            )

        return {
            "folder": folder,
            "groups": groups,
            "frames": sum(g["size"] for g in groups),
            "measured": len(members),
            "total": total,
            "unmeasured": unmeasured,
            "max_gap_seconds": gap,
            "faces_checked": checked,
        }

    def check_burst_faces(self, folder: str, models_dir: str, cache, progress=None, **filters) -> dict:
        """Look at the faces in every frame of this folder's bursts that has not
        been looked at as it is now, then answer as :meth:`find_bursts` does."""
        from analysis import eyes

        found = self.find_bursts(folder, **filters)
        paths = [m["filepath"] for g in found["groups"] for m in g["members"]]
        measured = eyes.measure_files(self.db, self.library_root, cache, paths, models_dir,
                                      progress=progress)
        result = self.find_bursts(folder, **filters)
        result["measured_now"] = measured
        return result

    def suggest_best(self, folder: str, count: int = 20, **filters) -> dict:
        """The ``count`` frames of this folder most worth keeping, one per moment.

        A burst offers only its keeper; a reject is never offered. Then, in order
        of weight: picked, stars, developed, the people at their best (where the
        faces have been looked at), sharp for this shoot; a soft frame, blown
        highlights or someone's closed eyes count against it. A suggestion only:
        the caller selects them for the person to act on.
        """
        from analysis import eyes

        records = self.db.get_photos_in_folder(folder, **filters)
        bursts = self.find_bursts(folder, **filters)
        not_keepers = {m["filepath"] for g in bursts["groups"] for m in g["members"]
                       if m["filepath"] != g["keep"]}
        closed = {fp for g in bursts["groups"] for fp in g.get("closed", [])}
        hints = self.folder_hints(folder, **filters)["hints"]
        stats = self.db.list_photo_stats_for_folder(folder)
        edited = set(self.photo_edits(folder))

        candidates = [r for r in records if r.flag != Flag.REJECT and r.filepath not in not_keepers]
        sharp = sorted(float((stats.get(r.filepath) or {}).get("sharpness") or 0.0) for r in candidates)
        faces = eyes.frames_of(self.db, [r.filepath for r in candidates])

        def rank(value: float) -> float:
            if len(sharp) < 2:
                return 0.5
            import bisect
            return bisect.bisect_left(sharp, value) / (len(sharp) - 1)

        scored = []
        for r in candidates:
            hint = hints.get(r.filepath) or {}
            people = eyes.assess({r.filepath: faces[r.filepath]})[r.filepath]["score"] if r.filepath in faces else 0.0
            score = (40 * (r.flag == Flag.PICK) + 10 * r.rating + 6 * (r.filepath in edited)
                     + 10 * min(people, 2.0)
                     + 10 * rank(float((stats.get(r.filepath) or {}).get("sharpness") or 0.0))
                     - 15 * bool(hint.get("soft_frame")) - 10 * bool(hint.get("blown_highlights"))
                     - 25 * (r.filepath in closed))
            scored.append((score, r))
        scored.sort(key=lambda t: (-t[0], t[1].filename))
        chosen = sorted((r for _, r in scored[:max(1, int(count))]), key=lambda r: r.filename)
        return {
            "folder": folder,
            "filepaths": [r.filepath for r in chosen],
            "considered": len(candidates),
            "faces_checked": sum(1 for r in candidates if r.filepath in faces),
        }

    def reject_burst_rest(self, keep: str, others: Iterable[str]) -> dict:
        """Reject every frame of one burst except the one being kept.

        The keeper is left exactly as it is. Rejecting the rest is the whole
        action, and a frame that stays unflagged stays visible in the grid.
        """
        rejected = []
        for filepath in others or []:
            if not filepath or filepath == keep:
                continue
            if self.set_flag(filepath, Flag.REJECT):
                rejected.append(filepath)
        return {"keep": keep, "rejected": rejected}

    def reject_burst_duplicates(self, folder: str, **filters) -> dict:
        """Keep the sharpest frame of every burst in a folder, reject the rest."""
        found = self.find_bursts(folder, **filters)
        rejected = []
        for group in found["groups"]:
            keep = group["keep"]
            for member in group["members"]:
                if member["filepath"] == keep or member["flag"] == Flag.REJECT:
                    continue
                if self.set_flag(member["filepath"], Flag.REJECT):
                    rejected.append(member["filepath"])
        return {
            "groups": len(found["groups"]),
            "rejected": len(rejected),
            "filepaths": rejected,
        }

    # ── One photograph's own edit ────────────────────────────────
    #
    # Non-destructive, and written in two places on purpose. The sidecar is the
    # durable copy: it travels with the photographs, another editor can read it,
    # and it survives losing this database. The catalog row is the index over it,
    # so drawing a grid of two thousand frames does not mean opening two thousand
    # files. Saving writes both, and `index_folder` reconciles them.

    def set_photo_edit(self, filepath: str, settings: Optional[dict]) -> dict:
        """Store an edit on one photograph, in the catalog and in its sidecar.

        Returns the edit as stored, after clamping. The sidecar write is not
        allowed to fail the save: a library on a read-only volume still remembers
        the edit in its catalog, and the next writable save puts it in the file.
        """
        from adjust.develop import normalize

        if is_video(filepath):
            raise ValueError("A video is not developed")
        clean = normalize(settings)
        self.db.set_photo_edit(filepath, clean)
        self._write_edit_to_sidecar(filepath, clean)
        return clean

    def clear_photo_edit(self, filepath: str) -> bool:
        """Take this app's edit off a photograph, in the catalog and the file.

        Only this app's own attributes leave the sidecar — a crop or a curve
        another editor wrote is not ours to delete.
        """
        removed = self.db.clear_photo_edit(filepath)
        self._write_edit_to_sidecar(filepath, {})
        return removed

    def photo_edit(self, filepath: str) -> dict:
        """This photograph's edit, or ``{}`` when it has none.

        Clamped and de-unknown-keyed on the way out, so a row written by another
        version of the app reads back as exactly the controls this one has.
        """
        from adjust.develop import normalize

        return normalize(self.db.get_photo_edit(filepath))

    def photo_edits(self, folder: Optional[str] = None) -> dict[str, dict]:
        """Which photographs in a folder are edited, and how."""
        from adjust.develop import normalize

        return {
            filepath: normalize(settings)
            for filepath, settings in self.db.list_photo_edits(folder).items()
        }

    def _write_edit_to_sidecar(self, filepath: str, settings: dict) -> bool:
        """Put an edit in its sidecar, replacing only this app's attributes."""
        from adjust.develop import CRS_KEYS, LIGHTROOM_CROP_KEYS, to_crs
        from catalog.xmp import read_xmp, sync_develop_to_xmp

        abs_path = self.library_root / filepath
        if not abs_path.exists():
            logger.warning(f"Cannot write edit — file missing: {filepath}")
            return False
        try:
            written = to_crs(settings)
            owned = CRS_KEYS
            # Lightroom's crop names are ours to replace only once this app has
            # put a crop there itself; a crop made in Lightroom, on a frame this
            # app never cropped, is left alone.
            if "crs:CropLeft" in written or "crs:PhotoolCropX" in read_xmp(abs_path).develop:
                owned = CRS_KEYS + LIGHTROOM_CROP_KEYS
            sync_develop_to_xmp(abs_path, written, owned=owned)
            return True
        except OSError as e:
            logger.warning(f"Cannot write edit to XMP for {filepath}: {e}")
            return False

    # ── Virtual copies ───────────────────────────────────────────
    #
    # A copy is another edit of the same file — a colour and a black-and-white, a
    # square for the grid and a 9:16 for Stories — not another file. Version 0 is
    # the photograph's own edit, untouched by any of this; copies are 1, 2, ….
    # They live in the catalog and in a small `<file>.copies.json` beside the
    # photograph, which index_folder reads back, so a catalog rebuild keeps them.

    @staticmethod
    def _copies_path(abs_path: Path) -> Path:
        return abs_path.with_suffix(abs_path.suffix + ".copies.json")

    def _read_copies_file(self, abs_path: Path) -> list[dict]:
        import json
        from adjust.develop import normalize

        path = self._copies_path(abs_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for c in raw.get("copies", []) if isinstance(raw, dict) else []:
            try:
                out.append({
                    "version": int(c["version"]),
                    "name": str(c.get("name", ""))[:60],
                    "settings": normalize(c.get("settings")),
                    "export": bool(c.get("export", True)),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return [c for c in out if c["version"] >= 1]

    def _write_copies_file(self, filepath: str) -> None:
        """Mirror a photograph's copies to disk; none left means no file."""
        import json

        abs_path = self.library_root / filepath
        path = self._copies_path(abs_path)
        copies = self.db.list_versions(filepath)
        try:
            if not copies:
                path.unlink(missing_ok=True)
            elif abs_path.exists():
                path.write_text(json.dumps({"copies": copies}, indent=1), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Cannot write copies for {filepath}: {e}")

    def photo_versions(self, filepath: str) -> list[dict]:
        from adjust.develop import normalize

        return [
            {**c, "settings": normalize(c["settings"])} for c in self.db.list_versions(filepath)
        ]

    def create_copy(self, filepath: str, from_version: int = 0, name: str = "") -> dict:
        """Start a new copy from another version's edit (0 is the photo itself)."""
        if from_version == 0:
            start = self.photo_edit(filepath)
        else:
            found = self.db.get_version(filepath, from_version)
            from adjust.develop import normalize
            start = normalize(found["settings"]) if found else {}
        version = self.db.create_version(filepath, start, name.strip()[:60])
        self._write_copies_file(filepath)
        return {"version": version, "settings": start}

    def set_copy_edit(self, filepath: str, version: int, settings: Optional[dict]) -> Optional[dict]:
        from adjust.develop import normalize

        clean = normalize(settings)
        if not self.db.update_version(filepath, version, clean, replace_settings=True):
            return None
        self._write_copies_file(filepath)
        return clean

    def update_copy(self, filepath: str, version: int, name: Optional[str] = None,
                    export: Optional[bool] = None) -> bool:
        ok = self.db.update_version(
            filepath, version, name=None if name is None else name.strip()[:60], export=export
        )
        if ok:
            self._write_copies_file(filepath)
        return ok

    def delete_copy(self, filepath: str, version: int) -> bool:
        ok = self.db.delete_version(filepath, version)
        if ok:
            self._write_copies_file(filepath)
        return ok

    # What a paste may carry that is tied to where things are in the frame: a
    # crop's rectangle and the spots' places. On a frame turned the other way
    # (portrait against landscape) they would land somewhere else entirely.
    _PLACED_KEYS = ("crop_x", "crop_y", "crop_w", "crop_h", "spots")

    def _carried(self, filepath: str, settings: dict, keys=None) -> dict:
        """This photograph's edit with ``keys`` taken from ``settings``.

        ``keys`` are the edit keys being carried; every other part of the
        photograph's own edit stays as it is, so pasting only the level leaves
        the tone alone. With no ``keys`` it is everything but the frame's own
        crop, level and spots: tone suits a burst, but a crop is a composition,
        and the frame it was drawn on is the only one it is known to fit.
        """
        from adjust.develop import normalize

        keys = self._carry_keys(keys)
        own = self.photo_edit(filepath)
        merged = {k: v for k, v in own.items() if k not in keys}
        merged.update({k: v for k, v in settings.items() if k in keys})
        return normalize(merged)

    @staticmethod
    def _carry_keys(keys) -> frozenset:
        from adjust.develop import EDIT_KEYS, OWN_KEYS

        known = set(EDIT_KEYS) | {"style"}
        if keys is None:
            return frozenset(known - set(OWN_KEYS))
        return frozenset(k for k in keys if k in known)

    def _turned(self, filepaths: list[str]) -> dict:
        """Per photograph, True when its EXIF turns it a quarter (a portrait
        frame from a landscape sensor), False when not, absent when unknown."""
        from thumbnails.extract import _batch_get_orientations

        sources = [self.library_root / f for f in filepaths]
        found = _batch_get_orientations(sources)
        return {f: found[str(s)] in (5, 6, 7, 8) for f, s in zip(filepaths, sources) if str(s) in found}

    def _paste(self, source: Optional[str], targets: list[str], settings: Optional[dict], keys=None,
               adjust=None) -> dict:
        """Put ``settings``' ``keys`` on every target. What each had before comes
        back as ``previous``, so the paste can be taken back (:meth:`restore_edits`),
        and what each has now as ``edits``.

        A crop or spots are not put on a frame turned the other way from
        ``source``: those targets keep their own, and are listed as such.

        ``adjust`` is the per-frame step of a paste and match
        (export.paste_match.adjuster): given a target, its edit with the copied
        parts put on, and its edit from before, it gives back the edit to keep and
        a note of what it did, which comes back in ``notes``.
        """
        from adjust.develop import normalize

        clean = normalize(settings)
        carry = self._carry_keys(keys)
        placed = carry & set(self._PLACED_KEYS)
        turned = self._turned([source, *targets]) if placed and source else {}
        written, previous, now, kept_framing, notes = [], {}, {}, [], {}
        for path in targets:
            if self.db.get_photo(path) is None or is_video(path):
                continue                # a video is never developed
            keys_here = carry
            if placed and source in turned and path in turned and turned[path] != turned[source]:
                keys_here = carry - placed
                kept_framing.append(path)
            previous[path] = self.photo_edit(path)
            edit = self._carried(path, clean, keys_here)
            if adjust is not None:
                edit, note = adjust(path, edit, previous[path])
                edit = normalize(edit)
                if note:
                    notes[path] = note
            now[path] = self.set_photo_edit(path, edit)
            written.append(path)
        return {"copied": written, "previous": previous, "edits": now, "kept_framing": kept_framing,
                "settings": clean, "notes": notes}

    def apply_edit_to_burst(self, source: str, settings: Optional[dict], keys=None, adjust=None) -> dict:
        """Put one frame's edit on that frame and on the rest of its burst.

        A burst is one moment shot several times, so the exposure and white
        balance that suit the frame the user settled on usually suit the frames
        beside it; setting eight controls by hand per frame is the reason people
        stop bothering. ``keys`` are what is carried (see :meth:`_carried`).

        Every frame of the burst is written, basket or not. The edit belongs to
        the photograph now, so a frame that has not been chosen for export still
        keeps it — choosing what to export and choosing how are separate
        decisions, and only the first one is the basket's.
        """
        from adjust.develop import normalize

        clean = normalize(settings)
        paths = self.burst_of(source)
        self.set_photo_edit(source, clean)
        result = self._paste(source, [p for p in paths if p != source], clean, keys, adjust)
        return {"source": source, "frames": paths, **result}

    def apply_edit_to_many(self, filepaths: list[str], settings: Optional[dict], keys=None,
                           source: Optional[str] = None, adjust=None) -> dict:
        """Put one edit on a whole selection of photographs: the parts named by
        ``keys`` (see :meth:`_carried`), from the frame ``source`` when it is known,
        each adjusted by ``adjust`` when given (see :meth:`_paste`). What each had
        before comes back too, for an undo."""
        return self._paste(source, filepaths, settings, keys, adjust)

    def restore_edits(self, edits: dict) -> list[str]:
        """Put back the edits a paste replaced: ``{filepath: edit}``, ``{}`` for
        a photograph that had none."""
        restored = []
        for filepath, settings in edits.items():
            if self.db.get_photo(filepath) is None:
                continue
            if settings:
                self.set_photo_edit(filepath, settings)
            else:
                self.clear_photo_edit(filepath)
            restored.append(filepath)
        return restored

    # ── Quality hints ────────────────────────────────────────────

    def folder_hints(self, folder: str, **filters) -> dict:
        """Soft-frame and blown-highlight warnings for the frames on screen.

        The filters are the grid's, for the same reason bursts take them: a hint
        sits next to a thumbnail the user can actually see, and a RAW/JPG pair is
        one frame on screen, not two opinions about it.

        Nothing is stored here. The measurements are the expensive part and they
        are already in the catalog; the reading of them is redone on every call,
        so a threshold can change without re-measuring the library.
        """
        records = self.db.get_photos_in_folder(folder, **filters)
        measurements = self.db.list_photo_stats_for_folder(folder)
        current = {
            path: row
            for path, row in measurements.items()
            if row.get("stats_version") == STATS_VERSION
        }
        # The reference is the folder's, not the filtered view's: what counts as
        # sharp here is a property of the shoot, and turning on a filter should
        # not quietly redefine it.
        baseline = folder_baseline(current.values())
        # Each frame's peers, from the whole folder for the same reason.
        everyone = records if not filters else self.db.get_photos_in_folder(folder)
        peer_medians = peer_sharpness(current, peers(everyone))
        visible = {
            record.filepath: current[record.filepath]
            for record in records
            if record.filepath in current
        }
        computed = compute_hints(visible, baseline, peer_medians)
        return {
            "folder": folder,
            "total": len(records),
            "measured": computed["measured"],
            "median_sharpness": computed["median_sharpness"],
            "soft": sum(1 for hint in computed["hints"].values() if hint["soft_frame"]),
            "blown": sum(1 for hint in computed["hints"].values() if hint["blown_highlights"]),
            "hints": computed["hints"],
        }

    # ── Queries ──────────────────────────────────────────────────

    def get_photo(self, filepath: str) -> Optional[PhotoRecord]:
        return self.db.get_photo(filepath)

    def get_photos_in_folder(self, folder: str, **kwargs) -> list[PhotoRecord]:
        return self.db.get_photos_in_folder(folder, **kwargs)

    def get_folder_stats(self, folder: str) -> dict:
        return self.db.get_folder_stats(folder)

    def get_folder_state(self, folder: str) -> dict:
        """Where the user last stopped in this folder, if anywhere."""
        return self.db.get_folder_state(folder)

    def set_folder_state(self, folder: str, last_filepath: Optional[str]) -> dict:
        """
        Remember where the user stopped.

        The path is stored as given: a photo may be deleted or moved after the
        position is saved, and the client resolves it against the list it is
        actually looking at. Validating it here would cost a lookup on every
        arrow keypress to save nothing.
        """
        return self.db.set_folder_state(folder, last_filepath)

    def get_photo_tags(self, filepath: str) -> list[str]:
        return self.db.get_photo_tags(filepath)

    def get_all_folders(self) -> list[dict]:
        return self.db.get_all_folders()

    def search(self, query: str, sort: str = "", desc: bool = True, owner: str = "local",
               describe: bool = False, folder: str = "") -> list:
        """Photos anywhere in the library matching a query (see :mod:`catalog.search`),
        or only in ``folder`` (exactly that folder) when one is given.

        Returns search Photos, pairs merged; raises SearchError on a query it
        cannot read. A shows: term is scored with the photographs' descriptions
        (analysis.clip) when smart search is set up.
        """
        from analysis import clip
        from catalog import search

        records, tags, edited, basket, mono, denoised = self.db.search_snapshot(owner)
        if folder:
            records = [r for r in records if r.folder == folder]
        scorer = (lambda text: clip.scores(self.db, text)) if clip.available() else None
        return search.search(search.gather(records, tags, edited, basket, mono, denoised), query, sort, desc,
                             scorer=scorer, describe=describe)

    def demoted(self, owner: str = "local"):
        """A test for the photos ``owner`` has demoted: in a folder they demoted
        whole, or by a camera they demoted in that folder. Takes a search Photo
        (whose camera is its pair's)."""
        shoots = self.db.demoted_shoots(owner)

        def test(photo) -> bool:
            folder = photo.record.folder
            return (folder, "*") in shoots or (folder, (photo.record.camera or "").strip()) in shoots
        return test

    def _photos_for(self, owner: str) -> list:
        """Every photo, pairs merged, without the ones ``owner`` demoted: what is
        picked for one person leaves out what they find less important."""
        from catalog import search

        records, tags, edited, basket, mono, denoised = self.db.search_snapshot(owner)
        photos = search.gather(records, tags, edited, basket, mono, denoised)
        demoted = self.demoted(owner)
        return [p for p in photos if not demoted(p)]

    # On this day and the years are read from the whole library, most of a
    # second each, and the Library's front page asks for both every time it is
    # shown — over Wi-Fi, after every folder. Neither has to be up to the second:
    # an answer is kept this long, and is asked again as soon as what this person
    # demoted changes.
    PICKS_SECONDS = 300

    def _picked(self, kind: str, owner: str, arg, work):
        key = (kind, owner, arg, frozenset(self.db.demoted_shoots(owner)))
        now = time.monotonic()
        with self._picks_lock:
            hit = self._picks.get(key)
            if hit is not None and now - hit[0] < self.PICKS_SECONDS:
                return hit[1]
        value = work()
        with self._picks_lock:
            self._picks = {k: v for k, v in self._picks.items() if now - v[0] < self.PICKS_SECONDS}
            self._picks[key] = (now, value)
        return value

    def forget_picks(self) -> None:
        """Dates changed (catalog/clock.py): what was picked by day is picked again."""
        with self._picks_lock:
            self._picks = {}

    def on_this_day(self, today, owner: str = "local") -> list:
        """Earlier years' photographs from ``today``'s date (:mod:`catalog.memories`),
        leaving out the shoots ``owner`` demoted."""
        from catalog import memories

        return self._picked("day", owner, today,
                            lambda: memories.on_this_day(self._photos_for(owner), today))

    def photo_years(self, owner: str = "local") -> list:
        """The years with photographs, the most recent first (:mod:`catalog.review`)."""
        from catalog import review

        return self._picked("years", owner, None, lambda: review.years(self._photos_for(owner)))

    def perspective_view(self, pid: int) -> Optional[tuple]:
        """``(folders, patterns, people)`` for perspective ``pid``: the shoots it
        is about, whose names it is about, and every photo's named people as
        ``{folder: {base_name: {names}}}``. None for no such perspective."""
        from catalog import perspectives

        every = self.db.list_perspectives()
        chosen = perspectives.find(every, pid)
        if chosen is None:
            return None
        people = perspectives.folder_people(self.db.people_in_photos())
        return perspectives.folders_of(chosen, every, people), chosen.get("people") or [], people

    def perspective_folders(self, pid: int) -> Optional[set]:
        """The folders perspective ``pid`` is about (:mod:`catalog.perspectives`),
        or None for no such perspective."""
        view = self.perspective_view(pid)
        return view[0] if view is not None else None

    def year_review(self, year: int, cards: list, owner: str = "local", perspective: int = 0) -> dict:
        """One year's page: numbers, best frames, people, shoots and places
        (:mod:`catalog.review`), without the shoots ``owner`` demoted, and only
        the shoots ``perspective`` is about when one is given."""
        from catalog import perspectives, review

        photos = self._photos_for(owner)
        faces = self.db.named_faces()
        view = self.perspective_view(perspective) if perspective else None
        if view is not None:
            folders, patterns, people = view
            # Its shoots, whole — the food and the venue have no one in them and
            # are still the dinner — but not a frame of only people the
            # perspective is not about: a school friend at a family party is in
            # a family shoot, and is not what the family's year is of. That holds
            # for people nobody has named too (a band on stage, a crowd), as long
            # as their faces are clear enough to be someone; a faint face in the
            # background is left to the shoot.
            from catalog.database import CLEAR_FACE_QUALITY
            from faces.detect import MIN_GROUP_PX

            strangers = self.db.photos_with_unnamed_people(CLEAR_FACE_QUALITY, MIN_GROUP_PX)

            def about_others(p):
                names = people.get(p.record.folder, {}).get(p.record.base_name) or set()
                if any(perspectives.matches(n, patterns) for n in names):
                    return False
                return bool(names) or (p.record.folder, p.record.base_name) in strangers

            photos = [p for p in photos if p.record.folder in folders and not about_others(p)]
            faces = [f for f in faces if perspectives.matches(f["name"], patterns)]
        in_year = [f for p in photos if p.date[:4] == str(year) for f in p.filepaths]
        return review.review(photos, year, self.db.photo_stats_brief(), faces, cards,
                             looks=self.db.clip_vectors(in_year))

    # ── XMP sync ─────────────────────────────────────────────────

    def _sync_to_xmp(self, filepath: str):
        """Write current DB state to XMP sidecar."""
        record = self.db.get_photo(filepath)
        if not record:
            return

        abs_path = self.library_root / filepath
        if not abs_path.exists():
            logger.warning(f"Cannot sync XMP — file missing: {filepath}")
            return

        # Read first, then write the cull state over it. The sidecar carries
        # more than the catalog does — an edit recorded at export time, and
        # anything another editor put there — and a star click must not be what
        # removes it.
        data = read_xmp(abs_path)
        data.rating = record.rating
        data.flag = record.flag
        data.color_label = record.color_label
        data.tags = self.db.get_photo_tags(filepath)
        write_xmp(abs_path, data)

    def rebuild_from_xmp(self, extensions: list[str]) -> dict:
        """
        Rebuild the entire catalog by scanning all folders.
        Works with any folder structure depth.
        """
        import os

        ext_set = {ext.lower() for ext in extensions}
        total = 0
        root = self.library_root.resolve()

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            has_images = any(
                Path(f).suffix.lower() in ext_set
                for f in filenames
                if not f.startswith(".")
            )
            if not has_images:
                continue

            rel = str(Path(dirpath).resolve().relative_to(root)).replace("\\", "/")
            if rel == ".":
                continue

            result = self.index_folder(rel, extensions)
            total += result.get("indexed", 0)

        return {"total_indexed": total}
