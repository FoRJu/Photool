"""
Bring an old collection of photographs into the library, in two steps.

    python -m tools.archive_import plan  "E:\\Old Photos"           (writes archive-plan.csv)
    python -m tools.archive_import apply archive-plan.csv  --dry-run
    python -m tools.archive_import apply archive-plan.csv

Run it from the ``backend`` folder on the computer that holds the library. It
reads ``config.yaml`` for where the library is, the same as the server. It
adds nothing to the server: there is no page or address that reads a folder
on this computer, so nothing on the network can ask for one.

**plan** reads the source (every file's size, a fingerprint of its first and
last 64 KB, and the camera's date where there is one) and writes one row per
folder to a CSV for a person to look over in Excel before anything is copied.
The rules are in importer/archive_plan.py: folders already laid out by day join
the library's own year folders, scans and everything else go under
``Archive/<year>/<date> <name>``, camera dumps are filed by day, copies found
twice are taken once, and lab data, music and caches are left out. Rows marked
``review`` need a look, most often a scan whose folder name has no date.
Change ``action``, ``dest`` or ``set_date`` in any row, save as CSV, and apply.
Folders this collection holds that are not family photographs (work files, a
music library), and names of its own that say nothing about what is in a folder,
go in ``archive-rules.txt`` beside the plan (see importer.archive_plan.Rules): it
stays on this computer, like the plan and the log.

A folder's date is the start of the last part of its ``dest``
(``Archive/2004/2004-06 WEDDING``): change it there. The ``date`` column is only
to read, because Excel rewrites dates in it when it saves.

**apply** copies what the plan says: never over a file, every copy checked
against its source's SHA-256, a scan given its folder's date (``set_date``),
favourites given stars, and every new folder indexed by the running server so
it is in the library at once. Run again, it carries on where it stopped. Every
file it touched is listed in ``archive-log.csv``.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config import RASTER_EXTENSIONS, VIDEO_EXTENSIONS, load_config  # noqa: E402
from importer.archive_plan import (  # noqa: E402
    NO_RULES, Row, Rules, SourceFile, SourceFolder, plan_folders, row_order, safe_name,
)

PROBE = 64 * 1024
STARS = 4


# ── Reading the source ────────────────────────────────────────────────

def fingerprint(path: Path, size: int) -> str:
    """Size and a hash of the first and last 64 KB: the same photograph copied
    twice has the same one, and reading it costs two small reads, not the file."""
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(PROBE))
        if size > 2 * PROBE:
            f.seek(-PROBE, os.SEEK_END)
            h.update(f.read(PROBE))
    return h.hexdigest()[:24]


def _exiftool() -> Optional[str]:
    try:
        import exiftool_path
        exiftool_path.ensure_on_path()
    except Exception:
        pass
    return shutil.which("exiftool")


def _parse_when(text) -> Optional[datetime]:
    if not text or not isinstance(text, str) or text.startswith("0000"):
        return None
    try:
        return datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y:%m:%d")
        except ValueError:
            return None


def read_exif(paths: list) -> dict:
    """``{path: (taken, make, software)}`` for what has a date, by exiftool in
    batches, or by Pillow (JPEG and TIFF) when there is no exiftool."""
    out = {}
    tool = _exiftool()
    if tool:
        from exif_batches import batches

        for batch in batches(paths):
            try:
                done = subprocess.run(
                    [tool, "-json", "-fast", "-charset", "filename=utf8", "-DateTimeOriginal",
                     "-CreateDate", "-Make", "-Software", *[str(p) for p in batch]],
                    capture_output=True, timeout=600)
                items = json.loads(done.stdout.decode("utf-8", "replace") or "[]")
            except (subprocess.SubprocessError, ValueError, OSError):
                continue
            for item in items:
                key = str(Path(item.get("SourceFile", ""))).replace("\\", "/").casefold()
                taken = _parse_when(item.get("DateTimeOriginal")) or _parse_when(item.get("CreateDate"))
                out[key] = (taken, str(item.get("Make") or ""), str(item.get("Software") or ""))
        return out
    from PIL import Image

    for p in paths:
        if Path(p).suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        try:
            with Image.open(p) as im:
                exif = im.getexif()
                sub = exif.get_ifd(0x8769)
                taken = _parse_when(sub.get(36867)) or _parse_when(exif.get(306))
                out[str(p).replace("\\", "/").casefold()] = (taken, str(exif.get(271) or ""),
                                                             str(exif.get(305) or ""))
        except Exception:
            continue
    return out


def media_extensions(config) -> set:
    return {e.lower() for e in config.photo_extensions}


def scan_source(source: Path, media: set, cache: dict, say=print) -> list:
    """Every folder under ``source`` with what is in it. ``cache`` (path → what was
    read) is filled in and reused: a second run reads only what changed."""
    folders = []
    todo_exif = []
    by_path = {}
    count = 0
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        rel = Path(dirpath).relative_to(source).as_posix()
        folder = SourceFolder(rel=rel)
        for name in sorted(filenames):
            if name.startswith(".") or name.startswith("._"):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in media:
                folder.other += 1
                continue
            try:
                st = path.stat()
            except OSError:
                folder.other += 1
                continue
            key = str(path)
            hit = cache.get(key)
            if hit and hit["size"] == st.st_size and hit["mtime"] == int(st.st_mtime):
                info = hit
            else:
                try:
                    fp = fingerprint(path, st.st_size)
                except OSError:
                    folder.other += 1
                    continue
                info = {"size": st.st_size, "mtime": int(st.st_mtime), "fp": fp, "exif": None}
                cache[key] = info
            if info.get("exif") is None:
                todo_exif.append(path)
            f = SourceFile(name=name, size=st.st_size,
                           modified=datetime.fromtimestamp(st.st_mtime).date(),
                           fingerprint=info["fp"])
            by_path[key] = (f, info)
            folder.media.append(f)
            count += 1
            if count % 2000 == 0:
                say(f"  read {count:,} files…")
        if folder.media or folder.other:
            folders.append(folder)
    if todo_exif:
        say(f"  reading the dates of {len(todo_exif):,} files…")
        found = read_exif(todo_exif)
        for p in todo_exif:
            taken, make, software = found.get(str(p).replace("\\", "/").casefold(), (None, "", ""))
            cache[str(p)]["exif"] = [taken.isoformat() if taken else "", make, software]
    for key, (f, info) in by_path.items():
        stamp, make, software = info.get("exif") or ["", "", ""]
        f.taken = datetime.fromisoformat(stamp) if stamp else None
        f.make, f.software = make, software
    return folders


def _cache_path(plan_csv: Path) -> Path:
    return plan_csv.with_suffix(".cache.json")


def _load_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("files", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(path: Path, source: Path, files: dict) -> None:
    path.write_text(json.dumps({"source": str(source), "files": files}), encoding="utf-8")


# ── plan ──────────────────────────────────────────────────────────────

def run_plan(source: Path, out: Path, config, archive: str = "Archive", say=print,
             rules: Rules = NO_RULES) -> list:
    from importer.copy import build_library_index

    media = media_extensions(config)
    cache_file = _cache_path(out)
    cache = _load_cache(cache_file)
    say(f"Reading {source} …")
    folders = scan_source(source, media, cache, say)
    _save_cache(cache_file, source, cache)
    say(f"Looking at what {config.library_root} already holds …")
    library = build_library_index(config.library_root, media)
    rows = plan_folders(folders, library, config.folder_pattern, archive, rules)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        fh.write(f"# source: {source}\n")
        w = csv.DictWriter(fh, fieldnames=Row.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())
    return rows


def summarize(rows: list) -> str:
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r.action][0] += 1
        by[r.action][1] += r.new if r.action != "skip" else r.files
    review = sum(1 for r in rows if r.review and r.action != "skip")
    dups = sum(r.duplicates for r in rows)
    lib = sum(r.in_library for r in rows)
    lines = [f"  {a:8} {n:5} folders  {f:7,} files" for a, (n, f) in sorted(by.items())]
    lines.append(f"  copies found twice: {dups:,}   already in the library: {lib:,}")
    lines.append(f"  rows to look at (review = yes): {review}")
    return "\n".join(lines)


# ── apply ─────────────────────────────────────────────────────────────

def read_plan(path: Path) -> tuple:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
        # Excel keeps the first line as a cell: quoted, with the other columns' commas.
        line = first.strip().strip(",").strip('"').strip()
        source = line.split(":", 1)[1].strip() if line.startswith("# source:") else ""
        if not source:
            fh.seek(0)
        rows = []
        for d in csv.DictReader(fh):
            row = Row(action=(d.get("action") or "").strip().lower(), source=d.get("source") or "")
            row.dest = (d.get("dest") or "").strip().strip("/\\").replace("\\", "/")
            row.date = (d.get("date") or "").strip()
            row.precision = (d.get("precision") or "").strip()
            row.set_date = (d.get("set_date") or "no").strip().lower()
            rows.append(row)
    return Path(source) if source else None, rows


_DEST_DATE = __import__("re").compile(r"^((?:19|20)\d\d(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?)?)(?:\s|$)")


def dest_date(dest: str) -> str:
    """The date a folder is given, read from the start of its name
    (``Archive/2002/2002-10 Lakeside`` → ``2002-10``). Read from there and not
    from the date column, which Excel turns into its own dates when it saves."""
    m = _DEST_DATE.match(dest.rsplit("/", 1)[-1])
    return m.group(1) if m else ""


def _exif_stamp(value: str) -> Optional[str]:
    """The date a scan is given: its day, else the first of its month, else
    the first of its year, at noon."""
    parts = value.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        date(y, m, d)
    except (ValueError, IndexError):
        return None
    return f"{y:04d}:{m:02d}:{d:02d} 12:00:00"


def _split_dest(template: str, when: Optional[datetime], fallback: str) -> str:
    if when is None:
        return fallback
    return template.format(year=f"{when.year:04d}", month=f"{when.month:02d}", day=f"{when.day:02d}",
                           date=when.date().isoformat())


def _existing_day_folder(library_root: Path, dest: str) -> str:
    """The library's folder for this day when one is there under another name
    (``2010-02-21`` for ``2010-02-21 Picnic``, or the other way round): the day
    stays one folder."""
    parent, leaf = (dest.rsplit("/", 1) + [""])[:2] if "/" in dest else ("", dest)
    day = leaf[:10]
    try:
        names = [e.name for e in os.scandir(library_root / parent) if e.is_dir()]
    except OSError:
        return dest
    if leaf in names:
        return dest
    same = [n for n in names if n[:10] == day and (len(n) == 10 or n[10] == " ")]
    return f"{parent}/{same[0]}" if len(same) == 1 else dest


class Server:
    """The running Photool on this computer, asked to index what was copied."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.ok = None

    def _call(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read() or b"{}")

    def reachable(self) -> bool:
        if self.ok is None:
            try:
                self._call("GET", "/api/catalog/folders")
                self.ok = True
            except (urllib.error.URLError, OSError, ValueError):
                self.ok = False
        return self.ok

    def index(self, folder: str) -> None:
        self._call("POST", "/api/catalog/index/" + urllib.parse.quote(folder, safe="/"))

    def rate(self, filepath: str, stars: int) -> None:
        self._call("PUT", "/api/catalog/rate", {"filepath": filepath, "rating": stars})


def run_apply(plan: Path, config, dry_run: bool = False, server: Optional[Server] = None,
              stars: int = STARS, say=print, source: Optional[Path] = None) -> dict:
    from importer.copy import _copy_with_hash, build_library_index, sha256_file

    found_source, rows = read_plan(plan)
    source = source or found_source
    if source is None or not source.is_dir():
        raise SystemExit(f"The plan does not say where its photographs are, or {source} is not there.")
    library_root = Path(config.library_root)
    media = media_extensions(config)
    cache = _load_cache(_cache_path(plan))
    folders = {f.rel: f for f in scan_source(source, media, cache, say)}
    _save_cache(_cache_path(plan), source, cache)
    library = build_library_index(library_root, media)

    # The archive's own folder, from the plan: everything else is the library's
    # day tree, where a day already there under another name is joined.
    archive_root = next((r.dest.split("/", 1)[0] for r in rows
                         if r.action in ("archive", "star") and r.dest), "Archive")

    # 1. Every file's destination, in the order the plan keeps copies by.
    seen = {}                     # fingerprint → library-relative path it goes to
    by_name = {}                  # (name, size) → library-relative path
    planned = defaultdict(list)   # dest folder → [(source path, SourceFile, row)]
    stars_for = []                # library-relative paths to star
    starred = []
    log = []
    counts = defaultdict(int)
    for row in sorted(rows, key=lambda r: row_order(r)):
        if row.action == "skip" or not row.action:
            continue
        folder = folders.get(row.source)
        if folder is None:
            log.append((row.source, "", "missing", "the folder is not in the source any more"))
            continue
        if row.action == "star":
            starred.append((row, folder))
            continue
        files = sorted(folder.media, key=lambda f: (f.ext in RASTER_EXTENSIONS, f.name.lower()))
        for f in files:
            src = source / folder.rel / f.name
            lib_hit = next((p for size, p in library.get(f.name.lower(), ()) if size == f.size), None)
            if f.fingerprint in seen:
                counts["duplicate"] += 1
                log.append((str(src), seen[f.fingerprint], "duplicate", "the same photograph is copied from elsewhere"))
                continue
            if lib_hit is not None:
                rel = Path(os.path.relpath(lib_hit, library_root)).as_posix()
                seen[f.fingerprint] = rel
                by_name[(f.name.lower(), f.size)] = rel
                counts["in library"] += 1
                log.append((str(src), rel, "in library", ""))
                continue
            if row.action == "split" or "{" in row.dest:
                fallback = f"{row.dest.split('/', 1)[0]}/Undated/{safe_name(Path(row.source).name) or 'Unnamed'}"
                dest = _split_dest(row.dest, f.taken, fallback)
            else:
                dest = row.dest
            if not dest:
                log.append((str(src), "", "skipped", "no dest in the plan"))
                continue
            if dest.split("/", 1)[0] != archive_root:
                dest = _existing_day_folder(library_root, dest)
            planned[dest].append((src, f, row))
            seen[f.fingerprint] = dest + "/" + f.name          # final name settled below
            by_name[(f.name.lower(), f.size)] = seen[f.fingerprint]

    # 2. Names inside each folder: a picture never lands on another, and an
    # edited scan or Photoshop document never takes the name of the JPEG or RAW
    # beside it (they would be shown as one photograph, the JPEG).
    jobs = []
    for dest, items in planned.items():
        try:
            taken_names = {e.name.lower() for e in os.scandir(library_root / dest)}
        except OSError:
            taken_names = set()
        stems = {n.rsplit(".", 1)[0] for n in taken_names
                 if "." + n.rsplit(".", 1)[-1] not in RASTER_EXTENSIONS}
        for src, f, row in items:
            name = f.name
            if f.ext in RASTER_EXTENSIONS and f.stem.lower() in stems:
                name = f"{f.stem}-{f.ext[1:]}{f.ext}"
            if f.ext not in RASTER_EXTENSIONS:
                stems.add(f.stem.lower())
            target = library_root / dest / name
            if name.lower() in taken_names:
                if target.exists() and target.stat().st_size == f.size and fingerprint(target, f.size) == f.fingerprint:
                    jobs.append((src, f, row, dest, name, "exists"))
                    taken_names.add(name.lower())
                    seen[f.fingerprint] = f"{dest}/{name}"
                    continue
                prefix = safe_name(src.parent.name).replace(" ", "_") or "copy"
                name = f"{prefix}_{name}"
                n = 2
                while name.lower() in taken_names:
                    name = f"{prefix}_{n}_{f.name}"
                    n += 1
            taken_names.add(name.lower())
            seen[f.fingerprint] = f"{dest}/{name}"
            by_name[(f.name.lower(), f.size)] = seen[f.fingerprint]
            jobs.append((src, f, row, dest, name, "copy"))

    # 3. Favourites: the original found elsewhere gets stars; one not found is copied too.
    by_plain_name = {}
    for (n, _), path in by_name.items():
        by_plain_name.setdefault(n, path)
    for row, folder in starred:
        for f in folder.media:
            src = source / folder.rel / f.name
            target = seen.get(f.fingerprint) or by_name.get((f.name.lower(), f.size))
            if target is None:
                lib_hit = next((p for size, p in library.get(f.name.lower(), ()) if size == f.size), None)
                if lib_hit is not None:
                    target = Path(os.path.relpath(lib_hit, library_root)).as_posix()
            if target is None:
                target = by_plain_name.get(f.name.lower())
            if target is not None:
                stars_for.append(target)
                log.append((str(src), target, "star", f"the original gets {stars} stars"))
                continue
            dest = row.dest or "Archive/Starred"
            jobs.append((src, f, row, dest, f.name, "copy"))
            stars_for.append(f"{dest}/{f.name}")
            seen[f.fingerprint] = f"{dest}/{f.name}"

    total = sum(1 for j in jobs if j[5] == "copy")
    size = sum(j[1].size for j in jobs if j[5] == "copy")
    left = [f"{counts['duplicate']:,} copies found twice"] if counts["duplicate"] else []
    if counts["in library"]:
        left.append(f"{counts['in library']:,} already in the library")
    say(f"{total:,} files to copy ({size / 1e9:.1f} GB) into {len(planned) + len(starred)} folders"
        + (f"; left out: {' and '.join(left)}." if left else "."))
    if dry_run:
        for src, f, row, dest, name, what in jobs:
            log.append((str(src), f"{dest}/{name}", "would copy" if what == "copy" else "exists", row.action))
        _write_log(plan, log)
        say(f"Nothing was copied (dry run). What would happen is in {plan.with_name('archive-log.csv')}.")
        return {"copy": total, "bytes": size, "dry_run": True}

    # 4. Copy, each checked against its source.
    done = 0
    dated = defaultdict(list)
    touched = set()
    started = time.monotonic()
    for src, f, row, dest, name, what in jobs:
        target = library_root / dest / name
        rel = f"{dest}/{name}"
        if what == "exists":
            log.append((str(src), rel, "exists", "copied before"))
            touched.add(dest)
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"{rel} appeared while copying")
            partial = target.with_name(target.name + ".part")
            digest = _copy_with_hash(src, partial)
            if sha256_file(partial) != digest:
                partial.unlink(missing_ok=True)
                raise OSError("the copy does not match its source")
            partial.rename(target)
            for side in (src.with_name(src.name + ".xmp"), src.with_name(src.stem + ".xmp")):
                if side.exists():
                    side_target = target.with_name(target.name + ".xmp") if side.name.endswith(
                        src.name + ".xmp") else target.with_name(Path(name).stem + ".xmp")
                    if not side_target.exists():
                        shutil.copy2(side, side_target)
            log.append((str(src), rel, "copied", row.action))
            touched.add(dest)
            when = dest_date(dest)
            if row.set_date == "yes" and when and f.ext not in VIDEO_EXTENSIONS and f.ext != ".bmp":
                dated[when].append(target)
        except OSError as e:
            log.append((str(src), rel, "failed", str(e)))
            counts["failed"] += 1
        done += 1
        if done % 200 == 0:
            rate = done / max(1e-6, time.monotonic() - started)
            say(f"  {done:,} / {total:,} copied ({rate:.0f} a second)")

    # 5. A scan's date is the folder's, not the scanner's.
    if dated:
        tool = _exiftool()
        if not tool:
            say("exiftool is not installed: the scans keep the dates their files had. "
                "Install it and run apply again to date them.")
        else:
            from exif_batches import batches

            for value, paths in dated.items():
                stamp = _exif_stamp(value)
                if not stamp:
                    continue
                for batch in batches(paths):
                    subprocess.run([tool, "-overwrite_original", "-P", "-q", "-m",
                                    f"-DateTimeOriginal={stamp}", f"-CreateDate={stamp}",
                                    *[str(p) for p in batch]], capture_output=True, timeout=600)
            say(f"Gave {sum(len(p) for p in dated.values()):,} scans the date of their folder.")

    # 6. Into the library, and the favourites' stars.
    server = server or Server("http://127.0.0.1:8000")
    if touched or stars_for:
        if server.reachable():
            for i, folder in enumerate(sorted(touched), 1):
                try:
                    server.index(folder)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    log.append(("", folder, "not indexed", str(e)))
                if i % 50 == 0:
                    say(f"  indexed {i} / {len(touched)} folders")
            rated = 0
            for path in dict.fromkeys(stars_for):
                try:
                    server.rate(path, stars)
                    rated += 1
                except (urllib.error.URLError, OSError, ValueError):
                    log.append(("", path, "not starred", "not in the catalog yet: open its folder and run apply again"))
            say(f"Indexed {len(touched)} folders; {rated} favourites given {stars} stars.")
        else:
            say("Photool is not running, so nothing was indexed or starred. Start it and run "
                "apply again: what is copied is not copied twice, and the rest is done then.")
    _write_log(plan, log)
    say(f"Done: {sum(1 for l in log if l[2] == 'copied'):,} copied, {counts['failed']} failed. "
        f"Every file is listed in {plan.with_name('archive-log.csv')}.")
    return {"copied": sum(1 for l in log if l[2] == "copied"), "failed": counts["failed"],
            "folders": sorted(touched), "starred": stars_for}


def _write_log(plan: Path, log: list) -> None:
    with open(plan.with_name("archive-log.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "library", "what", "detail"])
        w.writerows(log)


# ── The command ───────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.archive_import",
        description="Bring an old collection of photographs into the library: plan, look over, apply.")
    parser.add_argument("--config", help="config.yaml (default: the one the server uses)")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("plan", help="read a folder and write what would be done to a CSV")
    p.add_argument("source", help="the folder the old collection is in")
    p.add_argument("--out", default="archive-plan.csv", help="the plan to write (default archive-plan.csv)")
    p.add_argument("--archive", default="Archive", help="the library folder for everything not by day")
    p.add_argument("--rules", help="this collection's own rules (default: archive-rules.txt beside the plan, "
                                   "when there is one): folders to skip, names that say nothing, favourites")
    a = sub.add_parser("apply", help="copy what a (reviewed) plan says")
    a.add_argument("plan", help="the plan CSV")
    a.add_argument("--source", help="where the old collection is, if the plan's first line was lost")
    a.add_argument("--dry-run", action="store_true", help="say what would be copied, copy nothing")
    a.add_argument("--server", default="http://127.0.0.1:8000", help="the running Photool, to index into")
    a.add_argument("--stars", type=int, default=STARS, help="stars for a favourite (default 4)")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.step == "plan":
        source = Path(args.source)
        if not source.is_dir():
            parser.error(f"{source} is not a folder")
        out = Path(args.out)
        rules_file = Path(args.rules) if args.rules else out.with_name("archive-rules.txt")
        rules = NO_RULES
        if rules_file.is_file():
            rules = Rules.read(rules_file)
            print(f"Rules from {rules_file}: {len(rules.skip)} to skip, {len(rules.generic)} names "
                  f"that say nothing, {len(rules.star)} favourites folders")
        elif args.rules:
            parser.error(f"{rules_file} is not there")
        rows = run_plan(source, out, config, args.archive, rules=rules)
        print(f"\nThe plan is in {out.resolve()}:")
        print(summarize(rows))
        print("\nOpen it in Excel and look at the rows marked review. Change action or dest where you "
              "want (a date goes at the start of dest's last part: Archive/2004/2004-06 WEDDING), "
              "save as CSV, then:\n"
              f"  python -m tools.archive_import apply {out} --dry-run")
        return 0
    run_apply(Path(args.plan), config, dry_run=args.dry_run, server=Server(args.server), stars=args.stars,
              source=Path(args.source) if args.source else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
