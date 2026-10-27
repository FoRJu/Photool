"""
Culling with an agent: what is known about each frame, cheap pictures to look at,
and proposals the person accepts.

A vision model culling a folder of a thousand frames cannot afford to develop
each RAW to look at it, and should not spend its looking on what the library
already knows. So:

* ``cull.signals`` hands over everything measured: sharpness against the shoot,
  blown highlights, the burst a frame belongs to and its keeper, closed eyes,
  who is in it, and the *scene* (frames close together in time), so the model
  can drop technical failures without looking and compare like with like.
* ``render.thumbnail`` and ``render.contact_sheet`` show the camera's own JPEG
  from the thumbnail cache, never a develop: a sheet of a scene's frames, each
  numbered, is one picture for the model to compare and costs about what one
  preview does.
* ``cull.propose`` records the model's answer with its reason as a *proposal*
  (catalog/proposals.py). The person sees them in the grid and accepts or
  dismisses them; accepting never overwrites a mark the person made, and a pass
  can be undone. ``cull.evaluate`` compares proposals with a folder the person
  has already culled, which is how an agent earns trust before it is used.
"""

import io
import math
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from catalog import proposals as store
from config import is_video
from .library import FolderFilterParams, FolderParams
from .registry import CommandContext, CommandError, Rendered, command

_FLAG = {1: "pick", 0: None, -1: "reject"}


def _when(capture_date: str) -> Optional[datetime]:
    """A capture date as the catalog keeps it ("2026:03:08 12:42:46")."""
    text = (capture_date or "").strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == ":" and text[7] == ":":
        text = f"{text[:4]}-{text[5:7]}-{text[8:]}"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


# ── What is known about each frame ───────────────────────────────────

class SignalsParams(FolderFilterParams):
    scene_gap_seconds: int = Field(
        120, ge=5, le=3600,
        description="A gap longer than this between two frames starts a new scene",
    )
    blind: bool = Field(False, description="Leave out the person's own marks and any proposal: for "
                                           "culling a folder the person has culled, to evaluate against")
    limit: int = Field(300, ge=1, le=2000)
    offset: int = Field(0, ge=0)


@command("cull.signals", params=SignalsParams)
def signals(ctx: CommandContext, params: SignalsParams):
    """Everything the library knows about each frame of a folder, for culling it without
    looking at every frame: time and scene (frames close in time), the burst it belongs to
    and whether it is the burst's suggested keeper, sharpness as a rank among the frame's
    peers (the same camera, within half a minute, at a like focal length; 0 softest, 1
    sharpest; absent with too few peers) with a soft-frame warning, blown highlights, brightness, faces,
    named people, closed eyes (bursts only), the person's own marks and any pending
    proposal. ``kind`` is what the person said the shoot is: "school" (culled to share: about a
    quarter kept, a frame of family alone left out) or "family" (most frames with people
    kept); absent when not said. A video says so with ``video`` (its length in seconds); its
    measurements are of one frame of it, so it has no sharpness verdict. A key that is false, empty or unknown is left out to keep the answer short;
    ``measured`` false means run cull.measure first. ``scenes`` summarises each scene. In
    time order, paged with limit/offset."""
    from analysis import eyes

    catalog, db = ctx.catalog, ctx.catalog.db
    filters = params.as_kwargs()
    records = catalog.get_photos_in_folder(params.folder, **filters)
    stats = db.list_photo_stats_for_folder(params.folder)
    hints = catalog.folder_hints(params.folder, **filters)["hints"]
    bursts = catalog.find_bursts(params.folder, **filters)

    burst_of, keepers, closed = {}, set(), set()
    for number, group in enumerate(bursts["groups"], 1):
        keepers.add(group["keep"])
        closed.update(group.get("closed") or [])
        for member in group["members"]:
            burst_of[member["filepath"]] = number
    eyes_checked = set(eyes.frames_of(db, list(burst_of)))

    # Faces are found on whichever file of a RAW+JPEG pair the scan read, so they
    # are matched to the frame by its base name.
    faces_by_base: dict[str, list] = {}
    for face in db.list_faces_for_folder(params.folder):
        faces_by_base.setdefault(Path(face["filepath"]).stem, []).append(face)
    names = db.cluster_names({f["cluster_id"] for fs in faces_by_base.values() for f in fs})

    # Sharpness is ranked among the frame's peers (analysis.hints.peers: the same
    # camera, moments apart, at a like focal length), not the whole shoot, where a
    # telephoto frame always ranked near the bottom.
    from analysis.hints import MIN_PEERS, peers

    peer_map = peers(records if not filters else catalog.get_photos_in_folder(params.folder))

    def rank(path: str) -> Optional[float]:
        mine = stats.get(path, {}).get("sharpness")
        values = sorted(float(stats[p]["sharpness"]) for p in peer_map.get(path, [])
                        if stats.get(p, {}).get("sharpness") is not None)
        if mine is None or len(values) < MIN_PEERS:
            return None
        import bisect
        return round(bisect.bisect_left(values, float(mine)) / len(values), 2)

    proposed = {p["filepath"]: p for p in store.select(db, folder=params.folder)}

    timed = []
    for r in records:
        when = _when(r.capture_date)
        if when is None and stats.get(r.filepath, {}).get("source_mtime"):
            when = datetime.fromtimestamp(int(stats[r.filepath]["source_mtime"]))
        timed.append((when, r))
    timed.sort(key=lambda t: (t[0] is None, t[0] or datetime.min, t[1].filename))

    rows, scene, last = [], 0, None
    for when, r in timed:
        if scene == 0 or when is None or last is None or (when - last).total_seconds() > params.scene_gap_seconds:
            scene += 1
        last = when or last
        s = stats.get(r.filepath)
        hint = hints.get(r.filepath)
        faces = [f for f in faces_by_base.get(r.base_name, []) if float(f["score"] or 0) >= eyes.MIN_SCORE]
        people = sorted({names.get(f["cluster_id"], "") for f in faces} - {""})
        p = proposed.get(r.filepath)
        exposure = " ".join(str(v) for v in (r.shutter_speed, r.aperture,
                                               f"ISO{r.iso}" if r.iso else "", r.focal_length) if v)
        row = {
            "filepath": r.filepath,
            "time": when.strftime("%H:%M:%S") if when else None,
            "scene": scene,
            "burst": burst_of.get(r.filepath),
            "burst_keeper": (r.filepath in keepers) if r.filepath in burst_of else None,
            "exposure": exposure,
            "measured": bool(hint is not None),
            "sharpness_rank": rank(r.filepath) if s and hint is not None else None,
            "soft": hint.get("soft_frame") if hint else None,
            "blown": hint.get("blown_highlights") if hint else None,
            "brightness": round(float(s["mean_lum"]), 1) if s and hint is not None else None,
            "faces": len(faces),
            "people": people,
            "eyes_closed": (r.filepath in closed) if r.filepath in eyes_checked else None,
        }
        if is_video(r.extension):
            row.update(video=round(r.duration, 1) or True, sharpness_rank=None, soft=None, blown=None)
        if not params.blind:
            row.update(rating=r.rating, flag=_FLAG.get(r.flag), label=r.color_label or None,
                       proposal=({k: p[k] for k in ("rating", "flag", "color_label", "reason", "status")
                                  if p[k] is not None} if p else None))
        rows.append(row)

    scenes = {}
    for row in rows:
        s = scenes.setdefault(row["scene"], {"scene": row["scene"], "start": row["time"], "end": row["time"],
                                             "frames": 0, "bursts": set()})
        s["frames"] += 1
        s["end"] = row["time"] or s["end"]
        if row["burst"]:
            s["bursts"].add(row["burst"])
    for s in scenes.values():
        s["bursts"] = len(s["bursts"])

    page = rows[params.offset:params.offset + params.limit]
    compact = [_compact(row) for row in page]
    from places.store import notes

    return {
        "folder": params.folder,
        # What the person said this shoot is (the folder's About): "school" or
        # "family", which decides how it is culled; None when not said.
        "kind": (notes(db).get(params.folder) or {}).get("kind") or None,
        "total": len(rows),
        "offset": params.offset,
        "measured": sum(1 for row in rows if row["measured"]),
        "bursts": len(bursts["groups"]),
        "eyes_checked": len(eyes_checked),
        "with_faces": sum(1 for row in rows if row["faces"]),
        "scenes": list(scenes.values()),
        "photos": compact,
    }


# Kept even when 0 or false, because there they still say something: the softest
# frame ranks 0.0, and eyes_closed false means looked at and open.
_KEEP_FALSY = {"filepath", "scene", "sharpness_rank", "eyes_closed"}


def _compact(row: dict) -> dict:
    """A frame's signals without the keys that say nothing: unknown, false or empty
    (measured is only there when false)."""
    out = {}
    for key, value in row.items():
        if key == "measured":
            if not value:
                out[key] = False
        elif value is None:
            continue
        elif value or key in _KEEP_FALSY:
            out[key] = value
    return out


class MeasureParams(FolderFilterParams):
    faces: bool = Field(True, description="Also look at the faces in the folder's bursts, for closed eyes "
                                          "and the burst keeper. Slower: seconds per burst frame")


@command("cull.measure", params=MeasureParams, writes=True)
def measure(ctx: CommandContext, params: MeasureParams):
    """Measure a folder so cull.signals has something to say: sharpness, exposure and the
    hash that groups bursts, from the camera's thumbnails (fast; frames already measured
    are skipped), and with ``faces`` the eyes in every burst frame."""
    from analysis.service import analyze_folder

    if ctx.cache is None:
        raise CommandError("No thumbnail cache in this server")
    result = analyze_folder(ctx.catalog.db, ctx.cache, ctx.config.library_root, params.folder)
    if result.get("error"):
        raise CommandError(f"{result['error']}: {params.folder}")
    out = {"measured": {k: result[k] for k in ("total", "analyzed", "skipped", "failed")}}
    if params.faces:
        found = ctx.catalog.check_burst_faces(params.folder, ctx.config.models_dir, ctx.cache,
                                              **params.as_kwargs())
        out["bursts"] = len(found["groups"])
        out["eyes_checked_now"] = found.get("measured_now", 0)
    return out


class BestParams(FolderFilterParams):
    count: int = Field(20, ge=1, le=500)


@command("cull.best", params=BestParams)
def best(ctx: CommandContext, params: BestParams):
    """The library's own pick of a folder's frames most worth keeping, one per moment: a
    burst offers only its keeper, rejects never; weighed by the person's marks, the people
    at their best, sharpness, and against soft, blown or eyes-closed frames. A starting
    point to look at, not a decision."""
    return ctx.catalog.suggest_best(params.folder, params.count, **params.as_kwargs())


# ── Cheap pictures ───────────────────────────────────────────────────

def _camera_jpeg(ctx: CommandContext, filepath: str, long_side: int) -> bytes:
    """The frame as the camera rendered it, from the thumbnail cache (made on a miss)."""
    from thumbnails import ThumbnailSize, get_thumbnail

    if ctx.cache is None:
        raise CommandError("No thumbnail cache in this server")
    ctx.photo(filepath)
    source = ctx.library_file(filepath)
    size = ThumbnailSize.GRID if long_side <= ThumbnailSize.GRID.value else ThumbnailSize.PREVIEW
    data = get_thumbnail(source, ctx.cache, size)
    if not data:
        raise CommandError(f"Could not read a picture from {filepath}")
    return data


def _image(data: bytes, long_side: int):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    if max(img.size) > long_side:
        img.thumbnail((long_side, long_side), Image.LANCZOS)
    return img


def _jpeg(img, quality: int = 82) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class ThumbnailParams(BaseModel):
    filepath: str = Field(description="Path relative to the library root")
    size: int = Field(768, ge=128, le=2000, description="Longest side in pixels")


@command("render.thumbnail", params=ThumbnailParams)
def thumbnail(ctx: CommandContext, params: ThumbnailParams):
    """A frame as the camera rendered it, from the thumbnail cache: fast, no develop. For
    judging a frame while culling. Use render.preview to see an edit, render.loupe for focus."""
    img = _image(_camera_jpeg(ctx, params.filepath, params.size), params.size)
    return Rendered(_jpeg(img), {"filepath": params.filepath, "width": img.width, "height": img.height})


class SheetParams(BaseModel):
    filepaths: list[str] = Field(min_length=1, max_length=36,
                                 description="Frames to show, in the order to number them (1, 2, …)")
    cell: int = Field(384, ge=160, le=640, description="Longest side of each frame in pixels")
    columns: int = Field(0, ge=0, le=8, description="Frames per row; 0 picks a near-square grid")
    marks: bool = Field(True, description="Print each frame's stars and pick/reject under it")


_MARK_COLOURS = {1: (40, 160, 70), -1: (200, 50, 50)}


@command("render.contact_sheet", params=SheetParams)
def contact_sheet(ctx: CommandContext, params: SheetParams):
    """Several frames in one picture, numbered 1, 2, … in the order given, each with its file
    name: the way to compare the frames of a burst or a scene side by side and choose among
    them. From the thumbnail cache, no develop. ``cells`` maps each number to its filepath."""
    from PIL import Image, ImageDraw, ImageFont

    records = [ctx.photo(fp) for fp in params.filepaths]
    images = [_image(_camera_jpeg(ctx, fp, params.cell), params.cell) for fp in params.filepaths]
    n = len(images)
    columns = params.columns or math.ceil(math.sqrt(n))
    cell, pad = params.cell, 8
    caption = max(16, cell // 16)
    font = ImageFont.load_default(size=caption)
    badge_font = ImageFont.load_default(size=max(18, cell // 10))
    # Each row is as tall as its tallest frame, so a sheet of landscapes is not
    # half empty.
    heights = [max(img.height for img in images[i:i + columns]) for i in range(0, n, columns)]
    width = columns * (cell + pad) + pad
    height = sum(h + caption + 6 + pad for h in heights) + pad
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    cells, y0 = [], pad
    for i, (filepath, record, img) in enumerate(zip(params.filepaths, records, images)):
        col, row = i % columns, i // columns
        if col == 0 and i:
            y0 += heights[row - 1] + caption + 6 + pad
        x0 = pad + col * (cell + pad)
        ix, iy = x0 + (cell - img.width) // 2, y0 + (heights[row] - img.height) // 2
        sheet.paste(img, (ix, iy))

        number = str(i + 1)
        box = draw.textbbox((0, 0), number, font=badge_font)
        bw, bh = box[2] - box[0] + 12, box[3] - box[1] + 10
        draw.rectangle((ix, iy, ix + bw, iy + bh), fill=(255, 210, 0))
        draw.text((ix + 6 - box[0], iy + 5 - box[1]), number, fill=(0, 0, 0), font=badge_font)

        text = record.filename
        if params.marks:
            if record.rating:
                text += "  " + "*" * int(record.rating)
            if record.flag in _MARK_COLOURS:
                text += "  " + ("PICK" if record.flag == 1 else "REJECT")
        draw.text((ix + 2, y0 + heights[row] + 3), text, font=font,
                  fill=_MARK_COLOURS.get(record.flag, (220, 220, 220)) if params.marks else (220, 220, 220))
        cells.append({"n": i + 1, "filepath": filepath})

    return Rendered(_jpeg(sheet, 80), {"cells": cells, "width": width, "height": height})


# ── Proposals ────────────────────────────────────────────────────────

class ProposalItem(BaseModel):
    filepath: str = Field(description="Path relative to the library root")
    rating: Optional[int] = Field(None, ge=0, le=5, description="Stars to suggest; left out is no opinion")
    flag: Optional[int] = Field(None, ge=-1, le=1, description="1 pick, 0 unflag, -1 reject; left out is no opinion")
    color_label: Optional[int] = Field(None, ge=0, le=5)
    reason: str = Field(min_length=1, max_length=300,
                        description="Why, in a few words the person will read in the grid: "
                                    "'sharpest of the burst, both kids looking', 'eyes closed', 'missed focus'")

    @model_validator(mode="after")
    def _says_something(self):
        if self.rating is None and self.flag is None and self.color_label is None:
            raise ValueError(f"{self.filepath}: give at least one of rating, flag, color_label")
        return self


class ProposeParams(BaseModel):
    proposals: list[ProposalItem] = Field(min_length=1, max_length=2000)
    run: str = Field("", max_length=80, description="A name for this pass, to accept, undo or evaluate it "
                                                    "as one; a new one is made when left out")
    agent: str = Field("", max_length=80, description="Who proposes: the model's name")


@command("cull.propose", params=ProposeParams, writes=True)
def propose(ctx: CommandContext, params: ProposeParams):
    """Suggest ratings, picks and rejects, each with a reason, for the person to accept or
    dismiss in the grid. This is how an agent culls: nothing the person sees as their own
    rating changes until the proposal is accepted. A new proposal for a photo replaces its
    old one. ``already_decided`` counts photos the person has already marked, which
    accepting will leave as the person set them."""
    items, decided = [], 0
    for item in params.proposals:
        record = ctx.photo(item.filepath)
        current = {"rating": record.rating or 0, "flag": record.flag or 0, "color_label": record.color_label or 0}
        if any(getattr(item, f) is not None and current[f] != store.UNSET[f] for f in store.FIELDS):
            decided += 1
        items.append({**item.model_dump(), "folder": record.folder})
    run = params.run.strip() or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    count = store.propose(ctx.catalog.db, items, run=run, agent=params.agent.strip())
    return {"run": run, "proposed": count, "already_decided": decided}


class ScopeParams(BaseModel):
    """Which proposals: those of a folder, of a run, or of listed photos; together they narrow."""
    folder: Optional[str] = Field(None, description="Folder relative to the library root")
    run: Optional[str] = Field(None, description="The run cull.propose returned")
    filepaths: Optional[list[str]] = Field(None, max_length=5000)

    def select(self, ctx: CommandContext, status: Optional[str] = None) -> list[dict]:
        if not (self.folder or self.run or self.filepaths):
            raise CommandError("Say which proposals: a folder, a run or filepaths")
        return store.select(ctx.catalog.db, folder=self.folder, run=self.run,
                            filepaths=self.filepaths, status=status)


class ListProposalsParams(ScopeParams):
    status: Optional[Literal["pending", "accepted", "dismissed"]] = None


@command("cull.proposals", params=ListProposalsParams)
def list_proposals(ctx: CommandContext, params: ListProposalsParams):
    """The proposals of a folder, a run or some photos, with their state: pending (waiting
    for the person), accepted or dismissed."""
    found = params.select(ctx, params.status)
    counts = {s: sum(1 for p in found if p["status"] == s) for s in store.STATUSES}
    return {"counts": counts, "proposals": found}


class AcceptParams(ScopeParams):
    overwrite: bool = Field(False, description="Also replace marks the person has set. Only when the person "
                                               "has said so")


@command("cull.accept", params=AcceptParams, writes=True)
def accept_proposals(ctx: CommandContext, params: AcceptParams):
    """Make pending proposals real ratings, flags and labels (catalog and sidecar). Only
    when the person has asked for them to be applied. A mark the person has already set
    is kept unless ``overwrite``; the answer counts what was ``kept``. Undo with cull.undo."""
    return store.accept(ctx.catalog, params.select(ctx, "pending"), overwrite=params.overwrite)


@command("cull.dismiss", params=ScopeParams, writes=True)
def dismiss_proposals(ctx: CommandContext, params: ScopeParams):
    """Set pending proposals aside without applying them. They are kept for cull.evaluate."""
    return {"dismissed": store.dismiss(ctx.catalog.db, params.select(ctx, "pending"))}


@command("cull.undo", params=ScopeParams, writes=True)
def undo_proposals(ctx: CommandContext, params: ScopeParams):
    """Take back accepted proposals: each field returns to what it held before, unless the
    person has changed it since. The proposals become pending again."""
    return store.undo(ctx.catalog, params.select(ctx, "accepted"))


class EvaluateParams(ScopeParams):
    keep_rating: int = Field(1, ge=1, le=5, description="Stars at which a photo counts as kept (a pick always does)")


@command("cull.evaluate", params=EvaluateParams)
def evaluate(ctx: CommandContext, params: EvaluateParams):
    """How well proposals agree with the person's own cull, on a folder the person has
    finished: keep/drop agreement, precision (of the agent's keeps, how many the person
    kept) and recall (of the person's keeps, how many the agent found), flags, stars, and
    the disagreements with the agent's reasons. Propose on such a folder first, then
    evaluate, before culling folders the person has not."""
    return store.evaluate(ctx.catalog, params.select(ctx), keep_rating=params.keep_rating)


# ── How this person culls ────────────────────────────────────────────

# What calibration has learned about the person's taste is theirs, so it lives with
# their library, beside the catalog (.photomanager/culling.md), and is kept up to
# date there after each run. Until a library has one, the repository's template
# stands in: the procedure, with the lessons still to be learned.
GUIDE_NAME = "culling.md"
TEMPLATE = Path(__file__).resolve().parent.parent.parent / "calibration" / "culling.example.md"
_LESSONS = "## What we know about how you cull"


def guide_path(config) -> Path:
    """The library's own culling guide, or the template when it has none yet."""
    own = Path(config.db_path).parent / GUIDE_NAME
    return own if own.is_file() else TEMPLATE


class GuideParams(BaseModel):
    full: bool = Field(False, description="The whole guide: also how to run a calibration against a "
                                          "culled folder, and the log of past runs")


@command("cull.guide", params=GuideParams)
def guide(ctx: CommandContext, params: GuideParams):
    """How this person culls, learned from calibration runs and from their accepting and
    dismissing proposals: how much to keep at a school event and at a family one, what the
    stars mean, which frames they keep and drop, and where the signals mislead. Read it
    before culling a folder, and cull by it. In the text "you" is the person. ``full``
    also returns how to run a calibration and the log of past runs."""
    path = guide_path(ctx.config)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise CommandError(f"The culling guide ({path}) cannot be read")
    if not params.full and _LESSONS in text:
        text = text[text.index(_LESSONS):].split("\n## ", 1)[0].rstrip() + "\n"
    return {"guide": text}
