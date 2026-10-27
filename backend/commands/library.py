"""Finding photographs and culling them: folders, the grid's filters, ratings, tags, bursts."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .registry import CommandContext, CommandError, command


class Filters(BaseModel):
    """The grid's filters. Every one is optional; together they narrow the folder."""
    min_rating: int = Field(0, ge=0, le=5, description="Only photos with at least this many stars")
    flag: Optional[int] = Field(None, ge=-1, le=1, description="1 picked, 0 unflagged, -1 rejected")
    color: Optional[int] = Field(None, ge=0, le=5, description="Colour label: 1 red, 2 yellow, 3 green, 4 blue, 5 purple")
    tag: Optional[str] = Field(None, description="Only photos carrying this tag")
    unreviewed: bool = Field(False, description="Only photos with no rating, flag or label yet")
    pair_mode: Literal["all", "raw", "jpg", "paired", "raw_only", "jpg_only"] = Field(
        "paired",
        description="How RAW+JPEG pairs are listed. 'paired' shows each pair once, as the grid does",
    )

    def as_kwargs(self) -> dict:
        return {
            "min_rating": self.min_rating,
            "flag_filter": self.flag,
            "color_filter": self.color,
            "tag_filter": (self.tag or "").strip() or None,
            "unreviewed_only": self.unreviewed,
            "pair_mode": self.pair_mode,
        }


class FolderParams(BaseModel):
    folder: str = Field(description="Folder relative to the library root, e.g. '2025/2025-03-06'")


class FolderFilterParams(Filters, FolderParams):
    pass


class PhotosParams(FolderFilterParams):
    sort: Literal[
        "filename", "rating", "flag", "capture_date", "focal_length", "iso", "file_size", "base_name"
    ] = "filename"
    desc: bool = False
    limit: int = Field(200, ge=1, le=2000)
    offset: int = Field(0, ge=0)


class PhotoParams(BaseModel):
    filepath: str = Field(description="Path relative to the library root, as library.photos returns it")


class RateParams(BaseModel):
    filepaths: list[str] = Field(min_length=1, description="Photos to mark, relative to the library root")
    rating: Optional[int] = Field(None, ge=0, le=5, description="Stars, 0 clears them")
    flag: Optional[int] = Field(None, ge=-1, le=1, description="1 pick, 0 unflag, -1 reject")
    color_label: Optional[int] = Field(None, ge=0, le=5, description="0 clears the label")


class PhotoTagsParams(PhotoParams):
    tags: list[str] = Field(description="The photo's complete tag list; replaces what it had")


class TagFolderParams(FolderFilterParams):
    tag: str = Field(description="The tag to add (or remove) on every photo the filters match")
    remove: bool = False
    # The tag being applied is not also a filter here.
    filter_tag: Optional[str] = Field(None, description="Only touch photos already carrying this tag")

    def as_kwargs(self) -> dict:
        kwargs = super().as_kwargs()
        kwargs["tag_filter"] = (self.filter_tag or "").strip() or None
        return kwargs


class BurstParams(FolderFilterParams):
    max_gap_seconds: Optional[int] = Field(
        None, ge=1, le=600,
        description="How far apart two alike frames may be and still be one burst; the library default when left out",
    )


def photo_summary(p, tags=None, edit=None) -> dict:
    out = {
        "filepath": p.filepath,
        "filename": p.filename,
        "rating": p.rating,
        "flag": p.flag,
        "color_label": p.color_label,
        "camera": p.camera,
        "lens": p.lens,
        "focal_length": p.focal_length,
        "aperture": p.aperture,
        "shutter_speed": p.shutter_speed,
        "iso": p.iso,
        "capture_date": p.capture_date,
        "dimensions": p.dimensions,
    }
    if tags is not None:
        out["tags"] = tags
    if edit is not None:
        out["edited"] = bool(edit)
    return out


@command("library.folders")
def folders(ctx: CommandContext, params):
    """Every folder in the catalog with how many photos it holds. Start here to find photos."""
    return {"folders": ctx.catalog.get_all_folders()}


@command("library.photos", params=PhotosParams)
def photos(ctx: CommandContext, params: PhotosParams):
    """The photos of one folder that match the grid's filters, with their exposure data,
    rating, flag, label, tags and whether they carry an edit. Paged with limit/offset;
    ``total`` is the count before paging."""
    found = ctx.catalog.get_photos_in_folder(
        params.folder, sort_by=params.sort, sort_desc=params.desc, **params.as_kwargs()
    )
    page = found[params.offset:params.offset + params.limit]
    tags = ctx.catalog.db.get_tags_for_filepaths([p.filepath for p in page])
    edits = ctx.catalog.photo_edits(params.folder)
    return {
        "folder": params.folder,
        "total": len(found),
        "offset": params.offset,
        "photos": [photo_summary(p, tags.get(p.filepath, []), edits.get(p.filepath, {})) for p in page],
    }


@command("library.photo", params=PhotoParams)
def photo(ctx: CommandContext, params: PhotoParams):
    """One photo in full: metadata, tags, its edit and its virtual copies."""
    record = ctx.photo(params.filepath)
    out = photo_summary(record, ctx.catalog.get_photo_tags(params.filepath))
    out["edit"] = ctx.catalog.photo_edit(params.filepath)
    out["copies"] = ctx.catalog.photo_versions(params.filepath)
    out["file_size"] = record.file_size
    return out


class SearchParams(BaseModel):
    query: str = Field(description=(
        "Terms separated by spaces; a photo is found when all hold. "
        "rating:>=3  flag:pick|reject|none  label:red,green  tag:\"school play\"  camera:r6  lens:35  "
        "iso:>1600  f:<=2.8  focal:24..35  shutter:<1/100  date:2026-04 (or >=2026-03, 2025..2026)  "
        "folder:hawaii  type:raw|jpg|cr3  is:edited|unedited|unreviewed|reviewed|pick|reject|basket  "
        "shows:\"kids at the beach\" finds what the pictures show (smart search; best first)  "
        "-term negates; a comma means either; plain words match filename, folder, camera, lens, tags"
    ))
    sort: Literal["", "relevance", "date", "rating", "filename", "folder"] = Field("", description=(
        "'' is by relevance when the query has a shows: term, else by date"))
    desc: bool = True
    limit: int = Field(200, ge=1, le=5000)
    offset: int = Field(0, ge=0)


@command("library.search", params=SearchParams)
def search(ctx: CommandContext, params: SearchParams):
    """Find photos anywhere in the library with a query such as 'rating:>=3 iso:>3200 date:2026'.
    A RAW+JPEG pair is one result. ``understood`` echoes the query as it was read."""
    from catalog.search import SearchError, describe, unclear

    try:
        found = ctx.catalog.search(params.query, params.sort, params.desc)
    except SearchError as exc:
        raise CommandError(str(exc))
    page = found[params.offset:params.offset + params.limit]
    rows = []
    for photo in page:
        row = photo_summary(photo.record, sorted(photo.tags), photo.edited)
        row.update(rating=photo.rating, flag=photo.flag, color_label=photo.label,
                   folder=photo.record.folder, date=photo.date, files=photo.filepaths)
        rows.append(row)
    return {"query": params.query, "understood": describe(params.query),
            "total": len(found), "offset": params.offset, "photos": rows,
            "unclear": unclear(found)}


@command("library.stats", params=FolderParams)
def stats(ctx: CommandContext, params: FolderParams):
    """Counts for one folder: how many photos, picked, rejected, rated and so on."""
    return ctx.catalog.get_folder_stats(params.folder)


@command("library.bursts", params=BurstParams)
def bursts(ctx: CommandContext, params: BurstParams):
    """Near-identical frames shot seconds apart, grouped, each group with a suggested
    keeper (the sharpest). Needs the folder's thumbnails measured first."""
    return ctx.catalog.find_bursts(
        params.folder, max_gap_seconds=params.max_gap_seconds, **params.as_kwargs()
    )


@command("tags.list")
def list_tags(ctx: CommandContext, params):
    """Every tag in the library with how many photos carry it."""
    return {"tags": ctx.catalog.db.list_tags()}


@command("photo.rate", params=RateParams, writes=True)
def rate(ctx: CommandContext, params: RateParams):
    """Set stars, pick/reject flag and/or colour label on one or more photos. Fields left
    out are not touched. Written to the catalog and to each photo's XMP sidecar."""
    if params.rating is None and params.flag is None and params.color_label is None:
        raise CommandError("Nothing to set: give rating, flag or color_label")
    for filepath in params.filepaths:
        ctx.photo(filepath)
    for filepath in params.filepaths:
        if params.rating is not None:
            ctx.catalog.set_rating(filepath, params.rating)
        if params.flag is not None:
            ctx.catalog.set_flag(filepath, params.flag)
        if params.color_label is not None:
            ctx.catalog.set_color_label(filepath, params.color_label)
    return {"photos": [
        {"filepath": p.filepath, "rating": p.rating, "flag": p.flag, "color_label": p.color_label}
        for p in (ctx.catalog.get_photo(f) for f in params.filepaths)
    ]}


@command("photo.set_tags", params=PhotoTagsParams, writes=True)
def set_tags(ctx: CommandContext, params: PhotoTagsParams):
    """Replace one photo's tags. Tags that do not exist yet are created."""
    ctx.photo(params.filepath)
    ctx.catalog.set_photo_tags(params.filepath, params.tags)
    return {"filepath": params.filepath, "tags": ctx.catalog.get_photo_tags(params.filepath)}


@command("tags.apply", params=TagFolderParams, writes=True)
def apply_tag(ctx: CommandContext, params: TagFolderParams):
    """Add a tag to (or, with remove, take it off) every photo of a folder that the
    filters match."""
    tag = params.tag.strip()
    if not tag:
        raise CommandError("A tag name is required")
    act = ctx.catalog.untag_matching_photos if params.remove else ctx.catalog.tag_matching_photos
    return act(params.folder, tag, **params.as_kwargs())


@command("bursts.reject_duplicates", params=FolderFilterParams, writes=True)
def reject_duplicates(ctx: CommandContext, params: FolderFilterParams):
    """In every burst of a folder, keep the sharpest frame and reject the rest. Only
    frames the grouping is confident are the same shot are touched; a reject can be
    undone with photo.rate flag=0."""
    return ctx.catalog.reject_burst_duplicates(params.folder, **params.as_kwargs())
