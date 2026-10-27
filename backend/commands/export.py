"""The export basket and writing it out: what to export, then how."""

from pydantic import BaseModel, Field

from .registry import CommandContext, CommandError, command


class BasketParams(BaseModel):
    filepaths: list[str] = Field(min_length=1, description="Photos, relative to the library root")


class ExportParams(BaseModel):
    folder: str = Field(
        "",
        description="A folder name inside library_root/export to write the JPEGs to (the date and time "
                    "when left out). Not a path: exports cannot be written anywhere else, and never "
                    "overwrite a file",
    )
    max_resolution: int = Field(2048, ge=0, le=12000,
                                description="Longest side in pixels; 0 is full size. 2048 suits Google Photos and sharing")
    quality: int = Field(85, ge=1, le=100, description="JPEG quality")
    strip_exif: bool = Field(True, description="Keep only the dates and the credit: no location, no camera serial")
    photographer: str = Field("", description="The credit, written as EXIF Artist and Copyright")


class ProgressParams(BaseModel):
    job_id: str


def _basket(ctx: CommandContext) -> dict:
    items = ctx.catalog.db.list_basket()
    edits = ctx.catalog.photo_edits()
    return {"count": len(items), "items": [
        {"filepath": i.filepath, "edited": i.filepath in edits} for i in items
    ]}


@command("basket.list")
def basket_list(ctx: CommandContext, params):
    """The photos in the export basket, and which of them carry an edit."""
    return _basket(ctx)


@command("basket.add", params=BasketParams, writes=True)
def basket_add(ctx: CommandContext, params: BasketParams):
    """Put photos in the export basket. Already there is fine."""
    for filepath in params.filepaths:
        ctx.photo(filepath)
    for filepath in params.filepaths:
        record = ctx.catalog.get_photo(filepath)
        ctx.catalog.db.add_to_basket(record.filepath, record.folder, record.filename)
    return _basket(ctx)


@command("basket.remove", params=BasketParams, writes=True)
def basket_remove(ctx: CommandContext, params: BasketParams):
    """Take photos out of the export basket. Their edits stay."""
    for filepath in params.filepaths:
        ctx.catalog.db.remove_from_basket(filepath)
    return _basket(ctx)


@command("basket.clear", writes=True)
def basket_clear(ctx: CommandContext, params):
    """Empty the export basket. Edits, ratings and copies are not touched."""
    ctx.catalog.db.clear_basket()
    return _basket(ctx)


@command("export.run", params=ExportParams, writes=True)
def export_run(ctx: CommandContext, params: ExportParams):
    """Write every photo in the basket as a JPEG, in the background. Returns a job id;
    follow it with export.progress. Each photo is written as it stands: as developed
    if it has an edit, the camera's own picture if not. Virtual copies marked for
    export are written too."""
    from export.jobs import start_export

    try:
        return start_export(ctx.config, ctx.catalog, params)
    except ValueError as exc:
        raise CommandError(str(exc))


@command("export.progress", params=ProgressParams)
def export_progress(ctx: CommandContext, params: ProgressParams):
    """How far an export job has got: completed, failed, and the file being written."""
    from export.jobs import progress

    state = progress(params.job_id)
    if state is None:
        raise CommandError(f"No export job {params.job_id}")
    return state
