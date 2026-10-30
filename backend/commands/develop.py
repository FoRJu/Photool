"""
Developing a photograph: reading and writing its edit, Auto, and looking at the result.

An edit is the same plain dict everywhere — the one :func:`adjust.develop.normalize`
cleans, the catalog stores and the sidecar carries — so these commands pass it
through rather than describing it a second time. ``develop.controls`` is how an
agent learns what each key means.
"""

import io
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .registry import CommandContext, CommandError, Rendered, command

# What each control does, in the terms an agent needs to choose a value. The
# ranges are read from adjust.develop so they cannot drift from the sliders.
_MEANING = {
    "exposure": "Stops. +1 doubles the light; a real re-exposure in linear light, not a brightness lift",
    "temperature": "White balance. Negative cooler (bluer), positive warmer (yellower)",
    "tint": "White balance. Negative greener, positive more magenta",
    "contrast": "A curve through the mid-tones. Negative flatter, positive punchier",
    "highlights": "The bright half. Negative recovers a bright sky, positive brightens",
    "lights": "The bright middle tones, about two thirds of a stop over grey: a lit face, skin, pale cloth",
    "darks": "The dark middle tones, under grey: hair, dark cloth, the shade side of a face",
    "shadows": "The dark half. Positive opens up shadows, negative deepens them",
    "whites": "Only the very top of the range: sets the white point",
    "blacks": "Only the very bottom of the range: sets the black point",
    "flat": "How far to leave the camera's own rendering behind. 0 is the camera's look, 100 the plain decode",
    "compress": "Squeeze a wide scene into the picture's range while keeping local detail (HDR to a normal picture)",
    "clarity": "Mid-tone local contrast. Positive crisper, negative softer",
    "vibrance": "Saturation that favours the flat colours and spares skin",
    "saturation": "All colours alike",
    "fade": "Lifted blacks and eased whites: the matte look",
    "shadow_hue": "Split toning: the hue tinting the shadows (0-360). Needs shadow_sat",
    "shadow_sat": "Split toning: how strongly the shadows are tinted",
    "highlight_hue": "Split toning: the hue tinting the highlights (0-360). Needs highlight_sat",
    "highlight_sat": "Split toning: how strongly the highlights are tinted",
    "angle": "Straighten, degrees clockwise",
    "crop_x": "Crop: left edge of the kept rectangle, as a fraction of the levelled frame. All four crop keys or none",
    "crop_y": "Crop: top edge, as a fraction",
    "crop_w": "Crop: width, as a fraction",
    "crop_h": "Crop: height, as a fraction",
    "mono": ("Black and white: 1 develops the photo in grey, 0 in colour. The B&W mix (bw_*) sets how light "
             "each colour turns; split toning (shadow_hue/sat, highlight_hue/sat) then tints the grey"),
    "grain": "Film grain, 0 none to 100 heavy. The same grains every render, sized to the picture",
    "grain_size": "Grain size: 0 is a 35 mm film's grain, negative finer, positive coarser. Needs grain",
    "vignette": "Corners of the picture as cropped: negative darker, positive lighter",
    "nr_luma": ("Noise reduction, RAW only: how much of the luminance noise (grain) a learned denoiser "
                "takes out, 0 none to 100 all. 50-80 keeps a fine, natural grain. The first use on a "
                "photograph runs the model (~10 s), later changes are instant"),
    "nr_color": ("Noise reduction, RAW only: how much of the colour noise (coloured speckle and blotches) "
                 "is taken out, 0 none to 100 all. Usually 100 on a high-ISO frame"),
    "sharpen": ("Sharpening amount, 0 off to 150, Lightroom's scale (its RAW default is 40). On brightness "
                "only, after noise reduction. Judge it with render.loupe at 1:1: a downsized preview hides it"),
    "sharpen_radius": ("Sharpening radius as tenths of a pixel added to 1.0 px: -5 is 0.5 px, 0 is 1.0, 20 is "
                       "3.0. Small for fine detail (landscapes), larger for soft or downsized subjects. Needs sharpen"),
    "sharpen_detail": ("Lightroom's Detail less 25: -25 is Detail 0 (strong edges held back, no halos), 0 is "
                       "the default 25, 75 is Detail 100 (every edge sharpened fully). Needs sharpen"),
    "sharpen_masking": ("Edge masking, 0 sharpens everywhere, higher keeps it to edges and leaves skin and "
                        "sky smooth; 30-60 for portraits. Needs sharpen"),
    "lens_profile": ("1 corrects the lens's distortion as the camera corrected its own JPEG of this frame "
                     "(learned from that JPEG; RAW only). Straight lines near the edges of a wide-angle frame "
                     "come out straight. 0 leaves the RAW's geometry"),
    "distortion": ("Lens distortion by hand, on top of lens_profile: positive straightens a barrel (lines bowing "
                   "out), negative a pincushion. The picture is enlarged just enough to keep its corners"),
    "tf_vertical": ("Transform: tilt the picture plane about its horizontal axis. Negative straightens verticals that "
                    "lean in towards the top (a building or a stage shot from below), positive the other way, as "
                    "Lightroom's Vertical. develop.upright finds it, with the angle, from the frame's own lines"),
    "tf_horizontal": "Transform: the same about the vertical axis, for a wall or a stage shot from one side",
    "tf_aspect": ("Transform: taller (positive) or wider, for the squat look a strong vertical correction leaves. "
                  "The picture is enlarged just enough to keep its corners"),
}


# The colour mixer, band by band. Centres in degrees are Lightroom's.
for _band, _centre in zip(("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"),
                          (0, 30, 60, 120, 180, 240, 270, 300)):
    _MEANING[f"hue_{_band}"] = f"Colour mixer, {_band} (hue {_centre}°): shift those colours toward the neighbouring hue"
    _MEANING[f"sat_{_band}"] = f"Colour mixer, {_band}: more or less saturated; -100 turns that colour grey"
    _MEANING[f"lum_{_band}"] = f"Colour mixer, {_band}: darker or lighter, e.g. lum_blue -40 deepens a sky"
    _MEANING[f"bw_{_band}"] = (f"B&W mix, {_band}: how light {_band} turns in grey when mono is 1, like a "
                               f"colour filter on film; e.g. bw_blue -60 with bw_red +40 is a red filter's dark sky")

_CURVE_SHAPE = ("A list of [x, y] points on 0..255, input to output, at most 16, e.g. "
                "[[0, 0], [64, 52], [192, 205], [255, 255]] for a gentle S. Smooth and never "
                "overshoots between points; flat beyond the first and last")
_CURVE_MEANING = {
    "curve": "Tone curve on all channels. " + _CURVE_SHAPE,
    "curve_r": "Tone curve on red only. " + _CURVE_SHAPE,
    "curve_g": "Tone curve on green only. " + _CURVE_SHAPE,
    "curve_b": "Tone curve on blue only. " + _CURVE_SHAPE,
}


_MASKS_MEANING = (
    "Gradients (local adjustments): a list, at most 16. Positions are fractions of the whole frame "
    "before crop and level. {kind: 'linear', x1, y1, x2, y2, invert}: whole at (x1, y1), gone at (x2, y2), e.g. "
    "x1 0.5 y1 0 x2 0.5 y2 0.4 for a sky; invert turns it round. {kind: 'radial', cx, cy, rx, ry, angle, feather 0-100, invert}: "
    "whole inside the ellipse (rx a share of the width, ry of the height), fading out over feather; "
    "invert puts it outside. Each takes exposure (-4..4 stops) and contrast, highlights, lights, darks, shadows, "
    "temperature, clarity (-100..100) and saturation (-200..200, added to the edit's own); and, on the skin of the "
    "people it reaches only, smooth (0..100, calms blemishes and blotches), even (0..100, evens redness and dark "
    "patches) and texture (-100..100, softer or crisper pores), e.g. people with smooth 30 even 20 for a natural "
    "touch. {kind: 'brush', invert} reaches only where painted, or with invert everywhere but there. {kind: 'subject' or 'sky', invert}: where a "
    "model finds the picture's subject or sky (invert for the background or the land). {kind: 'people' or "
    "'faces', person, invert}: everyone in the frame, or their faces; person 1, 2, ... (counted from the left "
    "by face) for one person only; a people mask can take areas, a list of face_skin, eyes, brows, lips, teeth, "
    "hair, body_skin, clothes, for only those parts of them, e.g. faces with exposure 0.5 and clarity -20 to light and soften the faces "
    "of a group. Any of them takes a range: {kind: 'luminance', lo, hi, smooth} (0-100) for only a band of "
    "brightness, or {kind: 'color', cb, cr, lum, width} for only colours near one (sample cb, cr and lum with "
    "the develop view, or take them from the picture: BT.709 blue and red differences of display values). Any of "
    "them takes strokes: [{points: [[x, y], ...], size (radius, a share of the longer side, 0.0005 to 0.5, "
    "e.g. 0.03), feather 0-100, flow 1-100, erase}], laid in order, erasing ones taking away (from a radial "
    "too). Any of them can be cropped by a section: {x1, y1, x2, y2, feather}, the edges as fractions of the "
    "whole frame and feather 0-100 (the share of the section's shorter side its edges fade in over, so 100 "
    "fades from the middle); outside the section the mask reaches nothing, strokes and invert included. A "
    "section of the whole frame with feather 0 limits nothing, but is kept: it is a section switched on. "
    "Replacing the list replaces every gradient and brush")


_SPOTS_MEANING = (
    "Spots removed (dust, a blemish): a list, at most 200, done before every other control. "
    "{x, y: the spot's centre, fractions of the whole frame before crop and level; r: its radius, a "
    "share of the longer side, e.g. 0.008; sx, sy: the centre of a clean place nearby to paint it from "
    "(look at the picture to choose one); mode 'heal' (the source's texture in the spot's own light and "
    "colour, the default) or 'clone'; feather 0-100; opacity 1-100}. A frame's own, like its crop: not "
    "carried by paste or to a burst. Replacing the list replaces every spot")


def controls() -> list[dict]:
    from adjust import develop
    from adjust.geometry import GEOMETRY_KEYS, MAX_ANGLE

    out = []
    for key in develop.SETTINGS_SLIDERS:
        if key == "exposure":
            low, high = -develop.MAX_EXPOSURE_EV, develop.MAX_EXPOSURE_EV
        else:
            low, high = develop.SLIDER_RANGES.get(key, (-develop.MAX_TONE_VALUE, develop.MAX_TONE_VALUE))
        out.append({"key": key, "min": low, "max": high, "default": 0, "meaning": _MEANING.get(key, "")})
    for key in develop.CURVE_KEYS:
        out.append({"key": key, "min": 0, "max": 255, "default": None, "meaning": _CURVE_MEANING[key]})
    out.append({"key": develop.MASKS_KEY, "min": None, "max": None, "default": None, "meaning": _MASKS_MEANING})
    out.append({"key": develop.SPOTS_KEY, "min": None, "max": None, "default": None, "meaning": _SPOTS_MEANING})
    for key in GEOMETRY_KEYS:
        low, high = (-MAX_ANGLE, MAX_ANGLE) if key == "angle" else (0.0, 1.0)
        out.append({"key": key, "min": low, "max": high, "default": 0 if key == "angle" else None,
                    "meaning": _MEANING.get(key, "")})
    return out


class EditParams(BaseModel):
    filepath: str = Field(description="Path relative to the library root")
    version: int = Field(0, ge=0, description="0 is the photograph's own edit; 1, 2, … its virtual copies")


class SetEditParams(EditParams):
    settings: dict = Field(description="Control values by key, e.g. {'exposure': 0.5, 'shadows': 30}. See develop.controls")
    mode: Literal["merge", "replace"] = Field(
        "merge",
        description="'merge' changes only the keys given and keeps the rest of the edit; 'replace' makes these the whole edit",
    )


class AutoParams(EditParams):
    apply: bool = Field(False, description="Also store the suggestion as the edit (merged over the current one)")


class UprightParams(AutoParams):
    mode: str = Field("vertical", pattern="^(vertical|level)$",
                      description="vertical: make the frame's vertical lines vertical (tf_vertical and angle); "
                                  "level: make its level lines level (angle)")


class PasteParams(BaseModel):
    filepaths: list[str] = Field(min_length=1)
    settings: dict = Field(description="The edit to put on every photo. Each keeps its own crop and straighten")
    source: Optional[str] = Field(None, description="The photo the edit came from: needed to match to it")
    paste_as: Literal["copied", "keep_light", "match_exposure", "match_look"] = Field(
        "copied",
        description=(
            "copied: the edit as given. keep_light: each photo keeps its own exposure and white "
            "balance. match_exposure: each photo's light (exposure and white balance) brought to "
            "source's: by its capture settings while the light stayed the same, checked against "
            "the people and things both frames show; where those say the light itself changed, "
            "measured from them (the note says which). match_look: each photo matched to source "
            "as developed; same shoot only, since Match copies a picture's content as well as "
            "its look"),
    )


class PreviewParams(BaseModel):
    filepath: str = Field(description="Path relative to the library root")
    version: int = Field(0, ge=0)
    size: int = Field(1024, ge=128, le=2048, description="Longest side in pixels")
    settings: Optional[dict] = Field(
        None,
        description="Render with this edit instead of the stored one, without saving it. Use it to try a change before develop.set",
    )


class LoupeParams(BaseModel):
    filepath: str
    u: float = Field(0.5, ge=0.0, le=1.0, description="Horizontal centre of the patch, as a fraction of the picture")
    v: float = Field(0.5, ge=0.0, le=1.0, description="Vertical centre, as a fraction")
    size: int = Field(480, ge=64, le=1600, description="Side of the 1:1 patch, in source pixels")
    version: int = Field(0, ge=0)
    settings: Optional[dict] = None


def _unknown_keys(settings: dict) -> list[str]:
    from adjust.develop import EDIT_KEYS

    return sorted(k for k in settings if k not in EDIT_KEYS)


def stored_edit(ctx: CommandContext, filepath: str, version: int) -> dict:
    ctx.photo(filepath)
    if not version:
        return ctx.catalog.photo_edit(filepath)
    from adjust.develop import normalize

    found = ctx.catalog.db.get_version(filepath, version)
    if found is None:
        raise CommandError(f"{filepath} has no copy {version}")
    return normalize(found["settings"])


def _store(ctx: CommandContext, filepath: str, version: int, settings: dict) -> dict:
    if version:
        stored = ctx.catalog.set_copy_edit(filepath, version, settings)
        if stored is None:
            raise CommandError(f"{filepath} has no copy {version}")
        return stored
    if not settings:
        ctx.catalog.clear_photo_edit(filepath)
        return {}
    return ctx.catalog.set_photo_edit(filepath, settings)


def _source(ctx: CommandContext, filepath: str):
    from export.engine import resolve_export_source

    return resolve_export_source(ctx.library_file(filepath), ctx.config.export_prefer_raw)


@command("develop.controls")
def list_controls(ctx: CommandContext, params):
    """Every develop control: its key, range, default and what it does. Read this before
    develop.set. Values out of range are clamped; a control at 0 is no edit."""
    return {"controls": controls()}


@command("develop.get", params=EditParams)
def get_edit(ctx: CommandContext, params: EditParams):
    """A photo's current edit. Only controls that are not at zero are listed."""
    return {"filepath": params.filepath, "version": params.version,
            "settings": stored_edit(ctx, params.filepath, params.version)}


@command("develop.set", params=SetEditParams, writes=True)
def set_edit(ctx: CommandContext, params: SetEditParams):
    """Change a photo's edit. Non-destructive: stored in the catalog and the XMP sidecar,
    never in the image file. Returns the edit as stored, after clamping."""
    current = stored_edit(ctx, params.filepath, params.version)
    settings = {**current, **params.settings} if params.mode == "merge" else dict(params.settings)
    stored = _store(ctx, params.filepath, params.version, settings)
    return {"filepath": params.filepath, "version": params.version, "settings": stored,
            "ignored": _unknown_keys(params.settings)}


@command("develop.reset", params=EditParams, writes=True)
def reset_edit(ctx: CommandContext, params: EditParams):
    """Take a photo's edit off, back to the camera's own rendering."""
    stored_edit(ctx, params.filepath, params.version)
    _store(ctx, params.filepath, params.version, {})
    return {"filepath": params.filepath, "version": params.version, "settings": {}}


@command("develop.auto", params=AutoParams, writes=True)
def auto(ctx: CommandContext, params: AutoParams):
    """Suggest exposure and tone for a photo, measured from the frame itself. Conservative:
    a frame meant to be dark stays dark. With apply, the suggestion is stored."""
    from adjust.render import auto_develop
    from export.engine import linear_base

    current = stored_edit(ctx, params.filepath, params.version)
    frame = linear_base(_source(ctx, params.filepath), 512)
    if frame is None:
        raise CommandError(f"Could not read {params.filepath}")
    suggested = auto_develop(frame)
    out = {"filepath": params.filepath, "suggested": suggested}
    if params.apply:
        out["settings"] = _store(ctx, params.filepath, params.version, {**current, **suggested})
    return out


@command("develop.upright", params=UprightParams, writes=True)
def upright_command(ctx: CommandContext, params: UprightParams):
    """Straighten a photo from its own lines, as Lightroom's Upright does: vertical sets
    the tilt (tf_vertical) and the level (angle) that make its verticals vertical; level
    sets the angle that makes its horizon level. Says so, and changes nothing, when the
    frame's lines are too few or do not agree (people and trees are not verticals). With
    apply, the suggestion is stored."""
    from export.engine import upright_for

    current = stored_edit(ctx, params.filepath, params.version)
    found = upright_for(_source(ctx, params.filepath), current, params.mode)
    if found is None:
        raise CommandError(f"Could not read {params.filepath}")
    out = {"filepath": params.filepath, **found}
    if params.apply and found.get("found"):
        suggested = {k: found[k] for k in ("tf_vertical", "angle") if k in found}
        out["settings"] = _store(ctx, params.filepath, params.version, {**current, **suggested})
    return out


@command("develop.paste", params=PasteParams, writes=True)
def paste(ctx: CommandContext, params: PasteParams):
    """Put one edit on many photos at once, replacing theirs. Each photo keeps its own
    crop and straighten. ``paste_as`` can also adjust each photo towards ``source``;
    ``notes`` says what was done to each."""
    from adjust.develop import normalize
    from export.paste_match import adjuster, faces_from

    if params.paste_as != "copied" and params.paste_as != "keep_light" and not params.source:
        raise CommandError("Matching needs source: the photo the edit came from")
    adjust = adjuster(Path(ctx.config.library_root), params.source, normalize(params.settings), params.paste_as,
                      ctx.config.export_prefer_raw, faces_of=faces_from(ctx.catalog.db))
    result = ctx.catalog.apply_edit_to_many(params.filepaths, params.settings, source=params.source,
                                            adjust=adjust)
    result["missing"] = [f for f in params.filepaths if f not in result["copied"]]
    return result


@command("develop.presets")
def presets(ctx: CommandContext, params):
    """The saved edit presets, by name, with their settings. Apply one with develop.paste
    or develop.set."""
    return {"presets": ctx.catalog.db.list_develop_presets()}


def _jpeg(img, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class LookHealthParams(BaseModel):
    filepath: str


@command("develop.look_health", params=LookHealthParams)
def look_health(ctx: CommandContext, params: LookHealthParams):
    """Whether a RAW's camera look is drawn the way the camera drew it. ``even`` is how
    evenly its tone curve rises through the mid-tones; below ``even_min`` it has a
    plateau and ``off`` is true. ``banding`` is how unevenly the photograph with no
    edit keeps the camera JPEG's local contrast across its tones, in stops;
    ``shows`` is an off look whose plateau can be seen in this photograph (faces or
    skies posterized; ``flattest`` names the tone). Banding alone is not a fault: a
    camera JPEG with heavy noise reduction is softer than any develop. A look that
    is off: say so, rather than judge the photo by it."""
    from export.engine import look_health as health

    return {"filepath": params.filepath, **health(_source(ctx, params.filepath))}


@command("render.preview", params=PreviewParams)
def preview(ctx: CommandContext, params: PreviewParams):
    """Look at a photo as it is developed: the stored edit, or ``settings`` to try one
    without saving it. Renders the file an export would develop (the RAW of a pair)."""
    from adjust.develop import normalize
    from adjust.render import develop_frame
    from export.engine import camera_image, linear_base

    settings = (normalize(params.settings) if params.settings is not None
                else stored_edit(ctx, params.filepath, params.version))
    source = _source(ctx, params.filepath)
    img = None
    if settings:
        frame = linear_base(source, params.size)
        if frame is not None:
            img = develop_frame(frame, settings)
    if img is None:
        img = camera_image(source, params.size)
    if img is None:
        raise CommandError(f"Could not render {params.filepath}")
    return Rendered(_jpeg(img), {"filepath": params.filepath, "rendered": source.name,
                                 "width": img.width, "height": img.height, "settings": settings})


@command("render.loupe", params=LoupeParams)
def loupe(ctx: CommandContext, params: LoupeParams):
    """A 1:1 patch of the developed photo around a point, for judging sharpness, noise,
    skin and highlight detail that a downsized preview hides."""
    from adjust.develop import normalize
    from export.engine import loupe_image

    settings = (normalize(params.settings) if params.settings is not None
                else stored_edit(ctx, params.filepath, params.version))
    img = loupe_image(_source(ctx, params.filepath), settings, params.u, params.v, params.size)
    if img is None:
        raise CommandError(f"Could not render {params.filepath}")
    return Rendered(_jpeg(img, 90), {"filepath": params.filepath, "u": params.u, "v": params.v,
                                     "width": img.width, "height": img.height})
