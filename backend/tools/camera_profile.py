"""
Camera profiles, offline: learn them from RAW files, and measure how close they
come to the camera.

The app learns each camera's rendering as frames are opened (``adjust.look``,
pooled per camera and Picture Style by ``export.engine``). This tool does the
same work on a folder at once, and says how well it went, so that the model can
be judged against new material as it arrives rather than by eye:

  learn   open every RAW the way the app does, adding what each teaches to its
          camera's pool in the profile directory the app uses.

  report  measure, for every RAW, the look the app would use against the
          camera's own JPEG: colour error (OkLab distance x100; about 1 is at the
          edge of visible side by side) and brightness error (fraction of full
          scale). Read-only: it works on a copy of the stored pools.

          Frames shot as an exposure bracket (same camera, settings and moment,
          different exposures) are also tested the way an edit uses the look:
          each frame is moved to the bracket's middle exposure with the exposure
          control and compared with the camera's JPEG of that middle frame. That
          is the question "does a small exposure edit look like what the camera
          would have made?" asked of real data. A frame that is not the same
          scene (an extra shot taken in the same half minute) is skipped.

          ``lens`` is the camera's vignetting correction learned for the frame,
          in stops at the corner. ``edge`` is what is left of it: how much brighter
          than the camera's picture ours is at the edge than in the middle, in
          hundredths of full scale (near 0 when the correction matches; clearly
          positive where the camera darkens the corners itself).

          ``coarse`` is the colour error over 4 x 4 pixel blocks: tone and colour
          without the fine texture the camera's sharpening changes. Where it is
          well below ``dE``, the difference is texture, not the look.

          ``banding`` is how unevenly ours keeps the camera's local contrast across
          its tones, in stops (adjust.look.tone_detail): about 0.2 to 0.3 for a
          look that draws like the camera, 1 or more where a face or a sky is
          posterized. Mean error does not see that; this does.

  health  check every kept look (one per photograph opened) for the shape of a
          broken one: a tone curve with a plateau in the mid-tones (``even``,
          adjust.look.curve_evenness), marked OFF. With ``--render``, each
          photograph is drawn with its kept look as it is against the camera's
          JPEG (``banding``), and an OFF look whose plateau can be seen in its
          photograph is marked SHOWS. Read-only: nothing is learned or changed.
          The worst come first; open those again, or learn their folder, to have
          them learned again.

  list    the stored pools: which camera and style, and which frames.

  inspect what the app sees in a file from a camera it has not met: the model and
          style settings its pool is keyed on, whether the colour matrix is there,
          the sensor's levels, the JPEGs it could learn from and whether they are
          the same picture, and the lens setting. Run it on a few files before
          trusting a new camera, and send it with a `report` if something is off.

Run from ``backend``::

    python -m tools.camera_profile report ~/Pictures/2026-10-08 --json report.json
    python -m tools.camera_profile learn ~/Pictures/2026-10-08
    python -m tools.camera_profile health --render --json health.json
    python -m tools.camera_profile list

``--profiles`` points at another profile directory; the default is the app's,
``camera_profiles`` beside the thumbnail cache in ``config.yaml``.
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Frames of one bracket are taken within this many seconds of each other.
BRACKET_SECONDS = 30.0
# A bracket has at least this many frames and spans at least this many stops.
BRACKET_MIN_FRAMES = 3
BRACKET_MIN_SPAN = 0.6
# Frames are developed at about this size, as the app's preview is.
DEFAULT_SIZE = 1800


# ── Measuring ────────────────────────────────────────────────────────

_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
_TO_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]])


def _oklab(display: np.ndarray) -> np.ndarray:
    """sRGB-encoded 0..1 to OkLab."""
    v = np.clip(np.asarray(display, dtype=np.float64), 0.0, 1.0)
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ _TO_LMS.T) @ _TO_LAB.T


def _blocks(image: np.ndarray, k: int) -> np.ndarray:
    """Means over k x k blocks; a block with any NaN is NaN."""
    h, w = (image.shape[0] // k) * k, (image.shape[1] // k) * k
    return image[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def compare(shown: np.ndarray, camera: np.ndarray) -> dict:
    """How far ``shown`` is from ``camera`` (both display-encoded 0..1, the same
    size; NaN in ``camera`` marks pixels to leave out).

    ``delta_e_coarse`` is the same measure over 4 x 4 pixel blocks: the tone and
    colour, without the fine texture where the camera's sharpening and noise
    reduction differ from a plain develop. Foliage at full size raises
    ``delta_e`` well above it; a gap between the two is texture, not the look.
    """
    from adjust.render import LUMA

    valid = np.isfinite(camera).all(axis=-1)
    a, b = shown[valid], camera[valid]
    if len(a) == 0:
        return {"delta_e": float("nan"), "delta_e_p90": float("nan"),
                "delta_e_coarse": float("nan"), "luma": float("nan"), "banding": float("nan")}
    de = np.linalg.norm(_oklab(a) - _oklab(b), axis=-1) * 100.0
    sa, sb = _blocks(shown, 4), _blocks(camera, 4)
    ok = np.isfinite(sb).all(axis=-1)
    coarse = np.linalg.norm(_oklab(sa[ok]) - _oklab(sb[ok]), axis=-1) * 100.0
    return {
        "delta_e": round(float(de.mean()), 3),
        "delta_e_p90": round(float(np.percentile(de, 90)), 3),
        "delta_e_coarse": round(float(coarse.mean()), 3) if len(coarse) else float("nan"),
        "luma": round(float(np.mean(np.abs(a @ LUMA - b @ LUMA))), 5),
        "vignette": _corner_minus_centre(shown, camera, valid),
        **_banding(shown, camera),
    }


def _banding(shown: np.ndarray, camera: np.ndarray) -> dict:
    from adjust.look import tone_detail

    found = tone_detail(shown, camera)
    if found is None:
        return {"banding": float("nan")}
    return {"banding": found["banding"], "flattest": found["flattest"], "steepest": found["steepest"]}


def _corner_minus_centre(shown: np.ndarray, camera: np.ndarray, valid: np.ndarray) -> float:
    """How much brighter than the camera's picture ours is in the corners than in the
    middle (mean brightness difference, shown minus camera, of the outer ring less the
    inner one, in hundredths of full scale). Near 0 when the lens correction matches the
    camera's; positive when ours is too bright at the edge (the camera darkens the
    corners, as a Ricoh's Retro effect does), negative when too dark."""
    from adjust.render import LUMA

    h, w = valid.shape
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot((x - (w - 1) / 2) / max(1.0, (w - 1) / 2), (y - (h - 1) / 2) / max(1.0, (h - 1) / 2)) / np.sqrt(2.0)
    diff = np.where(valid, shown @ LUMA - np.nan_to_num(camera) @ LUMA, np.nan)
    inner, outer = valid & (r < 0.35), valid & (r > 0.8)
    if inner.sum() < 100 or outer.sum() < 100:
        return float("nan")
    return round(float((np.nanmean(diff[outer]) - np.nanmean(diff[inner])) * 100.0), 2)


# Frames of a bracket must show the same scene; below this correlation of their
# brightness pattern (exposure taken out, at a size where a handheld shift does
# not matter) a frame is another picture taken in the same half minute.
SAME_SCENE = 0.85


def same_scene(a: np.ndarray, b: np.ndarray, stops: float) -> float:
    """How alike two linear frames are once ``b`` is moved ``stops`` brighter:
    the correlation of their log brightness at about 96 pixels across."""
    from adjust.render import LUMA, resize_linear

    small_a = resize_linear(a, 96) @ LUMA
    small_b = resize_linear(b, 96) @ LUMA * (2.0 ** stops)
    if small_a.shape != small_b.shape:
        return 0.0
    la, lb = np.log2(np.maximum(small_a, 1e-4)), np.log2(np.maximum(small_b, 1e-4))
    usable = (la > -9) & (lb > -9) & (small_a < 0.9) & (small_b < 0.9)
    if usable.sum() < 50:
        return 0.0
    return float(np.corrcoef(la[usable], lb[usable])[0, 1])


# ── What a file is ───────────────────────────────────────────────────

@dataclass
class Shot:
    path: Path
    exposure: float               # log2(shutter * ISO / aperture²): +1 is a stop more light
    taken: float                  # seconds since the epoch
    settings: tuple               # what must match within a bracket
    measured: dict = field(default_factory=dict)


def _from_rawpy(path: Path):
    import rawpy

    with rawpy.imread(str(path)) as raw:
        other = raw.other                         # absent in older rawpy releases
        taken = other.timestamp
        stamp = taken.timestamp() if hasattr(taken, "timestamp") else float(taken or 0)
        return (float(other.shutter_speed), float(other.aperture), float(other.iso_speed),
                float(other.focal_length), stamp)


def _from_exiftool(path: Path):
    import subprocess
    from datetime import datetime

    out = subprocess.run(
        ["exiftool", "-s3", "-n", "-ExposureTime", "-FNumber", "-ISO", "-FocalLength",
         "-DateTimeOriginal", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout.split("\n")
    shutter, aperture, iso, focal = (float(v) for v in out[:4])
    stamp = datetime.strptime(out[4].strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    return shutter, aperture, iso, focal, stamp


def read_shot(path: Path) -> Optional[Shot]:
    """The exposure and capture settings of a RAW file, or ``None``.

    Read through rawpy where it offers them, else through exiftool (which the
    app already uses for thumbnails).
    """
    found = None
    for reader in (_from_rawpy, _from_exiftool):
        try:
            found = reader(path)
            if found[0] > 0 and found[1] > 0 and found[2] > 0:
                break
        except Exception:
            found = None
    if not found or not (found[0] > 0 and found[1] > 0 and found[2] > 0):
        return None
    shutter, aperture, iso, focal, stamp = found
    # The picture style belongs to what must match: a scene shot in several Image
    # Controls in a row is not a bracket, and moving one style's frame to another's
    # exposure compares two renderings, not two exposures.
    try:
        from export.engine import _style_of
        style = _style_of(path)
    except Exception:
        style = ""
    return Shot(path=path, exposure=math.log2(shutter * iso / (aperture * aperture)),
                taken=stamp, settings=(round(aperture, 1), round(focal), round(iso), style))


def brackets(shots: list) -> list:
    """Groups of shots that form an exposure bracket: same settings (and picture
    style) apart from shutter, taken together, spanning a real exposure range."""
    groups, current = [], []
    for shot in sorted(shots, key=lambda s: s.taken):
        if current and (shot.settings != current[-1].settings
                        or shot.taken - current[-1].taken > BRACKET_SECONDS):
            groups.append(current)
            current = []
        current.append(shot)
    if current:
        groups.append(current)
    out = []
    for group in groups:
        span = max(s.exposure for s in group) - min(s.exposure for s in group)
        if len(group) >= BRACKET_MIN_FRAMES and span >= BRACKET_MIN_SPAN:
            out.append(group)
    return out


def raw_files(paths) -> list:
    from export.engine import _RAW_EXTENSIONS

    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in _RAW_EXTENSIONS]
        elif p.suffix.lower() in _RAW_EXTENSIONS:
            found.append(p)
    return found


# ── The commands ─────────────────────────────────────────────────────

def _camera_jpeg(path: Path):
    from export import engine

    sibling = engine._sibling_jpeg_rgb(path)
    return sibling if sibling is not None else engine._embedded_camera_rgb(path)


def _frame(path: Path, size: int, relearn: bool = False):
    from export import engine

    return engine.linear_base(path, size, relearn)


def _shown(frame, settings=None) -> np.ndarray:
    from adjust.render import develop_frame

    return np.asarray(develop_frame(frame, settings or {}), dtype=np.float32) / 255.0


def _measure(frame, camera_rgb, settings=None) -> dict:
    from adjust.look import align_camera

    camera = align_camera(frame.rgb, camera_rgb)
    return compare(_shown(frame, settings), camera)


def learn(paths, profiles: Path, size: int = DEFAULT_SIZE, out=sys.stdout) -> list:
    """Open every RAW as the app does, so each joins its camera's pool.

    Stops before opening anything when the profile directory refuses writes:
    learning into memory only would look like success and keep nothing.
    """
    from export import engine

    writable, error = engine.set_profile_dir(profiles)
    if not writable:
        raise ProfilesNotWritable(f"cannot save camera profiles in {profiles}: {error}\n"
                                  f"Fix the volume, or pass --profiles with a folder you can write to.")
    rows = []
    for path in raw_files(paths):
        frame = _frame(path, size, relearn=True)
        look = frame.look if frame is not None else None
        decision = engine.pooling_decision(path) or {}
        rows.append({"file": str(path), "look_frames": getattr(look, "frames", 0), **decision})
        if look is None:
            line = "no look (no matching camera JPEG)"
        else:
            line = f"look from {look.frames} frame(s), lens {sum(look.vignette):+.2f}"
            if decision.get("lens"):
                line += f" ({decision['lens']})"
            if decision:
                line += f" [{decision.get('pool') or '-'}: {decision['reason']}"
                if "own_error" in decision:
                    line += (f"; colour error {decision['pooled_error']:.4f} pooled"
                             f" vs {decision['own_error']:.4f} own")
                line += "]"
        print(f"{path.name}: {line}", file=out)
    return rows


def _aspect(w: int, h: int) -> float:
    return max(w, h) / max(1, min(w, h))


def inspect_file(path: Path) -> dict:
    """What the app can tell about one RAW file and its JPEGs, as plain data."""
    from adjust.look import MAX_ASPECT_MISMATCH
    from export import engine

    info = {"file": str(path), "notes": []}
    style = engine._style_of(path)
    info["style"] = style
    info["camera_key"] = engine._camera_key(path)
    info["lens_key"] = engine._lens_key(path)
    if not style:
        info["notes"].append("exiftool could not read the settings (is it installed?): styles and "
                             "lens settings cannot be told apart, so frames pool by colour data only")
    raw_aspect = None
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            matrix = np.asarray(raw.rgb_xyz_matrix, dtype=float)[:3]
            info["colours"] = int(raw.num_colors)
            info["colour_matrix"] = bool(np.any(matrix != 0)) and int(raw.num_colors) == 3
            info["white_level"] = int(raw.white_level)
            info["black_levels"] = [int(v) for v in raw.black_level_per_channel]
            w, h = raw.sizes.width, raw.sizes.height
            info["size"] = [int(w), int(h)]
            raw_aspect = _aspect(w, h)
    except Exception as e:
        info["notes"].append(f"rawpy could not open it: {e}")
    if "colour_matrix" in info and not info["colour_matrix"]:
        info["notes"].append("no usable colour matrix in the file: highlights are rebuilt whole "
                             "rather than channel by channel, and pools are keyed on the settings alone")
    references = []
    for name, reader in (("sibling JPEG", engine._sibling_jpeg_rgb), ("embedded JPEG", engine._embedded_camera_rgb)):
        rgb = reader(path)
        if rgb is None:
            references.append({"kind": name, "found": False})
            continue
        h, w = rgb.shape[:2]
        same = raw_aspect is not None and abs(_aspect(w, h) / raw_aspect - 1.0) <= MAX_ASPECT_MISMATCH
        references.append({"kind": name, "found": True, "size": [int(w), int(h)], "same_shape": bool(same)})
        if raw_aspect is not None and not same:
            info["notes"].append(f"the {name} has another shape than the RAW (a crop mode, perhaps): "
                                 f"it will not be used")
    info["references"] = references
    usable = [r for r in references if r.get("found") and r.get("same_shape")]
    info["learnable"] = bool(usable)
    if not usable:
        info["notes"].append("nothing to learn the camera's look from: shoot RAW+JPEG, or keep the "
                             "JPEG beside the RAW; the plain decode is shown until then")
    elif max(usable[0]["size"]) < 1500:
        info["notes"].append("the JPEG reference is small; a full-size JPEG beside the RAW is better")
    return info


def inspect(paths, out=sys.stdout) -> list:
    rows = []
    for path in raw_files(paths):
        info = inspect_file(path)
        rows.append(info)
        print(path.name, file=out)
        print(f"  style        {info['style'] or '(unreadable)'}", file=out)
        print(f"  pool key     {info['camera_key'] or '(none: not told apart from other cameras)'}", file=out)
        if "colour_matrix" in info:
            print(f"  colour data  {'matrix present' if info['colour_matrix'] else 'no matrix'}, "
                  f"{info['colours']} colours; raw {info['size'][0]}x{info['size'][1]}, "
                  f"white {info['white_level']}, black {info['black_levels']}", file=out)
        for r in info["references"]:
            if r["found"]:
                print(f"  {r['kind']:<14}{r['size'][0]}x{r['size'][1]}"
                      f"{'' if r['same_shape'] else '  (not the same shape)'}", file=out)
            else:
                print(f"  {r['kind']:<14}none", file=out)
        print(f"  lens setting {info['lens_key'] or '(unknown: vignetting is searched per frame)'}", file=out)
        print(f"  look         {'can be learned' if info['learnable'] else 'cannot be learned'}", file=out)
        for note in info["notes"]:
            print(f"  note         {note}", file=out)
    return rows


class ProfilesNotWritable(RuntimeError):
    pass


def report(paths, profiles: Optional[Path], size: int = DEFAULT_SIZE, out=sys.stdout) -> dict:
    """Measure the look the app would use, frame by frame and bracket by bracket.

    Works on a copy of the stored pools, so a report never changes them.
    """
    from export import engine

    scratch = Path(tempfile.mkdtemp(prefix="photool-profiles-"))
    try:
        if profiles is not None and Path(profiles).is_dir():
            for f in Path(profiles).glob("*.npz"):
                shutil.copy2(f, scratch / f.name)
        engine.set_profile_dir(scratch)
        engine.clear_preview_cache()

        files = raw_files(paths)
        shots, frames_out = {}, []
        print(f"{'file':<24} {'EV':>6} {'look':>6} {'lens':>6} {'dE':>6} {'dE p90':>7} {'coarse':>7} "
              f"{'luma':>7} {'edge':>6} {'band':>5}", file=out)
        for path in files:
            frame = _frame(path, size)
            camera = _camera_jpeg(path)
            shot = read_shot(path)
            row = {"file": str(path), "exposure": None if shot is None else round(shot.exposure, 3)}
            if frame is None or camera is None or frame.look is None:
                row["skipped"] = "no decode" if frame is None else "no camera JPEG or no look"
                print(f"{path.name:<24} skipped: {row['skipped']}", file=out)
            else:
                decision = engine.pooling_decision(path) or {}
                row.update(look_frames=frame.look.frames, pooling=decision.get("reason"),
                           lens_corner=round(float(sum(frame.look.vignette)), 3),
                           **_measure(frame, camera))
                ev = "-" if row["exposure"] is None else f"{row['exposure']:.2f}"
                print(f"{path.name:<24} {ev:>6} {row['look_frames']:>6} {row['lens_corner']:>+6.2f} "
                      f"{row['delta_e']:>6.2f} {row['delta_e_p90']:>7.2f} {row['delta_e_coarse']:>7.2f} "
                      f"{row['luma']:>7.4f} {row['vignette']:>+6.1f} {row['banding']:>5.2f}  "
                      f"{row['pooling'] or ''}", file=out)
                if shot is not None:
                    shots[path] = shot
            frames_out.append(row)

        if files and not shots:
            print("\nno exposure settings could be read (needs a newer rawpy or exiftool on "
                  "PATH), so brackets cannot be found", file=out)
        brackets_out = []
        for group in brackets(list(shots.values())):
            group = sorted(group, key=lambda s: s.exposure)
            # A frame taken alongside the bracket but of another scene must not
            # decide which frame is the middle: find the scene first, then the
            # middle of what belongs to it.
            likeness = _likeness_to(group[len(group) // 2], group, size)
            members = [s for s in group if likeness[s.path] >= SAME_SCENE] or group
            middle = members[len(members) // 2]
            likeness = _likeness_to(middle, group, size)
            reference = _camera_jpeg(middle.path)
            print(f"\nbracket around {middle.path.name} ({len(members)} frames): "
                  f"each frame moved to its exposure with the exposure control", file=out)
            rows = []
            for shot in group:
                frame = _frame(shot.path, size)
                edit = round(middle.exposure - shot.exposure, 3)
                if likeness[shot.path] < SAME_SCENE:
                    alike = likeness[shot.path]
                    rows.append({"file": str(shot.path), "edit": edit,
                                 "skipped": f"not the same scene (likeness {alike:.2f})"})
                    print(f"  {shot.path.name:<22} edit {edit:+.2f} EV  skipped: not the same "
                          f"scene (likeness {alike:.2f})", file=out)
                    continue
                m = _measure(frame, reference, {"exposure": edit} if abs(edit) > 1e-3 else None)
                rows.append({"file": str(shot.path), "edit": edit, **m})
                print(f"  {shot.path.name:<22} edit {edit:+.2f} EV  dE {m['delta_e']:.2f}  "
                      f"coarse {m['delta_e_coarse']:.2f}  luma {m['luma']:.4f}", file=out)
            brackets_out.append({"reference": str(middle.path), "frames": rows})
        return {"frames": frames_out, "brackets": brackets_out}
    finally:
        engine.set_profile_dir(None)
        shutil.rmtree(scratch, ignore_errors=True)


def _likeness_to(middle: Shot, group: list, size: int) -> dict:
    """How alike each frame of ``group`` is to ``middle``, exposure taken out."""
    base = _frame(middle.path, size).rgb
    return {s.path: same_scene(base, _frame(s.path, size).rgb, middle.exposure - s.exposure)
            for s in group}


def _drawn_with(path: Path, look, size: int):
    """``path`` developed with no edit and with ``look`` exactly as given, beside the
    camera's JPEG aligned to it; ``(None, None)`` when either cannot be read.
    Nothing is learned: the frame is decoded directly, not through the app's
    cache, so no look is fitted and no pool is touched."""
    from adjust.look import align_camera, correct_lens
    from adjust.render import resize_linear
    from export import engine

    camera_rgb = _camera_jpeg(path)
    rgb = engine._load_linear(path, size)
    if camera_rgb is None or rgb is None:
        return None, None
    frame = engine.frame_of(path, resize_linear(rgb, size))
    frame.look = look
    frame.rgb = correct_lens(frame.rgb, look)
    return _shown(frame), align_camera(frame.rgb, camera_rgb)


def health(paths, profiles: Path, render: bool = False, size: int = DEFAULT_SIZE,
           out=sys.stdout) -> list:
    """Every kept look under ``profiles`` (only those of photographs under
    ``paths``, when given), worst first, with whether it looks broken."""
    from adjust.look import BANDING_MAX, EVEN_MIN, curve_evenness, tone_detail
    from export import engine

    wanted = [Path(p).resolve() for p in paths or []]
    rows = []
    for f in sorted((Path(profiles) / "looks").glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        try:
            meta, look = engine.read_kept_look(f)
        except Exception as e:
            rows.append({"file": f.name, "unreadable": str(e)})
            continue
        source = Path(meta.get("source", ""))
        if wanted and not any(source == w or w in source.parents for w in wanted):
            continue
        even = curve_evenness(look.curve)
        row = {"file": str(source), "even": None if even is None else round(even, 3),
               "pooling": (meta.get("decision") or {}).get("reason"),
               "version": meta.get("version")}
        try:
            stat = source.stat()
            row["stale"] = (stat.st_mtime_ns != meta.get("mtime_ns") or stat.st_size != meta.get("size")
                            or meta.get("version") != engine._LOOK_FILE_VERSION)
        except OSError:
            row["stale"] = True
            row["missing"] = True
        if render and not row.get("missing"):
            shown, camera = _drawn_with(source, look, size)
            detail = tone_detail(shown, camera) if shown is not None else None
            if detail is not None:
                row.update(detail)
        row["off"] = bool(even is not None and even < EVEN_MIN)
        row["shows"] = bool(row["off"] and row.get("banding", 0.0) > BANDING_MAX)
        rows.append(row)
        if render:
            print(f"  {source.name}: even {row['even']}  banding {row.get('banding', '-')}"
                  f"{'  OFF' if row['off'] else ''}{' SHOWS' if row['shows'] else ''}", file=out, flush=True)

    def badness(row):
        if "unreadable" in row:
            return (2, 0.0)
        return (int(row["off"]) + int(row["shows"]),
                row.get("banding", 0.0) - (row["even"] if row["even"] is not None else 1.0))

    rows.sort(key=badness, reverse=True)
    print(f"\n{'photograph':<44} {'even':>6} {'banding':>8}  pooling", file=out)
    for row in rows:
        if "unreadable" in row:
            print(f"{row['file']:<44} unreadable: {row['unreadable']}", file=out)
            continue
        name = "/".join(Path(row["file"]).parts[-2:])
        even = "-" if row["even"] is None else f"{row['even']:.3f}"
        band = f"{row['banding']:.2f}" if "banding" in row else "-"
        flags = (("OFF " if row["off"] else "") + ("SHOWS " if row["shows"] else "")
                 + ("stale " if row.get("stale") else ""))
        print(f"{name:<44} {even:>6} {band:>8}  {flags}{row['pooling'] or ''}", file=out)
    judged = [r for r in rows if "unreadable" not in r]
    off = [r for r in judged if r["off"]]
    print(f"\n{len(off)} of {len(judged)} kept looks are off (even < {EVEN_MIN})"
          + (f", {sum(r['shows'] for r in off)} visibly in their photograph "
             f"(banding > {BANDING_MAX})" if render else ""), file=out)
    folders: dict = {}
    for r in off:
        folders[str(Path(r["file"]).parent)] = folders.get(str(Path(r["file"]).parent), 0) + 1
    for folder, n in sorted(folders.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {folder}", file=out)
    return rows


def list_profiles(profiles: Path, out=sys.stdout) -> list:
    rows = []
    for f in sorted(Path(profiles).glob("*.npz")):
        try:
            with np.load(f, allow_pickle=False) as data:
                names = [Path(str(p)).name for p in data["frames"]]
        except Exception as e:
            names = [f"unreadable: {e}"]
        rows.append({"profile": f.stem, "frames": names})
        print(f"{f.stem}: {len(names)} frame(s): {', '.join(names)}", file=out)
    return rows


def write_json_checked(path: Path, data) -> None:
    """Write ``data`` as JSON and prove it arrived: flushed to the device, read back
    and parsed, before it replaces anything at ``path``.

    A file whose size was recorded but whose content was never written (a camera
    card, a write cache that was not flushed) looks like success to a plain write,
    and arrives as thousands of zero bytes. Raises OSError, naming the path, when
    what is on disk is not what was written.
    """
    import os

    path = Path(path)
    text = json.dumps(data, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if json.loads(tmp.read_text(encoding="utf-8")) != json.loads(text):
            raise ValueError("content differs")
        tmp.replace(path)
    except (OSError, ValueError) as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise OSError(f"could not save the report intact to {path}: {e}") from e


def _default_profiles() -> Path:
    from config import load_config

    return Path(load_config().thumbnail_cache) / "camera_profiles"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.camera_profile",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("learn", "report"):
        p = sub.add_parser(name)
        p.add_argument("paths", nargs="+", help="RAW files or folders (searched recursively)")
        p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="develop at this long side")
        p.add_argument("--profiles", type=Path, help="profile directory (default: the app's)")
        if name == "report":
            p.add_argument("--json", type=Path, help="also write the measurements here")
    p = sub.add_parser("health")
    p.add_argument("paths", nargs="*", help="only the photographs under these (default: every kept look)")
    p.add_argument("--render", action="store_true",
                   help="also draw each photograph with its kept look against the camera's JPEG (slow)")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="draw at this long side")
    p.add_argument("--profiles", type=Path, help="profile directory (default: the app's)")
    p.add_argument("--json", type=Path, help="also write the rows here")
    p = sub.add_parser("inspect")
    p.add_argument("paths", nargs="+", help="RAW files or folders")
    p = sub.add_parser("list")
    p.add_argument("--profiles", type=Path, help="profile directory (default: the app's)")
    args = parser.parse_args(argv)

    profiles = (getattr(args, "profiles", None) or _default_profiles()) if args.command != "inspect" else None
    if args.command == "learn":
        try:
            learn(args.paths, profiles, args.size)
        except ProfilesNotWritable as e:
            print(e, file=sys.stderr)
            return 2
    elif args.command == "report":
        result = report(args.paths, profiles, args.size)
        if args.json:
            try:
                write_json_checked(args.json, result)
            except OSError as e:
                print(f"\n{e}\nThe table above is complete; copy it, or pass --json with a path "
                      f"on a drive that is not removable.", file=sys.stderr)
                return 3
    elif args.command == "health":
        rows = health(args.paths, profiles, args.render, args.size)
        if args.json:
            try:
                write_json_checked(args.json, rows)
            except OSError as e:
                print(f"\n{e}", file=sys.stderr)
                return 3
    elif args.command == "inspect":
        inspect(args.paths)
    else:
        list_profiles(profiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
