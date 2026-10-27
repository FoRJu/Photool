"""
Paste and match: one edit carried to many frames, with each frame's shooting
variables brought to the frame it was copied from.

Copy and paste carries an interpretation forward: the same curves, mixer and
camera look on every frame. That is right for frames shot in the same light at
the same exposure, and a set rarely is: the meter drifts a third of a stop from
frame to frame, the light changes during an event, and the photographer moves.
So a paste can also adjust each frame, in one of these ways (``PASTE_AS``):

  copied          the edit as it was copied
  keep_light      the copied edit, but each frame keeps its own exposure and
                  white balance: for frames already corrected one by one
  match_exposure  each frame's light (exposure and white balance) brought to the
                  copied frame's, from the best evidence the frame has (see
                  route_light). The id predates the white balance; it is kept so
                  a choice remembered on a device, or a call naming it, still works.
  match_look      the frame matched to the copied frame as developed
                  (adjust.match): only for frames of the same shoot

Measured on R6 II frames of one burst with the meter off by a third or two thirds
of a stop: the capture settings took a frame back to within 0.02 coarse dE of
where it should be (camera look at 0%), a solve from the picture's brightness to
0.53, Match to 0.90, and a plain paste stayed at 1.96. Match across shoots moved
frames further from the copied interpretation than leaving them unedited, in 174
of 210 comparisons: it matches the picture, and cannot tell a picture's look
from its subject.
"""

import logging
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PASTE_AS = ("copied", "keep_light", "match_exposure", "match_look")
# What a frame keeps of its own under keep_light.
LIGHT_KEYS = ("exposure", "temperature", "tint")

# The capture settings are believed only when the picture agrees with them to
# within this many stops. Where the light was the same and only the meter moved,
# the two agreed within 0.26 stop in half the pairs of frames whose settings
# changed; where the photographer re-metered for light that changed, they
# disagreed by 0.5 to 1.2 stops, and correcting by the settings would have undone
# the photographer's own correction. Frames whose settings did not change differ
# in picture brightness by up to 0.21 stop in nine pairs of ten: the subject
# moving, not the light. The same limit decides when the anchors (export.anchors)
# say the light itself changed.
AGREE_STOPS = 0.25
# A white balance shift smaller than this on every channel is left alone: below
# it the anchors disagree among themselves by more. One larger than WB_MAX_STOPS
# is held to it: a frame that far off in colour is lit by something else, and
# the anchors are more likely to be wrong than the frame.
WB_MIN_STOPS = 0.05
WB_MAX_STOPS = 0.25
# Frames are measured at this size: faces of a stage shot from the back are a
# few dozen pixels across here, and it is a size the develop view keeps.
MEASURE_SIZE = 1600
# Match's result goes on at this strength, as the develop view puts it on.
MATCH_AMOUNT = 0.75
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def _light(source: Path) -> Optional[float]:
    """log2(shutter x ISO / aperture^2) from the file's capture settings: +1 is a
    stop more light reaching the sensor. ``None`` when they cannot be read."""
    from tools.camera_profile import read_shot

    shot = read_shot(source)
    return None if shot is None else shot.exposure


def _frame(source: Path) -> Optional[np.ndarray]:
    """The frame's linear light at MEASURE_SIZE, lens-corrected, as the develop
    view decodes it."""
    from export.engine import linear_base

    frame = linear_base(source, MEASURE_SIZE)
    return None if frame is None else frame.rgb


def _brightness(rgb: Optional[np.ndarray]) -> Optional[float]:
    """Median linear brightness of the frame's lit, unclipped pixels."""
    if rgb is None:
        return None
    y = rgb @ _LUMA.astype(np.float32)
    ok = (y > 1e-4) & (y < 0.9)
    return float(np.median(y[ok])) if ok.any() else None


def choose_exposure(capture: Optional[float], picture: Optional[float]) -> dict:
    """What to add to a frame's exposure when it shares nothing with the copied
    frame to measure against: the difference its capture settings record
    (``capture``), when the difference its picture shows (``picture``) agrees,
    both in stops towards the frame copied from."""
    if capture is None:
        return {"stops": 0.0, "used": None, "reason": "capture settings could not be read"}
    if abs(capture) < 0.01:
        return {"stops": 0.0, "used": "capture", "reason": "shot at the same settings"}
    if picture is None:
        return {"stops": 0.0, "used": None, "reason": "the picture could not be measured"}
    if abs(capture - picture) > AGREE_STOPS:
        return {"stops": 0.0, "used": None,
                "reason": f"the light changed: its settings say {capture:+.2f} EV, its picture {picture:+.2f} EV"}
    return {"stops": round(capture, 2), "used": "capture", "reason": f"{capture:+.2f} EV from capture settings"}


def _shared(anchors: dict) -> str:
    faces, points = anchors["faces"], anchors["points"]
    parts = []
    if faces:
        parts.append(f"{faces} shared face{'' if faces == 1 else 's'}")
    if points:
        parts.append(f"{points} matched points")
    return " and ".join(parts)


def _colour(wb: np.ndarray) -> str:
    warm = wb[0] - wb[2]
    if abs(warm) < WB_MIN_STOPS:
        return ""
    return " and warmer" if warm > 0 else " and cooler"


def route_light(capture: Optional[float], picture: Optional[float], anchors: Optional[dict]) -> dict:
    """The light to give a frame, from the best evidence it has, as ``{exposure,
    wb, used, changed, reason}``: ``exposure`` in stops and ``wb`` a gain per
    channel in stops (summing to nothing in brightness), both towards the frame
    copied from.

    The capture settings are exact while the light stays the same, so they are
    the answer whenever they can be checked and pass: against the anchors (the
    same people and things in both frames, export.anchors) when the frames share
    any, else against the picture's brightness. When the anchors disagree with
    them, the light itself changed, and the anchors are the measurement. Measured
    on a stage shoot, the same person's skin against the hero's: where the
    capture settings changed between the frames, 4.5 OkLab dE x100 this way,
    against 5.9 by the capture settings alone and 7.2 for a plain paste.
    """
    none = np.zeros(3)
    if anchors is None:
        found = choose_exposure(capture, picture)
        reason = found["reason"]
        if found["used"] == "capture" and abs(found["stops"]) >= 0.01:
            reason += " (nothing shared with the copied frame to check it)"
        return {"exposure": found["stops"], "wb": none, "used": found["used"],
                "changed": "light changed" in reason, "reason": reason}
    stops = np.asarray(anchors["stops"], dtype=np.float64)
    measured = float(anchors["exposure"])
    wb = stops - measured
    if not anchors.get("colour", True) or np.abs(wb).max() < WB_MIN_STOPS:
        wb = np.zeros(3)
    wb = np.clip(wb, -WB_MAX_STOPS, WB_MAX_STOPS)
    shared = _shared(anchors)
    if capture is not None and abs(measured - capture) <= AGREE_STOPS:
        if abs(capture) < 0.01:
            reason = f"shot at the same settings, as {shared} confirm"
        else:
            reason = f"{capture:+.2f} EV from capture settings, confirmed by {shared}"
        if wb.any():
            reason += f"; white balance{_colour(wb) or ' moved'} from them"
        return {"exposure": round(capture, 2), "wb": wb, "used": "capture", "changed": False, "reason": reason}
    reason = f"the light changed: {measured:+.2f} EV{_colour(wb)} from {shared}"
    if capture is not None:
        reason += f" (its settings say {capture:+.2f} EV)"
    return {"exposure": round(measured, 2), "wb": wb, "used": "anchors", "changed": True, "reason": reason}


def to_sliders(copied: dict, exposure: float, wb: np.ndarray) -> dict:
    """The exposure, temperature and tint that change a frame developed with
    ``copied`` by ``exposure`` stops and ``wb`` (a gain per channel in stops).

    Temperature and tint are gains on the linear frame (adjust.develop.adjustment:
    red 1 + warm, blue 1 - warm, green from tint), so the wanted ratios between
    the channels give them exactly, and the exposure takes up whatever brightness
    they move.
    """
    from adjust.develop import MAX_TINT_GAIN, MAX_WB_GAIN

    def gains(temperature, tint):
        warm = temperature / 100.0 * MAX_WB_GAIN
        green = -tint / 100.0 * MAX_TINT_GAIN
        return np.array([1.0 + warm, 1.0 + green, 1.0 - warm])

    t0, n0 = float(copied.get("temperature", 0) or 0), float(copied.get("tint", 0) or 0)
    e0 = float(copied.get("exposure", 0.0) or 0.0)
    before = gains(t0, n0)
    wanted = before * np.exp2(np.asarray(wb, dtype=np.float64))
    a, c = wanted[0] / wanted[1], wanted[2] / wanted[1]
    warm, green = (a - c) / (a + c), 2.0 / (a + c) - 1.0
    temperature = int(round(max(-100.0, min(100.0, warm / MAX_WB_GAIN * 100.0))))
    tint = int(round(max(-100.0, min(100.0, -green / MAX_TINT_GAIN * 100.0))))
    moved = np.log2(gains(temperature, tint) / before)
    total = exposure + np.asarray(wb, dtype=np.float64)
    out = {"exposure": round(e0 + float((total - moved) @ _LUMA), 2)}
    if temperature or "temperature" in copied:
        out["temperature"] = temperature
    if tint or "tint" in copied:
        out["tint"] = tint
    return out


def adjuster(library_root: Path, source: Optional[str], copied: dict, paste_as: str,
             prefer_raw: bool = True, faces_of: Optional[Callable] = None) -> Optional[Callable]:
    """The per-frame step of a paste made ``paste_as``: a function taking a
    target's path, its edit with the copied parts put on and its edit from
    before, and giving back ``(edit, note)``. ``None`` when the paste puts the
    edit on as copied.

    ``copied`` is the whole edit of the frame ``source`` (catalog paths) as it was
    copied; a match is made to that frame as it was developed then. ``faces_of``
    gives a catalog path's faces as ``(person, x, y, w, h)``, for the anchors.
    """
    from export.engine import resolve_export_source

    if paste_as in (None, "", "copied"):
        return None
    if paste_as not in PASTE_AS:
        raise ValueError(f"paste_as must be one of {', '.join(PASTE_AS)}")
    root = Path(library_root)

    def file_of(rel: str) -> Path:
        return resolve_export_source(root / rel, prefer_raw)

    if paste_as == "keep_light":
        def keep(path: str, edit: dict, own: dict):
            out = {k: v for k, v in edit.items() if k not in LIGHT_KEYS}
            out.update({k: own[k] for k in LIGHT_KEYS if k in own})
            return out, "kept its own exposure and white balance"
        return keep

    if not source:
        raise ValueError("matching needs the frame the edit was copied from")
    reference = file_of(source)

    if paste_as == "match_exposure":
        from export import anchors as anchoring

        hero = {}

        def hero_once():
            if not hero:
                rgb = _frame(reference)
                hero.update(rgb=rgb, light=_light(reference), brightness=_brightness(rgb),
                            faces=list(faces_of(source)) if faces_of else [],
                            features=anchoring.features(rgb) if rgb is not None else None)
            return hero

        def match_light(path: str, edit: dict, own: dict):
            if path == source:
                return edit, None
            try:
                h = hero_once()
                target = file_of(path)
                rgb = _frame(target)
                lt = _light(target)
                capture = None if h["light"] is None or lt is None else h["light"] - lt
                bt = _brightness(rgb)
                picture = math.log2(h["brightness"] / bt) if h["brightness"] and bt else None
                anchors = None
                if h["rgb"] is not None and rgb is not None:
                    faces = anchoring.face_pairs(h["rgb"], h["faces"], rgb, list(faces_of(path)) if faces_of else [])
                    scene = anchoring.scene_pairs(h["rgb"], h["features"], rgb, anchoring.features(rgb))
                    anchors = anchoring.solve(faces, scene)
                light = route_light(capture, picture, anchors)
            except Exception as e:
                logger.warning(f"Light not matched for {path}: {e}")
                return edit, "light left as pasted: the frame could not be measured"
            if light["used"] is None:
                return edit, f"light left as pasted: {light['reason']}"
            out = dict(edit)
            out.update(to_sliders(copied, light["exposure"], light["wb"]))
            return out, f"exposure {out['exposure']:+.2f} ({light['reason']})"
        return match_light

    # match_look
    from adjust.develop import CURVE_KEYS
    from adjust.match import match, scaled
    from adjust.render import develop_frame
    from export.engine import camera_image, linear_base

    look_size = 512

    def developed(path: Path, settings: dict):
        frame = linear_base(path, look_size)
        if frame is not None:
            return develop_frame(frame, settings)
        return camera_image(path, look_size)

    hero_look = developed(reference, copied)
    hero_look = None if hero_look is None else np.asarray(hero_look.convert("RGB"))
    shoot = Path(source).parent

    def match_look(path: str, edit: dict, own: dict):
        if path == source:
            return edit, None
        if hero_look is None:
            return edit, "look not matched: the copied frame could not be read"
        base = {k: v for k, v in edit.items() if k not in CURVE_KEYS and k not in ("saturation", "mono")}
        frame = developed(file_of(path), base)
        if frame is None:
            return edit, "look not matched: the frame could not be read"
        found = match(np.asarray(frame.convert("RGB")), hero_look)
        note = "look matched to the copied frame"
        if Path(path).parent != shoot:
            note += " (from another shoot: Match copies a picture's content as well as its look)"
        return {**base, **scaled(found, MATCH_AMOUNT)}, note
    return match_look


def faces_from(db) -> Callable:
    """``faces_of`` for adjuster, from the catalog: a photograph's known faces,
    found under its own path or, for a RAW, the JPEG beside it (faces are found on
    the half of a pair the grid shows)."""
    def faces_of(rel: str) -> list:
        rows = db.list_faces(rel)
        if not rows:
            stem, _, ext = rel.rpartition(".")
            if ext.lower() not in ("jpg", "jpeg"):
                for other in ("JPG", "jpg", "jpeg", "JPEG"):
                    rows = db.list_faces(f"{stem}.{other}")
                    if rows:
                        break
        return [(r["cluster_id"], r["x"], r["y"], r["w"], r["h"]) for r in rows if r.get("cluster_id")]
    return faces_of
