"""
Styles: a rendering learned once and put on any RAW file, from any camera.

A *look* (``adjust.look``) is what one camera's JPEG engine does to its own
sensor data: a colour matrix, a tone curve applied per channel and a hue stage.
A *style* is the same thing kept under a name and put on another camera's files:
a Ricoh GR's Positive Film on a Canon CR3, Canon's Neutral on a Sony ARW, or the
rendering of your own edited exports (a RAW with an edited JPEG beside it is a
frame that teaches that edit).

**Why it translates.** Every RAW is decoded into the same middle space before a
look ever sees it: the sensor's values are white-balanced with the camera's own
gains and taken through that camera's colour matrix (LibRaw's, from Adobe's
calibration of that sensor against CIE XYZ) into scene-linear sRGB/Rec.709
primaries with headroom above white, the lens's vignetting corrected. That is a
camera-independent description of the scene's light — up to the accuracy of one
matrix per sensor — and it is what the look's matrix starts from. So a look fitted
on a Ricoh's decode describes "what Positive Film does to light", not "what it
does to a Ricoh's numbers", and it can be applied to a Canon's decode as it is.

**What does not translate: where middle grey sits.** Cameras disagree about how
bright the same scene lands in their raw numbers — ISO calibration differs by
make, and Canon's Highlight Tone Priority deliberately records a stop darker to
keep highlights. A tone curve is a function of brightness, so applied at the wrong
level it is the wrong curve. Each frame is therefore *anchored*: a single exposure
factor is found so that the style renders the frame's middle brightness where the
frame's own camera rendered it (its JPEG, through its own learned look). The style
supplies the character — contrast, roll-off, colour — and the frame keeps the
brightness its photographer chose. A RAW with no JPEG to learn from is anchored
to the style's own typical brightness instead. The exposure slider still works on
top, and moves the frame along the style's curve.

What it does not do: a sensor's matrix is one 3x3 fitted at one illuminant, so two
cameras' decodes of a strongly saturated colour can disagree by a few ΔE before
any style is applied; the style cannot know that. And a style learned from frames
of one kind of scene carries that scene's tone choices (a style learned only from
night shots will lift shadows).
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from adjust.look import (
    EVEN_MIN,
    CameraLook,
    FrameSamples,
    HUE_TERMS,
    _white_stops,
    apply_look,
    curve_evenness,
    fit_look,
    look_error,
)
from adjust.render import LUMA, _shoulder, encode, resize_linear

logger = logging.getLogger(__name__)

# A style id is what goes into an edit and a sidecar: short, safe in a file name.
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")

# The anchor is searched over this many stops either side of no change.
ANCHOR_RANGE = 5.0
# ... on a copy of the frame this wide.
ANCHOR_PROBE = 160
# The anchor matches the frame's brightness at these quantiles of its luma, so
# one window or one shadow does not decide it.
_ANCHOR_QUANTILES = (0.35, 0.5, 0.65)

# Most frames a style is learned from; spread evenly over what was offered.
MAX_LEARN_FRAMES = 40

# Improving a style with frames from other folders (see improve()). On real
# shoots, a style learned from six frames got 40-45% of the way to one learned
# from twenty-one of the same folder once ten to twenty frames with the same
# camera settings were added from other folders; frames shot with other
# settings (Highlight Tone Priority off, another Picture Style) made it nearly
# three times worse. So only same-settings frames are ever offered, and the
# result is kept only if it does no worse on the style's own frames.
MAX_ADDED_FRAMES = 20
MAX_TOTAL_FRAMES = 80
# A candidate the current style misses by more than this many times what a fit
# to that frame alone misses by is left out: an edited JPEG beside the RAW, a
# frame from a setting the key cannot see. Same-settings frames measured 1.4-4.1.
SCREEN_MAX = 6.0
# Folds of the held-out check. A style from two frames is checked frame against frame.
FOLDS = 5

_DIR: Optional[Path] = None
_LOCK = threading.Lock()
_CACHE: dict = {}          # id -> (mtime_ns, Style)


class Style:
    """A named look, with what it was learned from."""

    __slots__ = ("id", "name", "look", "frames", "source", "created", "typical",
                 "key", "settings", "folders", "improved")

    def __init__(self, id: str, name: str, look: CameraLook, frames: int = 0,
                 source: str = "", created: float = 0.0, typical: float = 0.45,
                 key: str = "", settings: str = "", folders: Sequence[str] = (),
                 improved: Optional[dict] = None):
        self.id = id
        self.name = name
        self.look = look
        self.frames = frames
        self.source = source
        self.created = created
        # The median display brightness of the frames it was learned from: the
        # anchor for a frame that has no camera JPEG of its own.
        self.typical = typical
        # The camera and its rendering settings the frames were shot with: the
        # pool key, the readable settings string (its first field the model),
        # and the folders they came from. Frames from other folders are only
        # ever added when they share the key. Empty for styles learned before.
        self.key = key
        self.settings = settings
        self.folders = list(folders)
        # The last improvement: what was added and how it measured.
        self.improved = improved

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "frames": self.frames,
                "source": self.source, "created": self.created,
                "folders": len(self.folders), "improvable": bool(self.key),
                "improved": self.improved, "has_previous": has_previous(self.id)}


# ── Storage ──────────────────────────────────────────────────────────

def set_dir(path) -> None:
    """Keep styles in ``path`` (created on first save)."""
    global _DIR
    with _LOCK:
        _DIR = Path(path) if path else None
        _CACHE.clear()


def style_dir() -> Optional[Path]:
    return _DIR


def valid_id(value) -> bool:
    return isinstance(value, str) and bool(_ID.match(value))


def slug(name: str) -> str:
    """An id from a name: lower case, dashes, unique among the stored styles."""
    base = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40] or "style"
    if not base[0].isalnum():
        base = "s" + base
    candidate, n = base, 2
    while _DIR is not None and (_DIR / f"{candidate}.npz").exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def save(style: Style, samples: Sequence[FrameSamples] = ()) -> Path:
    """Write a style: the fitted look (so loading is instant) and the samples it
    was fitted to (so a later version of the fit can redo it)."""
    if _DIR is None:
        raise RuntimeError("no styles directory has been configured")
    _DIR.mkdir(parents=True, exist_ok=True)
    look = style.look
    path = _DIR / f"{style.id}.npz"
    tmp = _DIR / f"{style.id}.tmp.npz"
    arrays = dict(
        curve=np.asarray(look.curve, np.float32),
        matrix=np.asarray(look.matrix, np.float32),
        hue=(np.asarray(look.hue, np.float32) if look.hue is not None
             else np.zeros((0, HUE_TERMS), np.float32)),
        meta=np.array(json.dumps({
            "id": style.id, "name": style.name, "frames": style.frames,
            "source": style.source, "created": style.created or time.time(),
            "typical": style.typical, "error": float(look.error),
            "key": style.key, "settings": style.settings, "folders": style.folders,
            "improved": style.improved,
        })),
    )
    if samples:
        arrays["counts"] = np.array([len(s.linear) for s in samples])
        arrays["linear"] = np.concatenate([s.linear for s in samples]).astype(np.float32)
        arrays["camera"] = np.concatenate([s.camera for s in samples]).astype(np.float32)
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    with _LOCK:
        _CACHE.pop(style.id, None)
    return path


def load(style_id: str) -> Optional[Style]:
    """A stored style, or ``None``. Kept in memory until its file changes."""
    if not valid_id(style_id) or _DIR is None:
        return None
    path = _DIR / f"{style_id}.npz"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    with _LOCK:
        hit = _CACHE.get(style_id)
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            curve = data["curve"].astype(np.float32)
            hue = data["hue"].astype(np.float32)
            look = CameraLook(curve=curve, matrix=data["matrix"].astype(np.float32),
                              white_stops=_white_stops(curve), error=float(meta.get("error", 0.0)),
                              frames=int(meta.get("frames", 0)),
                              hue=hue if hue.size else None)
    except Exception as e:
        logger.warning(f"Style {path.name} unreadable: {e}")
        return None
    style = Style(style_id, meta.get("name") or style_id, look, int(meta.get("frames", 0)),
                  meta.get("source", ""), float(meta.get("created", 0.0)),
                  float(meta.get("typical", 0.45)), key=meta.get("key", ""),
                  settings=meta.get("settings", ""), folders=meta.get("folders") or [],
                  improved=meta.get("improved"))
    with _LOCK:
        _CACHE[style_id] = (mtime, style)
    return style


def list_styles() -> list[dict]:
    if _DIR is None or not _DIR.is_dir():
        return []
    out = []
    for path in sorted(_DIR.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        style = load(path.stem)
        if style is not None:
            out.append(style.summary())
    return sorted(out, key=lambda s: (s["name"].lower(), s["id"]))


def delete(style_id: str) -> bool:
    """Put a style aside in ``deleted/``, out of the list but not gone.

    A style is minutes of learning from frames that may since have moved, and a
    click in the wrong place used to erase it for good. Copying the file back
    out of ``deleted/`` brings it back, under the same id the edits still name.
    """
    if not valid_id(style_id) or _DIR is None:
        return False
    path = _DIR / f"{style_id}.npz"
    if not path.exists():
        return False
    aside = _DIR / "deleted"
    try:
        aside.mkdir(exist_ok=True)
        target = aside / path.name
        if target.exists():
            target = aside / f"{style_id}-{int(time.time())}.npz"
        path.replace(target)
        previous = _previous_path(style_id)
        if previous is not None and previous.exists():
            previous.unlink()
    except OSError:
        return False
    with _LOCK:
        _CACHE.pop(style_id, None)
    return True


def rename(style_id: str, name: str) -> bool:
    style = load(style_id)
    if style is None or _DIR is None:
        return False
    style.name = (name or "").strip()[:80] or style.name
    # Re-save keeps the samples: read them back first.
    save(style, samples_of(style_id))
    return True


def samples_of(style_id: str) -> list:
    """The per-frame samples a style was fitted to, as stored with it."""
    samples = []
    if _DIR is None or not valid_id(style_id):
        return samples
    try:
        with np.load(_DIR / f"{style_id}.npz", allow_pickle=False) as data:
            if "counts" in data.files:
                bounds = np.concatenate([[0], np.cumsum(data["counts"])])
                for i in range(len(data["counts"])):
                    a, b = int(bounds[i]), int(bounds[i + 1])
                    samples.append(FrameSamples(linear=data["linear"][a:b], camera=data["camera"][a:b]))
    except Exception:
        samples = []
    return samples


# ── The version before an improvement ────────────────────────────────

def _previous_path(style_id: str) -> Optional[Path]:
    return None if _DIR is None else _DIR / "previous" / f"{style_id}.npz"


def has_previous(style_id: str) -> bool:
    path = _previous_path(style_id)
    return bool(path and path.exists())


def keep_previous(style_id: str) -> bool:
    """Copy the style as it is now aside, so one improvement can be undone."""
    import shutil

    if _DIR is None or not valid_id(style_id):
        return False
    path, aside = _DIR / f"{style_id}.npz", _previous_path(style_id)
    if not path.exists():
        return False
    aside.parent.mkdir(exist_ok=True)
    shutil.copy2(path, aside)
    return True


def restore_previous(style_id: str) -> bool:
    """Put back the version kept before the last improvement."""
    aside = _previous_path(style_id)
    if not valid_id(style_id) or aside is None or not aside.exists():
        return False
    aside.replace(_DIR / f"{style_id}.npz")
    with _LOCK:
        _CACHE.pop(style_id, None)
    return True


# ── Learning ─────────────────────────────────────────────────────────

def from_samples(name: str, samples: Sequence[FrameSamples], source: str = "",
                 typical: Optional[float] = None, style_id: Optional[str] = None) -> Optional[Style]:
    """Fit one look to the samples of many frames and keep it as a style.
    ``None`` when the frames do not agree on a rendering the fit can trust."""
    samples = [s for s in samples if s is not None and len(s.linear)]
    if not samples:
        return None
    look = fit_look(samples)
    if look is None:
        return None
    if typical is None:
        medians = [float(np.median(s.camera @ LUMA)) for s in samples]
        typical = float(np.median(medians)) if medians else 0.45
    return Style(style_id or slug(name), (name or "").strip()[:80] or "Style", look,
                 frames=len(samples), source=source, created=time.time(), typical=typical)


def held_out_error(own: Sequence[FrameSamples], added: Sequence[FrameSamples] = (),
                   folds: int = FOLDS) -> float:
    """How well a style fitted to ``own`` (and ``added``) renders frames of its
    own it was not fitted to: each fold of ``own`` is left out in turn, the look
    is fitted to the rest plus every added frame, and measured on the fold left
    out. On both cameras tried it followed the error on frames never used at
    all: down with same-settings frames added, up sharply with other settings.
    ``inf`` when a fold cannot be fitted; ``nan`` with fewer than two frames."""
    own = [s for s in own if s is not None and len(s.linear)]
    if len(own) < 2:
        return float("nan")
    folds = max(2, min(folds, len(own)))
    errors = []
    for f in range(folds):
        test = [s for i, s in enumerate(own) if i % folds == f]
        train = [s for i, s in enumerate(own) if i % folds != f] + list(added)
        look = fit_look(train)
        if look is None:
            return float("inf")
        errors.append(look_error(look, test, colour=True))
    return float(np.mean(errors))


def screen(look: CameraLook, candidates: Sequence[FrameSamples]) -> list:
    """``(ratio, sample)`` for each candidate the style could take, best first:
    how far the style misses the frame, against how far a fit to that frame
    alone does. Frames over SCREEN_MAX are left out."""
    kept = []
    for s in candidates:
        own = fit_look([s])
        if own is None:
            continue
        ratio = look_error(look, [s], colour=True) / max(look_error(own, [s], colour=True), 1e-4)
        if ratio <= SCREEN_MAX:
            kept.append((ratio, s))
    kept.sort(key=lambda rs: rs[0])
    return kept


def improve(style: Style, own: Sequence[FrameSamples], candidates: Sequence[FrameSamples]):
    """Try the style with ``candidates`` (same camera and settings, other
    folders) added. Returns ``(new_style or None, report, added samples)``: a new style only when
    it does no worse on the style's own frames held out, and its curve is not one
    of the plateaued curves that posterize faces. All the screened frames are
    tried first, then the better half, before giving up."""
    report = {"offered": len(candidates), "screened": 0, "added": 0,
              "before": None, "after": None, "reason": ""}
    room = max(0, MAX_TOTAL_FRAMES - len(own))
    ranked = screen(style.look, candidates)
    report["screened"] = len(ranked)
    ranked = ranked[:min(MAX_ADDED_FRAMES, room)]
    if not ranked:
        report["reason"] = ("it already has as many frames as a style keeps" if not room else
                            "none of the frames found is rendered like this style")
        return None, report, []
    before = held_out_error(own)
    report["before"] = before
    if not np.isfinite(before):
        report["reason"] = "it was learned from too few frames to check an improvement against"
        return None, report, []
    tries = [ranked]
    if len(ranked) >= 4:
        tries.append(ranked[:len(ranked) // 2])
    for added in tries:
        extra = [s for _, s in added]
        after = held_out_error(own, extra)
        report["after"] = after
        if not after <= before:
            continue
        look = fit_look(list(own) + extra)
        even = curve_evenness(look.curve) if look is not None else None
        if look is None or (even is not None and even < EVEN_MIN):
            continue
        typical = float(np.median([float(np.median(s.camera @ LUMA)) for s in list(own) + extra]))
        better = Style(style.id, style.name, look, frames=len(own) + len(extra), source=style.source,
                       created=style.created, typical=typical, key=style.key,
                       settings=style.settings, folders=style.folders)
        report["added"] = len(extra)
        return better, report, extra
    report["reason"] = "adding them made it worse on its own frames"
    return None, report, []


def spread(items: list, limit: int = MAX_LEARN_FRAMES) -> list:
    """At most ``limit`` items, evenly spaced over the list (a long shoot is
    learned from its whole length, not its first forty frames)."""
    if len(items) <= limit:
        return list(items)
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


# ── Applying ─────────────────────────────────────────────────────────

def _probe(rgb: np.ndarray) -> np.ndarray:
    return resize_linear(np.ascontiguousarray(rgb, dtype=np.float32), ANCHOR_PROBE)


def _levels(display: np.ndarray) -> np.ndarray:
    luma = np.asarray(display, np.float32).reshape(-1, 3) @ LUMA
    return np.quantile(luma, _ANCHOR_QUANTILES)


def own_rendering(rgb: np.ndarray, look: Optional[CameraLook]) -> np.ndarray:
    """A frame as its own camera rendered it (its learned look), or the plain
    decode when it has none."""
    if look is not None:
        return apply_look(rgb, look, 1.0)
    return encode(_shoulder(rgb))


def anchor(rgb: np.ndarray, style: Style, own_look: Optional[CameraLook]) -> float:
    """The exposure factor that puts this frame's middle brightness, through the
    style, where its own camera put it.

    Solved by bisection in stops: the style's rendering of a brighter input is
    never darker, so the match is monotonic. Without a camera look of its own the
    frame is matched to the style's typical brightness instead.
    """
    probe = _probe(rgb)
    if own_look is not None:
        target = _levels(own_rendering(probe, own_look))
    else:
        target = np.full(len(_ANCHOR_QUANTILES), float(style.typical), np.float32)
    target_mid = float(np.mean(target))
    if not np.isfinite(target_mid) or target_mid <= 0.0:
        return 1.0

    def shown(stops: float) -> float:
        return float(np.mean(_levels(apply_look(probe * np.float32(2.0 ** stops), style.look, 1.0))))

    low, high = -ANCHOR_RANGE, ANCHOR_RANGE
    if shown(low) >= target_mid:
        return float(2.0 ** low)
    if shown(high) <= target_mid:
        return float(2.0 ** high)
    for _ in range(22):
        mid = (low + high) / 2.0
        if shown(mid) < target_mid:
            low = mid
        else:
            high = mid
    return float(2.0 ** ((low + high) / 2.0))


def render(rgb: np.ndarray, style: Style, factor: float, exposure: float = 0.0) -> np.ndarray:
    """Linear light to display colour through a style, at the frame's anchor."""
    return apply_look(np.asarray(rgb, np.float32) * np.float32(factor), style.look, 1.0,
                      exposure=exposure)
