"""
A second exposure laid over a frame, as a camera's multiple exposure does it.

Light adds up, so the two are put together in linear light, before the tone: the same edit
then shapes the result as one picture, and a RAW and a JPEG can be mixed since both reach
linear light. Its four modes are the ones Canon bodies offer:

  * ``add``      the two lights added, as film does. Bright where either is bright.
  * ``average``  added, then brought back down, so the whole keeps the brightness of one.
  * ``lighten``  whichever is brighter at each pixel.
  * ``darken``   whichever is darker at each pixel.

An edit names the second photograph by its path under the library and how it is laid on
(:func:`normalize_second`), placed to cover the frame, however differently shaped. It can be
kept to the people, the subject or the sky of the first, which is how a landscape comes to
fill a silhouette.

The second comes in one of two ways:

  * **as shot**, with no ``version``: its raw light, mixed before the tone, so the first
    photograph's edit shapes the two as one picture, as a camera's multiple exposure does.
  * **as edited**, with a ``version``: that version of it (0 its own edit, 1.. a virtual
    copy) finished by its own ``edit``, which the second keeps a copy of. Each picture then
    keeps its own look, crop and colour, and the two are mixed as light once both are done
    (:func:`laid_on`). A black-and-white copy laid over a colour frame stays black and white.
    The copy of the edit keeps the double exposure as it was made when the other photo is
    edited again; the panel says when the two differ and offers to take the new one.

The second file is needed each time the edit is drawn. If it has gone, the frame is drawn
without it, and :func:`missing` is how the panel knows to say so.
"""

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODES = ("add", "average", "lighten", "darken")
REACHES = ("", "subject", "people", "sky")
# (low, high, default) of each number. Amount is a percentage of the second's light, over 100
# for a faint second frame; scale is of the size that just covers the frame; x and y move it
# by a percentage of the frame.
_NUMBERS = {
    "amount": (1, 200, 100),
    "exposure": (-3.0, 3.0, 0.0),
    "x": (-100, 100, 0),
    "y": (-100, 100, 0),
    "scale": (100, 400, 100),
}
# A preview's second frame is read at this size, anything larger from the whole file.
_READ_SIZE = 2048
# Pictures kept read: a preview and a full-size one, and the one before them.
_KEPT = 3

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

_library_root: Optional[str] = None
_pictures: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_reading = threading.Lock()
_warned: set = set()


def configure(library_root: Optional[str]) -> None:
    global _library_root
    if library_root != _library_root:
        _library_root = library_root
        _pictures.clear()
        _warned.clear()


def _number(raw: dict, key: str):
    low, high, default = _NUMBERS[key]
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not np.isfinite(value):
        value = default
    value = min(high, max(low, value))
    return round(value, 2) if isinstance(default, float) else int(round(value))


def normalize_second(raw) -> Optional[dict]:
    """The second exposure of an edit with everything in range, or ``None`` if there is none.
    Text is read as the sidecar writes it. Its path stays under the library."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, dict):
        return None
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.strip().replace("\\", "/")
    parts = PurePosixPath(path).parts
    if path.startswith("/") or ":" in parts[0] or ".." in parts:
        return None
    out = {"path": path, "mode": raw.get("mode") if raw.get("mode") in MODES else MODES[0]}
    out.update({key: _number(raw, key) for key in _NUMBERS})
    out["flip"] = bool(raw.get("flip"))
    out["reach"] = raw.get("reach") if raw.get("reach") in REACHES else ""
    out["invert"] = bool(raw.get("invert")) and bool(out["reach"])
    version = raw.get("version")
    if isinstance(version, (int, float)) and not isinstance(version, bool) and np.isfinite(version)             and version >= 0:
        out["version"] = int(version)
        out["edit"] = _edit_of(raw.get("edit"))
    return out


def _edit_of(raw) -> dict:
    """A second's own edit, made clean. Its own second exposure is left out: one photograph
    laid over another, never a chain of them."""
    from adjust.develop import normalize

    if not isinstance(raw, dict):
        return {}
    clean = normalize({k: v for k, v in raw.items() if k != "second"})
    clean.pop("second", None)
    return clean


def developed(second: dict) -> bool:
    """Whether the second comes as edited, with a look of its own, rather than as shot."""
    return "version" in second


def second_to_text(second: dict) -> str:
    return json.dumps(second, separators=(",", ":"), sort_keys=True)


def _file_of(path: str) -> Optional[Path]:
    """The file under the library, or ``None`` if there is none."""
    if not _library_root:
        return None
    root = Path(_library_root).resolve()
    try:
        file = (root / path).resolve()
        file.relative_to(root)
        return file if file.is_file() else None
    except (OSError, ValueError):
        return None


def missing(second: Optional[dict]) -> bool:
    """Whether an edit's second photograph can no longer be found."""
    return bool(second) and _file_of(second["path"]) is None


def _picture(path: str, longest: float) -> Optional[np.ndarray]:
    """The second photograph as shot, linear, at about ``longest`` pixels or more."""
    file = _file_of(path)
    if file is None:
        if path not in _warned:
            _warned.add(path)
            logger.warning("[double] the second exposure %s is not there; drawing without it", path)
        return None
    from export.engine import _load_linear, resolve_export_source

    size = _READ_SIZE if longest <= _READ_SIZE else 0
    return _kept((str(file), file.stat().st_mtime_ns, size), path,
                 lambda: _load_linear(resolve_export_source(file), size))


def _kept(key: tuple, path: str, read) -> Optional[np.ndarray]:
    """``read()`` once for ``key``, then the picture kept."""
    with _reading:
        kept = _pictures.get(key)
        if kept is not None:
            _pictures.move_to_end(key)
            return kept
        rgb = read()
        if rgb is None:
            logger.warning("[double] the second exposure %s cannot be read", path)
            return None
        _pictures[key] = rgb
        while len(_pictures) > _KEPT:
            _pictures.popitem(last=False)
    return rgb


def _finished(second: dict, longest: float) -> Optional[np.ndarray]:
    """The second photograph as its own edit finishes it, back in linear light, at about
    ``longest`` pixels or more: its crop, its colour, its grain and all."""
    path = second["path"]
    file = _file_of(path)
    if file is None:
        if path not in _warned:
            _warned.add(path)
            logger.warning("[double] the second exposure %s is not there; drawing without it", path)
        return None
    size = _READ_SIZE if longest <= _READ_SIZE else 0
    edit = second["edit"]

    def read():
        from adjust.render import decode, develop_frame, resize_linear
        from export.engine import (_load_linear, _with_camera_look, _with_denoiser, frame_of,
                                   linear_base, resolve_export_source)

        source = resolve_export_source(file)
        if size:
            # The same decode the develop view keeps of that photo, when it has one.
            frame = linear_base(source, size)
        else:
            rgb = _load_linear(source, 0)
            frame = None
            if rgb is not None:
                frame = frame_of(source, resize_linear(rgb, 0))
                _with_camera_look(source, frame)
                _with_denoiser(source, frame, 0)
        if frame is None:
            return None
        picture = develop_frame(frame, edit)
        return decode(np.asarray(picture, dtype=np.float32) / 255.0).astype(np.float32)

    return _kept((str(file), file.stat().st_mtime_ns, size, second_to_text(edit)), path, read)


def place(second_rgb: np.ndarray, shape: tuple, window: tuple, second: dict) -> tuple:
    """``second_rgb`` laid over the part ``window`` of the whole frame, this part being
    ``shape`` pixels: ``(rgb, cover)``, ``cover`` being 1 where the second reaches. Scale 100
    just covers the frame; a move or a mirror can leave part of it uncovered."""
    import cv2

    h, w = shape
    left, top, right, bottom = window
    whole_w, whole_h = w / max(right - left, 1e-9), h / max(bottom - top, 1e-9)
    picture = second_rgb[:, ::-1] if second["flip"] else second_rgb
    sh, sw = picture.shape[:2]
    k = max(whole_w / sw, whole_h / sh) * second["scale"] / 100.0
    if k < 1.0:
        # Brought down first, or the pixels skipped would show as sparkle.
        sw, sh = max(1, round(sw * k)), max(1, round(sh * k))
        picture = cv2.resize(picture, (sw, sh), interpolation=cv2.INTER_AREA)
        k = max(whole_w / sw, whole_h / sh) * second["scale"] / 100.0
    centre_x = whole_w * (0.5 + second["x"] / 100.0)
    centre_y = whole_h * (0.5 + second["y"] / 100.0)
    xs = (np.arange(w, dtype=np.float32) + left * whole_w - centre_x) / k + sw / 2.0 - 0.5
    ys = (np.arange(h, dtype=np.float32) + top * whole_h - centre_y) / k + sh / 2.0 - 0.5
    map_x, map_y = np.meshgrid(xs, ys)
    picture = np.ascontiguousarray(picture, dtype=np.float32)
    rgb = cv2.remap(picture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Pixel centres: -0.5 is the very edge of the second, which still covers all of that pixel.
    cover = np.clip(np.minimum.reduce([map_x + 1.5, map_y + 1.5, sw - map_x + 0.5, sh - map_y + 0.5]), 0.0, 1.0)
    return rgb, cover.astype(np.float32)


def mix(first: np.ndarray, second_rgb: np.ndarray, weight: np.ndarray, second: dict) -> np.ndarray:
    """``first`` and ``second_rgb`` put together, by ``second``'s mode and amount, as much as
    ``weight`` (H, W) says at each pixel. Weight 0 leaves ``first`` exactly as it was."""
    amount = second["amount"] / 100.0
    other = second_rgb * float(2.0 ** second["exposure"])
    m = weight[..., None]
    mode = second["mode"]
    if mode == "add":
        return (first + other * amount * m).astype(np.float32)
    if mode == "average":
        target = (first + other * amount) / (1.0 + amount)
    else:
        # By brightness at each pixel, not channel by channel, which would make new colours.
        brighter = (other @ _LUMA) > (first @ _LUMA)
        pick = brighter if mode == "lighten" else ~brighter
        target = np.where(pick[..., None], other, first)
        m = m * min(amount, 1.0)
    return (first + (target - first) * m).astype(np.float32)


def pixels(frame, rgb: np.ndarray, window: tuple, second: dict) -> np.ndarray:
    """``rgb``, the part ``window`` of ``frame``'s picture at any resolution, with the second
    exposure on it. ``rgb`` itself when there is nothing to lay on: the file has gone, or the
    model that says where it reaches can't be had."""
    from adjust.render import LinearFrame, mask_reach

    h, w = rgb.shape[:2]
    longest = max(w / max(window[2] - window[0], 1e-9), h / max(window[3] - window[1], 1e-9))
    picture = _picture(second["path"], longest)
    if picture is None:
        return rgb
    placed, weight = place(picture, (h, w), window, second)
    if second["reach"]:
        part = LinearFrame(rgb, window=window, anchor_of=getattr(frame, "anchor_of", None) or frame)
        reach = mask_reach(part, {"kind": second["reach"], "invert": second["invert"]})
        if reach is None:
            return rgb
        weight = weight * reach
    return mix(rgb, placed, weight, second)


def laid_on(frame, enc: np.ndarray, second: dict) -> np.ndarray:
    """``enc``, ``frame`` as its own edit finished it (display-encoded, on the frame's grid,
    before its framing), with the second laid on as edited: the two mixed as light. ``enc``
    itself when there is nothing to lay on."""
    from adjust.render import decode, encode, mask_reach

    h, w = enc.shape[:2]
    window = frame.window
    longest = max(w / max(window[2] - window[0], 1e-9), h / max(window[3] - window[1], 1e-9))
    picture = _finished(second, longest)
    if picture is None:
        return enc
    placed, weight = place(picture, (h, w), window, second)
    if second["reach"]:
        reach = mask_reach(frame, {"kind": second["reach"], "invert": second["invert"]})
        if reach is None:
            return enc
        weight = weight * reach
    mixed = mix(decode(enc).astype(np.float32), placed, weight, second)
    return encode(np.clip(mixed, 0.0, 1.0)).astype(np.float32)


def exposed(frame, second: dict):
    """``frame`` with the second exposure on it, as a frame of its own; ``frame`` itself when
    there is nothing to lay on. Kept on the frame for these values."""
    from adjust.render import LinearFrame

    key = second_to_text(second)
    if frame._exposed is not None and frame._exposed[0] == key:
        return frame._exposed[1]
    rgb = pixels(frame, frame.rgb, frame.window, second)
    if rgb is frame.rgb:
        return frame
    out = LinearFrame(rgb, look=frame.look, window=frame.window, scale=frame.scale,
                      anchor_of=frame.anchor_of or frame, lens=frame.lens, wb=frame.wb)
    frame._exposed = (key, out)
    return out
