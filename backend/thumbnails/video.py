"""
Videos in the library: a frame to show for one, what it is, and a copy any
browser can play.

A video is shown and culled like a photograph but never decoded here. ffmpeg
(on the PATH, or not at all) does the work: one frame for the thumbnail, a probe
for the length and the codec. The file itself is what the viewer plays: Edge on
a PC with a GPU plays a DJI Pocket's HEVC 10-bit as it is. A browser that cannot
(a phone on the network, say) asks for a compatible copy instead, made once in
the background as H.264 (on the GPU when there is one) and kept in the
thumbnail cache.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The frame shown for a video: a second in, past a fade from black or the
# camera settling, unless the clip is shorter than that.
POSTER_AT_SECONDS = 1.0
# Compatible copies kept, oldest dropped past this.
COPIES_BUDGET_BYTES = 8 * 1024 ** 3


def _tool(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    # The usual Windows install, in case the server started without it on the PATH.
    for folder in (Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg",
                   Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg" / "bin"):
        candidate = folder / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def available() -> bool:
    return _tool("ffmpeg") is not None and _tool("ffprobe") is not None


def poster_jpeg(source: Path) -> Optional[bytes]:
    """A frame of the video as JPEG bytes, at its full size and upright; ``None``
    when ffmpeg is missing or cannot read the file."""
    ffmpeg = _tool("ffmpeg")
    if not ffmpeg:
        return None
    for at in (POSTER_AT_SECONDS, 0.0):
        try:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-ss", str(at), "-i", str(source), "-frames:v", "1",
                 "-pix_fmt", "yuvj420p", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "2", "-"],
                capture_output=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("No frame from %s: %s", source.name, exc)
            return None
        if result.returncode == 0 and result.stdout:
            return result.stdout
    logger.warning("No frame from %s: %s", source.name, result.stderr.decode(errors="replace")[-300:])
    return None


def probe(source: Path) -> dict:
    """The video's ``codec``, ``duration`` (seconds), ``width`` and ``height``;
    empty when it cannot be read."""
    ffprobe = _tool("ffprobe")
    if not ffprobe:
        return {}
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height:format=duration",
             "-of", "json", str(source)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}
    stream = (data.get("streams") or [{}])[0]
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except ValueError:
        duration = 0.0
    return {"codec": stream.get("codec_name", ""), "duration": duration,
            "width": stream.get("width", 0), "height": stream.get("height", 0)}


# ── Compatible copies ────────────────────────────────────────────────

_copies_dir: Optional[Path] = None
_jobs: dict = {}            # copy path -> "working" | "failed"
_jobs_lock = threading.Lock()
_one_at_a_time = threading.Semaphore(1)


def configure(cache_dir: Optional[str]) -> None:
    global _copies_dir
    _copies_dir = Path(cache_dir) if cache_dir else None


def _copy_path(source: Path) -> Optional[Path]:
    if _copies_dir is None:
        return None
    try:
        stat = source.stat()
    except OSError:
        return None
    key = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return _copies_dir / (hashlib.sha1(key.encode()).hexdigest()[:24] + ".mp4")


def compatible_copy(source: Path) -> tuple[str, Optional[Path]]:
    """``("ready", path)`` when the copy is made; otherwise its state,
    ``"working"`` (it is being made now: ask again), ``"failed"`` or
    ``"unavailable"``, and ``None``."""
    target = _copy_path(source)
    if target is None or not available():
        return "unavailable", None
    if target.is_file():
        return "ready", target
    with _jobs_lock:
        state = _jobs.get(target)
        if state is None:
            _jobs[target] = "working"
            threading.Thread(target=_make_copy, args=(source, target), daemon=True,
                             name="video-copy").start()
            state = "working"
    return state, None


def _make_copy(source: Path, target: Path) -> None:
    ffmpeg = _tool("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part.mp4")
    common = ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
    attempts = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"],      # the GPU's encoder
                ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])  # or the CPU's
    with _one_at_a_time:
        for encoder in attempts:
            try:
                result = subprocess.run(
                    [ffmpeg, "-v", "error", "-y", "-i", str(source), *encoder, *common, str(partial)],
                    capture_output=True, timeout=3600,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("Compatible copy of %s failed: %s", source.name, exc)
                continue
            if result.returncode == 0 and partial.is_file():
                os.replace(partial, target)
                logger.info("Made a compatible copy of %s (%s)", source.name, encoder[1])
                _prune(keep=target)
                with _jobs_lock:
                    _jobs.pop(target, None)
                return
            logger.info("Compatible copy of %s with %s failed: %s", source.name, encoder[1],
                        result.stderr.decode(errors="replace")[-300:])
    partial.unlink(missing_ok=True)
    with _jobs_lock:
        _jobs[target] = "failed"


def _prune(keep: Path) -> None:
    """Drop the oldest copies while they take more than COPIES_BUDGET_BYTES."""
    try:
        copies = sorted((p for p in keep.parent.glob("*.mp4") if not p.name.endswith(".part.mp4")),
                        key=lambda p: p.stat().st_atime)
        total = sum(p.stat().st_size for p in copies)
        for old in copies:
            if total <= COPIES_BUDGET_BYTES or old == keep:
                continue
            total -= old.stat().st_size
            old.unlink(missing_ok=True)
    except OSError:
        pass
