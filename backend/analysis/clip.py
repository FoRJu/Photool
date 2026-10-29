"""
Searching by what is in a photograph: "kids on a beach", "a red car", "snow".

CLIP (OpenAI's ViT-B/32, as exported to ONNX by Xenova) describes a picture
and a line of text as points in one space, near each other when the text says
what the picture shows. Every photograph is described once, in the background,
from its 400 px grid thumbnail, and kept (``photo_clip``, 512 numbers as
float16); a search describes its words and ranks the library by how near each
photograph is (catalog.search, the ``shows:`` term).

The models are 300 MB, so they are fetched only when smart search is set up
(:func:`setup`), into ``models_dir`` beside the face models, and checked
against a pinned revision's hashes. Both run in OpenCV's DNN module, which
needs every call to one network to have the same shape: images go through in
batches of exactly :data:`BATCH` and text at CLIP's full 77 tokens, padded with
the end-of-text token (which leaves the description unchanged).

A RAW+JPEG pair is one photograph: the JPEG is described, and the pair found
by it.
"""

import html
import json
import logging
import re
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from faces.models import ModelSpec, ModelsUnavailable, _download, verify

logger = logging.getLogger(__name__)

MODEL_ID = "clip-vit-b32"
_REPO = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
         "d15189d7028b43f1d3e65039190477f6af591c2a/")
VISION, TEXT, VOCAB, MERGES = ("clip_vision_fp16.onnx", "clip_text_fp16.onnx",
                               "clip_vocab.json", "clip_merges.txt")
MODELS = {
    VISION: ModelSpec(VISION, _REPO + "onnx/vision_model_fp16.onnx",
                      "35c4e0fb0aeee527dcde1693520b214a34424a786babd530f35366bad5844efd", 176080659),
    TEXT: ModelSpec(TEXT, _REPO + "onnx/text_model_fp16.onnx",
                    "df587ffbf248bf20d44fa6e16adc5ebc27ead691860e5333dbdaab5fd6bf3f6e", 127339794),
    VOCAB: ModelSpec(VOCAB, _REPO + "vocab.json",
                     "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363", 862328),
    MERGES: ModelSpec(MERGES, _REPO + "merges.txt",
                      "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a", 524619),
}
DOWNLOAD_MB = round(sum(s.size for s in MODELS.values()) / 1e6)

BATCH = 8
CONTEXT = 77
_START, _END = 49406, 49407
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

_models_dir: Optional[str] = None
_nets: dict = {}
_net_locks = {VISION: threading.Lock(), TEXT: threading.Lock()}
_setup = {"status": "idle", "error": "", "file": ""}


def configure(models_dir: str) -> None:
    global _models_dir
    if models_dir != _models_dir:
        _models_dir = models_dir
        _nets.clear()
        _tokenizer.cache_clear()


def _path(name: str) -> Path:
    return Path(_models_dir or ".") / name


def available() -> bool:
    """Whether every model file is here, as it should be."""
    return bool(_models_dir) and all(verify(_path(n), s) for n, s in MODELS.items())


def setup_state() -> dict:
    return dict(_setup)


def setup(on_ready=None) -> None:
    """Fetch the models in the background (a no-op when they are here), then
    call ``on_ready``. Progress is :func:`setup_state`."""
    if _setup["status"] == "downloading":
        return

    def work():
        try:
            for name, spec in MODELS.items():
                if not verify(_path(name), spec):
                    _setup.update(status="downloading", file=name, error="")
                    _download(spec, _path(name))
            _setup.update(status="ready", file="")
            if on_ready:
                on_ready()
        except (ModelsUnavailable, OSError) as exc:
            _setup.update(status="failed", error=str(exc), file="")

    _setup.update(status="downloading", error="")
    threading.Thread(target=work, name="clip-setup", daemon=True).start()


# ── Words to tokens: CLIP's byte-pair encoding ────────────────────────

def _bytes_to_unicode() -> dict:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTES = _bytes_to_unicode()
# CLIP's own split, with \p{L} and \p{N} written for Python's re.
_SPLIT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_",
                    re.IGNORECASE)


@lru_cache(maxsize=1)
def _tokenizer():
    vocab = json.loads(_path(VOCAB).read_text(encoding="utf-8"))
    lines = _path(MERGES).read_text(encoding="utf-8").split("\n")[1:]
    ranks = {tuple(line.split()): i for i, line in enumerate(l for l in lines if l.strip())}
    return vocab, ranks


@lru_cache(maxsize=20000)
def _bpe(token: str) -> tuple:
    _, ranks = _tokenizer()
    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    while len(word) > 1:
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        best = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if best not in ranks:
            break
        merged, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                merged.append(word[i] + word[i + 1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        word = tuple(merged)
    return word


def tokenize(text: str) -> list:
    """CLIP's token ids for ``text``, start and end marks included, at most 77."""
    vocab, _ = _tokenizer()
    text = re.sub(r"\s+", " ", html.unescape(text)).strip().lower()
    ids = [_START]
    for piece in _SPLIT.findall(text):
        piece = "".join(_BYTES[b] for b in piece.encode("utf-8"))
        ids.extend(vocab[t] for t in _bpe(piece) if t in vocab)
    return ids[:CONTEXT - 1] + [_END]


# ── Pictures and words to points ──────────────────────────────────────

def _net(name: str):
    import cv2

    net = _nets.get(name)
    if net is None:
        if not available():
            raise ModelsUnavailable("smart search is not set up")
        net = cv2.dnn.readNet(str(_path(name)))
        _nets[name] = net
    return net


def _prepare(rgb: np.ndarray) -> np.ndarray:
    """CLIP's view of a picture: its shorter side to 224, the middle 224 square."""
    import cv2

    h, w = rgb.shape[:2]
    scale = 224 / min(h, w)
    rgb = cv2.resize(rgb, (max(224, round(w * scale)), max(224, round(h * scale))),
                     interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    y, x = (h - 224) // 2, (w - 224) // 2
    square = rgb[y:y + 224, x:x + 224].astype(np.float32) / 255.0
    return ((square - _MEAN) / _STD).transpose(2, 0, 1)


def embed_images(pictures: list) -> np.ndarray:
    """Unit vectors (N, 512) for 8-bit RGB pictures of any size."""
    out = []
    for start in range(0, len(pictures), BATCH):
        chunk = [_prepare(p) for p in pictures[start:start + BATCH]]
        n = len(chunk)
        chunk += [np.zeros_like(chunk[0])] * (BATCH - n)      # always a full batch
        with _net_locks[VISION]:
            net = _net(VISION)
            net.setInput(np.stack(chunk))
            vectors = net.forward()[:n]
        out.append(vectors)
    v = np.concatenate(out).astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


@lru_cache(maxsize=256)
def _embed_text(text: str) -> bytes:
    ids = tokenize(text)
    ids += [_END] * (CONTEXT - len(ids))
    with _net_locks[TEXT]:
        net = _net(TEXT)
        net.setInput(np.array([ids], dtype=np.int64))
        v = net.forward()[0].astype(np.float32)
    return (v / max(float(np.linalg.norm(v)), 1e-9)).tobytes()


def embed_text(text: str) -> np.ndarray:
    """A unit vector for a description, asked as CLIP was taught: "a photo of …"."""
    text = text.strip()
    prompt = text if re.match(r"(?i)^(a |an )?(photo|picture|image)\b", text) else f"a photo of {text}"
    return np.frombuffer(_embed_text(prompt), dtype=np.float32)


# ── Kept descriptions ─────────────────────────────────────────────────

def pending(db) -> list[dict]:
    """The photographs with no description, one file per pair (its JPEG), with
    the size it has. A file replaced since (another size) is described again."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT p.filepath, p.folder, p.base_name, p.extension, p.file_size, "
            "c.source_size AS described FROM photos p "
            "LEFT JOIN photo_clip c ON c.filepath = p.filepath AND c.model = ? "
            "ORDER BY p.folder DESC, p.filepath", (MODEL_ID,)).fetchall()
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["folder"], r["base_name"]), []).append(dict(r))
    out = []
    for group in groups.values():
        group.sort(key=lambda r: (r["extension"].lower() not in (".jpg", ".jpeg"), r["filepath"]))
        if any(r["described"] is not None and r["described"] == r["file_size"] for r in group):
            continue
        out.append(group[0])
    return out


def store(db, rows: list) -> None:
    """Keep ``(filepath, source_size, vector)`` rows."""
    if not rows:
        return
    now = datetime.now().isoformat(timespec="seconds")
    with db._connect() as conn:
        conn.executemany(
            "INSERT INTO photo_clip (filepath, model, source_size, vector, described_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(filepath) DO UPDATE SET model = excluded.model, "
            "source_size = excluded.source_size, vector = excluded.vector, "
            "described_at = excluded.described_at",
            [(fp, MODEL_ID, int(size or 0), np.asarray(v, np.float16).tobytes(), now)
             for fp, size, v in rows])
    _index.stale = True


class _Index:
    """Every kept description in memory, as one matrix, for scoring the library
    against a search's words in one multiplication."""

    def __init__(self):
        self.stale = True
        self.paths: list = []
        self.matrix = np.zeros((0, 512), np.float32)
        self.lock = threading.Lock()
        self._neutral = None

    def neutral(self) -> np.ndarray:
        """Each photograph's nearness to NEUTRAL, worked out once per load."""
        if self._neutral is None or len(self._neutral) != len(self.paths):
            self._neutral = self.matrix @ embed_text(NEUTRAL)
        return self._neutral

    def load(self, db):
        with self.lock:
            if not self.stale:
                return
            with db._connect() as conn:
                rows = conn.execute("SELECT filepath, vector FROM photo_clip WHERE model = ?",
                                    (MODEL_ID,)).fetchall()
            self._neutral = None
            self.paths = [r["filepath"] for r in rows]
            self.matrix = (np.frombuffer(b"".join(r["vector"] for r in rows), np.float16)
                           .reshape(-1, 512).astype(np.float32) if rows else np.zeros((0, 512), np.float32))
            self.stale = False


_index = _Index()


# What any photograph is, as words: its nearness to this is taken off its
# nearness to a search's words. Some pictures — an empty blue sky, a blur, the
# dark — are near *every* description (0.27 to "a giraffe" in a library with
# none), and would otherwise top every search for something the library does
# not hold; a picture of what was asked for stays well clear.
NEUTRAL = "a photo"


def scores(db, text: str) -> dict:
    """``{filepath: relevance}`` for every described photograph: how much
    nearer it is to ``text`` than to any photo at all. Measured on a library of
    29,000: what was asked for scores 0.03-0.08, the edge of it about 0.025,
    and nothing in the library tops 0.035 for something it does not hold."""
    _index.load(db)
    if not _index.paths:
        return {}
    s = _index.matrix @ embed_text(text) - _index.neutral()
    return dict(zip(_index.paths, s.tolist()))


def counts(db) -> dict:
    """How many photographs (pairs as one) are described, of how many."""
    with db._connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM (SELECT 1 FROM photos GROUP BY folder, base_name)").fetchone()[0]
        done = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM photo_clip c JOIN photos p ON p.filepath = c.filepath "
            "AND p.file_size = c.source_size WHERE c.model = ? GROUP BY p.folder, p.base_name)",
            (MODEL_ID,)).fetchone()[0]
    return {"photos": int(total), "described": int(done)}


# ── The background describer ──────────────────────────────────────────

_lock = threading.Lock()
_stop = threading.Event()
_job: dict = {"status": "idle", "done": 0, "total": 0, "error": ""}
_wanted = False
_thread = None


def job() -> dict:
    return dict(_job)


def run(library_root, db, cache, stop_event: Optional[threading.Event] = None) -> dict:
    """Describe every pending photograph, on this thread, from its grid
    thumbnail (made if it is not cached yet)."""
    import cv2

    from thumbnails import ThumbnailSize, get_thumbnail

    import background as gate

    rows = pending(db)
    _job.update(status="running", total=len(rows), done=0, error="", started_at=time.time())
    root = Path(library_root)
    try:
        for start in range(0, len(rows), BATCH * 4):
            if stop_event is not None and stop_event.is_set():
                break
            if not gate.wait_turn("Smart search", stop_event and stop_event.is_set):
                break
            chunk = rows[start:start + BATCH * 4]
            pictures, kept = [], []
            for row in chunk:
                try:
                    jpeg = get_thumbnail(root / row["filepath"], cache, ThumbnailSize.GRID)
                except Exception:
                    jpeg = None
                bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
                if bgr is None:
                    continue
                pictures.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                kept.append(row)
            if pictures:
                vectors = embed_images(pictures)
                store(db, [(r["filepath"], r["file_size"], v) for r, v in zip(kept, vectors)])
            _job["done"] += len(chunk)
            time.sleep(0.02)          # the library stays quick to browse meanwhile
        _job["status"] = "stopped" if stop_event is not None and stop_event.is_set() else "completed"
    except ModelsUnavailable as exc:
        _job.update(status="idle", error=str(exc))
    except Exception as exc:
        logger.exception("Describing photographs failed")
        _job.update(status="failed", error=str(exc))
    finally:
        _job["finished_at"] = time.time()
    return job()


def wake(library_root, db, cache) -> None:
    """Describe whatever is pending, in the background, when smart search is
    set up. A call while a pass runs asks for one more after it."""
    global _thread, _wanted
    if not available() or not getattr(db, "writable", True):
        return
    with _lock:
        _wanted = True
        if _thread is not None:
            return
        _stop.clear()
        _thread = threading.Thread(target=_worker, args=(library_root, db, cache),
                                   name="clip-describe", daemon=True)
        _thread.start()


def _worker(library_root, db, cache) -> None:
    global _thread, _wanted
    while True:
        with _lock:
            if not _wanted or _stop.is_set():
                _thread = None
                return
            _wanted = False
        run(library_root, db, cache, stop_event=_stop)


def stop() -> None:
    _stop.set()
