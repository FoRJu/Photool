"""
Application configuration.
Loads from config.yaml with sensible defaults for Canon R6 II workflow.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # Root of your photo library (where Lightroom stored things)
    library_root: str = ""

    # Folder pattern under library_root. Supports: {year}, {month}, {day}
    # Results in e.g. library_root/2024/06/15/
    folder_pattern: str = "{year}/{month}/{day}"

    # Extra places to look for a memory card. Every drive or volume is looked at
    # anyway (importer.sd_detect.mounted_volumes); these are for anywhere else.
    sd_mount_paths: list[str] = field(default_factory=list)

    # File extensions to import. CR3 is the R-series RAW; CR2 is the older
    # Canon RAW that the 5D Mark II and its contemporaries write. Both go
    # through the same RAW pipeline (embedded-JPEG extraction, rawpy fallback),
    # so leaving CR2 out here silently drops every RAW from those bodies. DNG is
    # what a Ricoh GR writes (and Leica, Pentax and others can): the same pipeline,
    # with the camera's JPEG beside it as the look's reference. The other makers'
    # RAWs go through it too, for an older camera from the drawer. Videos (a
    # GoPro's, a DJI Pocket's) are copied off the card with the photographs and
    # shown beside them, to be watched and culled (see ``VIDEO_EXTENSIONS``).
    import_extensions: list[str] = field(
        default_factory=lambda: [".cr3", ".cr2", ".dng", ".jpg", ".jpeg",
                                 ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef",
                                 ".mp4", ".mov"]
    )

    # What to do with duplicates: skip | rename | overwrite
    duplicate_strategy: str = "skip"

    # Database path (SQLite catalog)
    db_path: str = ""

    # Thumbnail cache directory
    thumbnail_cache: str = ""

    # Upper bound on the thumbnail cache in MB; 0 means unlimited.
    # A 2000px preview at q90 is ~580 KB, so a five-figure library costs tens of
    # gigabytes. Set this to keep the cache inside a budget you have; over the
    # budget the least-recently-used previews are evicted first.
    thumbnail_cache_max_mb: int = 0

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000

    # Other computers. Off (the default), only this computer can use the app: a
    # request that arrives over the network is refused before it reaches any
    # photo. On, the app answers on the network (run `python serve.py`), and a
    # computer or phone must be *paired* first — it asks, and you approve it on
    # this computer after checking that both screens show the same code. A
    # paired device keeps a long random key in a cookie; only its hash is stored
    # here, and Devices in the header revokes it. See docs/guide.md, "Other computers".
    lan_access: bool = False
    # Tailscale logins (e.g. "you@gmail.com") let in without pairing when they
    # arrive through `tailscale serve` on this computer. Tailscale has already
    # proved who they are; everyone else on the tailnet still pairs.
    lan_trusted_logins: list[str] = field(default_factory=list)
    # HTTPS for `serve.py` on the network: a certificate and its key (from mkcert,
    # say). Without them the network traffic is plain HTTP; Tailscale is the
    # easier way to get it encrypted.
    tls_cert: str = ""
    tls_key: str = ""
    # The most pictures a paired device may download in one export. It receives
    # rendered JPEGs only, never originals; this keeps one click from copying
    # the whole library off this computer.
    remote_export_limit: int = 200

    # Thumbnail queue
    thumbnail_queue_enabled: bool = True
    thumbnail_queue_interval_seconds: float = 0.25
    thumbnail_queue_priority_low: int = 100
    thumbnail_queue_auto_generate: bool = True
    thumbnail_queue_idle_rescan_seconds: float = 60.0

    # Photo measurements. When a folder's thumbnails finish, its photos are
    # measured from those thumbnails — exposure, colour, sharpness, and a frame
    # hash. Nothing is visible until the culling and matching features use the
    # numbers, but collecting them in the background means those features never
    # have to wait for a scan.
    analysis_auto_enabled: bool = True

    # Export. When a photo has both a RAW and a camera JPEG, develop the RAW:
    # every frame then starts from the same kind of data instead of inheriting
    # the camera's per-shot auto exposure and white balance, which is what makes
    # matching a set of frames possible at all. Set false to export exactly the
    # file that is in the basket.
    export_prefer_raw: bool = True

    # Burst grouping. Two frames written this many seconds apart can still be
    # the same shot. The grouping also requires that they look the same, so this
    # only decides how far apart two identical-looking frames may be before they
    # become two shots of the same thing. A camera firing three frames a second
    # wants this low; someone walking around and framing the same view twice a
    # minute apart wants it higher.
    burst_max_gap_seconds: int = 10

    # Person clustering. Faces are found and described entirely on this machine
    # by OpenCV — no image and no embedding leaves it — and the two ONNX models
    # are fetched into models_dir the first time a scan runs. The scan reads a
    # 1024px rendering of each frame, which is what makes one person's scores
    # separable from another's; that rendering is cached like any other
    # thumbnail at about 127 KB a frame, so scanning a whole library costs a few
    # gigabytes of cache and can be repeated cheaply.
    models_dir: str = ""
    # Faces smaller than this in the analysed frame are dropped as soon as they
    # are found, before anything expensive is done with them. They cannot be
    # grouped — at a few dozen pixels the vector is mostly noise, and noise in a
    # group's average is what merges two people into one — and describing them
    # is the expensive part of a scan: one real frame produced 221 detections,
    # none wider than 28 pixels. 40px at 1024 is a head filling about 4% of the
    # frame's width.
    face_min_px: float = 40.0
    # Cosine similarity at which two faces are called the same person. Measured
    # on a real library: different people never exceeded 0.41, and one person
    # across a burst sat at 0.83-0.98. Set between those, and nearer the low end
    # than the high one, because a group that splits in two is a merge the user
    # can make in a click, while a group that merges two people is a name on the
    # wrong photographs.
    face_match_threshold: float = 0.5
    # Which model describes a face: "arcface" (ResNet-50, 512 numbers, the best
    # at telling similar faces apart — a one-time 290 MB download), "arcface_lite"
    # (MobileFaceNet, a 15 MB download, a little less sure) or "sface" (the
    # original OpenCV model). Changing it re-describes every frame on the next
    # scan; names already given out are kept.
    face_model: str = "arcface"
    # The join threshold for the ArcFace models. ``face_match_threshold`` above
    # was measured for SFace, whose strangers already sit around 0.1-0.4; two
    # strangers under ArcFace sit around 0, and one person's faces above 0.6,
    # so its bar is lower. Leave empty for the model's own default.
    face_threshold: Optional[float] = None
    # Which rendering a scan reads: "preview" (2000px, the default — a face at
    # the back of a room is twice as many pixels as at 1024) or "face" (1024px,
    # the original; quicker, but small faces fall under the floor).
    face_scan_size: str = "preview"
    # Look for faces in photographs no scan has seen — after an import, when a
    # folder is indexed, and at startup — once the face models are installed.
    face_auto_scan: bool = True

    def effective_face_threshold(self) -> float:
        """The join threshold for the configured face model."""
        if self.face_threshold is not None:
            return float(self.face_threshold)
        model = (self.face_model or "").strip().lower()
        if model == "sface":
            return float(self.face_match_threshold)
        return ARCFACE_THRESHOLDS.get(model, ARCFACE_THRESHOLDS["arcface"])

    @property
    def photo_extensions(self) -> list[str]:
        """What the library shows: everything imported, videos included."""
        return list(self.import_extensions)

    def __post_init__(self):
        if not self.db_path:
            self.db_path = str(Path(self.library_root) / ".photomanager" / "catalog.db")
        if not self.thumbnail_cache:
            self.thumbnail_cache = str(
                Path(self.library_root) / ".photomanager" / "thumbnails"
            )
        if not self.models_dir:
            # Beside the code, not beside the photographs: the models belong to
            # the application, and a library drive that is offline should not
            # stop the app from knowing where they are.
            self.models_dir = str(Path(__file__).resolve().parent.parent / "models")


# Videos are copied off a card beside the photographs and shown with them, to be
# watched, rated, tagged and exported as they are: never developed, and left out
# of a burst. Their thumbnail is a frame from the clip (thumbnails.video).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mts", ".m4v"}


def is_video(name) -> bool:
    """Whether a file (a path, a name or an extension) is a video."""
    suffix = str(name).lower()
    return suffix in VIDEO_EXTENSIONS or any(suffix.endswith(e) for e in VIDEO_EXTENSIONS)


# Join thresholds for the ArcFace models, against a group's average. Two
# strangers score around 0 (at most 0.12-0.17 across labelled portraits) and one
# person's frontal faces 0.6-0.9; set nearer the strangers' end than SFace's 0.5
# because the average of a group is steadier than any one pair.
ARCFACE_THRESHOLDS = {"arcface": 0.42, "arcface_lite": 0.40}


def load_config(path: Optional[str] = None) -> Config:
    """
    Load config from YAML file. Falls back to defaults if file not found.

    Search order:
      1. Explicit path argument
      2. ./config.yaml
      3. ~/.config/canon-photo-manager/config.yaml
    """
    search_paths = []
    if path:
        search_paths.append(Path(path))
    search_paths.append(Path("config.yaml"))
    search_paths.append(
        Path.home() / ".config" / "canon-photo-manager" / "config.yaml"
    )

    for p in search_paths:
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            print(f"[config] Loaded from {p}")
            # By field, not by class attribute: a list setting (import_extensions,
            # sd_mount_paths) has a default factory and no class attribute, so
            # asking hasattr() quietly dropped every one of them.
            known = {f.name for f in fields(Config)}
            return Config(**{k: v for k, v in data.items() if k in known})

    print("[config] No config file found, using defaults")
    return Config()
