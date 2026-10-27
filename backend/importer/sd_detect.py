"""
Camera card detection.

Every camera since the early 2000s follows the same rule (DCF): photographs go
under /DCIM/, in folders the camera names. The name is how a card tells which
camera wrote it, and is all that is needed to say so on the import screen:

  /DCIM/100CANON/     Canon (also 100EOS5D, CANONMSC for settings)
  /DCIM/100RICOH/     Ricoh GR
  /DCIM/100GOPRO/     GoPro
  /DCIM/DJI_001/      DJI Osmo Pocket 3 (and DJI's drones: 100MEDIA)
  /DCIM/100NIKON/, 100MSDCF/ (Sony), 100OLYMP/, 100_PANA/, 100_FUJI/ ...

We detect cards by looking for /DCIM/ at known mount points and on every drive.
"""

import os
import platform
import string
from dataclasses import dataclass, field
from pathlib import Path

from config import VIDEO_EXTENSIONS, Config

# Which maker a DCIM folder name points to, by a fragment of it. The first match
# wins, so the more specific fragments come first.
_MAKERS = (
    ("GOPRO", "GoPro"),
    ("DJI", "DJI"),
    ("CANON", "Canon"),
    ("EOS", "Canon"),
    ("RICOH", "Ricoh"),
    ("PENTX", "Pentax"),
    ("NIKON", "Nikon"),
    ("NCD", "Nikon"),
    ("MSDCF", "Sony"),
    ("OLYMP", "Olympus"),
    ("OMSYS", "OM System"),
    ("PANA", "Panasonic"),
    ("FUJI", "Fujifilm"),
    ("LEICA", "Leica"),
    ("CASIO", "Casio"),
    ("KDK", "Kodak"),
    ("SANYO", "Sanyo"),
    ("APPLE", "iPhone"),
)


@dataclass
class SDCardInfo:
    """Info about a detected SD card."""
    mount_point: Path
    dcim_path: Path
    image_folders: list[Path]
    total_files: int
    total_size_bytes: int
    # Populated during detection so the import does not have to walk a slow
    # card a second time just to get the same list back.
    file_paths: list[Path] = field(default_factory=list)
    # Who wrote the card, as far as its folder names say ("" when they don't).
    device: str = ""
    videos: int = 0

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024 * 1024 * 1024)


def _scan_folder(folder: Path, extensions: set[str]) -> list[Path]:
    """
    List importable files in one DCIM folder.

    os.scandir is used instead of Path.iterdir()+stat(): DirEntry.stat()
    reuses the metadata the directory listing already returned, which avoids a
    separate stat syscall per file on a card that can hold thousands.
    """
    found: list[Path] = []
    try:
        entries = os.scandir(folder)
    except OSError:
        return found
    with entries:
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                # A Mac leaves "._IMG_0001.JPG" beside each file on a card it
                # has opened: not a photograph, and nothing to copy.
                if entry.name.startswith("."):
                    continue
                if Path(entry.name).suffix.lower() not in extensions:
                    continue
                found.append(Path(entry.path))
            except OSError:
                continue
    return found


def device_name(folders: list[Path]) -> str:
    """The maker a card's DCIM folders point to, or "" when they don't say."""
    found: list[str] = []
    for folder in folders:
        name = folder.name.upper()
        maker = next((label for fragment, label in _MAKERS if fragment in name), "")
        if maker and maker not in found:
            found.append(maker)
    return " + ".join(found)


MAC_VOLUMES = Path("/Volumes")


def mounted_volumes() -> list[Path]:
    """Every place a card can appear on this system, so that no mount point has
    to be configured: each drive letter on Windows, each volume under /Volumes on
    a Mac (a card is named by the camera that formatted it: EOS_DIGITAL,
    NIKON D750, RICOH GR, or plain Untitled), and the removable-media folders on
    Linux. Only the ones holding a DCIM folder are cards."""
    system = platform.system()
    if system == "Windows":
        return [p for p in (Path(f"{letter}:/") for letter in string.ascii_uppercase) if p.exists()]
    if system == "Darwin":
        roots = [MAC_VOLUMES]
    else:
        user = os.environ.get("USER", "")
        roots = [Path("/media") / user, Path("/run/media") / user, Path("/media"), Path("/mnt")]
    found: list[Path] = []
    for root in roots:
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir() and not entry.name.startswith("."):
                    found.append(Path(entry.path))
            except OSError:
                continue
    return found


def detect_sd_cards(config: Config) -> list[SDCardInfo]:
    """
    Scan known mount points for camera cards.
    Returns list of detected cards with file counts.
    """
    candidates = [Path(p) for p in config.sd_mount_paths]
    for p in mounted_volumes():
        if p not in candidates:
            candidates.append(p)

    cards = []
    extensions = {ext.lower() for ext in config.import_extensions}

    for mount in candidates:
        dcim = mount / "DCIM"
        if not dcim.is_dir():
            continue

        # Every folder under DCIM is the camera's own: 100CANON, 100GOPRO,
        # DJI_001. Only the hidden ones a computer adds are not.
        try:
            children = list(os.scandir(dcim))
        except OSError:
            continue

        image_folders: list[Path] = []
        for entry in children:
            try:
                if not entry.is_dir(follow_symlinks=False) or entry.name.startswith("."):
                    continue
                image_folders.append(Path(entry.path))
            except OSError:
                continue
        image_folders.sort()

        if not image_folders:
            continue

        # Count importable files (single pass; the paths are kept for import)
        total_files = 0
        total_size = 0
        videos = 0
        file_paths: list[Path] = []

        for folder in image_folders:
            for path in _scan_folder(folder, extensions):
                try:
                    total_size += path.stat().st_size
                except OSError:
                    continue
                total_files += 1
                videos += path.suffix.lower() in VIDEO_EXTENSIONS
                file_paths.append(path)

        cards.append(SDCardInfo(
            mount_point=mount,
            dcim_path=dcim,
            image_folders=image_folders,
            total_files=total_files,
            total_size_bytes=total_size,
            file_paths=file_paths,
            device=device_name(image_folders),
            videos=videos,
        ))

    return cards


def find_importable_files(card: SDCardInfo, extensions: list[str]) -> list[Path]:
    """
    Get all importable files from an SD card, sorted by name (which is
    roughly chronological for Canon's naming scheme).
    """
    ext_set = {ext.lower() for ext in extensions}

    # Reuse the listing gathered during detection when it is still valid,
    # instead of re-walking the card.
    if card.file_paths:
        files = [p for p in card.file_paths if Path(p).suffix.lower() in ext_set]
    else:
        files = []
        for folder in card.image_folders:
            files.extend(_scan_folder(folder, ext_set))

    # Folder order first, then name within the folder: Canon names restart at
    # 0001 per card folder, so a global name sort would interleave sessions.
    folder_order = {str(f): i for i, f in enumerate(sorted(card.image_folders))}
    return sorted(
        files,
        key=lambda p: (folder_order.get(str(p.parent), len(folder_order)), p.name),
    )
