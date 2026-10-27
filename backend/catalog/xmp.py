"""
XMP sidecar file support.

XMP sidecars store ratings, flags, and labels in a standard format
that Lightroom, darktable, and other tools can read. This means:
  - Your ratings survive even if you delete the SQLite database
  - You can open the same folder in darktable and see your picks
  - Ratings travel with the files if you move them

Format follows the Adobe XMP specification:
  - xmp:Rating → 0-5 star rating
  - xmp:Label  → color label name (Red, Yellow, Green, Blue, Purple)
  - photoshop:Urgency → maps flag to a priority value (Lightroom convention)
  - dc:subject → keyword / event tags (Lightroom's keyword panel)
  - crs:* → Camera Raw develop settings, so an edit made here shows up in
    Lightroom instead of being invisible to it

Sidecar files are named: <original_name>.xmp
  e.g. IMG_0001.CR3 → IMG_0001.CR3.xmp
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

# XMP namespaces
NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
}

# Register namespaces so ET doesn't mangle them
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

# Color label names (Adobe/Lightroom convention)
COLOR_NAMES = {
    0: "",
    1: "Red",
    2: "Yellow",
    3: "Green",
    4: "Blue",
    5: "Purple",
}
COLOR_FROM_NAME = {v.lower(): k for k, v in COLOR_NAMES.items() if v}


@dataclass
class XMPData:
    """Parsed XMP sidecar data."""
    rating: int = 0         # 0-5
    flag: int = 0           # -1 (reject), 0 (unflagged), 1 (pick)
    color_label: int = 0    # 0-5
    label_name: str = ""    # "Red", "Yellow", etc.
    tags: list[str] = field(default_factory=list)  # dc:subject keywords
    # Develop settings under their own attribute names, e.g.
    # {"crs:Exposure2012": "+0.70"}. Held as written rather than as numbers:
    # the sidecar is where another editor reads them, and translating them back
    # into sliders is the edit model's business, not this module's.
    develop: dict[str, str] = field(default_factory=dict)


# ── Sidecar path ─────────────────────────────────────────────────────

def sidecar_path(image_path: Path) -> Path:
    """
    Get the XMP sidecar path for an image.
    Convention: IMG_0001.CR3 → IMG_0001.CR3.xmp
    """
    return image_path.with_suffix(image_path.suffix + ".xmp")


# ── Develop settings ─────────────────────────────────────────────────

CRS = "crs"
_ATTR_ESCAPES = {'"': "&quot;"}

# Distinguishes "the caller did not say which attributes it owns" from "the
# caller owns none of them" — None is a meaningful value for the latter.
_UNSET = object()


def _local_name(name: str) -> str:
    """'crs:Exposure2012' or '{uri}Exposure2012' as the bare local name."""
    if "}" in name:
        return name.split("}", 1)[1]
    return name.split(":", 1)[-1]


def _read_develop(desc: ET.Element) -> dict[str, str]:
    """The crs:* attributes of an rdf:Description, under their prefixed names."""
    uri = f"{{{NS[CRS]}}}"
    return {
        f"{CRS}:{_local_name(key)}": value
        for key, value in desc.attrib.items()
        if key.startswith(uri)
    }


# ── Read ─────────────────────────────────────────────────────────────

def read_xmp(image_path: Path) -> XMPData:
    """
    Read rating/flag/label from an XMP sidecar file.
    Returns defaults if sidecar doesn't exist.
    """
    xmp_path = sidecar_path(image_path)
    data = XMPData()

    if not xmp_path.exists():
        return data

    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Find the rdf:Description element
        desc = root.find(".//rdf:Description", NS)
        if desc is None:
            return data

        # Rating
        rating_str = desc.get(f"{{{NS['xmp']}}}Rating", "0")
        try:
            rating = int(rating_str)
            data.rating = max(0, min(5, rating))
        except ValueError:
            pass

        # Flag via photoshop:Urgency (Lightroom convention)
        # Urgency 1 = pick, Urgency 8 = reject, others = unflagged
        urgency_str = desc.get(f"{{{NS['photoshop']}}}Urgency", "")
        if urgency_str:
            try:
                urgency = int(urgency_str)
                if urgency == 1:
                    data.flag = 1   # pick
                elif urgency == 8:
                    data.flag = -1  # reject
            except ValueError:
                pass

        # Color label
        label_name = desc.get(f"{{{NS['xmp']}}}Label", "")
        data.label_name = label_name
        data.color_label = COLOR_FROM_NAME.get(label_name.lower(), 0)

        # Keywords / event tags. Lightroom writes an rdf:Bag; some tools use
        # Alt or Seq, and a bare text value is legal too, so accept all of them.
        subject = desc.find(f"{{{NS['dc']}}}subject")
        if subject is not None:
            values: list[str] = []
            for container in ("Bag", "Alt", "Seq"):
                bag = subject.find(f"{{{NS['rdf']}}}{container}")
                if bag is not None:
                    values = [
                        (li.text or "").strip()
                        for li in bag.findall(f"{{{NS['rdf']}}}li")
                    ]
                    break
            if not values and (subject.text or "").strip():
                values = [subject.text.strip()]
            data.tags = [v for v in values if v]

        # De-duplicate while keeping order
        if data.tags:
            data.tags = list(dict.fromkeys(data.tags))

        # Develop settings, whoever wrote them. Read back so a write of our own
        # can carry them forward, and so a test can see what landed.
        data.develop = _read_develop(desc)

    except ET.ParseError as e:
        logger.warning(f"Failed to parse XMP sidecar {xmp_path}: {e}")

    return data


# ── Write ────────────────────────────────────────────────────────────

_XMP_TEMPLATE = """\
<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      xmp:Rating="{rating}"
      {label_attr}
      {urgency_attr}
      {develop_attrs}
    >{subject_block}</rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def _clean_tags(tags: list[str]) -> list[str]:
    """Trim, drop empties, de-duplicate, keep order."""
    clean = []
    for t in tags or []:
        t = (t or "").strip()
        if t and t not in clean:
            clean.append(t)
    return clean


def _develop_attrs(develop: dict[str, str]) -> str:
    """The develop settings as attributes for the new-file template."""
    return "\n      ".join(
        f'{CRS}:{_local_name(name)}="{escape(str(value), _ATTR_ESCAPES)}"'
        for name, value in (develop or {}).items()
    )


def _set_develop(desc: ET.Element, develop: dict[str, str], owned: object = _UNSET) -> None:
    """Replace this app's develop attributes on a description element.

    ``owned`` names the attributes being replaced. Passed, only those come off
    and everything else in the crs: block survives — a crop, a curve, a lens
    profile another editor wrote. That matters now that a slider movement writes
    the sidecar immediately: a shared file that loses half its contents because
    someone nudged an exposure in this app is worse than no sidecar at all.

    Left unset, the whole crs: block is replaced. That is what a caller wants
    when it has already read the block and is writing it back unchanged, and it
    is what clearing an edit means when there is nothing to preserve.
    """
    uri = f"{{{NS[CRS]}}}"
    if owned is _UNSET:
        doomed = [k for k in desc.attrib if k.startswith(uri)]
    else:
        names = {f"{uri}{_local_name(n)}" for n in owned}
        doomed = [k for k in desc.attrib if k in names]
    for key in doomed:
        del desc.attrib[key]
    for name, value in (develop or {}).items():
        desc.set(f"{uri}{_local_name(name)}", str(value))


def _subject_block(tags: list[str]) -> str:
    """dc:subject XML fragment for the template, or '' when there are no tags."""
    clean = _clean_tags(tags)
    if not clean:
        return ""
    items = "".join(f"\n        <rdf:li>{escape(t)}</rdf:li>" for t in clean)
    return (
        "\n      <dc:subject>\n"
        f"        <rdf:Bag>{items}\n"
        "        </rdf:Bag>\n"
        "      </dc:subject>\n    "
    )


def write_xmp(image_path: Path, data: XMPData, owned: object = _UNSET) -> Path:
    """
    Write rating/flag/label to an XMP sidecar file.

    Creates the file if it doesn't exist, or updates the existing one.
    For simplicity, we rewrite the full file rather than patching XML —
    these files are tiny (<1KB) and writes are infrequent.

    ``owned`` limits which develop attributes are replaced; see
    :func:`_set_develop`. Left unset, the whole crs: block is this app's, which
    is right for a caller that read it first and is writing it back.
    """
    xmp_path = sidecar_path(image_path)

    # If sidecar exists, try to update it preserving other fields
    if xmp_path.exists():
        return _update_existing_xmp(xmp_path, data, owned)

    # Create new sidecar
    return _create_new_xmp(xmp_path, data)


def _create_new_xmp(xmp_path: Path, data: XMPData) -> Path:
    """Create a new XMP sidecar file."""
    label_attr = ""
    if data.color_label > 0:
        name = COLOR_NAMES.get(data.color_label, "")
        if name:
            label_attr = f'xmp:Label="{name}"'

    urgency_attr = ""
    if data.flag == 1:
        urgency_attr = 'photoshop:Urgency="1"'
    elif data.flag == -1:
        urgency_attr = 'photoshop:Urgency="8"'

    content = _XMP_TEMPLATE.format(
        rating=data.rating,
        label_attr=label_attr,
        urgency_attr=urgency_attr,
        develop_attrs=_develop_attrs(data.develop),
        subject_block=_subject_block(data.tags),
    )

    xmp_path.write_text(content, encoding="utf-8")
    logger.debug(f"Created XMP sidecar: {xmp_path.name}")
    return xmp_path


def _update_existing_xmp(xmp_path: Path, data: XMPData, owned: object = _UNSET) -> Path:
    """Update an existing XMP sidecar, preserving unrelated fields."""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        desc = root.find(".//rdf:Description", NS)

        if desc is None:
            # Malformed file, just recreate
            return _create_new_xmp(xmp_path, data)

        # Update rating
        desc.set(f"{{{NS['xmp']}}}Rating", str(data.rating))

        # Update label
        if data.color_label > 0:
            name = COLOR_NAMES.get(data.color_label, "")
            desc.set(f"{{{NS['xmp']}}}Label", name)
        else:
            # Remove label if set to none
            key = f"{{{NS['xmp']}}}Label"
            if key in desc.attrib:
                del desc.attrib[key]

        # Update flag via urgency
        urgency_key = f"{{{NS['photoshop']}}}Urgency"
        if data.flag == 1:
            desc.set(urgency_key, "1")
        elif data.flag == -1:
            desc.set(urgency_key, "8")
        else:
            if urgency_key in desc.attrib:
                del desc.attrib[urgency_key]

        # Update the develop settings, whoever set them.
        _set_develop(desc, data.develop, owned)

        # Update keywords / event tags (replace the whole dc:subject block)
        subject_key = f"{{{NS['dc']}}}subject"
        for existing in list(desc.findall(subject_key)):
            desc.remove(existing)
        clean_tags = _clean_tags(data.tags)
        if clean_tags:
            subject = ET.SubElement(desc, subject_key)
            bag = ET.SubElement(subject, f"{{{NS['rdf']}}}Bag")
            for tag in clean_tags:
                li = ET.SubElement(bag, f"{{{NS['rdf']}}}li")
                li.text = tag

        tree.write(xmp_path, xml_declaration=True, encoding="unicode")
        logger.debug(f"Updated XMP sidecar: {xmp_path.name}")
        return xmp_path

    except ET.ParseError:
        logger.warning(f"Corrupt XMP, recreating: {xmp_path}")
        return _create_new_xmp(xmp_path, data)


# ── Sync helpers ─────────────────────────────────────────────────────

def sync_rating_to_xmp(
    image_path: Path,
    rating: int = 0,
    flag: int = 0,
    color_label: int = 0,
    tags: Optional[list[str]] = None,
) -> Path:
    """
    Convenience: set specific fields and write to XMP.
    Reads existing sidecar first to preserve other values.
    tags=None leaves any existing keywords untouched.
    """
    data = read_xmp(image_path)
    data.rating = rating
    data.flag = flag
    data.color_label = color_label
    if tags is not None:
        data.tags = tags
    return write_xmp(image_path, data)


def sync_develop_to_xmp(
    image_path: Path, develop: dict[str, str], owned: object = _UNSET
) -> Path:
    """Record an edit in the sidecar, leaving the cull state alone.

    One way round on purpose: this app writes the edit it applied, and another
    editor reading the sidecar sees the same numbers. It is not a two-way
    import — values another tool wrote are read back only so the next write can
    carry them rather than contradict them.

    ``owned`` is the set of attribute names this app manages. Pass it and the
    rest of the crs: block belongs to whoever else is editing the file; leave it
    out and this app is the only author, which is what clearing an edit wants.
    """
    data = read_xmp(image_path)
    data.develop = dict(develop or {})
    return write_xmp(image_path, data, owned)


def sync_folder_from_xmp(folder: Path, extensions: list[str]) -> list[XMPData]:
    """
    Read all XMP sidecars in a folder.
    Useful for rebuilding the SQLite catalog from sidecars.
    """
    ext_set = {ext.lower() for ext in extensions}
    results = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in ext_set:
            results.append(read_xmp(f))
    return results
