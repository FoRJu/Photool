"""
Catalog module — SQLite database + XMP sidecar support.

Dual-write architecture: every rating/flag operation writes to both
SQLite (fast queries) and XMP sidecar (portable, survives DB loss).
"""

from .database import CatalogDB, PhotoRecord, Flag, ColorLabel, ThumbnailJobRecord, ExportBasketRecord
from .xmp import read_xmp, write_xmp, XMPData, sidecar_path
from .service import CatalogService

__all__ = [
    "CatalogDB",
    "PhotoRecord",
    "Flag",
    "ColorLabel",
    "ThumbnailJobRecord",
    "ExportBasketRecord",
    "CatalogService",
    "read_xmp",
    "write_xmp",
    "XMPData",
    "sidecar_path",
]
