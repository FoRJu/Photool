"""
Write-access probes for the paths the app depends on.

Both the catalog (catalog.db) and the thumbnail cache (thumbnails.db) live under
library_root, which is usually a secondary volume. That volume can become
unwritable without the app noticing: a read-only mount, a changed ACL, a
disconnected drive letter, or a disk that has started protecting itself. Reads
keep working, so the app looks healthy while every rating, import record and
thumbnail silently fails to persist.

Everything here answers one question — can this path actually be written? — and
returns a message that names the path and the OS reason.
"""

from __future__ import annotations

import errno
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_PROBE_PREFIX = ".photomanager-write-probe"


def describe_os_error(exc: OSError) -> str:
    """Render an OSError the way a user can act on it."""
    name = errno.errorcode.get(exc.errno, "EUNKNOWN")
    detail = exc.strerror or str(exc)
    return f"{name} ({exc.errno}): {detail}"


def probe_writable(path: str | Path) -> tuple[bool, str]:
    """
    Return ``(writable, error_message)`` for a file path.

    The parent directory is probed by creating and deleting a scratch file, and
    an existing file is additionally opened for writing — a directory can accept
    new files while an existing DB is locked or read-only.

    Never raises. A missing parent is created when possible; if it cannot be
    created that is reported as the failure.
    """
    target = Path(path)
    parent = target.parent if target.suffix else target

    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"cannot create {parent} — {describe_os_error(exc)}"

    probe = parent / f"{_PROBE_PREFIX}-{os.getpid()}.tmp"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass
        return False, (
            f"cannot write to {parent} — {describe_os_error(exc)}. "
            "The volume may be mounted read-only, offline, or denying writes to this account."
        )

    if target.is_file():
        try:
            with open(target, "r+b"):
                pass
        except OSError as exc:
            return False, f"cannot open existing file {target} for writing — {describe_os_error(exc)}"

    return True, ""


def summarize(label: str, path: str) -> dict:
    """Probe one configured path and return a JSON-safe status block."""
    writable, error = probe_writable(path)
    if not writable:
        logger.error("[storage] %s is not writable: %s", label, error)
    return {"label": label, "path": str(path), "writable": writable, "error": error}


def _ro_uri(path: str | Path, immutable: bool = False) -> str:
    uri = str(path).replace("\\", "/")
    query = "mode=ro&immutable=1" if immutable else "mode=ro"
    return f"file:{uri}?{query}"


def open_readonly(path: str | Path) -> sqlite3.Connection:
    """
    Open an existing SQLite file for reading on a volume that refuses writes.

    A WAL-mode database needs a writable ``-shm`` file beside it, so a plain
    ``mode=ro`` open fails outright when the volume is read-only and no sidecar
    files were left behind — the app could not even browse its own catalog.
    ``immutable=1`` states that the file will not change, which is true while the
    volume is read-only, and lets SQLite read the main file directly.

    The one caveat: it skips WAL recovery, so anything left un-checkpointed in a
    ``-wal`` file is not visible. That data is unreachable either way until the
    volume is writable again, and it is not lost.
    """
    errors: list[Exception] = []
    for immutable in (False, True):
        conn = None
        try:
            conn = sqlite3.connect(_ro_uri(path, immutable), uri=True)
            # Touch the database before handing it back. SQLite reports the
            # missing -shm on the first statement, not on connect(), so a
            # connection that looks fine here would fail in the caller's face.
            conn.execute("PRAGMA page_count").fetchone()
            if immutable:
                logger.debug(
                    "[storage] reading %s in immutable mode: it is a WAL database on "
                    "a read-only volume, so its -wal/-shm sidecars cannot be used",
                    path,
                )
            return conn
        except sqlite3.OperationalError as exc:
            errors.append(exc)
            if conn is not None:
                conn.close()

    raise sqlite3.OperationalError(f"cannot open {path} for reading: {errors[-1]}")
