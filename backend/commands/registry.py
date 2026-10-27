"""
The command registry: every action an agent can take, under one stable name.

A command is an id (``develop.set``), a short description, a Pydantic model for
its parameters and a plain function that does the work. Nothing about a
command knows how it was reached. The MCP server (``mcp_server.py``) and the
``/api/commands`` routes both read this one table, so a command added here is
an MCP tool and an HTTP call at the same time, with the same checks.

Handlers are synchronous on purpose. Almost everything they do is a SQLite
query, a sidecar write or a decode, all of which block; :func:`run` puts each
call on a worker thread so a slow render does not stall the server.
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError


class CommandError(Exception):
    """A command could not do what was asked; the message says why, plainly."""


class NoParams(BaseModel):
    pass


@dataclass(frozen=True)
class Command:
    id: str
    description: str
    params: type[BaseModel]
    handler: Callable[..., Any]
    # Changes the catalog, a sidecar or a file on disk. A writing command is
    # refused outright on a read-only catalog rather than reporting a save that
    # did not happen.
    writes: bool = False

    def describe(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "writes": self.writes,
            "params": self.params.model_json_schema(),
        }


@dataclass(frozen=True)
class Rendered:
    """A picture a command produced, with what to know about it."""
    jpeg: bytes
    info: dict = field(default_factory=dict)


@dataclass
class CommandContext:
    """What a command may reach: the running app's own services."""
    config: Any
    catalog: Any
    # The thumbnail cache: what the grid shows, without developing anything.
    cache: Any = None

    @classmethod
    def from_app(cls, app) -> "CommandContext":
        return cls(config=app.state.config, catalog=app.state.catalog,
                   cache=getattr(app.state, "thumb_cache", None))

    def library_file(self, filepath: str) -> Path:
        """A catalog-relative path, resolved inside the library or refused."""
        root = Path(self.config.library_root)
        path = root / filepath
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            raise CommandError(f"Not a path inside the library: {filepath}")
        if not path.exists():
            raise CommandError(f"File not found: {filepath}")
        return path

    def photo(self, filepath: str):
        """The catalog record for a photograph, or a CommandError naming it."""
        record = self.catalog.get_photo(filepath)
        if record is None:
            raise CommandError(
                f"Not in the catalog: {filepath}. Paths are relative to the library "
                "root, as library.photos returns them."
            )
        return record


COMMANDS: dict[str, Command] = {}


def command(id: str, *, params: type[BaseModel] = NoParams, writes: bool = False):
    """Register the decorated function as the command ``id``.

    The function's docstring is the description an agent reads, so it should say
    what the command does and when to reach for it, not how it is implemented.
    """
    def register(fn):
        if id in COMMANDS:
            raise ValueError(f"Command registered twice: {id}")
        COMMANDS[id] = Command(
            id=id,
            description=inspect.cleandoc(fn.__doc__ or ""),
            params=params,
            handler=fn,
            writes=writes,
        )
        return fn
    return register


async def run(ctx: CommandContext, id: str, arguments: Optional[dict] = None) -> Any:
    """Validate ``arguments`` against command ``id`` and run it off the event loop."""
    cmd = COMMANDS.get(id)
    if cmd is None:
        raise CommandError(f"No such command: {id}")
    try:
        params = cmd.params.model_validate(arguments or {})
    except ValidationError as exc:
        raise CommandError(f"Invalid parameters for {id}: {exc}") from None
    if cmd.writes and not ctx.catalog.db.writable:
        raise CommandError(f"Catalog is not writable: {ctx.catalog.db.write_error}")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(cmd.handler, ctx, params))
