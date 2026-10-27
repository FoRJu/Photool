"""
Every action an agent can take on the library, by name. See :mod:`commands.registry`.

Importing this package registers the commands; the modules below are imported
for that side effect.
"""

from .registry import (
    COMMANDS,
    Command,
    CommandContext,
    CommandError,
    Rendered,
    command,
    run,
)
from . import library, develop, export, cull  # noqa: F401  (registers the commands)

__all__ = ["COMMANDS", "Command", "CommandContext", "CommandError", "Rendered", "command", "run"]
