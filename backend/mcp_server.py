"""
The library over MCP, so an agent (Claude Code, Claude Desktop, …) can drive it.

Served by the backend itself at ``http://localhost:8000/mcp`` (streamable HTTP),
in the same process as the UI's API. That is deliberate: the agent works on the
same catalog, the same caches and the same sidecars as the person at the screen,
and a rating it sets is in the grid at the next refresh.

There is nothing here but plumbing. Each tool is one entry of the command
registry (:mod:`commands`), with the command's own parameters as the tool's
arguments; adding a command adds a tool.
"""

import inspect
import json
from typing import Annotated

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from commands import COMMANDS, Command, CommandContext, CommandError, Rendered, run

INSTRUCTIONS = """\
Photool is a photo library: import, cull, develop and export, Lightroom-style, for
Canon and Ricoh RAW (CR3, CR2, DNG) and JPEG. Photos are addressed by their path
relative to the library root, exactly as library.photos returns it.

Typical flow: library.folders -> library.photos (filters) -> look with
render.preview -> photo.rate / photo.set_tags to cull -> develop.controls, then
develop.set or develop.auto -> render.preview to check -> basket.add -> export.run
-> export.progress.

Culling a folder: read cull.guide first, which says how this person culls, and cull by it.
cull.measure (once) -> cull.signals, which gives each frame's scene,
burst, keeper, sharpness rank, soft/blown warnings, faces, people and closed eyes ->
drop the technical failures from those numbers without looking -> render.contact_sheet
of each scene or burst (numbered frames, cheap) to compare and choose -> render.loupe on
a finalist's face if focus is in doubt -> cull.propose with a short reason for every
frame you have an opinion on. Proposals wait for the person in the grid; do not use
photo.rate or cull.accept to cull unless the person asks for marks to be applied. On a
folder the person has already culled, cull.propose then cull.evaluate measures how well
your choices agree with theirs.

Edits are non-destructive: they live in the catalog and the photo's XMP sidecar,
never in the image file. A RAW+JPEG pair is one photo; previews and exports
develop the RAW.
"""


def _compact(result):
    """A command's answer as JSON without indentation. The SDK would indent it, which
    on a folder's culling signals doubles what the model has to read."""
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)


def _signature(cmd: Command) -> inspect.Signature:
    """The command's parameters as a function signature, which is what the MCP SDK
    reads to build a tool's input schema."""
    parameters = []
    for name, info in cmd.params.model_fields.items():
        # The field's constraints (ranges, lengths) travel in its metadata.
        annotation = Annotated[(info.annotation, *info.metadata, Field(description=info.description))]
        default = (inspect.Parameter.empty if info.is_required()
                   else info.get_default(call_default_factory=True))
        parameters.append(inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation,
        ))
    return inspect.Signature(parameters)


def _tool(app, cmd: Command):
    async def call(**arguments):
        # Nested models arrive as model instances; the command validates plain data.
        plain = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in arguments.items()}
        try:
            result = await run(CommandContext.from_app(app), cmd.id, plain)
        except CommandError as exc:
            raise ToolError(str(exc)) from None
        if isinstance(result, Rendered):
            return [Image(data=result.jpeg, format="jpeg"), _compact(result.info)]
        return _compact(result)

    call.__name__ = cmd.id.replace(".", "_")
    call.__signature__ = _signature(cmd)
    return call


def build(app) -> MCPServer:
    server = MCPServer("photool", instructions=INSTRUCTIONS)
    for cmd in COMMANDS.values():
        server.add_tool(
            _tool(app, cmd),
            name=cmd.id,
            description=cmd.description,
            annotations=ToolAnnotations(readOnlyHint=not cmd.writes, openWorldHint=False),
            structured_output=False,
        )
    return server


class Endpoint:
    """The ``/mcp`` route: hands each request to this run's MCP HTTP app.

    The SDK's session manager can be started only once, but the FastAPI app is
    started more than once in a process (every TestClient does it), so the HTTP
    app is made afresh in each lifespan and looked up here per request.
    """

    async def __call__(self, scope, receive, send):
        await scope["app"].state.mcp_http(scope, receive, send)
