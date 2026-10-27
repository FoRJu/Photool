"""
The command registry over plain HTTP: the same commands the MCP tools run.

For scripts and quick checks with curl, without an MCP client::

    curl localhost:8000/api/commands
    curl -X POST localhost:8000/api/commands/library.photos -d '{"folder": "2025/2025-03-06"}'

A command that renders returns its JPEG base64-encoded beside its notes.
"""

import base64

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from commands import COMMANDS, CommandContext, CommandError, Rendered, run

router = APIRouter()


@router.get("/commands")
async def list_commands():
    """Every command, with its parameter schema."""
    return {"commands": [cmd.describe() for cmd in COMMANDS.values()]}


@router.post("/commands/{command_id}")
async def run_command(command_id: str, request: Request):
    if command_id not in COMMANDS:
        raise HTTPException(status_code=404, detail=f"No such command: {command_id}")
    body = await request.body()
    try:
        arguments = await request.json() if body.strip() else {}
    except ValueError:
        raise HTTPException(status_code=400, detail="The body must be a JSON object")
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="The body must be a JSON object")
    try:
        result = await run(CommandContext.from_app(request.app), command_id, arguments)
    except CommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(result, Rendered):
        if request.query_params.get("raw"):
            return Response(result.jpeg, media_type="image/jpeg")
        return {**result.info, "jpeg_base64": base64.b64encode(result.jpeg).decode("ascii")}
    return result
