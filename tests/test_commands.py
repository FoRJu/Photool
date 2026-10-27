"""
The command registry, through both of its doors: ``/api/commands`` and MCP at
``/mcp``. Against a real (temporary) library, so a command that says it rated or
developed a photo is checked against the catalog and the rendering, not its reply.
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

FOLDER = "2025/2025-03-06"
PHOTO = f"{FOLDER}/IMG_0001.JPG"
OTHER = f"{FOLDER}/IMG_0002.JPG"


def _mean(data: bytes) -> float:
    return float(np.asarray(Image.open(io.BytesIO(data)).convert("L")).mean())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    rgb = np.full((300, 400, 3), 60, dtype=np.uint8)
    rgb[:60] = 230
    for name in (PHOTO, OTHER):
        Image.fromarray(rgb, "RGB").save(library / name, format="JPEG", quality=95)

    (tmp_path / "config.yaml").write_text(
        # Single-quoted: YAML takes the backslashes of a Windows path literally there.
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)

    import main

    # The MCP endpoint only answers to a local host name (DNS rebinding guard).
    with TestClient(main.app, base_url="http://localhost:8000") as c:
        assert c.post(f"/api/catalog/index/{FOLDER}").status_code == 200
        c.library = library
        yield c


def call(client, command_id, **arguments):
    res = client.post(f"/api/commands/{command_id}", json=arguments)
    assert res.status_code == 200, res.text
    return res.json()


class TestTheRegistry:
    def test_every_command_is_listed_with_a_schema(self, client):
        listed = {c["id"]: c for c in client.get("/api/commands").json()["commands"]}
        assert {"library.photos", "photo.rate", "develop.set", "render.preview", "export.run"} <= set(listed)
        assert listed["develop.set"]["writes"] is True
        assert listed["library.photos"]["writes"] is False
        assert "folder" in listed["library.photos"]["params"]["properties"]

    def test_an_unknown_command_is_a_404(self, client):
        assert client.post("/api/commands/nope.nothing", json={}).status_code == 404

    def test_bad_parameters_say_what_is_wrong(self, client):
        res = client.post("/api/commands/photo.rate", json={"filepaths": [PHOTO], "rating": 9})
        assert res.status_code == 400
        assert "rating" in res.json()["detail"]

    def test_a_path_outside_the_library_is_refused(self, client):
        res = client.post("/api/commands/render.preview", json={"filepath": "../config.yaml"})
        assert res.status_code == 400


class TestCulling:
    def test_listing_a_folder(self, client):
        folders = call(client, "library.folders")["folders"]
        assert any(FOLDER in json.dumps(f) for f in folders)
        listed = call(client, "library.photos", folder=FOLDER)
        assert listed["total"] == 2
        assert {p["filepath"] for p in listed["photos"]} == {PHOTO, OTHER}

    def test_rating_reaches_the_catalog_and_the_filters(self, client):
        call(client, "photo.rate", filepaths=[PHOTO], rating=4, flag=1)
        found = call(client, "library.photos", folder=FOLDER, min_rating=3)["photos"]
        assert [p["filepath"] for p in found] == [PHOTO]
        assert found[0]["flag"] == 1

    def test_rating_a_photo_that_is_not_there_changes_nothing(self, client):
        res = client.post("/api/commands/photo.rate",
                          json={"filepaths": [PHOTO, f"{FOLDER}/missing.JPG"], "rating": 5})
        assert res.status_code == 400
        assert call(client, "library.photo", filepath=PHOTO)["rating"] == 0

    def test_tags(self, client):
        call(client, "photo.set_tags", filepath=PHOTO, tags=["garden"])
        assert call(client, "library.photo", filepath=PHOTO)["tags"] == ["garden"]
        call(client, "tags.apply", folder=FOLDER, tag="spring")
        assert {t["name"] for t in call(client, "tags.list")["tags"]} >= {"garden", "spring"}


class TestDeveloping:
    def test_controls_describe_every_key(self, client):
        keys = {c["key"] for c in call(client, "develop.controls")["controls"]}
        assert {"exposure", "shadows", "vibrance", "angle", "crop_w"} <= keys

    def test_merge_keeps_the_rest_of_the_edit_and_replace_does_not(self, client):
        call(client, "develop.set", filepath=PHOTO, settings={"exposure": 1.0})
        merged = call(client, "develop.set", filepath=PHOTO, settings={"shadows": 30})["settings"]
        assert merged == {"exposure": 1.0, "shadows": 30}
        replaced = call(client, "develop.set", filepath=PHOTO, settings={"contrast": 10},
                        mode="replace")["settings"]
        assert replaced == {"contrast": 10}
        assert call(client, "develop.get", filepath=PHOTO)["settings"] == {"contrast": 10}

    def test_values_are_clamped_and_unknown_keys_reported(self, client):
        out = call(client, "develop.set", filepath=PHOTO, settings={"exposure": 99, "sparkle": 5})
        assert out["settings"]["exposure"] == 5.0
        assert out["ignored"] == ["sparkle"]

    def test_the_edit_is_what_the_ui_sees(self, client):
        call(client, "develop.set", filepath=PHOTO, settings={"exposure": 1.0})
        assert client.get("/api/photos/edit", params={"filepath": PHOTO}).json()["settings"] == {"exposure": 1.0}

    def test_reset(self, client):
        call(client, "develop.set", filepath=PHOTO, settings={"exposure": 1.0})
        call(client, "develop.reset", filepath=PHOTO)
        assert call(client, "develop.get", filepath=PHOTO)["settings"] == {}

    def test_preview_shows_the_edit_and_a_trial_edit_is_not_saved(self, client):
        plain = call(client, "render.preview", filepath=PHOTO, size=256)
        brighter = call(client, "render.preview", filepath=PHOTO, size=256, settings={"exposure": 1.5})
        assert _mean(base64.b64decode(brighter["jpeg_base64"])) > _mean(base64.b64decode(plain["jpeg_base64"])) + 10
        assert call(client, "develop.get", filepath=PHOTO)["settings"] == {}

    def test_preview_as_a_plain_jpeg(self, client):
        res = client.post("/api/commands/render.preview?raw=1", json={"filepath": PHOTO, "size": 256})
        assert res.headers["content-type"] == "image/jpeg"
        assert max(Image.open(io.BytesIO(res.content)).size) <= 256

    def test_auto_suggests_and_applies_on_request(self, client):
        suggested = call(client, "develop.auto", filepath=PHOTO)
        assert call(client, "develop.get", filepath=PHOTO)["settings"] == {}
        applied = call(client, "develop.auto", filepath=PHOTO, apply=True)
        assert applied["settings"] == call(client, "develop.get", filepath=PHOTO)["settings"]
        assert isinstance(suggested["suggested"], dict)

    def test_paste_onto_many(self, client):
        out = call(client, "develop.paste", filepaths=[PHOTO, OTHER], settings={"vibrance": 20})
        assert sorted(out["copied"]) == sorted([PHOTO, OTHER])
        assert call(client, "develop.get", filepath=OTHER)["settings"] == {"vibrance": 20}


class TestExporting:
    def test_basket_then_export(self, client, tmp_path):
        assert call(client, "basket.add", filepaths=[PHOTO])["count"] == 1
        out = client.library / "export" / "out"
        job = call(client, "export.run", folder="out")
        for _ in range(100):
            state = call(client, "export.progress", job_id=job["job_id"])
            if state["status"] == "completed":
                break
            time.sleep(0.1)
        assert state["completed"] == 1 and state["failed"] == 0
        assert len(list(out.glob("*.jpg"))) == 1
        assert call(client, "basket.clear")["count"] == 0

    def test_an_empty_basket_is_said_plainly(self, client, tmp_path):
        res = client.post("/api/commands/export.run", json={"folder": "o"})
        assert res.status_code == 400
        assert "empty" in res.json()["detail"]


# ── MCP ──────────────────────────────────────────────────────────────

_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _messages(res) -> list[dict]:
    """JSON-RPC messages from a reply, whether sent as JSON or as an event stream."""
    if res.headers.get("content-type", "").startswith("application/json"):
        return [res.json()]
    return [json.loads(line[5:]) for line in res.text.splitlines() if line.startswith("data:")]


class McpSession:
    def __init__(self, client):
        self.client = client
        self.headers = dict(_HEADERS)
        self.next_id = 0
        reply = self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        }, keep_session=True)
        assert reply["result"]["serverInfo"]["name"] == "photool"
        self.client.post("/mcp", headers=self.headers,
                         json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    def request(self, method, params=None, keep_session=False):
        self.next_id += 1
        res = self.client.post("/mcp", headers=self.headers, json={
            "jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {},
        })
        assert res.status_code == 200, res.text
        if keep_session and "mcp-session-id" in res.headers:
            self.headers["mcp-session-id"] = res.headers["mcp-session-id"]
        return next(m for m in _messages(res) if m.get("id") == self.next_id)

    def call(self, name, **arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]


class TestMcp:
    def test_every_command_is_a_tool(self, client):
        from commands import COMMANDS

        tools = McpSession(client).request("tools/list")["result"]["tools"]
        assert {t["name"] for t in tools} == set(COMMANDS)
        rate = next(t for t in tools if t["name"] == "photo.rate")
        # Optional, so the bound sits on the integer arm of anyOf.
        rating = rate["inputSchema"]["properties"]["rating"]
        assert {"type": "integer", "minimum": 0, "maximum": 5}.items() <= rating["anyOf"][0].items()
        assert rate["annotations"]["readOnlyHint"] is False

    def test_a_tool_call_changes_the_library(self, client):
        session = McpSession(client)
        result = session.call("develop.set", filepath=PHOTO, settings={"exposure": 0.5})
        assert not result.get("isError")
        assert json.loads(result["content"][0]["text"])["settings"] == {"exposure": 0.5}
        assert call(client, "develop.get", filepath=PHOTO)["settings"] == {"exposure": 0.5}

    def test_a_render_comes_back_as_an_image(self, client):
        result = McpSession(client).call("render.preview", filepath=PHOTO, size=256)
        kinds = [c["type"] for c in result["content"]]
        assert kinds[0] == "image"
        assert result["content"][0]["mimeType"] == "image/jpeg"
        Image.open(io.BytesIO(base64.b64decode(result["content"][0]["data"])))

    def test_a_failure_is_a_tool_error_not_a_crash(self, client):
        result = McpSession(client).call("library.photo", filepath=f"{FOLDER}/missing.JPG")
        assert result["isError"] is True
        assert "Not in the catalog" in result["content"][0]["text"]

    def test_only_a_local_host_name_is_served(self, client):
        res = client.post("/mcp", headers={**_HEADERS, "Host": "evil.example"}, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        assert res.status_code == 421

    def test_nested_parameters(self, client):
        session = McpSession(client)
        session.call("basket.add", filepaths=[PHOTO])
        result = session.call("export.run", folder="out",
                              style={"vignette": 20})
        assert not result.get("isError"), result
        assert json.loads(result["content"][0]["text"])["total"] == 1
