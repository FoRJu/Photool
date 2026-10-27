"""
Tests for who may use the app: this computer, paired devices, and nobody else.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import access
from config import Config

LAN = ("192.168.1.20", 50000)
HOME = ("127.0.0.1", 50001)


def _app(tmp_path, **config) -> FastAPI:
    app = FastAPI()
    app.state.config = Config(library_root=str(tmp_path), **config)
    app.state.devices = access.DeviceStore(str(tmp_path / "devices.db"))
    access.install(app)
    app.include_router(access.router, prefix="/api")

    @app.get("/api/photos")
    def photos():
        return {"photos": ["IMG_0001.CR3"]}

    @app.post("/mcp")
    def mcp():
        return {"tools": ["search"]}

    @app.get("/")
    def page():
        return {"page": True}

    return app


def _client(app, address=LAN):
    return TestClient(app, client=address)


def _pair(app, name="Mom's iPad"):
    """Ask from the network, approve at home, collect the key."""
    lan, home = _client(app, LAN), _client(app, HOME)
    asked = lan.post("/api/auth/pair/request", json={"name": name}).json()
    pending = home.get("/api/auth/pending").json()["requests"]
    assert asked["code"] in [p["code"] for p in pending]
    assert home.post(f"/api/auth/pair/{asked['id']}/approve").status_code == 200
    claimed = lan.post("/api/auth/pair/claim", json={"id": asked["id"], "claim": asked["claim"]})
    assert claimed.json()["status"] == "approved"
    return lan


class TestThisComputer:
    def test_this_computer_needs_nothing(self, tmp_path):
        app = _app(tmp_path)
        assert _client(app, HOME).get("/api/photos").status_code == 200
        assert _client(app, ("::1", 1)).get("/api/photos").status_code == 200
        assert _client(app, ("::ffff:127.0.0.1", 1)).get("/api/photos").status_code == 200

    def test_a_local_proxy_carrying_someone_else_is_not_this_computer(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        relayed = _client(app, HOME).get("/api/photos", headers={"X-Forwarded-For": "192.168.1.20"})
        assert relayed.status_code == 401
        mine = _client(app, HOME).get("/api/photos", headers={"X-Forwarded-For": "127.0.0.1"})
        assert mine.status_code == 200


class TestTheNetwork:
    def test_with_network_access_off_nothing_is_shown(self, tmp_path):
        app = _app(tmp_path, lan_access=False)
        lan = _client(app)
        assert lan.get("/").status_code == 403
        assert lan.get("/api/photos").status_code == 403
        assert lan.get("/api/auth/status").status_code == 403

    def test_an_unpaired_device_gets_the_page_and_no_photos(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan = _client(app)
        assert lan.get("/").status_code == 200
        assert lan.get("/api/photos").status_code == 401
        status = lan.get("/api/auth/status").json()
        assert status == {"local": False, "paired": False, "device": "", "lan_access": True, "pending": 0}

    def test_the_mcp_endpoint_is_guarded_like_the_api(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        assert _client(app).post("/mcp").status_code == 401
        assert _client(app, HOME).post("/mcp").status_code == 200
        assert _pair(app).post("/mcp").status_code == 200

    def test_a_paired_device_sees_the_photos(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan = _pair(app)
        assert lan.get("/api/photos").status_code == 200
        assert lan.get("/api/auth/status").json()["device"] == "Mom's iPad"
        assert access.COOKIE in lan.cookies

    def test_the_key_is_stored_only_as_a_hash(self, tmp_path):
        import sqlite3

        app = _app(tmp_path, lan_access=True)
        lan = _pair(app)
        token = lan.cookies[access.COOKIE]
        dump = "\n".join(sqlite3.connect(tmp_path / "devices.db").iterdump())
        assert token not in dump

    def test_a_device_cannot_approve_itself_or_manage_devices(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan = _client(app)
        asked = lan.post("/api/auth/pair/request", json={"name": "Stranger"}).json()
        assert lan.post(f"/api/auth/pair/{asked['id']}/approve").status_code == 401
        paired = _pair(app, "Laptop")
        assert paired.post(f"/api/auth/pair/{asked['id']}/approve").status_code == 403
        assert paired.get("/api/auth/devices").status_code == 403

    def test_only_the_asking_device_collects_the_key(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan, home = _client(app), _client(app, HOME)
        asked = lan.post("/api/auth/pair/request", json={"name": "Phone"}).json()
        home.post(f"/api/auth/pair/{asked['id']}/approve")
        thief = _client(app, ("192.168.1.66", 1))
        stolen = thief.post("/api/auth/pair/claim", json={"id": asked["id"], "claim": "guess"})
        assert stolen.json()["status"] == "expired" and access.COOKIE not in thief.cookies
        assert lan.post("/api/auth/pair/claim",
                        json={"id": asked["id"], "claim": asked["claim"]}).json()["status"] == "approved"

    def test_a_denied_device_stays_out(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan, home = _client(app), _client(app, HOME)
        asked = lan.post("/api/auth/pair/request", json={"name": "Phone"}).json()
        home.post(f"/api/auth/pair/{asked['id']}/deny")
        assert lan.post("/api/auth/pair/claim",
                        json={"id": asked["id"], "claim": asked["claim"]}).json()["status"] == "denied"
        assert lan.get("/api/photos").status_code == 401

    def test_revoking_a_device_shuts_it_out(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan = _pair(app)
        home = _client(app, HOME)
        device = home.get("/api/auth/devices").json()["devices"][0]
        assert device["name"] == "Mom's iPad" and device["last_address"] == LAN[0]
        assert home.delete(f"/api/auth/devices/{device['id']}").status_code == 200
        assert lan.get("/api/photos").status_code == 401

    def test_a_flood_of_requests_is_refused(self, tmp_path):
        app = _app(tmp_path, lan_access=True)
        lan = _client(app)
        codes = [lan.post("/api/auth/pair/request", json={"name": "x"}).status_code for _ in range(7)]
        assert codes[:access.MAX_PER_ADDRESS] == [200] * access.MAX_PER_ADDRESS
        assert codes[-1] == 429


class TestTailscale:
    def test_a_listed_login_through_tailscale_serve_is_let_in(self, tmp_path):
        app = _app(tmp_path, lan_access=False, lan_trusted_logins=["Me@Example.com"])
        through = _client(app, HOME).get("/api/photos", headers={
            "Tailscale-User-Login": "me@example.com", "X-Forwarded-For": "100.64.0.7"})
        assert through.status_code == 200

    def test_another_tailnet_user_must_pair(self, tmp_path):
        app = _app(tmp_path, lan_access=False, lan_trusted_logins=["me@example.com"])
        other = _client(app, HOME).get("/api/photos", headers={
            "Tailscale-User-Login": "guest@example.com", "X-Forwarded-For": "100.64.0.8"})
        assert other.status_code == 401

    def test_the_login_header_is_not_believed_from_the_network(self, tmp_path):
        app = _app(tmp_path, lan_access=True, lan_trusted_logins=["me@example.com"])
        forged = _client(app, LAN).get("/api/photos", headers={"Tailscale-User-Login": "me@example.com"})
        assert forged.status_code == 401
