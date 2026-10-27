"""
Who may use the app: this computer, and the devices it has paired.

The app holds a family's photographs, so "on the same Wi-Fi" is not a reason to
show them. Three kinds of caller:

* **This computer.** A connection from the loopback address with no proxy in
  between is the person sitting at the machine that holds the library. It can
  do everything, including approving and revoking other devices.
* **A paired device.** Another computer or phone asked to join, showed a code,
  and someone at this computer approved it after checking the same code was on
  both screens. It then holds a long random key in a cookie (HttpOnly,
  SameSite=Strict, a year); only the key's SHA-256 is stored here, so a copy of
  the database is not a key. It can browse, cull and edit, but not manage
  devices.
* **Everyone else** gets the app's page and the pairing screen, and a 401 for
  every API call — no thumbnail, no file name, no face.

Why not the alternatives. A MAC address is not an identity: any device can
claim any MAC, phones randomise theirs per network, and the address never
crosses a router. A one-time code by e-mail or text needs a mail or SMS service
on the internet to send it, and proves only that someone can read that inbox;
approving on the computer that holds the photos proves that someone in the house
said yes. For access from outside the house, put Tailscale in front
(`tailscale serve`): it encrypts the traffic, identifies the person by their
Google/Apple/Microsoft login, and needs no open port. Requests that come through
it arrive from loopback *with* Tailscale's headers, so they are never mistaken
for this computer — they pair like any device, or are let in by login when the
config lists that login.

When ``lan_access`` is off, a direct connection from the network is refused
outright, page included, so binding the server to 0.0.0.0 by mistake shows
nothing.
"""

import hashlib
import ipaddress
import logging
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

COOKIE = "photool_device"
COOKIE_DAYS = 365
# A pairing request lives this long, unanswered.
REQUEST_SECONDS = 300
# At most this many requests wait at once, and one address may make this many
# in REQUEST_WINDOW seconds: enough to retry a typo, not enough to flood the
# person at the computer with prompts.
MAX_PENDING = 6
MAX_PER_ADDRESS = 5
REQUEST_WINDOW = 600

# API paths a device that is not paired may call: to learn that it is not, and
# to ask to be.
PUBLIC_API = ("/api/auth/status", "/api/auth/pair/request", "/api/auth/pair/claim")

router = APIRouter()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host.split("%")[0])
        if address.version == 6 and address.ipv4_mapped is not None:
            address = address.ipv4_mapped        # ::ffff:127.0.0.1
        return address.is_loopback
    except ValueError:
        # Starlette's TestClient reports its peer as "testclient"; a real
        # connection's peer is always an address.
        return host in ("localhost", "testclient")


def _forwarded_for(request: Request) -> list[str]:
    """Every address a proxy says the request passed through."""
    out: list[str] = []
    for value in request.headers.getlist("x-forwarded-for"):
        out.extend(p.strip() for p in value.split(",") if p.strip())
    for value in request.headers.getlist("forwarded"):
        for part in value.split(";"):
            part = part.strip()
            if part.lower().startswith("for="):
                out.append(part[4:].strip('"[]'))
    return out


def client_address(request: Request) -> str:
    """The caller's address as best known: the first hop a local proxy names,
    else the connection's own."""
    peer = request.client.host if request.client else ""
    hops = _forwarded_for(request)
    if _is_loopback(peer) and hops:
        return hops[0]
    return peer


def is_local(request: Request) -> bool:
    """The person at this computer: a loopback connection that no proxy relayed
    (a proxy on this machine — the dev server, `tailscale serve` — may be
    carrying someone else's request)."""
    peer = request.client.host if request.client else ""
    if not _is_loopback(peer):
        return False
    if request.headers.get("tailscale-user-login"):
        return False
    return all(_is_loopback(hop) for hop in _forwarded_for(request))


class DeviceStore:
    """Paired devices and pairing requests, in a small SQLite file of their own
    beside the catalog."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._recent: dict[str, list[float]] = {}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS paired_devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL DEFAULT '',
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL DEFAULT 0,
                    last_address TEXT NOT NULL DEFAULT '',
                    user_agent TEXT NOT NULL DEFAULT ''
                )"""
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Devices ──────────────────────────────────────────────────────

    def device_for(self, token: Optional[str], address: str = "") -> Optional[dict]:
        """The paired device holding ``token``, its last-seen time refreshed (at
        most once a minute, so browsing a grid is not a write per thumbnail)."""
        if not token or len(token) > 200:
            return None
        digest = _hash(token)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM paired_devices WHERE token_hash = ?", (digest,)).fetchone()
            if row is None:
                return None
            now = time.time()
            if now - float(row["last_seen"]) > 60 or row["last_address"] != address:
                conn.execute("UPDATE paired_devices SET last_seen = ?, last_address = ? WHERE id = ?",
                             (now, address, row["id"]))
            return dict(row)

    def add_device(self, name: str, address: str, user_agent: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO paired_devices (name, token_hash, created_at, last_seen, last_address, user_agent)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, _hash(token), now, now, address, user_agent[:300]),
            )
        return token

    def list_devices(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, last_seen, last_address, user_agent FROM paired_devices ORDER BY last_seen DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def revoke(self, device_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM paired_devices WHERE id = ?", (int(device_id),)).rowcount > 0

    # ── Pairing ──────────────────────────────────────────────────────

    def _expire(self) -> None:
        now = time.time()
        for rid in [r for r, p in self._pending.items() if now - p["created"] > REQUEST_SECONDS]:
            del self._pending[rid]

    def request(self, name: str, address: str, user_agent: str) -> dict:
        """A new pairing request: an id, the code both screens show, and a claim
        secret only the asking device knows (so nobody else can collect the key
        once the request is approved)."""
        with self._lock:
            self._expire()
            now = time.time()
            recent = [t for t in self._recent.get(address, []) if now - t < REQUEST_WINDOW]
            if len(recent) >= MAX_PER_ADDRESS:
                raise HTTPException(status_code=429, detail="Too many requests from this device; wait a few minutes")
            if len(self._pending) >= MAX_PENDING:
                raise HTTPException(status_code=429, detail="Too many devices are waiting; try again shortly")
            self._recent[address] = recent + [now]
            rid = secrets.token_urlsafe(12)
            claim = secrets.token_urlsafe(24)
            code = f"{secrets.randbelow(1_000_000):06d}"
            self._pending[rid] = {
                "id": rid, "name": name, "address": address, "user_agent": user_agent[:300],
                "code": code, "claim_hash": _hash(claim), "created": now, "status": "waiting",
                "token": None,
            }
            return {"id": rid, "code": code, "claim": claim, "expires_in": REQUEST_SECONDS}

    def pending(self) -> list[dict]:
        with self._lock:
            self._expire()
            return [
                {k: p[k] for k in ("id", "name", "address", "user_agent", "code", "created")}
                for p in self._pending.values() if p["status"] == "waiting"
            ]

    def answer(self, rid: str, approve: bool) -> bool:
        with self._lock:
            self._expire()
            req = self._pending.get(rid)
            if req is None or req["status"] != "waiting":
                return False
            if approve:
                req["token"] = self.add_device(req["name"], req["address"], req["user_agent"])
                req["status"] = "approved"
            else:
                req["status"] = "denied"
            return True

    def claim(self, rid: str, claim: str) -> tuple[str, Optional[str]]:
        """``(status, token)``: the key once, to the device that asked."""
        with self._lock:
            self._expire()
            req = self._pending.get(rid)
            if req is None or not secrets.compare_digest(req["claim_hash"], _hash(claim or "")):
                return "expired", None
            if req["status"] == "approved":
                del self._pending[rid]
                return "approved", req["token"]
            if req["status"] == "denied":
                del self._pending[rid]
                return "denied", None
            return "waiting", None


# ── The gate ─────────────────────────────────────────────────────────

def _trusted_login(request: Request, config) -> Optional[str]:
    """A Tailscale login listed in the config, when the request came through
    `tailscale serve` on this machine (the header is only believed on a
    loopback connection: a device on the network could send it to a server
    listening on 0.0.0.0)."""
    login = (request.headers.get("tailscale-user-login") or "").strip().lower()
    peer = request.client.host if request.client else ""
    if not login or not _is_loopback(peer):
        return None
    allowed = {str(x).strip().lower() for x in (getattr(config, "lan_trusted_logins", None) or [])}
    return login if login in allowed else None


def who(request: Request) -> dict:
    """What this caller is: ``{"local", "device", "login"}``."""
    state = request.app.state
    if is_local(request):
        return {"local": True, "device": None, "login": None}
    store: Optional[DeviceStore] = getattr(state, "devices", None)
    device = store.device_for(request.cookies.get(COOKIE), client_address(request)) if store else None
    login = _trusted_login(request, state.config)
    return {"local": False, "device": device, "login": login}


def owner_of(request: Request) -> tuple[str, str]:
    """Whose basket and exports a request belongs to, and what to call them:
    ``("local", "This computer")``, ``("device:<id>", name)`` or
    ``("login:<login>", login)``. The gate has already turned everyone else away."""
    caller = who(request)
    if caller["local"]:
        return "local", "This computer"
    if caller["device"] is not None:
        return f"device:{caller['device']['id']}", caller["device"]["name"] or "A paired device"
    if caller["login"]:
        return f"login:{caller['login']}", caller["login"]
    return "unknown", "Unknown"


def install(app) -> None:
    """Put the gate in front of every request."""

    @app.middleware("http")
    async def gate(request: Request, call_next):
        config = getattr(request.app.state, "config", None)
        if config is None:   # before startup: nothing to protect yet
            return await call_next(request)
        peer = request.client.host if request.client else ""
        if not _is_loopback(peer) and not getattr(config, "lan_access", False):
            return Response("Photool only answers on this computer (lan_access is off).",
                            status_code=403, media_type="text/plain")
        path = request.url.path
        if not (path.startswith("/api/") or path == "/mcp" or path.startswith("/mcp/")):
            return await call_next(request)     # the page itself holds no photos
        if not path.startswith("/api/"):
            # The MCP endpoint drives the whole library: this computer, or a
            # caller this gate has already let in, and nobody else.
            if is_local(request):
                return await call_next(request)
            caller = who(request)
            if caller["device"] is None and caller["login"] is None:
                return JSONResponse({"detail": "This device is not paired with Photool"}, status_code=401)
            return await call_next(request)
        if is_local(request):
            return await call_next(request)
        if path in PUBLIC_API:
            return await call_next(request)
        caller = who(request)
        if caller["device"] is None and caller["login"] is None:
            return JSONResponse({"detail": "This device is not paired with Photool"}, status_code=401)
        if path.startswith("/api/auth/"):
            return JSONResponse({"detail": "Only the computer that holds the library can manage devices"},
                                status_code=403)
        return await call_next(request)


# ── Endpoints ────────────────────────────────────────────────────────

class PairRequest(BaseModel):
    name: str = ""


class ClaimRequest(BaseModel):
    id: str
    claim: str


def _store(request: Request) -> DeviceStore:
    store = getattr(request.app.state, "devices", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Device pairing is not available")
    return store


@router.get("/auth/status")
async def auth_status(request: Request):
    caller = who(request)
    store = getattr(request.app.state, "devices", None)
    return {
        "local": caller["local"],
        "paired": caller["local"] or caller["device"] is not None or caller["login"] is not None,
        "device": (caller["device"] or {}).get("name") or caller["login"] or "",
        "lan_access": bool(getattr(request.app.state.config, "lan_access", False)),
        "pending": len(store.pending()) if (store and caller["local"]) else 0,
    }


@router.post("/auth/pair/request")
async def pair_request(req: PairRequest, request: Request):
    if is_local(request):
        raise HTTPException(status_code=400, detail="This computer does not need pairing")
    name = " ".join((req.name or "").split())[:60] or "Unnamed device"
    return _store(request).request(name, client_address(request),
                                   request.headers.get("user-agent", ""))


@router.post("/auth/pair/claim")
async def pair_claim(req: ClaimRequest, request: Request):
    status, token = _store(request).claim(req.id, req.claim)
    response = JSONResponse({"status": status})
    if token:
        response.set_cookie(
            COOKIE, token, max_age=COOKIE_DAYS * 86400, httponly=True, samesite="strict",
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
            path="/",
        )
    return response


# The endpoints below are reached only from this computer: the gate refuses
# /api/auth/* to everyone else except the three public ones above.

@router.get("/auth/pending")
async def pair_pending(request: Request):
    return {"requests": _store(request).pending()}


@router.post("/auth/pair/{request_id}/approve")
async def pair_approve(request_id: str, request: Request):
    if not _store(request).answer(request_id, True):
        raise HTTPException(status_code=404, detail="That request has expired")
    return {"ok": True}


@router.post("/auth/pair/{request_id}/deny")
async def pair_deny(request_id: str, request: Request):
    if not _store(request).answer(request_id, False):
        raise HTTPException(status_code=404, detail="That request has expired")
    return {"ok": True}


@router.get("/auth/devices")
async def devices(request: Request):
    return {"devices": _store(request).list_devices()}


@router.delete("/auth/devices/{device_id}")
async def revoke(device_id: int, request: Request):
    if not _store(request).revoke(device_id):
        raise HTTPException(status_code=404, detail="No such device")
    return {"ok": True}
