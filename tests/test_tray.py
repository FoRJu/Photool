"""
The tray icon's menu: Quit stops the server and lets its port go, Restart starts a
new server only once the old one has stopped, and the icon is drawn without a file.
The menu's actions are driven directly, with a real uvicorn server on a spare port;
the icon itself is not shown.
"""

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import serve  # noqa: E402
import tray  # noqa: E402


async def _app(scope, receive, send):
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serving(port, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if serve.already_running(port):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def running():
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(_app, host="127.0.0.1", port=port, log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _serving(port)

    def stop():
        server.should_exit = True
        thread.join(tray.STOP_WAIT)
        return not thread.is_alive()

    yield port, stop
    server.should_exit = True


def test_quit_stops_the_server_and_frees_its_port(running):
    port, stop = running
    gone = []
    controls = tray.Controls(f"http://localhost:{port}", None, stop, lambda: None, lambda: gone.append(1))
    controls.quit()
    assert gone == [1] and controls.quitting
    assert not serve.already_running(port)


def test_restart_starts_the_new_server_only_once_the_old_one_has_gone(running):
    port, stop = running
    seen = []

    def spawn():
        seen.append(serve.already_running(port))     # the port, when the new one starts

    controls = tray.Controls(f"http://localhost:{port}", None, stop, spawn, lambda: seen.append("icon gone"))
    controls.restart()
    assert seen == [False, "icon gone"]
    controls.restart()                                # a second click does nothing more
    assert seen == [False, "icon gone"]


def test_a_server_that_will_not_stop_is_not_started_twice():
    started = []
    controls = tray.Controls("http://localhost:1", None, lambda: False, lambda: started.append(1), lambda: None)
    controls.restart()
    assert started == []


class _Item:
    def __init__(self, text, action=None, **_):
        self.text, self.action = text, action


class _Icon:
    """Stands in for pystray.Icon: runs ``setup`` and waits for stop()."""

    def __init__(self, *_):
        self.stopped = threading.Event()
        _Icon.last = self

    def run(self, setup):
        threading.Thread(target=setup, args=(self,), daemon=True).start()
        self.stopped.wait(30)

    def stop(self):
        self.stopped.set()


@pytest.fixture
def fake_pystray(monkeypatch):
    menu = type("Menu", (), {"SEPARATOR": None, "__init__": lambda self, *items: setattr(self, "items", items)})
    module = type(sys)("pystray")
    module.Icon, module.MenuItem, module.Menu = _Icon, _Item, menu
    monkeypatch.setitem(sys.modules, "pystray", module)


def _run_and_click(watched, label=None, spawned=None):
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(_app, host="127.0.0.1", port=port, log_config=None))
    if spawned is not None:
        tray.subprocess.Popen, real = (lambda *a, **k: spawned.append(a)), tray.subprocess.Popen
    ended = []
    runner = threading.Thread(target=lambda: ended.append(tray.run(server, "http://x", [], None, [], watched=watched)))
    runner.start()
    assert _serving(port)
    time.sleep(0.2)
    if label:
        item = next(i for i in _Icon.last.menu.items if getattr(i, "text", "") == label)
        item.action()
    else:
        server.should_exit = True           # stops by itself, nobody asked
    runner.join(30)
    if spawned is not None:
        tray.subprocess.Popen = real
    return ended[0]


def test_run_says_how_it_ended_for_the_watcher(fake_pystray):
    assert _run_and_click(True, "Quit") == "quit"
    assert _run_and_click(True, "Restart") == "restart"
    assert _run_and_click(True) == "stopped"


def test_a_watched_restart_leaves_starting_to_the_watcher(fake_pystray):
    spawned = []
    _run_and_click(True, "Restart", spawned)
    assert spawned == []
    _run_and_click(False, "Restart", spawned)
    assert len(spawned) == 1


class _Stuck:
    """A server that never stops, however it is asked."""

    should_exit = False

    def run(self):
        threading.Event().wait()


def test_a_stuck_server_is_ended_so_the_watcher_can_start_a_new_one(fake_pystray, monkeypatch):
    monkeypatch.setattr(tray, "STOP_WAIT", 0.1)
    monkeypatch.setattr(tray, "FORCE_WAIT", 0.1)
    # The icon never goes either: the process has to end regardless.
    monkeypatch.setattr(_Icon, "stop", lambda self: None)
    for label, how in (("Restart", "restart"), ("Quit", "quit")):
        forced = threading.Event()
        said = []
        _Icon.last = None

        def force_exit(what):
            said.append(what)
            forced.set()

        threading.Thread(target=tray.run, args=(_Stuck(), "http://x", [], None, []),
                         kwargs={"watched": True, "force_exit": force_exit}, daemon=True).start()
        end = time.time() + 5
        while time.time() < end and not getattr(getattr(_Icon, "last", None), "menu", None):
            time.sleep(0.01)
        item = next(i for i in _Icon.last.menu.items if getattr(i, "text", "") == label)
        item.action()
        assert forced.wait(5)
        assert said == [how]


def _stops_with_a_stream_open(**options) -> float:
    """How long a real server takes to stop while a response is still being sent."""
    import asyncio

    import uvicorn

    async def streams(scope, receive, send):
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"start", "more_body": True})
        await asyncio.Event().wait()

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(streams, host="127.0.0.1", port=port, log_config=None, **options))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _serving(port)
    with socket.create_connection(("127.0.0.1", port)) as client:
        client.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        assert client.recv(1024)
        begun = time.time()
        server.should_exit = True
        thread.join(4)
        took = time.time() - begun
        stuck = thread.is_alive()
        server.force_exit = True
        thread.join(5)
    return float("inf") if stuck else took


def test_an_open_stream_cannot_hold_a_stopping_server():
    assert _stops_with_a_stream_open() == float("inf")     # what happens with no limit
    assert _stops_with_a_stream_open(timeout_graceful_shutdown=1) < 4


def test_the_servers_limit_leaves_the_tray_time():
    assert serve.GRACEFUL_SECONDS < tray.STOP_WAIT


def test_the_icon_is_drawn():
    img = tray.icon_image(64)
    assert img.size == (64, 64) and img.getpixel((32, 4))[3] == 255 and img.getpixel((0, 0))[3] == 0


def test_a_restart_runs_the_venvs_own_python():
    exe = Path(tray.python_for_restart())
    assert exe.is_file()
    assert Path(sys.prefix) in exe.parents or exe == Path(sys.executable)
