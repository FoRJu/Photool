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


def test_the_icon_is_drawn():
    img = tray.icon_image(64)
    assert img.size == (64, 64) and img.getpixel((32, 4))[3] == 255 and img.getpixel((0, 0))[3] == 0


def test_a_restart_runs_the_venvs_own_python():
    exe = Path(tray.python_for_restart())
    assert exe.is_file()
    assert Path(sys.prefix) in exe.parents or exe == Path(sys.executable)
