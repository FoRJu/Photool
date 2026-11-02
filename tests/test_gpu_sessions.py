"""
The graphics card's models are let go of together, and one call runs at a time. Letting go
of one DirectML session breaks the others already open: the next call is an access violation
that takes the whole process down, so the real thing is tried in a process of its own.
"""

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust import gpu, noise, objects, parts

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
MODELS = ROOT / "models"


def _has_directml() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()


real_card = pytest.mark.skipif(
    not _has_directml() or not (MODELS / noise.MODEL.name).exists()
    or not all((MODELS / spec.name).exists() for spec in objects.MODELS[objects.DEFAULT_MODEL]["files"]),
    reason="needs DirectML and the noise and SAM models")


class TestReleasedTogether:
    def test_a_release_drops_every_models_sessions(self):
        noise._session = "a session"
        parts._sessions["face"] = "a session"
        objects._sessions[("sam", "encoder")] = "a session"
        objects.release()
        assert noise._session is None and not parts._sessions and not objects._sessions

    def test_nothing_of_the_masks_to_let_go_of_leaves_the_rest_alone(self):
        objects._sessions.clear()
        noise._session = "a session"
        try:
            objects.release()
            assert noise._session == "a session"
        finally:
            noise._session = None

    def test_changing_the_masks_model_drops_the_others_too(self, monkeypatch):
        import types

        real = sys.modules.get("onnxruntime")
        fake = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            SessionOptions=lambda: types.SimpleNamespace(),
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=1),
            InferenceSession=lambda *a, **k: types.SimpleNamespace(get_providers=lambda: ["CPU"]))
        monkeypatch.setitem(sys.modules, "onnxruntime", fake)
        monkeypatch.setattr(objects, "ensure", lambda model: Path("."))
        models = list(objects.MODELS)
        if len(models) < 2:
            pytest.skip("one mask model")
        objects._sessions.clear()
        noise._session = "a session"
        objects._sessions[(models[1], "encoder")] = "the other model"
        objects._sessions_of(models[0], ("encoder", "decoder"))
        assert noise._session is None
        assert all(k[0] == models[0] for k in objects._sessions)
        objects._sessions.clear()
        if objects._idle is not None:
            objects._idle.cancel()
        assert real is None or real is not fake


class TestOneCallAtATime:
    def test_the_models_share_the_cards_lock(self):
        assert objects._lock is gpu.LOCK and objects._run_lock is gpu.LOCK

    def test_a_release_waits_for_a_call_under_way(self):
        order = []
        noise._session = "a session"
        started = threading.Event()

        def call():
            with gpu.LOCK:
                started.set()
                time.sleep(0.2)
                order.append("call done")

        def release():
            started.wait(5)
            gpu.release_all()
            order.append("released")

        threads = [threading.Thread(target=call), threading.Thread(target=release)]
        [t.start() for t in threads]
        [t.join(5) for t in threads]
        assert order == ["call done", "released"]

    def test_a_denoise_frame_gives_way_between_batches(self, monkeypatch):
        """The card is taken for each batch, so a mask's click waits for a batch, not a frame."""
        import numpy as np

        held = []

        class Session:
            def run(self, names, feeds):
                held.append(gpu.LOCK._is_owned())
                n, c, h, w = feeds["input"].shape
                return [np.zeros((n, c, h, w), np.float16)]

        monkeypatch.setattr(noise, "_get_session", lambda: Session())
        rgb = np.zeros((1100, 1100, 3), np.float32)
        noise._run_model(rgb, 1.0)
        assert all(held) and len(held) >= 2
        assert not gpu.LOCK._is_owned()


def _run(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import faulthandler, gc, sys
        import numpy as np
        sys.path.insert(0, {str(BACKEND)!r})
        faulthandler.enable()
        from adjust import noise, objects
        noise.configure({str(MODELS)!r}, None)
        objects.configure({str(MODELS)!r}, None)
        batch = np.zeros((4, 3, 512, 512), np.float16)
        cond = np.full((4, 1), 1.0, np.float16)
        def denoise():
            from adjust import gpu
            for _ in range(10):
                with gpu.LOCK:
                    noise._get_session().run(["output"], {{"input": batch, "cond": cond}})
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)


@real_card
class TestOnTheRealCard:
    def test_denoising_after_a_mask_lets_go_does_not_crash(self):
        done = _run("""
            denoise()
            objects._session_pair(objects.DEFAULT_MODEL)
            objects.release()
            denoise()
            print("survived")
        """)
        assert done.returncode == 0, done.stderr[-800:]
        assert "survived" in done.stdout
