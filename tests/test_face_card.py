"""
ArcFace on the graphics card (faces.detect): the card when it can, the CPU
otherwise, and never at the cost of a large model holding the card.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

cv2 = pytest.importorskip("cv2")

from adjust import gpu
from faces import detect


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(detect, "_card_session", None)
    monkeypatch.setattr(detect, "_card_path", None)
    monkeypatch.setattr(detect, "_card_failed", False)
    monkeypatch.setattr(gpu, "_large_open", False)
    yield


class FakeSession:
    runs = 0

    def __init__(self, path, sess_options=None, providers=None):
        self.providers = providers

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [type("I", (), {"name": "input.1"})()]

    def run(self, _, feed):
        FakeSession.runs += 1
        x = feed["input.1"]
        assert x.shape[0] == 1          # one crop at a time: the file's batch is fixed at one
        return [np.full((1, 512), float(x.mean()), np.float32)]


class FakeOrt:
    InferenceSession = FakeSession

    class SessionOptions:
        log_severity_level = 0

    @staticmethod
    def get_available_providers():
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


def blob(n=2):
    return np.random.rand(n, 3, 112, 112).astype(np.float32)


def test_runs_on_the_card_one_crop_at_a_time(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    FakeSession.runs = 0
    out = detect._card_describe("model.onnx", blob(2))
    assert out.shape == (2, 512) and FakeSession.runs == 2
    # the session is kept for the next face
    detect._card_describe("model.onnx", blob(2))
    assert FakeSession.runs == 4


def test_a_large_model_on_the_card_is_left_alone(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    released = []
    monkeypatch.setattr(gpu, "release_all", lambda: released.append(1))
    gpu._large_open = True
    assert detect._card_describe("model.onnx", blob()) is None
    assert not released and detect._card_session is None


def test_a_failing_card_falls_back_to_the_cpu_for_good(monkeypatch):
    class Broken(FakeOrt):
        @staticmethod
        def InferenceSession(*a, **k):
            raise RuntimeError("The parameter is incorrect.")

    monkeypatch.setitem(sys.modules, "onnxruntime", Broken)
    assert detect._card_describe("model.onnx", blob()) is None
    assert detect._card_failed
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    assert detect._card_describe("model.onnx", blob()) is None   # not tried again


def test_a_session_that_fails_mid_run_is_let_go_of_with_the_others(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    detect._card_describe("model.onnx", blob())
    assert detect._card_session is not None
    released = []
    monkeypatch.setattr(gpu, "release_all", lambda: released.append(1) or detect._forget_card_session())

    def broken_run(*a):
        raise RuntimeError("The GPU device instance has been suspended")

    monkeypatch.setattr(detect._card_session, "run", broken_run)
    assert detect._card_describe("model.onnx", blob()) is None
    assert released == [1] and detect._card_session is None


def test_no_card_means_the_cpu(monkeypatch):
    class CpuOnly(FakeOrt):
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", CpuOnly)
    assert detect._card_describe("model.onnx", blob()) is None


def test_describe_loads_opencv_only_when_the_card_declines(monkeypatch):
    loaded = []

    class Net:
        def setInput(self, b):
            self.n = len(b)

        def forward(self):
            return np.ones((self.n, 512), np.float32)

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda p: loaded.append(p) or Net())
    face = np.zeros((112, 112, 3), np.uint8)
    arc = detect._ArcFace("model.onnx")

    monkeypatch.setattr(detect, "_card_describe", lambda p, b: np.ones((len(b), 512), np.float32))
    v = arc.describe(face)
    assert not loaded and v.shape == (512,)

    monkeypatch.setattr(detect, "_card_describe", lambda p, b: None)
    v = arc.describe(face)
    assert loaded == ["model.onnx"] and v.shape == (512,)
    arc.describe(face)
    assert len(loaded) == 1


def test_workers_follow_the_card(monkeypatch):
    from faces import service

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    monkeypatch.setattr(detect, "_recognizer", detect.DEFAULT_RECOGNIZER)
    assert detect.card_available()
    assert service._face_workers() == service.FACE_WORKERS_WITH_CARD
    detect._card_failed = True
    assert not detect.card_available()
    assert service._face_workers() == service.FACE_WORKERS
    # SFace stays with OpenCV
    detect._card_failed = False
    monkeypatch.setattr(detect, "_recognizer", "sface")
    assert not detect.card_available()


def test_the_card_gives_opencvs_vectors():
    """On a machine with the models and a card: the same vectors either way."""
    ort = pytest.importorskip("onnxruntime")
    path = Path(__file__).parent.parent / "models" / "w600k_r50.onnx"
    if not path.exists() or "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("no ArcFace model or no card here")
    rng = np.random.default_rng(1)
    face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    on_card = detect._ArcFace(path).describe(face)
    detect._card_failed = True
    on_cpu = detect._ArcFace(path).describe(face)
    cos = float(on_card @ on_cpu / np.linalg.norm(on_card) / np.linalg.norm(on_cpu))
    assert cos > 0.9999
