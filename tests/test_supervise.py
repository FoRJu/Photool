"""
The watcher starts a dead server again, leaves a clean exit alone, and waits longer
when it keeps dying. Then the real thing: a child that crashes natively is started
again, numpy gets one OpenBLAS thread, and a crash leaves Python's stacks behind.
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import serve  # noqa: E402

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
ACCESS_VIOLATION = 0xC0000005


class Rig:
    """A worker that exits with each code in turn, each after ``ran`` seconds."""

    def __init__(self, codes, ran=1.0):
        self.codes, self.ran, self.now = list(codes), ran, 0.0
        self.notes, self.waits = [], []

    def start(self):
        self.now += self.ran
        return self.codes.pop(0)

    def run(self):
        return serve.supervise(self.start, self.notes.append, self.waits.append, lambda: self.now)


def test_a_clean_exit_is_left_alone():
    rig = Rig([0])
    assert rig.run() == 0
    assert rig.notes == [] and rig.waits == []


def test_a_crash_is_started_again_and_said_so():
    rig = Rig([ACCESS_VIOLATION, 0])
    assert rig.run() == 0
    assert rig.waits == [serve.FIRST_DELAY]
    assert "0xc0000005" in rig.notes[0]


def test_a_restart_from_the_tray_does_not_wait_or_count_as_a_crash():
    rig = Rig([serve.RESTART_EXIT, serve.RESTART_EXIT, 0])
    assert rig.run() == 0
    assert rig.waits == [] and rig.notes == []


def test_crash_after_crash_waits_longer_up_to_a_limit_and_never_gives_up():
    rig = Rig([1] * 12 + [0])
    assert rig.run() == 0
    assert rig.waits[:4] == [2, 4, 8, 16]
    assert rig.waits[-1] == serve.LONGEST_DELAY and len(rig.waits) == 12


def test_a_server_that_settled_starts_counting_again():
    rig = Rig([1, 1, 0], ran=serve.SETTLED_SECONDS + 1)
    rig.run()
    assert rig.waits == [serve.FIRST_DELAY, serve.FIRST_DELAY]


def test_blas_threads_comes_before_anything_that_could_load_numpy():
    for name in ("serve.py", "main.py"):
        tree = ast.parse((BACKEND / name).read_text(encoding="utf-8"))
        first = next(n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        assert isinstance(first, ast.Import) and first.names[0].name == "blas_threads", name


def _clean_env():
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)
    return env


def test_numpy_really_gets_one_blas_thread_through_serve():
    code = textwrap.dedent(f"""
        import sys, glob, os, ctypes
        sys.path.insert(0, {str(BACKEND)!r})
        import serve
        import numpy
        libs = glob.glob(os.path.join(os.path.dirname(numpy.__file__), '..', 'numpy.libs', '*openblas*'))
        lib = ctypes.CDLL(libs[0])
        print(lib.scipy_openblas_get_num_threads64_())
    """)
    out = subprocess.run([sys.executable, "-c", code], env=_clean_env(), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


def test_a_setting_of_ones_own_is_kept():
    code = (f"import sys; sys.path.insert(0, {str(BACKEND)!r}); import blas_threads, os; "
            "print(os.environ['OPENBLAS_NUM_THREADS'])")
    out = subprocess.run([sys.executable, "-c", code], env={**_clean_env(), "OPENBLAS_NUM_THREADS": "4"},
                         capture_output=True, text=True)
    assert out.stdout.strip() == "4"


def test_a_real_native_crash_is_recorded_and_seen_by_the_watcher(tmp_path):
    log = tmp_path / "serve.log"
    code = textwrap.dedent(f"""
        import sys, ctypes
        from pathlib import Path
        sys.path.insert(0, {str(BACKEND)!r})
        import serve
        serve._record_crashes(Path({str(log)!r}))
        def about_to_crash():
            ctypes.string_at(0)
        about_to_crash()
    """)
    runs = []

    def start():
        runs.append(1)
        if len(runs) == 1:
            return subprocess.call([sys.executable, "-c", code], cwd=tmp_path,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return 0

    notes = []
    assert serve.supervise(start, notes.append, lambda _: None) == 0
    assert len(runs) == 2 and len(notes) == 1
    crash = (tmp_path / "serve-crash.log").read_text(encoding="utf-8")
    assert "about_to_crash" in crash


def test_a_large_crash_log_starts_over(tmp_path):
    log = tmp_path / "serve.log"
    old = tmp_path / "serve-crash.log"
    old.write_text("x" * (serve.CRASH_LOG_BYTES + 1))
    code = (f"import sys; sys.path.insert(0, {str(BACKEND)!r}); import serve; from pathlib import Path; "
            f"serve._record_crashes(Path({str(log)!r}))")
    subprocess.run([sys.executable, "-c", code], check=True)
    assert old.stat().st_size == 0
