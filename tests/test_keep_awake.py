"""
The computer is held awake while the app is used or asked-for work runs, and let
go after. The clock and the Windows call are stood in for; middleware and the job
check are driven for real.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import background  # noqa: E402
import keep_awake  # noqa: E402


class Rig:
    def __init__(self, minutes=15):
        self.now = 1000.0
        self.busy = False
        self.calls = []
        self.awake = keep_awake.KeepAwake(minutes, lambda: self.busy, self.calls.append, lambda: self.now)


def test_nothing_is_held_until_someone_uses_the_app():
    rig = Rig()
    rig.awake.check()
    assert rig.calls == [] and not rig.awake.holding


def test_held_while_in_use_and_let_go_after_the_window():
    rig = Rig()
    rig.awake.touch()
    rig.awake.check()
    assert rig.calls == [True]
    rig.now += 14 * 60
    rig.awake.check()
    assert rig.calls == [True]
    rig.now += 2 * 60
    rig.awake.check()
    assert rig.calls == [True, False]


def test_use_extends_the_window():
    rig = Rig()
    rig.awake.touch()
    rig.awake.check()
    rig.now += 14 * 60
    rig.awake.touch()
    rig.now += 14 * 60
    rig.awake.check()
    assert rig.calls == [True]


def test_requested_work_holds_it_past_the_window():
    rig = Rig()
    rig.awake.touch()
    rig.awake.check()
    rig.busy = True
    rig.now += 60 * 60
    rig.awake.check()
    assert rig.calls == [True]
    rig.busy = False
    rig.awake.check()
    assert rig.calls == [True, False]


def test_a_failing_job_check_does_not_hold_it_or_stop_the_loop():
    rig = Rig()

    def broken():
        raise RuntimeError("no")

    rig.awake.busy = broken
    rig.awake.check()
    assert rig.calls == []


def test_a_page_left_open_is_not_use():
    assert not background.counts_as_use("GET", "/api/background")
    assert not background.counts_as_use("GET", "/api/activity")


def test_turned_off_starts_nothing():
    keep_awake.start(0, lambda: True)
    assert keep_awake._current is None
    keep_awake.touch()                      # and is harmless


def test_the_real_windows_call_can_be_made():
    import os

    if os.name != "nt":
        return
    keep_awake.hold_system_awake(True)
    keep_awake.hold_system_awake(False)


def test_requested_work_reads_each_kind_of_job():
    from api import routes
    from export import jobs as export_jobs

    assert routes.requested_work_running() is False

    export_jobs._progress["t"] = {"status": "running", "owner": None, "files": [], "started_at": 0}
    try:
        assert routes.requested_work_running() is True
    finally:
        del export_jobs._progress["t"]

    routes._import_jobs["t"] = {"status": "running", "started_at": 0}
    try:
        assert routes.requested_work_running() is True
    finally:
        del routes._import_jobs["t"]

    routes._face_job.update(status="running", automatic=True)
    try:
        assert routes.requested_work_running() is False     # automatic: lets it sleep
        routes._face_job["automatic"] = False
        assert routes.requested_work_running() is True
    finally:
        routes._face_job.update(status="idle", automatic=False)
