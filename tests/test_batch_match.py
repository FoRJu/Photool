"""
Matching many photos to one reference from the folder view: nothing is carried across but
what is asked for, a look at the result saves nothing, and the whole batch can be taken back.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from export import paste_match
from test_paste_match import DRIFTED, ELSEWHERE, FOLDER, HERO, RELIT, env  # noqa: F401  (a fixture)


def _match(client, **body):
    return client.post("/api/photos/edit/match", json=body)


def _finish(client, job_id):
    end = time.time() + 60
    while time.time() < end:
        state = client.get(f"/api/photos/edit/match/progress/{job_id}").json()
        if state["status"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def _edit(client, filepath):
    return client.get("/api/photos/edit", params={"filepath": filepath}).json()["settings"]


@pytest.fixture
def client(env):
    return env[0] if isinstance(env, tuple) else env


class TestWhatIsMatched:
    def test_a_look_at_the_result_saves_nothing(self, client):
        res = _match(client, filepaths=[DRIFTED], reference=HERO, kind="look", amount=75, dry_run=True)
        assert res.status_code == 200
        assert "curve_r" in res.json()["results"][DRIFTED]["settings"] or "mono" in res.json()["results"][DRIFTED]["settings"]
        assert _edit(client, DRIFTED) == {}

    def test_the_look_leaves_the_rest_of_a_frames_edit(self, client):
        client.put("/api/photos/edit", json={"filepath": DRIFTED, "exposure": 0.5, "vignette": -20})
        client.put("/api/photos/edit", json={"filepath": HERO, "exposure": 2.0, "clarity": 30})
        out = _finish(client, _match(client, filepaths=[DRIFTED], reference=HERO, kind="look", amount=75).json()["job_id"])
        assert out["status"] == "completed" and out["done"] == 1
        edit = _edit(client, DRIFTED)
        assert edit["exposure"] == 0.5 and edit["vignette"] == -20
        assert "clarity" not in edit                       # not a paste: the reference's edit stays its own

    def test_light_changes_only_the_light(self, client):
        client.put("/api/photos/edit", json={"filepath": DRIFTED, "vignette": -20, "curve_r": [[0, 0], [128, 140], [255, 255]]})
        out = _finish(client, _match(client, filepaths=[DRIFTED], reference=HERO, kind="light").json()["job_id"])
        assert out["status"] == "completed"
        edit = _edit(client, DRIFTED)
        assert edit["vignette"] == -20 and edit["curve_r"] == [[0, 0], [128, 140], [255, 255]]

    def test_strength_scales_the_look(self, client):
        def furthest(amount):
            res = _match(client, filepaths=[DRIFTED], reference=HERO, kind="look", amount=amount, dry_run=True).json()
            s = res["results"][DRIFTED]["settings"]
            return max((abs(y - x) for k in ("curve_r", "curve_g", "curve_b", "curve") for x, y in s.get(k, [])), default=0)

        client.put("/api/photos/edit", json={"filepath": HERO, "exposure": 1.5})
        full, half, none = furthest(100), furthest(50), furthest(0)
        assert full > 0 and none == 0
        assert half == pytest.approx(full / 2, abs=2)

    def test_the_reference_is_never_matched_to_itself(self, client):
        out = _finish(client, _match(client, filepaths=[HERO, DRIFTED], reference=HERO, kind="look").json()["job_id"])
        assert out["total"] == 1 and HERO not in out["edits"]

    def test_another_shoot_is_allowed_and_said_so(self, client):
        res = _match(client, filepaths=[ELSEWHERE], reference=HERO, kind="look", dry_run=True).json()
        assert "another shoot" in res["results"][ELSEWHERE]["note"]

    def test_the_note_speaks_of_the_reference_not_a_copy(self, client):
        res = _match(client, filepaths=[DRIFTED], reference=HERO, kind="look", dry_run=True).json()
        note = res["results"][DRIFTED]["note"]
        assert "the reference" in note and "copied" not in note


class TestTheBatch:
    def test_the_whole_batch_can_be_taken_back(self, client):
        client.put("/api/photos/edit", json={"filepath": DRIFTED, "exposure": 0.5})
        out = _finish(client, _match(client, filepaths=[DRIFTED, RELIT], reference=HERO, kind="look").json()["job_id"])
        assert set(out["previous"]) == {DRIFTED, RELIT}
        last = client.get("/api/photos/edit/match/last").json()["last"]
        assert last["photos"] == 2 and last["reference"] == HERO
        done = client.post("/api/photos/edit/match/undo").json()
        assert set(done["restored"]) == {DRIFTED, RELIT} and done["kept"] == []
        assert _edit(client, DRIFTED) == {"exposure": 0.5} and _edit(client, RELIT) == {}
        assert client.get("/api/photos/edit/match/last").json()["last"] is None
        assert client.post("/api/photos/edit/match/undo").status_code == 404

    def test_undo_never_wipes_work_done_since(self, client):
        _finish(client, _match(client, filepaths=[DRIFTED, RELIT], reference=HERO, kind="look").json()["job_id"])
        later = {**_edit(client, RELIT), "vignette": -30}
        client.put("/api/photos/edit", json={"filepath": RELIT, **later})
        kept = _edit(client, RELIT)
        done = client.post("/api/photos/edit/match/undo").json()
        assert list(done["restored"]) == [DRIFTED] and done["kept"] == [RELIT]
        assert _edit(client, DRIFTED) == {} and _edit(client, RELIT) == kept

    def test_undo_outlives_the_server(self, client):
        _finish(client, _match(client, filepaths=[DRIFTED], reference=HERO, kind="look").json()["job_id"])
        from api import routes

        routes._match_jobs.clear()                         # what a restart forgets
        assert client.get("/api/photos/edit/match/last").json()["last"]["photos"] == 1
        assert list(client.post("/api/photos/edit/match/undo").json()["restored"]) == [DRIFTED]

    def test_a_new_match_replaces_the_last_and_forgetting_keeps_the_edits(self, client):
        _finish(client, _match(client, filepaths=[DRIFTED], reference=HERO, kind="look").json()["job_id"])
        _finish(client, _match(client, filepaths=[RELIT], reference=HERO, kind="look").json()["job_id"])
        assert client.get("/api/photos/edit/match/last").json()["last"]["photos"] == 1
        matched = _edit(client, RELIT)
        assert client.delete("/api/photos/edit/match/last").status_code == 200
        assert client.get("/api/photos/edit/match/last").json()["last"] is None
        assert _edit(client, RELIT) == matched

    def test_a_match_that_changed_nothing_leaves_the_last_alone(self, client, monkeypatch):
        _finish(client, _match(client, filepaths=[DRIFTED], reference=HERO, kind="look").json()["job_id"])
        real = paste_match.matcher

        def broken(*a, **k):
            real(*a, **k)

            def stepped(path, own):
                raise RuntimeError("cannot read")
            return stepped

        monkeypatch.setattr(paste_match, "matcher", broken)
        _finish(client, _match(client, filepaths=[RELIT], reference=HERO, kind="look").json()["job_id"])
        assert list(client.post("/api/photos/edit/match/undo").json()["restored"]) == [DRIFTED]

    def test_stopping_keeps_what_was_done_and_can_be_undone(self, client, monkeypatch):
        real = paste_match.matcher
        job = {}

        def slow(*a, **k):
            step = real(*a, **k)

            def stepped(path, own):
                out = step(path, own)
                client.post(f"/api/photos/edit/match/cancel/{job['id']}")
                return out
            return stepped

        monkeypatch.setattr(paste_match, "matcher", slow)
        res = _match(client, filepaths=[DRIFTED, RELIT], reference=HERO, kind="look").json()
        job["id"] = res["job_id"]
        out = _finish(client, res["job_id"])
        assert out["status"] == "stopped" and out["done"] <= 2 and len(out["previous"]) == out["done"]

    def test_one_frame_that_fails_does_not_stop_the_rest(self, client, monkeypatch):
        real = paste_match.matcher

        def flaky(*a, **k):
            step = real(*a, **k)

            def stepped(path, own):
                if path == DRIFTED:
                    raise RuntimeError("cannot read")
                return step(path, own)
            return stepped

        monkeypatch.setattr(paste_match, "matcher", flaky)
        out = _finish(client, _match(client, filepaths=[DRIFTED, RELIT], reference=HERO, kind="look").json()["job_id"])
        assert out["status"] == "completed" and out["failed"] == [DRIFTED] and RELIT in out["edits"]

    def test_requested_work_keeps_the_computer_awake_while_it_runs(self, client):
        from api import routes

        routes._match_jobs["t"] = {"status": "running"}
        try:
            assert routes.requested_work_running()
        finally:
            del routes._match_jobs["t"]


class TestExamples:
    STATS = {f"p/{n}": {"lum_p50": lum, "wb_r": r, "wb_b": b} for n, lum, r, b in (
        ("a", 0.5, 1.0, 1.0), ("b", 0.1, 1.0, 1.0), ("c", 0.9, 1.0, 1.0), ("d", 0.55, 1.0, 1.0),
        ("e", 0.45, 3.0, 0.5), ("f", 0.52, 1.0, 1.0))}

    def _why(self, targets, stats, reference=None, count=4):
        return {e["why"]: e["filepath"] for e in paste_match.choose_examples(targets, stats, reference, count)}

    def test_the_likeliest_to_go_wrong_and_a_typical_one(self):
        why = self._why(list(self.STATS), self.STATS, {"wb_r": 1.0, "wb_b": 1.0})
        assert why["darkest"] == "p/b" and why["brightest"] == "p/c"
        assert why["colour furthest away"] == "p/e" and why["typical"] in self.STATS

    def test_never_more_than_asked_or_twice_the_same(self):
        picked = paste_match.choose_examples(list(self.STATS), self.STATS, None, 4)
        files = [e["filepath"] for e in picked]
        assert len(files) == 4 == len(set(files))

    def test_nothing_measured_is_spread_through_the_selection(self):
        targets = [f"p/{i}" for i in range(9)]
        picked = [e["filepath"] for e in paste_match.choose_examples(targets, {}, None, 4)]
        assert picked[0] == "p/0" and picked[-1] == "p/8" and len(set(picked)) == 4

    def test_fewer_photos_than_examples_shows_them_all(self):
        targets = ["p/a", "p/b"]
        assert {e["filepath"] for e in paste_match.choose_examples(targets, self.STATS, None, 4)} == set(targets)

    def test_through_the_route(self, client):
        res = client.post("/api/photos/edit/match/examples", json={"filepaths": [DRIFTED, RELIT], "reference": HERO})
        assert res.status_code == 200
        assert {e["filepath"] for e in res.json()["examples"]} == {DRIFTED, RELIT}


class TestRefused:
    def test_unknown_kind(self, client):
        assert _match(client, filepaths=[DRIFTED], reference=HERO, kind="vibes").status_code == 400

    def test_no_reference_in_the_library(self, client):
        assert _match(client, filepaths=[DRIFTED], reference=f"{FOLDER}/nope.JPG").status_code == 404

    def test_outside_the_library(self, client):
        assert _match(client, filepaths=[DRIFTED], reference="../../secret.jpg").status_code >= 400

    def test_nothing_to_match(self, client):
        assert _match(client, filepaths=[HERO], reference=HERO).status_code == 400

    def test_a_photo_not_in_the_catalog_is_left_out(self, client):
        res = _match(client, filepaths=[DRIFTED, f"{FOLDER}/nope.JPG"], reference=HERO, kind="look", dry_run=True).json()
        assert list(res["results"]) == [DRIFTED]
