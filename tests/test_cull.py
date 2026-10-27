"""
Culling with an agent: signals, cheap pictures, and proposals that wait for the
person. Through ``/api/commands``, the door the MCP tools share, against a real
(temporary) library.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

FOLDER = "2025/2025-03-06"
NAMES = [f"IMG_000{i}.JPG" for i in range(1, 5)]
PHOTOS = [f"{FOLDER}/{n}" for n in NAMES]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    (library / FOLDER).mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i, name in enumerate(NAMES):
        rgb = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        rgb[: 40 * (i + 1)] = 200
        # A second apart, so the frames are each other's peers (analysis.hints.peers).
        exif = Image.Exif()
        exif.get_ifd(0x8769)[36867] = f"2025:03:06 10:00:0{i}"
        Image.fromarray(rgb, "RGB").save(library / FOLDER / name, format="JPEG", quality=90,
                                         exif=exif.tobytes())

    (tmp_path / "config.yaml").write_text(
        f"library_root: '{library}'\n"
        f"db_path: '{tmp_path / 'data' / 'catalog.db'}'\n"
        f"thumbnail_cache: '{tmp_path / 'data' / 'thumbs'}'\n"
        "sd_mount_paths: []\n"
        "analysis_auto_enabled: false\n"
    )
    monkeypatch.chdir(tmp_path)

    import main

    with TestClient(main.app, base_url="http://localhost:8000") as c:
        assert c.post(f"/api/catalog/index/{FOLDER}").status_code == 200
        yield c


def call(client, command_id, **arguments):
    res = client.post(f"/api/commands/{command_id}", json=arguments)
    assert res.status_code == 200, res.text
    return res.json()


def marks(client, filepath):
    p = call(client, "library.photo", filepath=filepath)
    return p["rating"], p["flag"]


class TestSignals:
    def test_every_frame_is_listed_and_unmeasured_ones_say_so(self, client):
        out = call(client, "cull.signals", folder=FOLDER)
        assert out["total"] == 4
        assert {p["filepath"] for p in out["photos"]} == set(PHOTOS)
        assert all(p["measured"] is False and "sharpness_rank" not in p for p in out["photos"])

    def test_measuring_fills_them_in(self, client):
        call(client, "cull.measure", folder=FOLDER, faces=False)
        out = call(client, "cull.signals", folder=FOLDER)
        assert out["measured"] == 4
        ranks = sorted(p["sharpness_rank"] for p in out["photos"])
        assert ranks[0] == 0.0 and ranks[-1] == 1.0
        assert all("measured" not in p and p["brightness"] > 0 for p in out["photos"])

    def test_blind_leaves_out_the_persons_marks(self, client):
        call(client, "photo.rate", filepaths=[PHOTOS[0]], rating=4)
        call(client, "cull.propose", proposals=[{"filepath": PHOTOS[0], "flag": 1, "reason": "x"}])
        seen = {p["filepath"]: p for p in call(client, "cull.signals", folder=FOLDER)["photos"]}
        assert seen[PHOTOS[0]]["rating"] == 4 and seen[PHOTOS[0]]["proposal"]["flag"] == 1
        blind = {p["filepath"]: p for p in call(client, "cull.signals", folder=FOLDER, blind=True)["photos"]}
        assert "rating" not in blind[PHOTOS[0]] and "proposal" not in blind[PHOTOS[0]]

    def test_scenes_split_on_a_gap(self, client):
        out = call(client, "cull.signals", folder=FOLDER, scene_gap_seconds=5)
        assert out["scenes"] and sum(s["frames"] for s in out["scenes"]) == 4
        assert [p["scene"] for p in out["photos"]] == sorted(p["scene"] for p in out["photos"])


class TestPictures:
    def test_a_thumbnail_comes_from_the_cache(self, client):
        out = call(client, "render.thumbnail", filepath=PHOTOS[0], size=256)
        img = Image.open(io.BytesIO(base64.b64decode(out["jpeg_base64"])))
        assert max(img.size) == 256

    def test_a_contact_sheet_numbers_its_frames(self, client):
        out = call(client, "render.contact_sheet", filepaths=PHOTOS, cell=200)
        assert [c["n"] for c in out["cells"]] == [1, 2, 3, 4]
        assert [c["filepath"] for c in out["cells"]] == PHOTOS
        img = Image.open(io.BytesIO(base64.b64decode(out["jpeg_base64"])))
        assert img.size == (out["width"], out["height"])
        assert img.width < 3 * 200  # two columns for four frames

    def test_a_sheet_refuses_a_photo_not_in_the_catalog(self, client):
        res = client.post("/api/commands/render.contact_sheet",
                          json={"filepaths": [PHOTOS[0], f"{FOLDER}/nope.JPG"]})
        assert res.status_code == 400


class TestProposals:
    def test_a_proposal_changes_nothing_until_accepted(self, client):
        out = call(client, "cull.propose", proposals=[
            {"filepath": PHOTOS[0], "rating": 4, "flag": 1, "reason": "sharpest"},
            {"filepath": PHOTOS[1], "flag": -1, "reason": "eyes closed"},
        ], agent="test")
        assert out["proposed"] == 2 and out["already_decided"] == 0
        assert marks(client, PHOTOS[0]) == (0, 0)
        signals = {p["filepath"]: p for p in call(client, "cull.signals", folder=FOLDER)["photos"]}
        assert signals[PHOTOS[0]]["proposal"]["reason"] == "sharpest"
        assert signals[PHOTOS[0]]["proposal"]["status"] == "pending"

        got = call(client, "cull.accept", run=out["run"])
        assert got == {"accepted": 2, "changed": 3, "kept": 0}
        assert marks(client, PHOTOS[0]) == (4, 1)
        assert marks(client, PHOTOS[1]) == (0, -1)

    def test_a_proposal_says_something(self, client):
        res = client.post("/api/commands/cull.propose",
                          json={"proposals": [{"filepath": PHOTOS[0], "reason": "nothing"}]})
        assert res.status_code == 400

    def test_accepting_keeps_what_the_person_set(self, client):
        call(client, "photo.rate", filepaths=[PHOTOS[0]], rating=2)
        out = call(client, "cull.propose", proposals=[
            {"filepath": PHOTOS[0], "rating": 5, "flag": 1, "reason": "best"}])
        assert out["already_decided"] == 1
        got = call(client, "cull.accept", folder=FOLDER)
        assert got["kept"] == 1 and got["changed"] == 1
        assert marks(client, PHOTOS[0]) == (2, 1)

    def test_overwrite_replaces_it_and_undo_puts_it_back(self, client):
        call(client, "photo.rate", filepaths=[PHOTOS[0]], rating=2)
        call(client, "cull.propose", proposals=[{"filepath": PHOTOS[0], "rating": 5, "reason": "best"}])
        call(client, "cull.accept", filepaths=[PHOTOS[0]], overwrite=True)
        assert marks(client, PHOTOS[0]) == (5, 0)
        assert call(client, "cull.undo", folder=FOLDER)["restored"] == 1
        assert marks(client, PHOTOS[0]) == (2, 0)
        assert call(client, "cull.proposals", folder=FOLDER)["counts"]["pending"] == 1

    def test_undo_leaves_a_mark_the_person_changed_since(self, client):
        call(client, "cull.propose", proposals=[{"filepath": PHOTOS[0], "rating": 5, "reason": "best"}])
        call(client, "cull.accept", folder=FOLDER)
        call(client, "photo.rate", filepaths=[PHOTOS[0]], rating=3)
        got = call(client, "cull.undo", folder=FOLDER)
        assert got["kept"] == 1 and got["restored"] == 0
        assert marks(client, PHOTOS[0]) == (3, 0)

    def test_dismissed_proposals_are_not_applied(self, client):
        out = call(client, "cull.propose", proposals=[{"filepath": PHOTOS[0], "flag": -1, "reason": "blur"}])
        assert call(client, "cull.dismiss", run=out["run"]) == {"dismissed": 1}
        assert call(client, "cull.accept", folder=FOLDER)["accepted"] == 0
        assert marks(client, PHOTOS[0]) == (0, 0)

    def test_a_scope_is_required(self, client):
        assert client.post("/api/commands/cull.accept", json={}).status_code == 400

    def test_evaluate_against_the_persons_cull(self, client):
        call(client, "photo.rate", filepaths=[PHOTOS[0], PHOTOS[1]], rating=3)
        call(client, "photo.rate", filepaths=[PHOTOS[3]], flag=-1)
        call(client, "cull.propose", proposals=[
            {"filepath": PHOTOS[0], "rating": 3, "reason": "good"},      # both keep
            {"filepath": PHOTOS[1], "flag": -1, "reason": "soft"},       # person keeps, agent drops
            {"filepath": PHOTOS[2], "rating": 2, "reason": "ok"},        # agent keeps, person did not
            {"filepath": PHOTOS[3], "flag": -1, "reason": "blur"},       # both drop
        ])
        out = call(client, "cull.evaluate", folder=FOLDER)
        assert out["compared"] == 4
        assert out["keep"]["agreement"] == 0.5
        assert out["keep"]["precision"] == 0.5 and out["keep"]["recall"] == 0.5
        assert set(out["keep"]["by_moment"]) == {"precision", "recall"}
        assert out["flags"]["agent_by_person"]["reject"]["reject"] == 1
        assert {d["filepath"] for d in out["disagreements"]} == {PHOTOS[1], PHOTOS[2]}
        assert out["rating"]["exact"] == 0.5


def test_a_twin_of_the_same_moment_counts_as_found():
    """The person's pick between near-identical frames is close to arbitrary, so
    by moment a keep matches a keep of the same camera within two frame numbers
    and three seconds."""
    from types import SimpleNamespace

    from catalog.proposals import _matched, _moment

    def at(name, when):
        return _moment(SimpleNamespace(base_name=name, capture_date=f"2026:05:25 09:41:{when:02d}"))

    agent = [at("MK2A3950", 10), at("MK2A3990", 40)]
    person = [at("MK2A3951", 11), at("R0003950", 10), at("MK2A3960", 20)]
    assert _matched(agent, person) == 1            # 3950 ~ 3951; 3990 has no twin
    assert _matched(person, agent) == 1            # another camera's 3950 is not a twin
    assert _moment(SimpleNamespace(base_name="MK2A3950", capture_date="")) is None


def test_the_tools_reach_mcp(client):
    """Each command is an MCP tool; the culling ones are there with their schemas."""
    from commands import COMMANDS

    for name in ("cull.guide", "cull.signals", "cull.propose", "render.contact_sheet", "cull.evaluate"):
        assert name in COMMANDS
    assert COMMANDS["cull.propose"].writes and not COMMANDS["cull.signals"].writes


def test_the_guide_says_how_the_person_culls(client, tmp_path):
    """cull.guide gives an agent the lessons from calibration; full adds the procedure and log.
    A library without its own guide gets the template; one with a guide gets that."""
    lessons = call(client, "cull.guide")["guide"]
    assert lessons.startswith("## What we know about how you cull")
    assert "## Log" not in lessons
    whole = call(client, "cull.guide", full=True)["guide"]
    assert whole.startswith("# Culling calibration") and "## Log" in whole

    (tmp_path / "data" / "culling.md").write_text(
        "# Culling calibration\n\n## What we know about how you cull\n\n- Keep the bows.\n\n## Log\n",
        encoding="utf-8")
    assert call(client, "cull.guide")["guide"] == "## What we know about how you cull\n\n- Keep the bows.\n"
