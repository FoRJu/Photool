"""
Tests for burst grouping: the pure clustering in ``analysis.groups``, and the
catalog actions built on it (find bursts in a folder, reject the rest of one
burst, clean up a whole folder).

The hashes here are written out as hex so the Hamming distance between two of
them is something a reader can count, not a magic number.
"""

import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analysis.groups import MAX_DISTANCE, MAX_GAP_SECONDS, find_groups, suggest_keeper
from analysis.image_stats import STATS_VERSION, hamming_distance
from catalog.database import CatalogDB
from catalog.service import CatalogService
from catalog.xmp import read_xmp

# All bits clear, and neighbours of it at a known distance.
_SAME = "0000000000000000"
_ONE_BIT = "0000000000000001"
_AT_LIMIT = "0000000000000fff"        # 12 bits differ
_PAST_LIMIT = "0000000000001fff"      # 13 bits differ


def _member(name, mtime, dhash=_SAME, sharpness=1.0):
    return {
        "filepath": f"2024-06-15/{name}",
        "mtime": mtime,
        "dhash": dhash,
        "sharpness": sharpness,
    }


# ── Clustering ───────────────────────────────────────────────────────

class TestFindGroups:
    def test_the_same_shot_seconds_apart_groups(self):
        groups = find_groups(
            [_member("IMG_0001.CR3", 1000), _member("IMG_0002.CR3", 1002), _member("IMG_0003.CR3", 1004)]
        )
        assert len(groups) == 1
        assert [m["filepath"] for m in groups[0]] == [
            "2024-06-15/IMG_0001.CR3",
            "2024-06-15/IMG_0002.CR3",
            "2024-06-15/IMG_0003.CR3",
        ]

    def test_input_order_does_not_matter(self):
        groups = find_groups(
            [_member("IMG_0003.CR3", 1004), _member("IMG_0001.CR3", 1000), _member("IMG_0002.CR3", 1002)]
        )
        assert len(groups) == 1
        assert [m["mtime"] for m in groups[0]] == [1000, 1002, 1004]

    def test_the_same_frame_minutes_apart_is_not_a_burst(self):
        # A static stage between two songs looks identical and is not a burst.
        assert find_groups([_member("IMG_0001.CR3", 1000), _member("IMG_0009.CR3", 1600)]) == []

    def test_a_wider_window_reaches_what_the_default_misses(self):
        pair = [_member("IMG_0001.CR3", 1000), _member("IMG_0002.CR3", 1025)]
        assert find_groups(pair) == []
        groups = find_groups(pair, max_gap_seconds=30)
        assert len(groups) == 1
        assert [m["mtime"] for m in groups[0]] == [1000, 1025]

    def test_a_wider_window_still_measures_from_the_seed(self):
        # Ten seconds apart in a chain: a 15-second window reaches the next
        # frame, not the whole five minutes the chain covers.
        rows = [_member(f"IMG_{i:04d}.CR3", 1000 + i * 10) for i in range(6)]
        assert [len(g) for g in find_groups(rows, max_gap_seconds=15)] == [2, 2, 2]

    def test_distance_at_the_limit_groups(self):
        assert len(find_groups([_member("a", 1000), _member("b", 1001, _AT_LIMIT)])) == 1

    def test_distance_past_the_limit_does_not(self):
        assert hamming_distance(_SAME, _PAST_LIMIT) == MAX_DISTANCE + 1
        assert find_groups([_member("a", 1000), _member("b", 1001, _PAST_LIMIT)]) == []

    def test_a_frame_without_a_hash_is_left_out_not_guessed(self):
        groups = find_groups(
            [
                _member("a", 1000),
                _member("b", 1001, ""),
                _member("c", 1002),
            ]
        )
        assert len(groups) == 1
        assert [m["filepath"] for m in groups[0]] == ["2024-06-15/a", "2024-06-15/c"]

    def test_a_frame_lands_in_one_group_only(self):
        members = [_member(f"IMG_{i:02d}", 1000 + i) for i in range(6)]
        groups = find_groups(members)
        seen = [m["filepath"] for g in groups for m in g]
        assert len(groups) == 1
        assert len(seen) == len(set(seen)) == 6

    def test_the_window_is_measured_from_the_seed_not_chained(self):
        # Nine seconds apart, over three minutes of shooting. A transitive
        # closure would call all of this one burst; it is not.
        groups = find_groups(
            [_member("IMG_0001.CR3", 1000), _member("IMG_0002.CR3", 1009), _member("IMG_0003.CR3", 1018)]
        )
        assert [len(g) for g in groups] == [2]
        assert [m["mtime"] for m in groups[0]] == [1000, 1009]

    def test_two_bursts_in_one_folder_are_two_groups(self):
        groups = find_groups(
            [
                _member("IMG_0001.CR3", 1000),
                _member("IMG_0002.CR3", 1001),
                _member("IMG_0010.CR3", 1000, _PAST_LIMIT),
                _member("IMG_0011.CR3", 1001, _PAST_LIMIT),
            ]
        )
        assert [len(g) for g in groups] == [2, 2]

    def test_a_single_frame_is_not_a_group(self):
        assert find_groups([_member("only", 1000)]) == []
        assert find_groups([]) == []


class TestSuggestKeeper:
    def test_the_sharpest_frame_is_kept(self):
        group = [
            _member("soft", 1000, sharpness=2.0),
            _member("sharp", 1001, sharpness=40.0),
            _member("softer", 1002, sharpness=1.0),
        ]
        assert suggest_keeper(group)["filepath"] == "2024-06-15/sharp"

    def test_a_tie_keeps_the_earliest_frame(self):
        group = [_member("b", 1005), _member("a", 1000)]
        assert suggest_keeper(group)["filepath"] == "2024-06-15/a"

    def test_no_sharpness_figures_keeps_the_first_frame(self):
        group = [_member("a", 1000, sharpness=None), _member("b", 1001, sharpness=None)]
        assert suggest_keeper(group)["filepath"] == "2024-06-15/a"

    def test_an_empty_group_has_no_keeper(self):
        assert suggest_keeper([]) is None
        assert suggest_keeper([None]) is None


# ── Catalog service ──────────────────────────────────────────────────

class TestServiceBursts:
    def _setup(self, tmp: Path, names=("IMG_0001", "IMG_0002", "IMG_0003", "IMG_0004"),
               burst_max_gap_seconds=None):
        lib = tmp / "library"
        folder = lib / "2024-06-15"
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.CR3").write_bytes(b"fake raw " * 100)
            (folder / f"{name}.JPG").write_bytes(b"fake jpg " * 50)
        db = CatalogDB(str(tmp / "catalog.db"))
        if burst_max_gap_seconds is None:
            return CatalogService(db, str(lib)), lib
        return CatalogService(db, str(lib), burst_max_gap_seconds), lib

    def _measure(self, service, name, mtime, dhash=_SAME, sharpness=1.0):
        service.db.upsert_photo_stats(
            f"2024-06-15/{name}.CR3",
            {"stats_version": STATS_VERSION, "dhash": dhash, "sharpness": sharpness},
            source_mtime=mtime,
            source_size=900,
        )

    def test_a_burst_is_reported_with_a_suggested_keeper(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            self._measure(service, "IMG_0001", 1000, sharpness=3.0)
            self._measure(service, "IMG_0002", 1001, sharpness=9.0)
            self._measure(service, "IMG_0003", 1002, sharpness=2.0)
            self._measure(service, "IMG_0004", 1003, _PAST_LIMIT)

            found = service.find_bursts("2024-06-15", pair_mode="raw_only")
            assert len(found["groups"]) == 1
            group = found["groups"][0]
            assert group["size"] == 3
            assert group["keep"] == "2024-06-15/IMG_0002.CR3"
            assert found["frames"] == 3
            assert found["measured"] == 4
            assert found["total"] == 4
            assert found["unmeasured"] == 0

    def test_the_default_window_is_reported_with_the_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            found = service.find_bursts("2024-06-15", pair_mode="raw_only")
            # The payload says which window produced these groups, so a folder
            # that shows fewer bursts than expected can be explained.
            assert found["max_gap_seconds"] == MAX_GAP_SECONDS

    def test_the_library_window_is_used_for_every_burst_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp), burst_max_gap_seconds=30)
            service.index_folder("2024-06-15", [".cr3"])
            self._measure(service, "IMG_0001", 1000)
            self._measure(service, "IMG_0002", 1025)
            self._measure(service, "IMG_0003", 1001)

            found = service.find_bursts("2024-06-15", pair_mode="raw_only")
            assert found["max_gap_seconds"] == 30
            assert [g["size"] for g in found["groups"]] == [3]

            # Rejecting duplicates across the folder uses the same window, so
            # the two actions cannot disagree about what a burst is.
            cleaned = service.reject_burst_duplicates("2024-06-15", pair_mode="raw_only")
            assert cleaned["rejected"] == 2

    def test_one_request_can_ask_for_a_wider_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            self._measure(service, "IMG_0001", 1000)
            self._measure(service, "IMG_0002", 1025)

            assert service.find_bursts("2024-06-15", pair_mode="raw_only")["groups"] == []
            wider = service.find_bursts(
                "2024-06-15", max_gap_seconds=30, pair_mode="raw_only"
            )
            assert wider["max_gap_seconds"] == 30
            assert [g["size"] for g in wider["groups"]] == [2]

    def test_unmeasured_photos_are_counted_not_grouped(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            self._measure(service, "IMG_0001", 1000)

            found = service.find_bursts("2024-06-15", pair_mode="raw_only")
            assert found["groups"] == []
            assert found["measured"] == 1
            assert found["unmeasured"] == 3

    def test_measurements_from_older_code_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                service.db.upsert_photo_stats(
                    f"2024-06-15/{name}.CR3",
                    {"stats_version": STATS_VERSION - 1, "dhash": _SAME, "sharpness": 1.0},
                    source_mtime=mtime,
                    source_size=900,
                )

            found = service.find_bursts("2024-06-15", pair_mode="raw_only")
            assert found["groups"] == []
            assert found["unmeasured"] == 4

    def test_pair_mode_stops_a_cr3_grouping_with_its_own_jpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3", ".jpg"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                self._measure(service, name, mtime)

            found = service.find_bursts("2024-06-15", pair_mode="paired")
            members = [m["filepath"] for g in found["groups"] for m in g["members"]]
            assert all(not m.endswith(".JPG") for m in members)

    def test_a_burst_that_is_all_rejected_is_not_offered_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                self._measure(service, name, mtime)
                service.set_flag(f"2024-06-15/{name}.CR3", -1)

            assert service.find_bursts("2024-06-15", pair_mode="raw_only")["groups"] == []

    def test_reject_rest_rejects_the_burst_and_writes_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            keep = "2024-06-15/IMG_0001.CR3"
            others = ["2024-06-15/IMG_0002.CR3", "2024-06-15/IMG_0003.CR3"]

            result = service.reject_burst_rest(keep, others + [keep, ""])
            assert sorted(result["rejected"]) == sorted(others)

            for filepath in others:
                assert service.get_photo(filepath).flag == -1
                assert read_xmp(lib / filepath).flag == -1
            # The keeper is left exactly as it was.
            assert service.get_photo(keep).flag == 0
            assert read_xmp(lib / keep).flag == 0

    def test_reject_duplicates_keeps_one_frame_per_burst(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, lib = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            self._measure(service, "IMG_0001", 1000, sharpness=4.0)
            self._measure(service, "IMG_0002", 1001, sharpness=8.0)
            self._measure(service, "IMG_0003", 1002, sharpness=1.0)
            self._measure(service, "IMG_0004", 1003, _PAST_LIMIT)

            result = service.reject_burst_duplicates("2024-06-15", pair_mode="raw_only")
            assert result["groups"] == 1
            assert result["rejected"] == 2
            assert sorted(result["filepaths"]) == [
                "2024-06-15/IMG_0001.CR3",
                "2024-06-15/IMG_0003.CR3",
            ]
            assert service.get_photo("2024-06-15/IMG_0002.CR3").flag == 0
            assert service.get_photo("2024-06-15/IMG_0004.CR3").flag == 0
            assert read_xmp(lib / "2024-06-15/IMG_0003.CR3").flag == -1

    def test_reject_duplicates_is_a_no_op_on_an_unmeasured_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            result = service.reject_burst_duplicates("2024-06-15", pair_mode="raw_only")
            assert result == {"groups": 0, "rejected": 0, "filepaths": []}

    # ── Copying one edit onto the rest of the burst ──────────────────

    def _basketed(self, service, *names):
        for name in names:
            service.db.add_to_basket(f"2024-06-15/{name}.CR3", "2024-06-15", f"{name}.CR3")

    def _edits(self, service):
        return service.photo_edits("2024-06-15")

    def test_the_burst_of_a_frame_is_the_frames_shot_with_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (
                ("IMG_0001", 1000), ("IMG_0002", 1001), ("IMG_0003", 1002),
            ):
                self._measure(service, name, mtime)
            self._measure(service, "IMG_0004", 1003, dhash=_PAST_LIMIT)

            assert sorted(service.burst_of("2024-06-15/IMG_0002.CR3")) == [
                "2024-06-15/IMG_0001.CR3",
                "2024-06-15/IMG_0002.CR3",
                "2024-06-15/IMG_0003.CR3",
            ]
            # The fourth frame differs too much to be in it, so it comes back
            # alone rather than with company.
            assert service.burst_of("2024-06-15/IMG_0004.CR3") == [
                "2024-06-15/IMG_0004.CR3"
            ]

    def test_the_burst_is_the_group_and_not_what_the_grid_is_showing(self):
        """A frame hidden by a filter is still in the burst. Copying to "the
        rest of the burst" while the grid is filtered to picks would otherwise
        quietly skip the frames that most need the edit."""
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                self._measure(service, name, mtime)
                service.set_flag(f"2024-06-15/{name}.CR3", -1)
            self._measure(service, "IMG_0003", 1002)

            # Filtered to the picks, there is nothing left to group.
            assert service.find_bursts(
                "2024-06-15", flag_filter=1, pair_mode="raw_only"
            )["measured"] == 0
            assert len(service.burst_of("2024-06-15/IMG_0003.CR3")) == 3

    def test_an_unmeasured_frame_belongs_to_no_burst(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            assert service.burst_of("2024-06-15/IMG_0001.CR3") == [
                "2024-06-15/IMG_0001.CR3"
            ]
            assert service.burst_of("2024-06-15/NOPE.CR3") == [
                "2024-06-15/NOPE.CR3"
            ]

    def test_copying_an_edit_puts_it_on_the_rest_of_the_burst(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (
                ("IMG_0001", 1000), ("IMG_0002", 1001), ("IMG_0003", 1002),
            ):
                self._measure(service, name, mtime)
            self._measure(service, "IMG_0004", 1003, dhash=_PAST_LIMIT)

            edit = {"exposure": 0.5, "shadows": 30}
            result = service.apply_edit_to_burst("2024-06-15/IMG_0001.CR3", edit)

            assert sorted(result["copied"]) == [
                "2024-06-15/IMG_0002.CR3", "2024-06-15/IMG_0003.CR3",
            ]
            edits = self._edits(service)
            for name in ("IMG_0001", "IMG_0002", "IMG_0003"):
                assert edits[f"2024-06-15/{name}.CR3"] == edit
            # The frame that was not part of the burst keeps no edit, even
            # though it was measured in the same pass.
            assert edits.get("2024-06-15/IMG_0004.CR3", {}) == {}

    def test_spots_stay_on_their_own_frame(self):
        """Dust sits where it is on one frame: the tone goes to the burst, the
        spots removed from one frame do not, and a frame keeps its own."""
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                self._measure(service, name, mtime)
            own = {"x": 0.7, "y": 0.2, "r": 0.01, "sx": 0.75, "sy": 0.2, "mode": "heal"}
            service.set_photo_edit("2024-06-15/IMG_0002.CR3", {"spots": [own]})

            spot = {"x": 0.3, "y": 0.4, "r": 0.01, "sx": 0.33, "sy": 0.4, "mode": "heal"}
            service.apply_edit_to_burst("2024-06-15/IMG_0001.CR3", {"exposure": 0.5, "spots": [spot]})
            assert service.photo_edit("2024-06-15/IMG_0001.CR3")["spots"] == [spot]
            assert service.photo_edit("2024-06-15/IMG_0002.CR3") == {"exposure": 0.5, "spots": [own]}

    def test_a_burst_frame_outside_the_basket_still_gets_the_edit(self):
        """The edit belongs to the photograph now. Choosing what to export and
        choosing how it should look are separate decisions, and only the first
        one is the basket's — so a frame nobody has chosen to export still keeps
        the correction made on the frame beside it."""
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                self._measure(service, name, mtime)
            self._basketed(service, "IMG_0001")

            result = service.apply_edit_to_burst(
                "2024-06-15/IMG_0001.CR3", {"exposure": 1.0}
            )
            assert result["copied"] == ["2024-06-15/IMG_0002.CR3"]
            assert service.photo_edit("2024-06-15/IMG_0002.CR3") == {"exposure": 1.0}
            # ...and the basket is still exactly what the user put in it.
            assert service.db.basket_count() == 1

    def test_copying_clears_an_edit_that_is_no_longer_wanted(self):
        """The sliders walked back to zero have to reach the rest of the burst
        too, or the frames would keep an edit the panel no longer shows. A
        cleared edit is no edit, so the rows go rather than being kept empty."""
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = self._setup(Path(tmp))
            service.index_folder("2024-06-15", [".cr3"])
            for name, mtime in (("IMG_0001", 1000), ("IMG_0002", 1001)):
                self._measure(service, name, mtime)
            service.apply_edit_to_burst("2024-06-15/IMG_0001.CR3", {"exposure": 1.0})
            assert len(self._edits(service)) == 2

            service.apply_edit_to_burst("2024-06-15/IMG_0001.CR3", {})
            assert self._edits(service) == {}
            assert service.photo_edit("2024-06-15/IMG_0002.CR3") == {}
