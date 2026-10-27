"""
Regression tests for the API request helpers.

_run_blocking hands blocking filesystem work (directory walks, EXIF reads) to a
thread pool so the event loop stays responsive. An earlier revision built the
partial only when keyword arguments were present and passed the bare function
otherwise, silently dropping every positional argument. /library/folders and
/health both answered 500, which showed up in the UI as a Library with no
folders and no pictures — the failure was invisible without the traceback.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from api.routes import _run_blocking
from adjust.develop import normalize, point_of, tone_table
from catalog.database import _decode_settings


class TestRunBlocking:
    def test_positional_arguments_reach_the_call(self, tmp_path):
        def write_marker(name, suffix):
            target = Path(tmp_path) / (name + suffix)
            target.write_text("ok")
            return target.name

        # The shape /library/folders uses: two positional arguments, no kwargs.
        name = asyncio.run(_run_blocking(write_marker, "marker", ".txt"))
        assert name == "marker.txt"
        assert (tmp_path / "marker.txt").read_text() == "ok"

    def test_keyword_arguments_reach_the_call(self, tmp_path):
        def write_marker(name, *, suffix):
            (Path(tmp_path) / (name + suffix)).write_text("ok")
            return True

        assert asyncio.run(_run_blocking(write_marker, "kw", suffix=".txt"))
        assert (tmp_path / "kw.txt").read_text() == "ok"

    def test_arguments_and_keywords_together(self, tmp_path):
        def write(name, suffix, *, text):
            (Path(tmp_path) / (name + suffix)).write_text(text)
            return text

        assert asyncio.run(
            _run_blocking(write, "both", ".txt", text="hello")
        ) == "hello"
        assert (tmp_path / "both.txt").read_text() == "hello"

    def test_no_arguments_still_works(self):
        assert asyncio.run(_run_blocking(lambda: 7)) == 7

    def test_runs_off_the_event_loop(self):
        """The call must not run inline, or a long walk blocks every request."""
        async def main():
            caller = threading.current_thread()
            worker = await _run_blocking(threading.current_thread)
            return caller, worker

        caller, worker = asyncio.run(main())
        assert caller is not worker


class TestStoredEdit:
    """An edit is stored as the sliders the panel showed, not the curve they
    become, so reopening the develop view puts them back where they were left.
    It is read back thousands of times — once per grid render, once per export —
    so a row written by an older build, or edited by hand, must read as "no
    edit" rather than fail a caller or export something no control asked for."""

    def test_the_sliders_come_back_as_they_were_left(self):
        raw = json.dumps({"exposure": 0.5, "temperature": 20, "tint": -10})
        assert _decode_settings(raw) == {
            "exposure": 0.5, "temperature": 20, "tint": -10,
        }

    def test_an_empty_row_is_no_edit(self):
        assert _decode_settings("") == {}

    def test_a_row_that_is_not_an_object_is_no_edit(self):
        assert _decode_settings("not json") == {}
        assert _decode_settings("[1, 2]") == {}
        assert _decode_settings('"text"') == {}
        assert _decode_settings("null") == {}

    def test_a_partial_row_returns_only_the_controls_it_names(self):
        """A control at zero is not stored, so it is not reported either — the
        panel reads a missing control as its neutral position."""
        assert normalize(_decode_settings('{"exposure": 0.5}')) == {"exposure": 0.5}

    def test_a_value_out_of_range_is_clamped_not_trusted(self):
        """A row edited by hand could say anything. Clamping keeps it inside the
        range the sliders can reach rather than exporting something no control
        could have asked for."""
        assert normalize(_decode_settings('{"exposure": 40, "shadows": 900}')) == {
            "exposure": 5.0, "shadows": 100,
        }

    def test_an_unreadable_control_is_dropped_not_raised(self):
        assert normalize(_decode_settings('{"exposure": "bright", "shadows": 30}')) == {
            "shadows": 30,
        }

    def test_what_comes_back_is_the_shape_the_engine_reads(self):
        settings = normalize(
            _decode_settings('{"exposure": 1, "temperature": 30, "tint": -5}')
        )
        assert point_of(settings) is not None
        assert tone_table(settings) is None

        settings = normalize(_decode_settings('{"shadows": 30}'))
        assert tone_table(settings) is not None
