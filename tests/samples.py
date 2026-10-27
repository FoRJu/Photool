"""
Real photographs for the few tests that need one: a RAW and what the camera made
of it can't be faked with a synthetic frame.

They are not in the repository. Point ``PHOTOOL_SAMPLES`` at a folder of your own
frames under these names, and the tests that use them run; without it they skip.

  lens-24mm.CR3, lens-24mm.JPG   an R6 II + RF24-105 at 24 mm, RAW and its camera JPEG
  group-of-three.CR3             three people close up, a crowd behind them
  building-from-below.CR3        a building shot from below, its verticals converging
"""

import os
from pathlib import Path

import pytest


def sample(name: str) -> Path:
    """The sample frame ``name``, or the test is skipped."""
    root = os.environ.get("PHOTOOL_SAMPLES", "")
    path = Path(root) / name
    if not root or not path.exists():
        pytest.skip(f"no {name} in PHOTOOL_SAMPLES")
    return path
