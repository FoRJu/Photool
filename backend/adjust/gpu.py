"""
The graphics card's models, handled together.

Noise reduction, object masks and the people parsers each hold onnxruntime sessions on
DirectML, and those sessions are not independent of one another:

  * **Letting go of a session breaks the others already open.** The next model call on
    one of them is an access violation that takes the whole process down, with no
    Python error. Noise reduction, then a mask (which lets its models go after three
    idle minutes), then noise reduction again: it crashed every time this was tried,
    and all five crashes logged at that address followed a release. A session opened
    after the release is fine. So nothing is let go of alone: :func:`release_all` drops
    every session, and each model opens a new one the next time it is wanted.
  * **Two model calls at once can lose the device.** Two sessions of one model run from
    two threads failed with "The GPU device instance has been suspended". So all model
    calls, and the opening and closing of sessions, are under :data:`LOCK`.

A session must not be held in a variable beyond :data:`LOCK`: if a release came in
between, the last reference to the old session would go, and destroy it, outside the
lock, breaking whatever had been opened since.

  * **A large model does not share the card.** With the depth model and SAM open on 8 GB,
    noise reduction went from 13 s to over two minutes, though either one alone cost it
    nothing. So a large model opens with :func:`open_large`, and every other model calls
    :func:`make_room` before it opens.
"""

import gc
import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

# Re-entrant: a model's lock is this one, and its call for a session happens inside it.
LOCK = threading.RLock()
_holders: list = []
_large_open = False


def register(drop: Callable[[], None]) -> None:
    """``drop`` forgets every session its module holds."""
    if drop not in _holders:
        _holders.append(drop)


def release_all() -> None:
    """Let go of every session on the card, all at once."""
    global _large_open
    with LOCK:
        for drop in _holders:
            drop()
        _large_open = False
        gc.collect()


def open_large() -> None:
    """A large model is about to open: nothing else stays on the card with it."""
    global _large_open
    with LOCK:
        logger.info("[gpu] a large model opens: every other model on the card is let go of")
        release_all()
        _large_open = True


def make_room() -> None:
    """Another model is about to open: a large one goes, and opens again when it is wanted."""
    with LOCK:
        if _large_open:
            # Seen in the log next to a model opening again, this is the card being shared badly.
            logger.info("[gpu] another model opens: the large one is let go of")
            release_all()
