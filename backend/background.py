"""
Background work steps aside while someone is using the app.

The library keeps itself up to date on its own: faces, smart search, camera
data, places, measurements and thumbnails for folders nobody has opened yet. On
a large library that is hours of work, and it shares the machine with the person
browsing. A face scan held most of the CPU while the folder being looked at
waited for its own thumbnails, so the app felt slow exactly when it was in use.

So every background loop asks :func:`wait_turn` before each unit of work (a
frame, a batch, a folder). It returns at once when nobody is using the app, and
otherwise waits until the page has been quiet for ``QUIET_SECONDS``. The person
can also pause it outright from the page and resume it later.

What counts as "using the app" is decided by :func:`counts_as_use` from the
request: the page polls status endpoints every couple of seconds, and if those
counted the background would never run while the page was open.

Work the person asked for (an import, an export, a scan started by a click,
learning a style) runs straight through: they are waiting for it.
"""

import re
import threading
import time
from typing import Callable, Optional

# How long the app must go without a request from someone using it before
# background work picks up again. Long enough to cover the gaps while a grid
# loads or someone looks at a picture, short enough that work resumes soon
# after they stop.
QUIET_SECONDS = 5.0

# How often a waiting worker looks again.
_POLL_SECONDS = 0.25

_lock = threading.Lock()
_resume = threading.Event()        # cleared while the person has paused it
_resume.set()
_last_use = 0.0
_paused_at = 0.0
# Who is waiting right now, by name, so the page can say what stepped aside.
_waiting: dict[str, int] = {}

# Requests the page makes on a timer, or that only read a job's progress.
_POLLING = re.compile(
    r"/(status|progress|activity|health|background|cache-stats)(/|$)"
    r"|/queue/|/learn(/|$)|/auth/|/basket/count$|/library/clocks$|/smart-search$"
)


def counts_as_use(method: str, path: str) -> bool:
    """Whether a request means someone is using the app right now."""
    if not path.startswith("/api/"):
        return False
    return not _POLLING.search(path)


def touch() -> None:
    """Someone used the app just now."""
    global _last_use
    _last_use = time.monotonic()


def pause() -> None:
    global _paused_at
    with _lock:
        if _resume.is_set():
            _paused_at = time.time()
        _resume.clear()


def resume() -> None:
    _resume.set()


def paused() -> bool:
    return not _resume.is_set()


def in_use() -> bool:
    """Whether someone used the app in the last ``QUIET_SECONDS``."""
    return time.monotonic() - _last_use < QUIET_SECONDS


def must_wait() -> bool:
    return paused() or in_use()


def wait_turn(name: str = "", should_stop: Optional[Callable[[], bool]] = None) -> bool:
    """Wait until background work may run. False when ``should_stop`` asked to
    stop while waiting, so the caller can leave instead of doing one more unit.
    """
    if not must_wait():
        return True
    with _lock:
        _waiting[name] = _waiting.get(name, 0) + 1
    try:
        while must_wait():
            if should_stop and should_stop():
                return False
            if paused():
                _resume.wait(_POLL_SECONDS)
            else:
                time.sleep(_POLL_SECONDS)
        return not (should_stop and should_stop())
    finally:
        with _lock:
            _waiting[name] -= 1
            if _waiting[name] <= 0:
                del _waiting[name]


def status() -> dict:
    with _lock:
        waiting = sorted(n for n in _waiting if n)
    return {
        "paused": paused(),
        "paused_at": _paused_at if paused() else 0.0,
        # The app is in use, so background work is stepping aside on its own.
        "yielding": in_use() and not paused(),
        "quiet_seconds": QUIET_SECONDS,
        "waiting": waiting,
    }
