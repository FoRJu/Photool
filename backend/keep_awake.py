"""
Keep Windows from going to sleep while Photool is being used.

Windows' idle timer counts keyboard and mouse on this computer. Someone culling
from a phone or running an export from a tablet is neither, so the computer went
to sleep under them. While someone has used the app in the last ``minutes``, or
work they asked for is running, this holds the system awake with
``SetThreadExecutionState``; then it lets go, and the idle timer decides as usual.

A page left open does not count (its polling is not use, see background.py), nor
does automatic work such as a face scan: that would keep the computer on for hours
for nobody. Choosing Sleep, or a program asking for it, still works: this only
holds off the idle timer. Not used off Windows.
"""

import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("serve")

CHECK_SECONDS = 20.0
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def hold_system_awake(hold: bool) -> None:
    """Applies to the calling thread until it says otherwise or ends."""
    import ctypes

    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if hold else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


class KeepAwake:
    def __init__(self, minutes: float, busy: Callable[[], bool],
                 hold: Callable[[bool], None] = hold_system_awake,
                 clock: Callable[[], float] = time.monotonic):
        self.window = minutes * 60.0
        self.busy = busy
        self.hold = hold
        self.clock = clock
        self.last_use: Optional[float] = None
        self.holding = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def touch(self) -> None:
        self.last_use = self.clock()

    def wanted(self) -> bool:
        if self.last_use is not None and self.clock() - self.last_use < self.window:
            return True
        try:
            return bool(self.busy())
        except Exception:
            logger.exception("Could not tell whether work is running")
            return False

    def check(self) -> None:
        wanted = self.wanted()
        if wanted != self.holding:
            self.hold(wanted)
            self.holding = wanted
            logger.info("Keeping this computer awake: in use" if wanted
                        else "No longer keeping this computer awake")

    def _run(self) -> None:
        # The hold belongs to this thread, so every call is made from here.
        while not self._stop.wait(CHECK_SECONDS):
            self.check()
        if self.holding:
            self.hold(False)
            self.holding = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="keep-awake", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)


_current: Optional[KeepAwake] = None


def touch() -> None:
    """Someone used the app just now."""
    if _current is not None:
        _current.touch()


def start(minutes: float, busy: Callable[[], bool]) -> None:
    global _current
    if os.name != "nt" or minutes <= 0:
        return
    _current = KeepAwake(minutes, busy)
    _current.start()


def stop() -> None:
    global _current
    if _current is not None:
        _current.stop()
        _current = None
