"""Timestamped logging shared across the training modules.

Every line is prefixed with the wall-clock time and the offset since the run started, e.g.

    [14:05:01 +0:03:12] [gen 5] self-play done …

so a long unattended run reads back as a timeline (when something happened *and* how far into the run
it was). `set_start()` is called once at startup; everything else just calls `log()`. Worker processes
don't log — progress is reported from the parent — so the start time never needs to cross a process
boundary.
"""

from __future__ import annotations

import time

_START = time.time()


def set_start(t: float | None = None) -> None:
    global _START
    _START = t if t is not None else time.time()


def start_time() -> float:
    return _START


def offset_str(seconds: float) -> str:
    """`H:MM:SS` offset (hours never zero-padded so short runs stay compact)."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now} +{offset_str(time.time() - _START)}] {msg}", flush=True)
