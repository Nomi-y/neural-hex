"""Timestamped logging shared across the training modules.

Every line is prefixed with the wall-clock time and the offset since the run started, e.g.

    [14:05:01 +0:03:12] [gen 5] self-play done …

so a long unattended run reads back as a timeline (when something happened *and* how far into the run
it was). `set_start()` is called once at startup; everything else just calls `log()`. Worker processes
don't log — progress is reported from the parent — so the start time never needs to cross a process
boundary.
"""

from __future__ import annotations

import os
import time


def _env_start() -> float:
    """Training start epoch shared with spawned children via TRAIN_START_EPOCH (set by train.py).
    Lets a worker/inference-server process report the SAME `+offset` as the parent instead of one
    measured from its own (later) start.  Falls back to now for a standalone import."""
    raw = os.environ.get("TRAIN_START_EPOCH")
    try:
        return float(raw) if raw else time.time()
    except (TypeError, ValueError):
        return time.time()


def _env_gen() -> int | None:
    """Current generation shared with spawned children via TRAIN_GENERATION (set by train.py), so
    worker log lines carry `[gen N]` too.  None outside a generation / when unset."""
    raw = os.environ.get("TRAIN_GENERATION")
    try:
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


_START = _env_start()


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


_GEN: int | None = _env_gen()  # current generation; parent sets it via set_gen, children inherit via env


def set_gen(gen: int | None) -> None:
    global _GEN
    _GEN = gen


def log(msg: str, gen: int | None = None) -> None:
    """Timestamped log line: [HH:MM:SS +offset] [gen N] msg.

    If `gen` is not passed, falls back to the module-level `_GEN` (set by the main loop, or inherited
    from TRAIN_GENERATION in a spawned worker).  When neither is set, the [gen N] segment is omitted
    so startup/shutdown lines stay compact.
    """
    now = time.strftime("%H:%M:%S")
    gen = gen if gen is not None else _GEN
    prefix = f"[{now} +{offset_str(time.time() - _START)}]"
    if gen is not None:
        prefix += f" [gen {gen}]"
    print(f"{prefix} {msg}", flush=True)
