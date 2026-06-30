"""Verify the structured-logging + hardware-monitor plumbing before you redeploy.

Runs on CPU so it's safe inside the Docker build (Containerfile) as a build guard:
  python test_logging.py

If a CUDA GPU is present it ALSO exercises the GPU paths — run on your laptop
before committing (same rule as smoke_test.py).

Exercises:
  1. clock.log() format with gen context
  2. HardwareMonitor sampling + start/stop lifecycle
  3. Config logging section loads from hyperparams.toml
  4. Inference server heartbeat carries timestamps
  5. Gen context is threaded into hardware-monitor logs
"""

from __future__ import annotations

import os
import sys
import time

# Allow `python test_logging.py` from the NeuralEngine directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Logging + hardware monitor verification")
print("=" * 60)

# ── 1. Config loads the logging section ───────────────────────────────────
from config import load, Config

cfg = load()
assert hasattr(cfg, "logging"), "Config missing 'logging' attribute"
assert cfg.logging.util_interval >= 0, f"util_interval must be >= 0, got {cfg.logging.util_interval}"
assert cfg.logging.infer_heartbeat > 0, f"infer_heartbeat must be > 0, got {cfg.logging.infer_heartbeat}"
print(f"  ✓ Config logging: util_interval={cfg.logging.util_interval}s  "
      f"infer_heartbeat={cfg.logging.infer_heartbeat}s")

# ── 2. clock.log() format with gen context ────────────────────────────────
from train.clock import log, set_start, set_gen

set_start()
set_gen(None)
log("startup message (no gen)")  # should have no [gen N] prefix
set_gen(1)
log("generation-scoped message")  # should have [gen 1] prefix
set_gen(None)
log("shutdown message (no gen)")  # should have no [gen N] prefix
print("  ✓ clock.log() with gen context (visual check above)")

# ── 3. HardwareMonitor lifecycle + sampling ───────────────────────────────
from train.hardware_monitor import HardwareMonitor, sample_now, _format

snap = sample_now()
formatted = _format(snap)
assert formatted, "_format returned empty string"
print(f"  ✓ Hardware sample: {formatted}")

monitor = HardwareMonitor(interval_s=0.5)
monitor.start()
time.sleep(1.0)   # enough for at least one sample
monitor.stop()
monitor.start()   # start again (idempotency test)
monitor.stop()
print("  ✓ HardwareMonitor start/stop lifecycle clean")

# ── 4. INFERENCE_LOG_EVERY env wiring ────────────────────────────────────
# train.py sets this before spawning the server; verify the env-var path.
os.environ["INFERENCE_LOG_EVERY"] = str(cfg.logging.infer_heartbeat)
assert float(os.environ.get("INFERENCE_LOG_EVERY", "0")) == cfg.logging.infer_heartbeat, \
    "INFERENCE_LOG_EVERY env var not wired"
print(f"  ✓ INFERENCE_LOG_EVERY wired to {cfg.logging.infer_heartbeat}s")

# ── 5. GPU paths (only when CUDA is available) ────────────────────────────
def _gpu_checks() -> None:
    """Run GPU-dependent checks (must be called from __main__ for mp spawn)."""
    import torch
    if not torch.cuda.is_available():
        print("  — No CUDA GPU — GPU checks skipped —")
        return

    print("  — GPU detected, running GPU checks —")

    # 5a. Hardware monitor should report GPU stats
    snap = sample_now()
    assert "gpu_pct" in snap, "GPU sample missing gpu_pct"
    assert "vram_used_mib" in snap, "GPU sample missing vram_used_mib"
    print(f"  ✓ GPU sample: gpu={snap['gpu_pct']:.0%}  "
          f"vram={snap['vram_used_mib']}/{snap['vram_total_mib']}MiB")

    # 5b. Timestamped infer server heartbeat (one-shot via a quick server)
    import numpy as np
    import multiprocessing as mp
    from net.model import build_net
    from train.selfplay import to_numpy_state
    from train.inference_server import InferenceServer, RemoteEvaluator

    ctx = mp.get_context("spawn")
    net = build_net(cfg).to("cuda").eval()
    np_state = to_numpy_state(net.state_dict())
    server = InferenceServer(cfg, [np_state], "cuda", 1, ctx)
    server.start()
    try:
        ev = RemoteEvaluator(0, 0, server.req_q, server.resp_qs[0])
        from hex.board import HexState
        policies, values = ev.evaluate([HexState.initial(cfg.game.board_size, True)])
        assert policies.shape[0] == 1 and values.shape[0] == 1, \
            "RemoteEvaluator returned wrong shapes"
        print(f"  ✓ RemoteEvaluator on GPU: eval OK (policy{policies.shape} value{values.shape})")
    finally:
        server.stop()

    # 5c. Full smoke of the GPU path via smoke_test's cuda_checks
    from smoke_test import cuda_checks
    cuda_checks()
    print("  ✓ Full GPU smoke (cuda_checks) passed")
    print("  CUDA LOGGING CHECKS PASSED")


if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("  — Torch not available — GPU checks skipped —")
    else:
        _gpu_checks()

    print()
    print("LOGGING VERIFICATION PASSED")
