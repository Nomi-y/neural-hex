#!/usr/bin/env bash
# ============================================================================
#  NeuralEngine — container entrypoint
# ============================================================================
# One image, two roles (HEX_ROLE, default "train"):
#   train  — self-play training loop (train.train). Starts CUDA MPS so the
#            multi-worker self-play / arena fan-out can share the GPU.
#   engine — the deployed play/analysis engine (engine.play_engine). A single
#            process, so no MPS; it just needs a checkpoint + credentials.
# Both roles first log GPU properties so the logfile records the hardware.
# The engine image bakes HEX_ROLE=engine (see Containerfile build arg); the
# training image leaves it at the default. Override at runtime with -e HEX_ROLE=.
# ============================================================================
set -euo pipefail
cd /app

# Stream logs live: Python block-buffers stdout when it's a pipe (docker logs), so progress
# otherwise lags in big chunks. Unbuffered for the process and every spawned worker.
export PYTHONUNBUFFERED=1

HEX_ROLE="${HEX_ROLE:-train}"

# ── Log GPU properties (best-effort, nvidia-smi may not be in the image) ───
if command -v nvidia-smi &>/dev/null; then
  echo "[entrypoint] GPU info:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  echo "[entrypoint] GPU memory:"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
else
  # Fallback: Python can query CUDA device properties.
  python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'[entrypoint] CUDA device {i}: {p.name}  VRAM={p.total_memory//(1024**3):.0f}GB  cuda={p.major}.{p.minor}')
else:
    print('[entrypoint] CUDA not available — running on CPU')
" 2>/dev/null || true
fi

# ── Engine role: single process, no MPS — just run the play engine ─────────
if [ "$HEX_ROLE" = "engine" ]; then
  echo "[entrypoint] starting play engine…"
  exec python -u -m engine.play_engine
fi

# ── Training role: start CUDA MPS (best-effort) so fan-out workers share the GPU ──
MPS_CTRL=""
for candidate in nvidia-cuda-mps-control nvidia-mps; do
  if command -v "$candidate" &>/dev/null; then
    MPS_CTRL="$candidate"
    break
  fi
done

if [ -n "$MPS_CTRL" ]; then
  # Check if MPS is already running (pipe directory exists and server is alive).
  if [ -d /tmp/nvidia-mps ] && [ -S /tmp/nvidia-mps/control ] 2>/dev/null; then
    echo "[entrypoint] CUDA MPS appears to be already running"
  else
    echo "[entrypoint] starting CUDA MPS daemon ($MPS_CTRL)..."
    "$MPS_CTRL" -d 2>/dev/null && echo "[entrypoint] CUDA MPS started" || echo "[entrypoint] CUDA MPS not available (non-fatal)"
  fi
else
  echo "[entrypoint] nvidia-cuda-mps-control not found — MPS not started (non-fatal)"
fi

echo "[entrypoint] starting training…"
exec python -u -m train.train
