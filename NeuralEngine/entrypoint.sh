#!/usr/bin/env bash
# ============================================================================
#  NeuralEngine — container entrypoint
# ============================================================================
# 1. Best-effort: start CUDA MPS if the binary is available (host driver passes
#    it through, or it was installed in the image). MPS lets multiple processes
#    submit CUDA work concurrently on the same GPU — critical for multi-worker
#    self-play / arena fan-out.
# 2. Log GPU properties (name, VRAM, CUDA version) so the logfile captures what
#    hardware the run executed on.
# 3. Launch training.
# ============================================================================
set -euo pipefail
cd /app

# ── Optional: start CUDA MPS daemon ────────────────────────────────────────
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

# ── Log GPU properties (best-effort, nvidia-smi may not be in the image) ───
if command -v nvidia-smi &>/dev/null; then
  echo "[entrypoint] GPU info:"
  nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader 2>/dev/null || true
  echo "[entrypoint] GPU memory:"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
else
  # Fallback: Python can query CUDA device properties.
  python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'[entrypoint] CUDA device {i}: {p.name}  VRAM={p.total_mem//(1024**3):.0f}GB  cuda={p.major}.{p.minor}')
else:
    print('[entrypoint] CUDA not available — running on CPU')
" 2>/dev/null || true
fi

echo "[entrypoint] starting training…"
exec python -m train.train
