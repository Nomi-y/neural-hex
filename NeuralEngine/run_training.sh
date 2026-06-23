#!/usr/bin/env bash
# Launch self-play training. Auto-detects GPU/CPU and uses all cores. Override any config.py knob via
# environment variables (see the README table). Safe to stop (Ctrl-C) and re-run — it resumes from
# checkpoints/latest.pt and the deployable model is always checkpoints/best.pt.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Wall-clock budget in hours (the directive's "couple of hours"); override as needed.
export TRAIN_HOURS="${TRAIN_HOURS:-4}"

exec python -m train.train
