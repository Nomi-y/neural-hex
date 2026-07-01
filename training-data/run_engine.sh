#!/usr/bin/env bash
# Launch the Hex engine from any checkpoint in training-data/checkpoints/.
#
# Usage:
#   ./run_engine.sh              # lists checkpoints, prompts for a number
#   ./run_engine.sh 5            # picks checkpoint #5 (gen_0005.pt)
#   ./run_engine.sh best         # picks best.pt
#
# Reads ENGINE_ID / ENGINE_TOKEN from ../NeuralEngine/engine.env.
# The engine connects to ENGINE_WS (default ws://localhost:3001).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/../NeuralEngine"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
VENV="$ENGINE_DIR/.venv"

# ── Activate the NeuralEngine venv ────────────────────────────────────────
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "ERROR: venv not found at $VENV.  Run: cd $ENGINE_DIR && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
export PYTHONPATH="$ENGINE_DIR:${PYTHONPATH:-}"

# ── Load engine credentials ──────────────────────────────────────────────
if [ -f "$ENGINE_DIR/engine.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ENGINE_DIR/engine.env"
  set +a
fi

if [ -z "${ENGINE_ID:-}" ] || [ -z "${ENGINE_TOKEN:-}" ]; then
  echo "ENGINE_ID and ENGINE_TOKEN are required (set them in $ENGINE_DIR/engine.env or the environment)." >&2
  exit 2
fi

# ── List checkpoints ─────────────────────────────────────────────────────
shopt -s nullglob
files=("$CKPT_DIR"/*.pt)
if [ ${#files[@]} -eq 0 ]; then
  echo "No checkpoints found in $CKPT_DIR/." >&2
  echo "Sync them from S3 first:" >&2
  echo "  aws s3 sync s3://q8ayv4s4m7/ $CKPT_DIR/ --profile runpods3 --endpoint-url https://s3api-eu-ro-1.runpod.io" >&2
  exit 1
fi

choice="${1:-}"

# Direct name match: "./run_engine.sh best" picks best.pt
if [ -n "$choice" ] && [ -f "$CKPT_DIR/${choice}.pt" ]; then
  MODEL_PATH="$CKPT_DIR/${choice}.pt"
else
  # Sort: gen_NNNN.pt by generation number, then best.pt
  mapfile -t sorted < <(printf '%s\n' "${files[@]}" | sort -t'_' -k2 -n 2>/dev/null || printf '%s\n' "${files[@]}" | sort)

  echo "Available checkpoints:"
  echo "  CKPT_DIR = $CKPT_DIR"
  echo
  for idx in "${!sorted[@]}"; do
    f="${sorted[$idx]}"
    name=$(basename "$f")
    size=$(du -h "$f" | cut -f1)
    when=$(date -r "$f" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "?")
    printf "  [%d] %-18s  %5s  %s\n" "$((idx + 1))" "$name" "$size" "$when"
  done
  echo

  if [ -z "$choice" ]; then
    read -rp "Choose a checkpoint number (or 'best'): " choice
  fi

  if [ "$choice" = "best" ] && [ -f "$CKPT_DIR/best.pt" ]; then
    MODEL_PATH="$CKPT_DIR/best.pt"
  elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#sorted[@]}" ]; then
    MODEL_PATH="${sorted[$((choice - 1))]}"
  else
    echo "Invalid choice: '$choice' (expected 1..${#sorted[@]} or 'best')." >&2
    exit 1
  fi
fi

# ── Launch ────────────────────────────────────────────────────────────────
export MODEL_PATH
export ENGINE_WS="${ENGINE_WS:-ws://localhost:3001}"

echo "Launching engine:"
echo "  model   = $MODEL_PATH"
echo "  backend = $ENGINE_WS"
echo "  engine  = ${ENGINE_ID:0:8}…"
echo

cd "$ENGINE_DIR"
exec python -m engine.play_engine
