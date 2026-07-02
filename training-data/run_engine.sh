#!/usr/bin/env bash
# Launch the Hex engine from any checkpoint in training-data/.
#
# Scans checkpoints/ (run 1) and run2/checkpoints/ (run 2).  Sorted by
# generation number across both runs; run2 entries are tagged [run2].
#
# Usage:
#   ./run_engine.sh              # lists all checkpoints, prompts for a number
#   ./run_engine.sh 5            # picks checkpoint #5 in the sorted list
#   ./run_engine.sh best         # picks best.pt (run1 then run2)
#
# Reads ENGINE_ID / ENGINE_TOKEN from ../NeuralEngine/engine.env.
# The engine connects to ENGINE_WS (default ws://localhost:3001).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/../NeuralEngine"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
RUN2_DIR="$SCRIPT_DIR/run2/checkpoints"
VENV="$ENGINE_DIR/.venv"

# ── Activate the NeuralEngine venv ────────────────────────────────────────
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "ERROR: venv not found at $VENV." >&2
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
  echo "ENGINE_ID and ENGINE_TOKEN are required." >&2
  exit 2
fi

# ── Collect checkpoints from both runs ──────────────────────────────────
shopt -s nullglob
files1=("$CKPT_DIR"/*.pt)
files2=()
[ -d "$RUN2_DIR" ] && files2=("$RUN2_DIR"/*.pt)
all=("${files1[@]}" "${files2[@]}")
if [ ${#all[@]} -eq 0 ]; then
  echo "No checkpoints found." >&2
  exit 1
fi

choice="${1:-}"
MODEL_PATH=""

# Direct name match like "./run_engine.sh best"
if [ -n "$choice" ]; then
  [ -f "$CKPT_DIR/${choice}.pt" ] && MODEL_PATH="$CKPT_DIR/${choice}.pt"
  [ -z "$MODEL_PATH" ] && [ -f "$RUN2_DIR/${choice}.pt" ] && MODEL_PATH="$RUN2_DIR/${choice}.pt"
fi

if [ -z "$MODEL_PATH" ]; then
  # Sort by generation number
  mapfile -t sorted < <(printf '%s\n' "${all[@]}" | sort -t'_' -k2 -n 2>/dev/null || printf '%s\n' "${all[@]}" | sort)

  echo "Available checkpoints:"
  for idx in "${!sorted[@]}"; do
    f="${sorted[$idx]}"
    name=$(basename "$f")
    tag=""
    [[ "$f" == "$RUN2_DIR"/* ]] && tag=" [run2]"
    size=$(du -h "$f" | cut -f1)
    when=$(date -r "$f" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "?")
    printf "  [%d] %-18s %s %5s  %s\n" "$((idx + 1))" "$name" "$tag" "$size" "$when"
  done
  echo

  if [ -z "$choice" ]; then
    read -rp "Choose a checkpoint number: " choice
  fi

  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#sorted[@]}" ]; then
    MODEL_PATH="${sorted[$((choice - 1))]}"
  else
    echo "Invalid choice: '$choice' (expected 1..${#sorted[@]})." >&2
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
