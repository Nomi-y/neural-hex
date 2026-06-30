#!/usr/bin/env bash
# Pick a checkpoint and launch the external engine with it.
#
# Usage:
#   ./run_engine.sh            # lists checkpoints, prompts for a number
#   ./run_engine.sh 2          # picks checkpoint #2 non-interactively
#
# Credentials (ENGINE_ID / ENGINE_TOKEN / optional ENGINE_WS) are read from ./engine.env if present
# (that file is gitignored — keep your token out of version control), otherwise from the environment.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ -f engine.env ]; then
  set -a; # shellcheck disable=SC1091
  source engine.env; set +a
fi

if [ -z "${ENGINE_ID:-}" ] || [ -z "${ENGINE_TOKEN:-}" ]; then
  echo "ENGINE_ID and ENGINE_TOKEN are required (set them in engine.env or the environment)." >&2
  exit 2
fi

shopt -s nullglob
files=(checkpoints/*.pt)
if [ ${#files[@]} -eq 0 ]; then
  echo "No checkpoints found in checkpoints/. Train first (./build_container.sh, or python -m train.train)." >&2
  exit 1
fi

echo "Available checkpoints:"
for idx in "${!files[@]}"; do
  f="${files[$idx]}"
  size=$(du -h "$f" | cut -f1)
  when=$(date -r "$f" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "?")
  printf "  [%d] %s  (%s, %s)\n" "$((idx + 1))" "$f" "$size" "$when"
done

choice="${1:-}"
if [ -z "$choice" ]; then
  read -rp "Choose a checkpoint number: " choice
fi
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#files[@]}" ]; then
  echo "Invalid choice: '$choice' (expected 1..${#files[@]})." >&2
  exit 1
fi

export MODEL_PATH="${files[$((choice - 1))]}"
export ENGINE_WS="${ENGINE_WS:-ws://localhost:3001}"
echo "Launching engine: model=$MODEL_PATH  backend=$ENGINE_WS  engine=${ENGINE_ID:0:8}…"
exec python -m engine.play_engine
