#!/usr/bin/env bash
# Launch self-play training with a chosen preset.
#
# Usage:
#   ./run_training.sh           # prints the presets, prompts for a number
#   ./run_training.sh 5         # picks preset #5 non-interactively
#
# Each preset exports the relevant config.py knobs (DEVICE, net size, sims, batch sizes, time budget).
# Every preset PINS its device (DEVICE=cpu or DEVICE=cuda) so the choice is deterministic — a CPU preset
# always runs on CPU even on a box that has a GPU, and a CUDA preset fails loudly if there's no GPU
# rather than silently crawling on CPU.
#
# Anything you set in the environment beforehand still wins — every assignment is  export VAR="${VAR:-...}"
# so e.g.  TRAIN_HOURS=2 ./run_training.sh 5  overrides only the budget.
#
# Safe to stop (Ctrl-C) and re-run: it resumes from checkpoints/latest.pt and the deployable model is
# always checkpoints/best.pt. NOTE: resuming only works if the NET size matches the checkpoint — a
# preset with a different NET_CHANNELS/NET_BLOCKS cannot load an existing checkpoint and starts fresh,
# so don't switch net size mid-run (move checkpoints/ aside first if you mean to).
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Many CPU self-play actors open a lot of files/pipes/semaphores at once; lift the soft FD limit toward
# the hard limit so a high core count doesn't hit EMFILE ("Too many open files"). Best-effort.
ulimit -n "$(ulimit -Hn 2>/dev/null || echo 65535)" 2>/dev/null || true

cat <<'MENU'
Choose a training preset. Time/iteration figures are rough and scale with the VPS you run on
(core count for CPU presets, GPU model for CUDA presets) — the per-generation log prints the real clock.

  Personal computer, no GPU — verify it works locally, then move to a VPS ──────
  [1] Smoke run          tiny 32x4 net, 60 sims, 32 games/gen, 0.5h, DEVICE=cpu
                         → minutes per gen, a handful of gens. Confirms the pipeline, NOT strength.
  [2] Local check        64x6 net, 120 sims, 64 games/gen, 1h, DEVICE=cpu
                         → a few gens; sanity-check that loss falls / arena promotes before renting a box.

  CPU VPS — many-core box, uses all cores ─────────────────────────────────────
  [3] Balanced           64x6 net, 150 sims, 192 games/gen, 8h, DEVICE=cpu
                         → ~10-30 gens depending on core count; a real but modest engine.
  [4] Strong             96x8 net, 200 sims, 256 games/gen, 24h, DEVICE=cpu
                         → for a big CPU box left running long; stronger ceiling, slow per gen.

  CUDA GPU VPS ────────────────────────────────────────────────────────────────
  [5] Balanced (96x8)    200 sims, 64 parallel, 256 games/gen, 6h, DEVICE=cuda   [resume-compatible]
                         → tens of gens on a typical cloud GPU. Matches the default net, so it RESUMES
                           existing checkpoints/.
  [6] Large              128x10 net, 300 sims, 128 parallel, 512 games/gen, 12h, DEVICE=cuda
                         → higher ceiling for a bigger/faster GPU; the larger net CANNOT resume a 96x8
                           checkpoint — it starts fresh.
MENU

choice="${1:-}"
if [ -z "$choice" ]; then
  read -rp "Preset number [1-6]: " choice
fi

case "$choice" in
  1)  # Personal computer, no GPU — quick pipeline smoke run
      export DEVICE="${DEVICE:-cpu}"
      export NET_CHANNELS="${NET_CHANNELS:-32}";  export NET_BLOCKS="${NET_BLOCKS:-4}"
      export MCTS_SIMS="${MCTS_SIMS:-60}";        export PARALLEL_GAMES="${PARALLEL_GAMES:-16}"
      export GAMES_PER_GEN="${GAMES_PER_GEN:-32}"; export TRAIN_STEPS="${TRAIN_STEPS:-100}"
      export BATCH_SIZE="${BATCH_SIZE:-256}";     export ARENA_GAMES="${ARENA_GAMES:-12}"
      export ARENA_SIMS="${ARENA_SIMS:-60}";      export TRAIN_HOURS="${TRAIN_HOURS:-0.5}"
      label="1: personal computer (no GPU) — smoke run" ;;
  2)  # Personal computer, no GPU — local correctness check before renting a VPS
      export DEVICE="${DEVICE:-cpu}"
      export NET_CHANNELS="${NET_CHANNELS:-64}";  export NET_BLOCKS="${NET_BLOCKS:-6}"
      export MCTS_SIMS="${MCTS_SIMS:-120}";       export PARALLEL_GAMES="${PARALLEL_GAMES:-24}"
      export GAMES_PER_GEN="${GAMES_PER_GEN:-64}"; export TRAIN_STEPS="${TRAIN_STEPS:-200}"
      export BATCH_SIZE="${BATCH_SIZE:-256}";     export ARENA_GAMES="${ARENA_GAMES:-16}"
      export ARENA_SIMS="${ARENA_SIMS:-100}";     export TRAIN_HOURS="${TRAIN_HOURS:-1}"
      label="2: personal computer (no GPU) — local check" ;;
  3)  # CPU VPS — balanced, all cores
      export DEVICE="${DEVICE:-cpu}"
      export NET_CHANNELS="${NET_CHANNELS:-64}";  export NET_BLOCKS="${NET_BLOCKS:-6}"
      export MCTS_SIMS="${MCTS_SIMS:-150}";       export PARALLEL_GAMES="${PARALLEL_GAMES:-32}"
      export GAMES_PER_GEN="${GAMES_PER_GEN:-192}"; export TRAIN_STEPS="${TRAIN_STEPS:-300}"
      export BATCH_SIZE="${BATCH_SIZE:-384}";     export ARENA_GAMES="${ARENA_GAMES:-30}"
      export ARENA_SIMS="${ARENA_SIMS:-100}";     export TRAIN_HOURS="${TRAIN_HOURS:-8}"
      label="3: CPU VPS — balanced (all cores)" ;;
  4)  # CPU VPS — strong, all cores, long run
      export DEVICE="${DEVICE:-cpu}"
      export NET_CHANNELS="${NET_CHANNELS:-96}";  export NET_BLOCKS="${NET_BLOCKS:-8}"
      export MCTS_SIMS="${MCTS_SIMS:-200}";       export PARALLEL_GAMES="${PARALLEL_GAMES:-32}"
      export GAMES_PER_GEN="${GAMES_PER_GEN:-256}"; export TRAIN_STEPS="${TRAIN_STEPS:-400}"
      export BATCH_SIZE="${BATCH_SIZE:-512}";     export ARENA_GAMES="${ARENA_GAMES:-40}"
      export ARENA_SIMS="${ARENA_SIMS:-120}";     export TRAIN_HOURS="${TRAIN_HOURS:-24}"
      label="4: CPU VPS — strong (all cores)" ;;
  5)  # CUDA VPS — balanced, resume-compatible with the default 96x8 net
      export DEVICE="${DEVICE:-cuda}"
      export NET_CHANNELS="${NET_CHANNELS:-96}";  export NET_BLOCKS="${NET_BLOCKS:-8}"
      export MCTS_SIMS="${MCTS_SIMS:-200}";       export PARALLEL_GAMES="${PARALLEL_GAMES:-64}"
      export GAMES_PER_GEN="${GAMES_PER_GEN:-256}"; export TRAIN_STEPS="${TRAIN_STEPS:-400}"
      export BATCH_SIZE="${BATCH_SIZE:-512}";     export ARENA_GAMES="${ARENA_GAMES:-40}"
      export ARENA_SIMS="${ARENA_SIMS:-120}";     export TRAIN_HOURS="${TRAIN_HOURS:-6}"
      label="5: CUDA VPS — balanced 96x8 (resume-compatible)" ;;
  6)  # CUDA VPS — large net for a bigger/faster GPU
      export DEVICE="${DEVICE:-cuda}"
      export NET_CHANNELS="${NET_CHANNELS:-128}"; export NET_BLOCKS="${NET_BLOCKS:-10}"
      export MCTS_SIMS="${MCTS_SIMS:-300}";       export PARALLEL_GAMES="${PARALLEL_GAMES:-128}"
      export GAMES_PER_GEN="${GAMES_PER_GEN:-512}"; export TRAIN_STEPS="${TRAIN_STEPS:-600}"
      export BATCH_SIZE="${BATCH_SIZE:-768}";     export ARENA_GAMES="${ARENA_GAMES:-48}"
      export ARENA_SIMS="${ARENA_SIMS:-160}";     export TRAIN_HOURS="${TRAIN_HOURS:-12}"
      label="6: CUDA VPS — large net (128x10)" ;;
  *)  echo "Invalid choice: '$choice' (expected 1-6)." >&2; exit 1 ;;
esac

echo
echo "Preset $label"
echo "  device=${DEVICE} net=${NET_CHANNELS}x${NET_BLOCKS} sims=${MCTS_SIMS} parallel=${PARALLEL_GAMES}"
echo "  games/gen=${GAMES_PER_GEN} train_steps=${TRAIN_STEPS} arena=${ARENA_GAMES}@${ARENA_SIMS} budget=${TRAIN_HOURS}h"
echo

exec python -m train.train
