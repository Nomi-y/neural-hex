#!/usr/bin/env bash
# ============================================================================
#  NeuralEngine — container build script
# ============================================================================
#
#  This is the single entry point for building a training container.
#  It reads hyperparams.toml (the single source of truth for all settings),
#  optionally applies a hardware preset, and builds the container image.
#
#  Usage:
#    ./build_container.sh                        # build with current hyperparams.toml
#    ./build_container.sh --preset cuda-balanced # apply preset before building
#    ./build_container.sh --cuda                 # install CUDA torch wheel
#    ./build_container.sh --tag myengine:v2      # custom image tag
#    ./build_container.sh --list-presets          # show available presets
#
#  Presets are defined in hyperparams.toml (see the [presets.*] sections).
#  Applying a preset modifies hyperparams.toml IN PLACE — commit your changes
#  to version control first if you want to keep the original.
#
#  After building, run the container:
#    podman run --device nvidia.com/gpu=all -v ./checkpoints:/app/checkpoints neuralengine
#
#  To resume stopped training, just run the same command again — checkpoints
#  are persisted in the bind-mounted directory.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="neuralengine"
PRESET=""
CUDA=false
LIST_PRESETS=false
DRY_RUN=false

usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)
      PRESET="$2"; shift 2 ;;
    --tag)
      IMAGE="$2"; shift 2 ;;
    --cuda)
      CUDA=true; shift ;;
    --list-presets)
      LIST_PRESETS=true; shift ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    -h|--help)
      usage ;;
    *)
      echo "Unknown argument: $1  (use --help)" >&2; exit 2 ;;
  esac
done

# ── List presets ────────────────────────────────────────────────────────────

if $LIST_PRESETS; then
  echo "Available presets (from hyperparams.toml):"
  echo
  python3 -c "
import tomllib, sys
with open('hyperparams.toml', 'rb') as f:
    data = tomllib.load(f)
presets = data.get('presets', {})
if not presets:
    print('  (none defined)')
    sys.exit(0)
for name, cfg in presets.items():
    desc = cfg.get('description', '(no description)')
    print(f'  {name}')
    print(f'      {desc}')
    for section in ('net', 'mcts', 'selfplay', 'train'):
        if section in cfg:
            for k, v in cfg[section].items():
                print(f'      {section}.{k} = {v}')
    if 'device' in cfg:
        print(f'      device = {cfg[\"device\"]}')
    print()
"
  exit 0
fi

# ── Apply preset ────────────────────────────────────────────────────────────

if [ -n "$PRESET" ]; then
  if [ ! -f hyperparams.toml ]; then
    echo "ERROR: hyperparams.toml not found.  Are you in the project root?" >&2
    exit 1
  fi

  echo "=== Applying preset: $PRESET ==="
  echo

  python3 -c "
import tomllib, sys, os

preset_name = '${PRESET}'

with open('hyperparams.toml', 'rb') as f:
    data = tomllib.load(f)

presets = data.get('presets', {})
if preset_name not in presets:
    print(f'ERROR: preset \"{preset_name}\" not found in hyperparams.toml.')
    print(f'Available: {list(presets.keys())}')
    sys.exit(1)

preset = presets[preset_name]
print(f'Preset: {preset.get(\"description\", preset_name)}')

# Apply each section from the preset into the main config
for section in ('net', 'mcts', 'selfplay', 'train', 'game'):
    if section in preset:
        if section not in data:
            data[section] = {}
        for key, value in preset[section].items():
            old = data[section].get(key, '(not set)')
            data[section][key] = value
            print(f'  {section}.{key}: {old} -> {value}')

if 'device' in preset:
    old_dev = data.get('device', '(auto)')
    data['device'] = preset['device']
    print(f'  device: {old_dev} -> {preset[\"device\"]}')

# Write back as TOML (manual formatting to preserve structure)
# We rebuild the file to keep it clean.
import io
out = io.StringIO()

# Preserve header comments — read the first comment block
with open('hyperparams.toml', 'r') as f:
    for line in f:
        if line.startswith('#') or line.strip() == '':
            out.write(line)
        else:
            break

# Write each section
section_order = ['game', 'net', 'mcts', 'selfplay', 'train', 'engine', 'device', 'presets']
for sec in section_order:
    if sec not in data:
        continue
    if not isinstance(data[sec], dict):
        out.write(f'{sec} = {repr(data[sec])}\n\n')
        continue
    out.write(f'[{sec}]\n')
    for key, value in data[sec].items():
        if isinstance(value, bool):
            out.write(f'{key} = {str(value).lower()}\n')
        elif isinstance(value, str):
            out.write(f'{key} = \"{value}\"\n')
        elif isinstance(value, float):
            out.write(f'{key} = {value}\n')
        elif isinstance(value, int):
            out.write(f'{key} = {value}\n')
        elif isinstance(value, dict):
            # Nested dict (preset sections)
            pass
    out.write('\n')

with open('hyperparams.toml', 'w') as f:
    f.write(out.getvalue())

print()
print('hyperparams.toml updated.')
"
  echo
fi

# ── Show active config ──────────────────────────────────────────────────────

echo "=== Active configuration ==="
python3 -c "
import tomllib
with open('hyperparams.toml', 'rb') as f:
    data = tomllib.load(f)

def show(section, keys):
    d = data.get(section, {})
    for k in keys:
        if k in d:
            print(f'  {section}.{k} = {d[k]}')

show('game', ['board_size', 'swap_rule'])
show('net', ['channels', 'blocks', 'value_hidden'])
show('mcts', ['simulations', 'c_puct', 'dirichlet_alpha', 'dirichlet_epsilon',
              'solver_empty_threshold', 'use_virtual_connection'])
show('selfplay', ['parallel_games', 'temperature_moves', 'resign_threshold'])
show('train', ['hours', 'games_per_generation', 'batch_size', 'train_steps_per_generation',
               'learning_rate', 'lr_schedule', 'lr_min', 'lr_warmup_steps', 'grad_clip',
               'weight_decay', 'value_loss_weight', 'arena_games', 'arena_win_rate',
               'arena_simulations', 'replay_buffer_size', 'num_actors', 'seed'])
device = data.get('device', '(auto-detect)')
print(f'  device = {device}')
"
echo

if $DRY_RUN; then
  echo "=== Dry run — skipping build ==="
  exit 0
fi

# ── Build the container ─────────────────────────────────────────────────────

CONTAINERFILE="Containerfile"

# Build args
BUILD_ARGS=""
if $CUDA; then
  BUILD_ARGS="$BUILD_ARGS --build-arg CUDA_INDEX=https://download.pytorch.org/whl/cu121"
fi

echo "=== Building image: $IMAGE ==="
echo
echo "This will take a few minutes (mostly downloading PyTorch)."
echo "The smoke test runs inside the build to catch issues early."
echo

podman build \
  $BUILD_ARGS \
  -t "$IMAGE" \
  -f "$CONTAINERFILE" \
  .

echo
echo "=== Build complete ==="
echo
echo "Image:  $IMAGE"
echo
echo "To start training (CPU):"
echo "  mkdir -p checkpoints"
echo "  podman run -v ./checkpoints:/app/checkpoints $IMAGE"
echo
echo "To start training (GPU):"
echo "  mkdir -p checkpoints"
echo "  podman run --device nvidia.com/gpu=all -v ./checkpoints:/app/checkpoints $IMAGE"
echo
echo "To resume stopped training, run the same command — checkpoints persist."
echo "To override a setting at runtime:  podman run -e TRAIN_HOURS=12 ... $IMAGE"
echo "To monitor GPU usage:              watch -n1 nvidia-smi"
echo
echo "Logs can be piped:  podman run ... $IMAGE 2>&1 | tee train.log"
echo "Summary from logs:  ./training_summary.sh -f train.log"
