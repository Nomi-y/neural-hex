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
#    ./build_container.sh --cuda                 # install CUDA torch wheel (cu121, fleet-safe)
#    ./build_container.sh --cuda --cuda-wheel cu128  # Blackwell (RTX 5090/B200)
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
CUDA_WHEEL="cu121"
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
    --cuda-wheel)
      CUDA_WHEEL="$2"; CUDA=true; shift 2 ;;
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
    out.write(f'[{sec}]\\n')
    for key, value in data[sec].items():
        if isinstance(value, bool):
            out.write(f'{key} = {str(value).lower()}\\n')
        elif isinstance(value, str):
            out.write(f'{key} = \"{value}\"\\n')
        elif isinstance(value, float):
            out.write(f'{key} = {value}\\n')
        elif isinstance(value, int):
            out.write(f'{key} = {value}\\n')
        elif isinstance(value, dict):
            # Nested dict — write as inline table (valid TOML for preset values).
            pairs = []
            for k, v in value.items():
                if isinstance(v, bool):
                    pairs.append(f'{k} = {str(v).lower()}')
                elif isinstance(v, str):
                    pairs.append(f'{k} = \"{v}\"')
                elif isinstance(v, float):
                    if v == int(v) and abs(v) < 1e12:
                        pairs.append(f'{k} = {v}')
                    else:
                        pairs.append(f'{k} = {v}')
                elif isinstance(v, int):
                    pairs.append(f'{k} = {v}')
                elif isinstance(v, dict):
                    # Nested nested dict (e.g. train = { hours = 1.0, ... })
                    inner = []
                    for ik, iv in v.items():
                        if isinstance(iv, bool):
                            inner.append(f'{ik} = {str(iv).lower()}')
                        elif isinstance(iv, str):
                            inner.append(f'{ik} = \"{iv}\"')
                        elif isinstance(iv, (int, float)):
                            inner.append(f'{ik} = {iv}')
                    pairs.append(f'{k} = {{ {', '.join(inner)} }}')
            out.write(f'{key} = {{ {', '.join(pairs)} }}\\n')
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
               'arena_simulations', 'replay_buffer_size', 'num_actors', 'seed',
               'save_every_checkpoint', 'log_dir'])
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

# Auto-detect container runtime: prefer podman, fall back to docker.
CONTAINER_CMD="podman"
if ! command -v podman &>/dev/null; then
  if command -v docker &>/dev/null; then
    CONTAINER_CMD="docker"
  else
    echo "ERROR: Neither podman nor docker found.  Install one of them." >&2
    exit 1
  fi
fi

# Build args
BUILD_ARGS=""
if $CUDA; then
  BUILD_ARGS="$BUILD_ARGS --build-arg CUDA_BUILD=true --build-arg CUDA_WHEEL=$CUDA_WHEEL"
  echo "=== CUDA torch wheel: $CUDA_WHEEL (must be <= host driver's CUDA version; check nvidia-smi) ==="
  echo
fi

echo "=== Building image: $IMAGE  (using $CONTAINER_CMD) ==="
echo
echo "This will take a few minutes (mostly downloading PyTorch)."
echo "The smoke test runs inside the build to catch issues early."
echo

$CONTAINER_CMD build \
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
echo "  mkdir -p checkpoints logs"
echo "  $CONTAINER_CMD run -v ./checkpoints:/app/checkpoints -v ./logs:/app/logs $IMAGE"
echo
echo "To start training (GPU):"
echo "  mkdir -p checkpoints logs"
if [ "$CONTAINER_CMD" = "docker" ]; then
  echo "  $CONTAINER_CMD run --gpus all -v ./checkpoints:/app/checkpoints -v ./logs:/app/logs $IMAGE"
else
  echo "  $CONTAINER_CMD run --device nvidia.com/gpu=all -v ./checkpoints:/app/checkpoints -v ./logs:/app/logs $IMAGE"
fi
echo
echo "To resume stopped training, run the same command — checkpoints persist."
echo "To override a setting at runtime:  $CONTAINER_CMD run -e TRAIN_HOURS=12 ... $IMAGE"
echo "To monitor GPU usage:              watch -n1 nvidia-smi"
echo
echo "Logs can be piped:  $CONTAINER_CMD run ... $IMAGE 2>&1 | tee train.log"
echo "Summary from logs:  ./training_summary.sh -f train.log"
