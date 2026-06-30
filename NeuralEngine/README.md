# NeuralEngine — a self-training Hex engine (AlphaZero-style) 

A neural-network Hex engine that trains itself by self-play and plugs into the main backend as an
**external engine** (it both plays and analyses), following `../Backend/Engines.md`. Default **13×13**,
configurable via `BOARD_SIZE` (the `cuda-affordable-11` / `cuda-5090` presets train **11×11**).

It is self-contained: it does **not** run the main stack. It reuses the backend's *rules* (a faithful
Python port of `HexBoard` / `WinDetection` / the swap rule) and the backend's *engine protocol* (one
WebSocket, `MakeMove`/`Swap` and `AnalyzeRequest`/`AnalysisResult`), but training is its own process,
packaged as a container image you run on a rented GPU box (e.g. RunPod) — see *Train with Docker* below.

> Python conventions: this subproject is idiomatic PyTorch/Python (snake_case, PEP8), not the TS app's
> PascalCase — the project conventions describe the TypeScript codebase. The Hex *rules* here mirror the
> backend exactly so the engine plays the identical game.

## How it works

Standard AlphaZero loop, sized for "a couple of hours" of compute:

1. **Self-play** — the current *best* network plays games against itself; PUCT MCTS (policy priors +
   value estimate, Dirichlet root noise, temperature) produces a search policy at each move. Every
   position is stored as `(planes, MCTS policy, eventual result)`. Every game plays to the last stone
   (no early resignation — all data is used).
2. **Train** — the network learns to predict the MCTS policy (cross-entropy) and the game result
   (value MSE) from a replay buffer of recent games. Uses **AMP mixed precision** and
   **torch.compile** on CUDA for fast training.
3. **Arena gate** — the freshly trained network only becomes the new *best* (and thus the self-play
   generator) if it beats the incumbent over a set of games. This is the "reward-based evolution".
4. Repeat until the time budget elapses, checkpointing throughout.

### The directive's "built-in features"

- **Lookahead** is the MCTS itself — `simulations` per move bounds both strength and the O(bⁿ) cost.
- **Solve endgames / find winning paths** — `hex/solver.py` is a bounded exact alpha-beta solver with a
  transposition table; it kicks in only when few cells remain (`SOLVER_EMPTIES`) so it stays cheap, and
  a solved node becomes an *exact* terminal in MCTS (won/lost endgames are then played perfectly).
- **Patterns (bridges, forks)** — `hex/bridges.py` recognises bridges and their carriers, plays the
  forced response when a bridge is intruded upon, and exposes an optimistic virtual-connection check
  used to order the solver and as an endgame signal. The network learns higher-level shapes itself.
- **Canonical orientation** — Hex's transpose+colour-swap symmetry is exploited so the network always
  sees "side-to-move connecting top↔bottom", roughly halving what it must learn.
- **Move list + temperature** — search returns a ranked move list; the deployed engine picks from it by
  `ENGINE_TEMPERATURE` (0 = strongest/most-visited, >0 = sample).

### The swap rule

Implemented as Hex's transpose+colour-swap symmetry rather than the backend's seat reassignment (see
`hex/board.py`): stealing the opening is equivalent to relabelling the board under that symmetry, which
keeps each colour's fixed connection direction intact — what an orientation-aware network needs. The two
are game-theoretically identical, so the trained engine swaps correctly against the real backend.

## Files

| Path | Role |
|---|---|
| `config.py` | Every hyperparameter; auto-detects CUDA/MPS/CPU and worker count. |
| `hyperparams.toml` | Single source of truth for all settings + hardware presets. |
| `hex/board.py` | N×N rules — adjacency, win detection, swap, action space (N² cells + swap). |
| `hex/bridges.py` | Bridge patterns, forced responses, virtual-connection check. |
| `hex/solver.py` | Bounded exact endgame solver (alpha-beta + transposition table). |
| `net/encoding.py` | Canonical board→planes and the canonical↔real action mapping. |
| `net/model.py` | Residual ConvNet with policy + value heads. |
| `net/evaluator.py` | Batched network inference for MCTS (softmax on GPU). |
| `search/mcts.py` | PUCT MCTS — batched across games, terminal/solver-aware, array-backed nodes. |
| `train/selfplay.py` | Parallel batched self-play; CPU search workers + GPU inference server, or CPU all-cores. |
| `train/inference_server.py` | Single-GPU batched inference server + `RemoteEvaluator` (CPU workers, one CUDA context). |
| `train/replay_buffer.py` | Replay buffer with 180° symmetry augmentation. |
| `train/arena.py` | Candidate-vs-best gating matches. |
| `train/train.py` | The training loop + checkpointing (`python -m train.train`). |
| `engine/play_engine.py` | The deployed external engine (play + analysis). |
| `smoke_test.py` | Fast end-to-end check (CPU, plus the GPU paths when a CUDA device is present; also a build guard). |
| `test_inference_server.py` | Asserts the inference server == a local evaluator (also a build guard). |

## Run the tests locally (optional)

Training runs in the container (below); the build already runs `smoke_test.py` and
`test_inference_server.py` as guards. To run them outside Docker, make a venv (Python 3.10–3.14):

```bash
cd NeuralEngine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python smoke_test.py             # ~seconds; exercises every component on a 5×5 board
python test_inference_server.py  # GPU inference server == local evaluator (runs on CPU here)
```

## Train with Docker (recommended)

The container bakes in `hyperparams.toml` (your config) and handles torch install (CPU or CUDA).

```bash
# Build with a hardware preset baked in:
./build_container.sh --preset cuda-balanced --cuda

# Or for a top-end VPS GPU:
./build_container.sh --preset cuda-moon --cuda

# Deploying one image across hosts whose driver varies (e.g. RunPod)? Pin the CUDA
# wheel to the LOWEST driver you'll land on — a newer driver always runs an older
# wheel, but a wheel newer than the driver silently falls back to CPU. cu121 (the
# default) is safe across the modern fleet; Blackwell (RTX 5090/B200) needs cu128:
./build_container.sh --preset cuda-5090 --cuda --cuda-wheel cu128

# A run that still lands on too-old a driver aborts loudly (it won't waste the GPU
# silently training on CPU). Set ALLOW_CPU=1 to force an intentional CPU run.

# Run (CPU):
mkdir -p checkpoints logs
docker run -v ./checkpoints:/app/checkpoints -v ./logs:/app/logs neuralengine

# Run (GPU):
docker run --gpus all -v ./checkpoints:/app/checkpoints -v ./logs:/app/logs neuralengine

# Resume stopped training — same command. Checkpoints and logs persist in bind mounts.
# Override a setting:  docker run -e TRAIN_HOURS=12 ... neuralengine
```

The container auto-detects docker vs podman. Logs are written to both stdout and the `logs/`
directory (one file per run, timestamped) so you can review all generations later.

### CUDA wheel & startup safety

Prebuilt images run on hosts whose driver you don't control (and on RunPod the driver can change
between runs), so the wheel is pinned to the **GPU**, not the build host — and two guards stop a
misbuilt image from silently wasting the GPU.

**Pin the CUDA wheel.** A torch wheel bundles a CUDA runtime *and* a fixed set of GPU architectures; a
wheel newer than the host driver falls back to CPU, and a wheel lacking the GPU's arch can't run a
single kernel.

- `CUDA_WHEEL` build arg (default **`cu121`**) runs on any modern driver via backward compatibility;
  override with `build_container.sh --cuda-wheel <tag>` or the CI **`cuda_wheel`** input.
- **Blackwell (RTX 5090/B200, sm_120) needs `cu128`** — the `cuda-5090` preset selects it automatically
  (locally and in CI) unless you pass a wheel explicitly.

**Two fail-loud startup guards** abort in seconds with a fix instead of crawling on CPU or crashing
deep in self-play:

- *Driver too old* — torch can't use the GPU → abort: rebuild with a lower `--cuda-wheel`.
- *Arch unsupported* — `cuda.is_available()` is `True` but every kernel fails (e.g. `cu121` on a 5090);
  a tiny probe conv catches it → abort: rebuild with `cu128`.
- Set **`ALLOW_CPU=1`** (with `DEVICE=cpu`) to intentionally train on CPU and skip both.

## Deploy via GHCR (CI-built image)

Instead of building locally and copying to the VPS, let GitHub Actions build the image and push
it to the GitHub Container Registry (GHCR); the VPS just pulls it.

**Build it** (workflow at the repo root: `.github/workflows/build-image.yml`; `NeuralEngine/` is its
build context):

- **Manual:** Actions → *Build and Push Image* → *Run workflow* → pick a preset (and CUDA on/off).
- **Auto:** push to the `images` branch — builds `hyperparams.toml` exactly as committed.

The chosen preset is baked into `hyperparams.toml` inside the image (via the repo-root
`.github/apply_preset.py`).
Each build is pushed under three tags: `latest`, the preset name (or `images`), and `sha-<commit>`.

**One-time setup — make the package public** (so the VPS can pull without a login):
GitHub → your profile → *Packages* → `neural-hex` → *Package settings* → *Change visibility* → **Public**.
(The image bakes in only `hyperparams.toml` — no secrets. Training credentials are passed at runtime.)

**Pull and run on the VPS** (no `docker login` needed once the package is public):

```bash
docker pull ghcr.io/<owner>/neural-hex:latest    # <owner> lowercase, e.g. nomi-y
mkdir -p checkpoints logs
docker run --gpus all -v ./checkpoints:/app/checkpoints -v ./logs:/app/logs \
  ghcr.io/<owner>/neural-hex:latest
# Pin a specific config:  ...neural-hex:cuda-affordable-11   (or :sha-<commit>)
```

If you keep the package private instead, log in first with a PAT that has `read:packages`:
`echo "$GHCR_PAT" | docker login ghcr.io -u <owner> --password-stdin`.

## Hardware presets

Presets are defined in `hyperparams.toml` and can be applied at build time:

| Preset | Description |
|---|---|
| `smoke` | Tiny 32×4 net, 60 sims, CPU — verify the pipeline in seconds. |
| `cpu-balanced` | 64×6 net, 200 sims, all CPU cores — real engine on a many-core box. |
| `cuda-balanced` | 96×8 net, 250 sims — fits ~6-8 GB VRAM (RTX 3060/4060, T4). |
| `cuda-large` | 128×10 net, 400 sims — RTX 4090 / A10 class. |
| `cuda-moon` | 160×16 net, 600 sims — saturate an A100/H100/RTX PRO 6000. |
| `cuda-affordable-11` | 96×8 net, 200 sims, **11×11** board — affordable ~24h run (RTX 6000 Ada / A6000 / A40). |
| `cuda-5090` | 256×20 SE net, 256 sims, **11×11** — RTX 5090 (sized to keep the GPU busy); auto-builds **cu128** (Blackwell sm_120). |

> The **cuda-balanced** preset uses the same net size as the default TOML, so it can resume
> existing checkpoints. Other presets start fresh. The `*-11` / `5090` presets train an **11×11**
> board — register the deployed engine with `MinBoardSize`/`MaxBoardSize` 11 to match.

## CUDA performance features

The training loop automatically enables these on CUDA (no config needed):

- **cuDNN auto-tuner** (`benchmark=True`) — picks the fastest convolution algorithm for your GPU.
- **TF32** — uses TensorFloat32 on Ampere+ GPUs for faster matmuls.
- **AMP mixed precision** — fp16 where safe, ~1.5–2× faster training steps, less VRAM.
- **torch.compile** — JIT-compiles the network into fused CUDA kernels (~20–50% faster forward pass).
- **Non-blocking transfers** — async CPU→GPU data movement (overlaps compute with transfer).
- **MCTS Node arrays** — fixed-size arrays instead of Python dicts for faster selection/backup.
- **GPU softmax** — evaluator keeps softmax on GPU instead of a Python loop.
- **Work-stealing chunks** — 2–3× more self-play chunks than workers so fast cores grab extra work.

### Self-play: CPU search workers + one GPU inference server

Self-play here is **CPU-bound** — MCTS pegs the cores while each network forward pass is tiny. The
naive "put the net on the GPU in every worker" approach creates one CUDA context per worker
(~0.5 GB+ each), so with ≈ one worker per core it OOMs any card (even 80 GB) the moment you run on a
high-vCPU box. Two ways to avoid that, selected automatically:

- **CUDA (default): one inference server.** A single GPU process hosts the net(s); the many CPU
  workers do tree search and ship leaf positions to it, which **batches the forward across all
  workers**. One CUDA context (no OOM), and the network forward runs on the otherwise-idle GPU so
  cores stay on search. Arena's two nets (candidate, best) are served by the same process.
- **CPU fallback.** Set **`INFERENCE_SERVER=0`** and workers evaluate the (small) net on CPU while
  the GPU is used only for the training step. Robust and simple; slightly more CPU per worker.

Knobs (all env, so they work with baked images):

| Env | Default | Meaning |
|-----|---------|---------|
| `INFERENCE_SERVER` | on for CUDA | `0`/`1` — GPU inference server vs per-worker CPU eval. |
| `INFERENCE_MAX_BATCH` | 2048 | Max leaves the server coalesces into one forward (bounds VRAM). |
| `INFERENCE_RESULT_TIMEOUT` | 300 | Seconds a worker waits for a server result before erroring (dead-server guard). |
| `SELFPLAY_DEVICE` | cpu (CUDA box) | Worker eval device when the server is **off**. |
| `NUM_ACTORS` | cores − 1 | Self-play / arena worker processes. |

> The proper scaling path is the inference server; `SELFPLAY_DEVICE=cuda` (per-worker GPU eval) is
> only sane with a tiny `NUM_ACTORS` and is superseded by the server. CUDA MPS is no longer required.

## Key configuration (hyperparams.toml)

| Setting | Default | Meaning |
|---|---|---|
| `game.board_size` | 13 | Board size (changing requires a fresh start). |
| `net.channels` / `net.blocks` | 96 / 8 | Network width / depth. |
| `mcts.simulations` | 250 | Simulations per self-play move. |
| `selfplay.parallel_games` | 32 | Games batched per GPU forward pass. |
| `train.hours` | 48 | Wall-clock training budget. |
| `train.games_per_generation` | 256 | Self-play games per generation. |
| `train.batch_size` | 512 | Training batch size. |
| `train.train_steps_per_generation` | 500 | Optimizer steps per generation. |
| `train.learning_rate` | 0.001 | Initial learning rate (Adam). |
| `train.lr_schedule` | cosine | cosine / step / constant. |
| `train.arena_games` / `arena_win_rate` | 64 / 0.55 | Gating match size / promotion threshold. |
| `train.save_every_checkpoint` | false | Save each gen as `checkpoints/gen_NNNN.pt`. |
| `train.log_dir` | logs | Directory for persistent log files (bind-mount it). |
| `mcts.solver_empty_threshold` | 7 | Run exact solver at ≤ this many empty cells. |

All values can be overridden at runtime with environment variables (e.g. `-e TRAIN_HOURS=12`).

## Deploy the trained engine against the backend

```bash
# 1. register an external engine on the running backend (admin), with analysis enabled:
#    (set Min/MaxBoardSize to the board you trained — e.g. 11 for the *-11 / cuda-5090 presets)
curl -X POST http://<backend>/Admin/Api/Engines/External \
  -H "X-Admin-Session: <token>" -H "Content-Type: application/json" \
  -d '{ "Name": "NeuralHex", "MinBoardSize": 13, "MaxBoardSize": 13, "SupportsAnalysis": true }'
#   -> { "EngineId": "...", "Token": "..." }

# 2. run the engine (loads checkpoints/best.pt):
ENGINE_WS=ws://<backend>:3001 ENGINE_ID=<id> ENGINE_TOKEN=<token> \
  MODEL_PATH=checkpoints/best.pt python -m engine.play_engine
```

Or use the chooser, which lists the checkpoints and launches the one you pick:

```bash
cp engine.env.example engine.env   # then put your EngineId/Token in engine.env (gitignored)
./run_engine.sh                     # lists checkpoints/*.pt, prompts for a number
./run_engine.sh 1                   # or pick non-interactively
```

`engine.env` holds `ENGINE_ID` / `ENGINE_TOKEN` / `ENGINE_WS` and is gitignored — keep your token out
of version control (re-register on the backend if it ever leaks).

Now challenge "NeuralHex" from the web client (13×13), and use **Analyse** on a finished 13×13 match to
get its evaluation overlaid on the replay. `ENGINE_SIMS` / `ENGINE_MOVE_SECONDS` / `ENGINE_TEMPERATURE`
tune play strength, thinking time, and determinism.

## What a trained engine looks like (filesystem + behaviour)

**On disk** — training writes into `checkpoints/` and `logs/`:

- `best.pt` — the strongest network so far (what you deploy). A dict `{model, config, generation}`; a
  few MB to tens of MB depending on `NET_CHANNELS/BLOCKS`. **This is the whole "AI".**
- `latest.pt` — the most recent network + optimizer + replay metadata, for resuming. Larger than
  `best.pt` (it also holds optimizer state).
- `gen_NNNN.pt` — individual generation snapshots (only when `save_every_checkpoint = true`).
- `logs/train_YYYYMMDD_HHMMSS.log` — timestamped training logs.

There is no separate "weights folder" or dataset on disk — self-play data lives in memory in the replay
buffer during the run. The console log and persisted log files are your training record.

**In behaviour**, as it trains:

- *Early (random net)*: near-random moves, scattered on the board; loses quickly.
- *After the first promotions*: it starts occupying the centre, building short chains, and — thanks to
  the solver — never misplays a position that's a few moves from a win/loss.
- *With more generations*: recognisable Hex strategy emerges — playing and respecting **bridges**,
  contesting the **short diagonal**, blocking the opponent's connection while extending its own, and
  making sensible **swap** decisions on strong openings. It won't be MoHex-strength in a few hours on
  13×13 (that game is hard), but it should clearly beat the bundled `Heuristic` engine and give a club
  player a real game.
- *As an analyst*: the value head yields a win-probability curve across a match, the policy/visit counts
  give a ranked best-move list, and the badges flag the moves that swung the evaluation — exactly the
  overlay the replay viewer already draws.
- *Move feel*: a single move is one MCTS search (`ENGINE_SIMS`, default 400) — sub-second to a few
  seconds depending on hardware; raise/lower `ENGINE_SIMS` to trade strength for speed.

A quick sign training is working: the per-generation `arena` win-rate periodically exceeds the
promotion threshold (so `best.pt`'s `generation` keeps advancing), and `vloss` falls as the value head
learns to predict outcomes.
