# NeuralEngine — a self-training Hex engine (AlphaZero-style)

A neural-network Hex engine that trains itself by self-play and plugs into the main backend as an
**external engine** (it both plays and analyses), following `../Backend/Engines.md`. Fixed **13×13**.

It is self-contained: it does **not** run the main stack. It reuses the backend's *rules* (a faithful
Python port of `HexBoard` / `WinDetection` / the swap rule) and the backend's *engine protocol* (one
WebSocket, `MakeMove`/`Swap` and `AnalyzeRequest`/`AnalysisResult`), but training is its own process so
it can be scp'd to a beefier box and run there.

> Python conventions: this subproject is idiomatic PyTorch/Python (snake_case, PEP8), not the TS app's
> PascalCase — the project conventions describe the TypeScript codebase. The Hex *rules* here mirror the
> backend exactly so the engine plays the identical game.

## How it works

Standard AlphaZero loop, sized for "a couple of hours" of compute:

1. **Self-play** — the current *best* network plays games against itself; PUCT MCTS (policy priors +
   value estimate, Dirichlet root noise, temperature) produces a search policy at each move. Every
   position is stored as `(planes, MCTS policy, eventual result)`.
2. **Train** — the network learns to predict the MCTS policy (cross-entropy) and the game result
   (value MSE) from a replay buffer of recent games.
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
| `hex/board.py` | 13×13 rules — adjacency, win detection, swap, action space (169 cells + swap). |
| `hex/bridges.py` | Bridge patterns, forced responses, virtual-connection check. |
| `hex/solver.py` | Bounded exact endgame solver (alpha-beta + transposition table). |
| `net/encoding.py` | Canonical board→planes and the canonical↔real action mapping. |
| `net/model.py` | Residual ConvNet with policy (170) + value heads. |
| `net/evaluator.py` | Batched network inference for MCTS. |
| `search/mcts.py` | PUCT MCTS — batched across games, terminal/solver-aware. |
| `train/selfplay.py` | Parallel batched self-play; GPU single-actor or CPU all-cores. |
| `train/replay_buffer.py` | Replay buffer with 180° symmetry augmentation. |
| `train/arena.py` | Candidate-vs-best gating matches. |
| `train/train.py` | The training loop + checkpointing (`python -m train.train`). |
| `engine/play_engine.py` | The deployed external engine (play + analysis). |
| `smoke_test.py` | Fast end-to-end check on a tiny board (run after setup). |

## Setup (laptop or VPS)

```bash
cd NeuralEngine
./setup.sh                       # venv + CPU torch + numpy + websockets
# On an NVIDIA GPU, install the CUDA build instead (setup.sh prints the exact command).
source .venv/bin/activate
python smoke_test.py             # ~seconds; exercises every component on a 5×5 board
```

Needs Python 3.10–3.12 (PyTorch wheels). The code mirrors the backend rules but has no other coupling.

## Train on a VPS

```bash
# 1. copy the project (model checkpoints are not needed — training creates them)
scp -r NeuralEngine user@vps:~/NeuralEngine

# 2. on the VPS
cd ~/NeuralEngine && ./setup.sh                         # GPU: install the CUDA torch build
source .venv/bin/activate && python smoke_test.py       # verify
TRAIN_HOURS=4 ./run_training.sh                         # uses GPU if present, else all CPU cores
#   (run under tmux/nohup so it survives logout)
```

It resumes from `checkpoints/latest.pt` if restarted, and the deployable model is always
`checkpoints/best.pt`. Watch the per-generation log: `samples`, `buffer`, `ploss`/`vloss`, `arena`
win-rate, and whether the new net was `PROMOTED`.

### Key knobs (env vars → `config.py`)

| Var | Default | Meaning |
|---|---|---|
| `TRAIN_HOURS` | 4 | Wall-clock training budget. |
| `DEVICE` | auto | `cuda` / `mps` / `cpu`. |
| `NUM_ACTORS` | auto | Self-play processes (CPU: cores−1; GPU: 1 big batched actor). |
| `MCTS_SIMS` | 200 | Simulations per self-play move (strength vs speed). |
| `PARALLEL_GAMES` | 64 | Games batched together per actor (GPU batch size). |
| `GAMES_PER_GEN` | 256 | Self-play games per generation. |
| `NET_CHANNELS` / `NET_BLOCKS` | 96 / 8 | Network width / depth. |
| `BATCH_SIZE` / `TRAIN_STEPS` | 512 / 400 | Optimiser batch and steps per generation. |
| `ARENA_GAMES` / `ARENA_WIN_RATE` | 40 / 0.55 | Gating match size / promotion threshold. |
| `SOLVER_EMPTIES` | 7 | Run the exact solver at ≤ this many empty cells. |

Bigger box → raise `MCTS_SIMS`, `PARALLEL_GAMES`, `NET_CHANNELS/BLOCKS`, `GAMES_PER_GEN`.

## Deploy the trained engine against the backend

```bash
# 1. register an external engine on the running backend (admin), with analysis enabled:
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

**On disk** — training writes only into `checkpoints/`:

- `best.pt` — the strongest network so far (what you deploy). A dict `{model, config, generation}`; a
  few MB to tens of MB depending on `NET_CHANNELS/BLOCKS`. **This is the whole "AI".**
- `latest.pt` — the most recent network + optimizer + replay metadata, for resuming. Larger than
  `best.pt` (it also holds optimizer state).
- `.gitignore` keeps the `.pt` files out of git (they're build artifacts).

There is no separate "weights folder" or dataset on disk — self-play data lives in memory in the replay
buffer during the run. The console log is your training record (generations, losses, arena win-rates).

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
