"""Central configuration for the self-training Hex engine.

Every knob lives here so the project can be scp'd to a VPS and tuned in one place. Values are read
from environment variables where it is useful to override them per machine (e.g. board, worker counts,
time budget), but sensible auto-detected defaults mean it also just runs.

Device/worker selection auto-detects CUDA and core count, so the same code saturates either a single
GPU box or a many-core CPU box (the "use all resources" requirement).
"""

from __future__ import annotations

import os
import multiprocessing as mp
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _detect_device() -> str:
    override = os.environ.get("DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class GameConfig:
    # Fixed 13x13 per the directive: large enough to never be solved, small enough to self-train.
    board_size: int = _env_int("BOARD_SIZE", 13)
    swap_rule: bool = os.environ.get("SWAP_RULE", "true").lower() != "false"

    @property
    def num_cells(self) -> int:
        return self.board_size * self.board_size

    @property
    def num_actions(self) -> int:
        # One action per cell plus a single "swap" action (index == num_cells).
        return self.num_cells + 1

    @property
    def swap_action(self) -> int:
        return self.num_cells


@dataclass
class NetConfig:
    channels: int = _env_int("NET_CHANNELS", 96)
    blocks: int = _env_int("NET_BLOCKS", 8)
    # Input planes: my stones, opponent stones, my-edge mask, opp-edge mask, ones-bias. See encoding.py.
    in_planes: int = 5
    value_hidden: int = _env_int("NET_VALUE_HIDDEN", 128)


@dataclass
class MctsConfig:
    # Simulations per move. The dominant cost; bounds both strength and thinking time. Self-play uses
    # this; the deployed engine has its own (see EngineConfig) so play stays responsive.
    simulations: int = _env_int("MCTS_SIMS", 200)
    c_puct: float = _env_float("C_PUCT", 1.5)
    # Dirichlet exploration noise mixed into the root prior during self-play (AlphaZero defaults,
    # alpha scaled down a little for the larger 13x13 action space).
    dirichlet_alpha: float = _env_float("DIRICHLET_ALPHA", 0.2)
    dirichlet_epsilon: float = _env_float("DIRICHLET_EPS", 0.25)
    # Run the exact endgame solver at a node once empties drop to here (bounded, so O(b^n) stays small).
    solver_empty_threshold: int = _env_int("SOLVER_EMPTIES", 7)
    solver_node_budget: int = _env_int("SOLVER_NODES", 200_000)
    # Use the bridge virtual-connection check as a fast terminal/value signal.
    use_virtual_connection: bool = os.environ.get("USE_VC", "true").lower() != "false"


@dataclass
class SelfPlayConfig:
    # Games stepped in lockstep inside one actor so their leaf evaluations batch into one NN call
    # (this is what fills a GPU). On CPU boxes, many actors run instead (see TrainConfig.num_actors).
    parallel_games: int = _env_int("PARALLEL_GAMES", 64)
    # Plies for which moves are sampled with temperature 1 (exploration) before switching to greedy.
    temperature_moves: int = _env_int("TEMPERATURE_MOVES", 20)
    temperature: float = _env_float("TEMPERATURE", 1.0)
    # Resign a hopeless self-play game early to save compute (disabled if 0). Value is for side to move.
    resign_threshold: float = _env_float("RESIGN_THRESHOLD", -0.92)
    resign_min_ply: int = _env_int("RESIGN_MIN_PLY", 12)


@dataclass
class TrainConfig:
    # Wall-clock budget; the loop self-play/train/gate until this elapses, checkpointing throughout, so
    # the run can be stopped at any time and the latest best.pt deployed. "A couple of hours" by default.
    hours: float = _env_float("TRAIN_HOURS", 4.0)
    # Actors = self-play processes. On GPU one big-batch actor is usually best; on CPU, all cores.
    num_actors: int = _env_int("NUM_ACTORS", 0)  # 0 => auto (see resolve())
    games_per_generation: int = _env_int("GAMES_PER_GEN", 256)
    replay_buffer_size: int = _env_int("REPLAY_BUFFER", 200_000)
    batch_size: int = _env_int("BATCH_SIZE", 512)
    train_steps_per_generation: int = _env_int("TRAIN_STEPS", 400)
    learning_rate: float = _env_float("LR", 1e-3)
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-4)
    value_loss_weight: float = _env_float("VALUE_LOSS_WEIGHT", 1.0)
    # Arena gating: a freshly trained net replaces "best" only if it beats it by this win rate.
    arena_games: int = _env_int("ARENA_GAMES", 40)
    arena_win_rate: float = _env_float("ARENA_WIN_RATE", 0.55)
    arena_simulations: int = _env_int("ARENA_SIMS", 120)
    checkpoint_dir: str = os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))
    seed: int = _env_int("SEED", 0)


@dataclass
class EngineConfig:
    # The deployed external engine: keep thinking time modest (directive). Fewer sims than training.
    simulations: int = _env_int("ENGINE_SIMS", 400)
    # Play temperature: 0 => always the most-visited move; >0 => sample the move list (the directive's
    # "list of next moves where temperature chooses the next move").
    temperature: float = _env_float("ENGINE_TEMPERATURE", 0.0)
    move_budget_seconds: float = _env_float("ENGINE_MOVE_SECONDS", 5.0)
    model_path: str = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt"))
    backend_ws: str = os.environ.get("ENGINE_WS", "ws://localhost:3001")
    engine_id: str = os.environ.get("ENGINE_ID", "")
    token: str = os.environ.get("ENGINE_TOKEN", "")


@dataclass
class Config:
    game: GameConfig = field(default_factory=GameConfig)
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    device: str = field(default_factory=_detect_device)

    def resolve_actors(self) -> int:
        """How many self-play processes to launch. Auto: 1 on GPU/MPS (one big batched actor saturates
        it), else (cores - 1) on CPU so every core does self-play while one stays free for training."""
        if self.train.num_actors > 0:
            return self.train.num_actors
        if self.device in ("cuda", "mps"):
            return 1
        return max(1, mp.cpu_count() - 1)


def load() -> Config:
    return Config()
