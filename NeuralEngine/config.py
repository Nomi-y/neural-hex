"""Central configuration for the self-training Hex engine.

Every knob lives here (defaults) and in hyperparams.toml (single source of truth).
Values are resolved in this priority order:
  1. environment variable (overrides everything — handy for one-off runs)
  2. hyperparams.toml (baked into the container image at build time)
  3. hardcoded defaults (this file — safe fallback values)

Device/worker selection auto-detects CUDA and core count, so the same code
saturates either a single GPU box or a many-core CPU box.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional

# ── TOML defaults layer ──────────────────────────────────────────────────────

_TOML: dict = {}

def _load_toml() -> dict:
    """Load hyperparams.toml as a flat dotted-key dict, e.g. {"net.channels": 96}."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib       # Python <3.11 fallback
        except ImportError:
            return {}
    # Look next to this file; also check the CWD (container may copy it elsewhere).
    candidates = [
        os.path.join(os.path.dirname(__file__), "hyperparams.toml"),
        os.path.join(os.getcwd(), "hyperparams.toml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            flat: dict = {}
            for section, values in raw.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        flat[f"{section}.{key}"] = value
            return flat
    return {}

_TOML = _load_toml()


def _get(key: str, default, toml_path: Optional[str] = None, conv=None):
    """Resolve a config value: env var → TOML → hardcoded default."""
    raw = os.environ.get(key)
    if raw is not None and raw != "":
        return conv(raw) if conv else raw
    if toml_path:
        val = _TOML.get(toml_path)
        if val is not None:
            return conv(val) if conv else val
    return default


def _env_int(name: str, default: int, toml: Optional[str] = None) -> int:
    return _get(name, default, toml, int)


def _env_float(name: str, default: float, toml: Optional[str] = None) -> float:
    return _get(name, default, toml, float)


def _env_str(name: str, default: str, toml: Optional[str] = None) -> str:
    return _get(name, default, toml, str)


def _env_bool(name: str, default: bool, toml: Optional[str] = None) -> bool:
    raw = os.environ.get(name)
    if raw is not None and raw != "":
        return raw.lower() != "false"
    if toml:
        val = _TOML.get(toml)
        if val is not None:
            return bool(val)
    return default


def _detect_device() -> str:
    override = os.environ.get("DEVICE")
    if override:
        return override
    toml_dev = _TOML.get("device")
    if toml_dev:
        return str(toml_dev)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _cgroup_cpu_quota() -> Optional[float]:
    """CPU limit (in cores) from the cgroup CFS quota, or None if unlimited/unknown. Containers often
    cap CPU *time* via a quota without shrinking the affinity mask, so cpu_count()/affinity miss it."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max" and int(period) > 0:
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """CPUs actually usable by this process — respects CPU affinity / cpuset AND a CFS quota, so it
    returns the container's allocation (e.g. 16 vCPU) rather than the host's logical core count
    (e.g. 64). Without this, a 16-vCPU box spawns ~63 workers and thrashes."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        n = min(n, max(1, int(quota)))
    return max(1, n)


# ── Config dataclasses ───────────────────────────────────────────────────────

@dataclass
class GameConfig:
    board_size: int = _env_int("BOARD_SIZE", 13, "game.board_size")
    swap_rule: bool = _env_bool("SWAP_RULE", True, "game.swap_rule")

    @property
    def num_cells(self) -> int:
        return self.board_size * self.board_size

    @property
    def num_actions(self) -> int:
        return self.num_cells + 1

    @property
    def swap_action(self) -> int:
        return self.num_cells


@dataclass
class NetConfig:
    channels: int = _env_int("NET_CHANNELS", 96, "net.channels")
    blocks: int = _env_int("NET_BLOCKS", 8, "net.blocks")
    in_planes: int = 5  # derived, not user-tunable
    value_hidden: int = _env_int("NET_VALUE_HIDDEN", 128, "net.value_hidden")
    use_se: bool = _env_bool("NET_SE", False, "net.use_se")  # squeeze-excitation blocks (board-wide context)


@dataclass
class MctsConfig:
    simulations: int = _env_int("MCTS_SIMS", 200, "mcts.simulations")
    c_puct: float = _env_float("C_PUCT", 1.5, "mcts.c_puct")
    dirichlet_alpha: float = _env_float("DIRICHLET_ALPHA", 0.2, "mcts.dirichlet_alpha")
    dirichlet_epsilon: float = _env_float("DIRICHLET_EPS", 0.25, "mcts.dirichlet_epsilon")
    solver_empty_threshold: int = _env_int("SOLVER_EMPTIES", 7, "mcts.solver_empty_threshold")
    solver_node_budget: int = _env_int("SOLVER_NODES", 200_000, "mcts.solver_node_budget")
    use_virtual_connection: bool = _env_bool("USE_VC", True, "mcts.use_virtual_connection")
    fpu_reduction: float = _env_float("FPU", 0.0, "mcts.fpu_reduction")  # 0 disables FPU (legacy q=0)
    reuse_tree: bool = _env_bool("REUSE_TREE", False, "mcts.reuse_tree")  # carry MCTS subtree across self-play moves
    pipeline_shards: int = _env_int("PIPELINE_SHARDS", 1, "mcts.pipeline_shards")  # GPU inference pipelining depth


@dataclass
class SelfPlayConfig:
    parallel_games: int = _env_int("PARALLEL_GAMES", 64, "selfplay.parallel_games")
    temperature_moves: int = _env_int("TEMPERATURE_MOVES", 20, "selfplay.temperature_moves")
    temperature: float = _env_float("TEMPERATURE", 1.0, "selfplay.temperature")
    resign_enabled: bool = _env_bool("RESIGN_ENABLED", False, "selfplay.resign_enabled")
    resign_threshold: float = _env_float("RESIGN_THRESHOLD", -0.92, "selfplay.resign_threshold")
    resign_min_ply: int = _env_int("RESIGN_MIN_PLY", 12, "selfplay.resign_min_ply")
    resign_playthrough: float = _env_float("RESIGN_PLAYTHROUGH", 0.1, "selfplay.resign_playthrough")


@dataclass
class TrainConfig:
    hours: float = _env_float("TRAIN_HOURS", 4.0, "train.hours")
    max_generations: int = _env_int("TRAIN_MAX_GENS", 0, "train.max_generations")  # 0 = unlimited (budget only)
    num_actors: int = _env_int("NUM_ACTORS", 0, "train.num_actors")
    games_per_generation: int = _env_int("GAMES_PER_GEN", 256, "train.games_per_generation")
    replay_buffer_size: int = _env_int("REPLAY_BUFFER", 200_000, "train.replay_buffer_size")
    batch_size: int = _env_int("BATCH_SIZE", 512, "train.batch_size")
    train_steps_per_generation: int = _env_int("TRAIN_STEPS", 400, "train.train_steps_per_generation")

    # Optimizer
    learning_rate: float = _env_float("LR", 1e-3, "train.learning_rate")
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-4, "train.weight_decay")
    grad_clip: float = _env_float("GRAD_CLIP", 1.0, "train.grad_clip")

    # LR schedule
    lr_schedule: str = _env_str("LR_SCHEDULE", "cosine", "train.lr_schedule")
    lr_min: float = _env_float("LR_MIN", 1e-5, "train.lr_min")
    lr_warmup_steps: int = _env_int("LR_WARMUP_STEPS", 100, "train.lr_warmup_steps")
    lr_decay: float = _env_float("LR_DECAY", 0.5, "train.lr_decay")
    lr_step_gens: int = _env_int("LR_STEP_GENS", 20, "train.lr_step_gens")

    # Loss
    value_loss_weight: float = _env_float("VALUE_LOSS_WEIGHT", 1.0, "train.value_loss_weight")

    # Arena gating
    arena_games: int = _env_int("ARENA_GAMES", 40, "train.arena_games")
    arena_win_rate: float = _env_float("ARENA_WIN_RATE", 0.55, "train.arena_win_rate")
    arena_simulations: int = _env_int("ARENA_SIMS", 120, "train.arena_simulations")

    # Watchdog
    selfplay_timeout: float = _env_float("SELFPLAY_TIMEOUT", 1800.0, "train.selfplay_timeout")
    arena_timeout: float = _env_float("ARENA_TIMEOUT", 900.0, "train.arena_timeout")

    checkpoint_dir: str = _env_str("CHECKPOINT_DIR",
        os.path.join(os.path.dirname(__file__), "checkpoints"), "train.checkpoint_dir")
    save_every_checkpoint: bool = _env_bool("SAVE_EVERY_CKPT", False, "train.save_every_checkpoint")
    log_dir: str = _env_str("LOG_DIR", "logs", "train.log_dir")
    seed: int = _env_int("SEED", 0, "train.seed")


@dataclass
class EngineConfig:
    simulations: int = _env_int("ENGINE_SIMS", 400, "engine.simulations")
    temperature: float = _env_float("ENGINE_TEMPERATURE", 0.0, "engine.temperature")
    move_budget_seconds: float = _env_float("ENGINE_MOVE_SECONDS", 5.0, "engine.move_budget_seconds")
    model_path: str = os.environ.get("MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt"))
    backend_ws: str = _env_str("ENGINE_WS", "ws://localhost:3001", "engine.backend_ws")
    engine_id: str = os.environ.get("ENGINE_ID", "")
    token: str = os.environ.get("ENGINE_TOKEN", "")


@dataclass
class LoggingConfig:
    """Logging intervals — all env-overridable, all in hyperparams.toml [logging].
    No preset overrides for these; they're runtime-tuning knobs, not hardware-specific."""
    util_interval: float = _env_float("HW_LOG_INTERVAL", 60.0, "logging.util_interval")
    infer_heartbeat: float = _env_float("INFERENCE_LOG_EVERY", 30.0, "logging.infer_heartbeat")
    selfplay_progress_interval: float = _env_float("SELFPLAY_PROGRESS_INTERVAL", 30.0, "logging.selfplay_progress_interval")
    train_log_interval: float = _env_float("TRAIN_LOG_INTERVAL", 120.0, "logging.train_log_interval")


@dataclass
class Config:
    game: GameConfig = field(default_factory=GameConfig)
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = field(default_factory=_detect_device)

    def resolve_actors(self) -> int:
        """How many self-play/arena worker processes to launch — one CPU search worker per usable core.

        Uses available_cpus() (cgroup/affinity-aware), NOT mp.cpu_count(): on a 16-vCPU container the
        host's 64 logical cores would otherwise spawn 63 workers and thrash. Reserve one core for the
        main loop, plus one per GPU inference server when it's on (a CPU-starved server can't keep its
        GPU fed) — so on a multi-GPU box, where CPUs scale with GPUs, the reserve scales too. MPS
        (Apple) stays single-actor. Override with NUM_ACTORS."""
        if self.train.num_actors > 0:
            return self.train.num_actors
        if self.device == "mps":
            return 1
        reserve = (1 + self.num_gpus()) if self.use_inference_server() else 1
        return max(1, available_cpus() - reserve)

    def worker_eval_device(self) -> str:
        """Inference device for the fanned-out self-play / arena WORKER processes.

        Self-play here is CPU-bound (MCTS pegs the cores; the GPU idles during search), and
        resolve_actors() launches ≈ one worker per core. Each worker that touches CUDA creates
        its own CUDA context (~0.5 GB+ each: runtime + cuDNN/cuBLAS + caching allocator + Triton),
        so fanning hundreds of them across a single GPU OOMs the card regardless of its size —
        even 80 GB. So workers evaluate the (small) net on CPU while training keeps cfg.device.
        Override with SELFPLAY_DEVICE to force GPU self-play (only sane with a small NUM_ACTORS)."""
        override = os.environ.get("SELFPLAY_DEVICE")
        if override:
            return override
        if self.device == "cuda":
            return "cpu"
        return self.device

    def use_inference_server(self) -> bool:
        """Whether fan-out self-play/arena routes leaf evaluations through ONE GPU inference server
        (CPU workers do MCTS; a single GPU process batches the network forward across all of them).

        Default on for CUDA: it uses the GPU without a CUDA context per worker, and moves the forward
        off the CPU so cores stay on search. INFERENCE_SERVER=0/1 forces it — set 0 to fall back to the
        per-worker CPU evaluators (worker_eval_device). Irrelevant on CPU/MPS (no GPU to centralise)."""
        override = os.environ.get("INFERENCE_SERVER")
        if override is not None:
            return override.strip().lower() in ("1", "true", "yes")
        return self.device == "cuda"

    def num_gpus(self) -> int:
        """Number of GPUs available for inference servers. Auto-detected via
        torch.cuda.device_count(); override with NUM_GPUS env var.  On CPU/MPS
        returns 1 (single server running on that device)."""
        override = os.environ.get("NUM_GPUS")
        if override:
            return max(1, int(override))
        if self.device != "cuda":
            return 1
        try:
            import torch
            if torch.cuda.is_available():
                return max(1, torch.cuda.device_count())
        except Exception:
            pass
        return 1


def load() -> Config:
    return Config()
