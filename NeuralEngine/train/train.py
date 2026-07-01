"""Self-play reinforcement-learning loop (AlphaZero-style) for Hex.

Each generation:
  1. the *best* network generates self-play games (exploration via Dirichlet noise + temperature),
  2. the *current* network trains on a replay buffer of recent games (policy cross-entropy + value MSE),
  3. an arena gates promotion: the current network only becomes the new best if it beats the incumbent,
  4. checkpoints are written (latest.pt always; best.pt on promotion; gen_N.pt if save_every_checkpoint).

The loop runs until the wall-clock budget (TRAIN_HOURS) elapses and can be stopped/resumed at any time;
the deployed engine simply loads checkpoints/best.pt. Run via `python -m train.train` (the container entrypoint).
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

# Allow `python -m train.train` and `python train/train.py` alike.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load, Config, available_cpus
from net.model import build_net, BareModule, CleanStateDict
from train.replay_buffer import ReplayBuffer
from train import selfplay, arena
from train.clock import log, set_start, offset_str, set_gen
from train.hardware_monitor import HardwareMonitor

LATEST = "latest.pt"
BEST = "best.pt"


def _setup_cuda() -> None:
    """One-time CUDA performance knobs.  Safe to call even without a GPU."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True


def _gpu_memory_str() -> str:
    """Return a compact VRAM summary string, or '' if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return ""
    try:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return f"VRAM: {allocated / (1024**3):.1f}G allocated, {reserved / (1024**3):.1f}G reserved"
    except Exception:
        return ""


def _log_gpu_device() -> None:
    """Log CUDA device properties once at startup."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        total_gb = p.total_memory / (1024 ** 3)
        log(f"[train]   cuda:{i}  {p.name}  {total_gb:.1f}GB  compute {p.major}.{p.minor}")


def _gpu_hardware_present() -> bool:
    """True if an NVIDIA GPU is visible to the OS, regardless of whether torch can use it.
    Independent of the torch/driver CUDA check, so it still fires when a wheel/driver
    mismatch has disabled CUDA inside torch."""
    import glob
    if glob.glob("/proc/driver/nvidia/gpus/*") or glob.glob("/dev/nvidia[0-9]*"):
        return True
    import shutil, subprocess
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and "GPU " in out.stdout:
                return True
        except Exception:
            pass
    return False


def _assert_cuda_usable() -> None:
    """torch.cuda.is_available() can be True yet every kernel fail — e.g. a cu121 wheel on a
    Blackwell sm_120 GPU (RTX 5090): the arch isn't in the wheel's kernel list, so the real launch
    dies with 'no kernel image is available'. Force a tiny real conv now and, if it fails, abort with
    a rebuild hint instead of crashing deep in self-play."""
    try:
        import torch.nn.functional as F
        x = torch.zeros(1, 1, 4, 4, device="cuda")
        w = torch.zeros(1, 1, 3, 3, device="cuda")
        F.conv2d(x, w).sum().item()  # launch + sync; raises here if the arch is unsupported
    except Exception as exc:
        try:
            archs = ", ".join(torch.cuda.get_arch_list())
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            gpu = f"{name} (sm_{cc[0]}{cc[1]})"
        except Exception:
            archs, gpu = "?", "?"
        log("[train] " + "=" * 60)
        log("[train] FATAL: CUDA is available but this torch wheel has no kernels for the GPU's architecture.")
        log(f"[train]   GPU: {gpu}   wheel supports: {archs}")
        log(f"[train]   ({type(exc).__name__}: {exc})")
        log("[train]   Rebuild the image with a CUDA wheel that covers this GPU:")
        log("[train]     • Blackwell (RTX 5090/B200, sm_120) needs cu128 — ./build_container.sh --cuda --cuda-wheel cu128")
        log("[train]       (or set the CI 'cuda_wheel' input to cu128).")
        log("[train]   To intentionally train on CPU instead, set  ALLOW_CPU=1  with DEVICE=cpu.")
        log("[train] " + "=" * 60)
        raise SystemExit(1)


def _assert_device_or_die(device: str) -> None:
    """Refuse to silently train on CPU when a GPU is physically present — that almost
    always means the image's CUDA wheel is newer than the host driver (e.g. a RunPod
    box whose driver changed between runs), and a rented GPU would otherwise sit idle
    while training crawls on CPU. Set ALLOW_CPU=1 to override for an intentional CPU run."""
    # device=cuda but torch can't actually use it (e.g. wheel newer than driver):
    # fail with the same actionable message instead of crashing deep in training.
    if device == "cuda" and torch.cuda.is_available():
        _assert_cuda_usable()  # is_available() True is not enough — verify kernels actually run
        return
    if device != "cuda":
        if os.environ.get("ALLOW_CPU", "").lower() in ("1", "true", "yes"):
            return
        if not _gpu_hardware_present():
            return
    built_for = getattr(torch.version, "cuda", None) or "cpu-only"
    log("[train] " + "=" * 60)
    log("[train] FATAL: an NVIDIA GPU is present but torch cannot use it — refusing to train on CPU.")
    log(f"[train]   torch {torch.__version__} was built for CUDA {built_for}; this host's driver is older.")
    log("[train]   The GPU you are paying for would sit idle. Fix the wheel/driver mismatch:")
    log("[train]     • Check the host driver's CUDA: run  nvidia-smi  (top-right 'CUDA Version').")
    log("[train]     • Rebuild the image with a wheel <= that, e.g.  ./build_container.sh --cuda --cuda-wheel cu121")
    log("[train]     • Blackwell (RTX 5090/B200) needs cu128 AND a driver supporting CUDA 12.8.")
    log("[train]   To intentionally train on CPU anyway, set  ALLOW_CPU=1.")
    log("[train] " + "=" * 60)
    raise SystemExit(1)


def _setup_log_file(cfg: Config) -> None:
    """If log_dir is configured, tee all log output to a timestamped file there."""
    if not cfg.train.log_dir:
        return
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg.train.log_dir, f"train_{ts}.log")

    import builtins
    _orig_print = builtins.print

    def tee_print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        with open(log_path, "a") as f:
            _orig_print(*args, file=f, **kwargs)

    builtins.print = tee_print
    log(f"[train] logging to {log_path}")


def _save(path: str, payload: dict) -> None:
    """Atomic checkpoint save (model + buffer + metadata)."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _throttled(label: str, total: int, every_seconds: float = 10.0):
    """A progress(done, total) callback that logs at most once every `every_seconds` (and at 100%)."""
    state = {"last": 0.0}
    phase_start = time.time()

    def cb(done: int, _total: int) -> None:
        now = time.time()
        if done < total and (now - state["last"]) < every_seconds:
            return
        state["last"] = now
        rate = done / max(1e-9, now - phase_start)
        log(f"[{label}] {done}/{total} ({done / total:.0%}, {rate:.1f}/s)")

    return cb


# ── Learning rate schedule ───────────────────────────────────────────────────

def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current LR from the first param group."""
    return float(optimizer.param_groups[0]["lr"])


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, generation: int):
    """Return a function `step(global_step: int, total_steps: int)` that adjusts the LR for the
    current optimizer step.  Called once per generation; the schedule covers that generation's
    training steps."""
    initial_lr = cfg.train.learning_rate
    lr_schedule = cfg.train.lr_schedule
    warmup = cfg.train.lr_warmup_steps
    lr_min = cfg.train.lr_min
    steps = cfg.train.train_steps_per_generation
    global_gen = generation  # captured for logging

    if lr_schedule == "constant":
        return lambda step, _total: None  # no-op

    if lr_schedule == "step":
        # Global-generation step decay
        decay = cfg.train.lr_decay
        step_gens = max(1, cfg.train.lr_step_gens)
        current_lr = initial_lr * (decay ** (generation // step_gens))
        _set_lr(optimizer, current_lr)

        return lambda step, _total: None  # handled once at gen start

    if lr_schedule == "cosine":
        total_steps = steps

        def cosine_step(step: int, _total_steps: int) -> None:
            s = step  # 0-indexed step within this generation
            if warmup > 0 and s < warmup:
                # Linear warmup from 0 → initial_lr
                lr = initial_lr * (s + 1) / warmup
            else:
                # Cosine decay from initial_lr → lr_min
                progress = (s - warmup) / max(1, total_steps - warmup)
                lr = lr_min + 0.5 * (initial_lr - lr_min) * (1.0 + math.cos(math.pi * progress))
            _set_lr(optimizer, lr)

        return cosine_step

    # Unknown schedule → constant
    return lambda step, _total: None


# ── Weight statistics ────────────────────────────────────────────────────────

def _weight_stats(net: torch.nn.Module) -> dict:
    """Collect mean/std of parameters and their gradients for observability."""
    w_mean, w_std = 0.0, 0.0
    g_norm = 0.0
    total = 0
    std_total = 0
    g_sq = 0.0
    for p in net.parameters():
        if not p.requires_grad:
            continue
        w = p.detach().float()
        w_mean += float(w.mean().item())
        if w.numel() >= 2:
            w_std += float(w.std().item())
            std_total += 1
        total += 1
        if p.grad is not None:
            g_sq += float((p.grad.detach().float() ** 2).sum().item())
    if total > 0:
        w_mean /= total
        w_std /= max(1, std_total)
    g_norm = math.sqrt(g_sq)
    return {"w_mean": w_mean, "w_std": w_std, "g_norm": g_norm}


# ── Config validation on resume ──────────────────────────────────────────────

def _validate_checkpoint_config(ckpt_config: dict | None, cfg: Config) -> bool:
    """Return True if the checkpoint is compatible with the current config."""
    if ckpt_config is None:
        return True
    checks = [
        ("board_size", cfg.game.board_size),
        ("channels", cfg.net.channels),
        ("blocks", cfg.net.blocks),
        ("use_se", cfg.net.use_se),
    ]
    for key, current in checks:
        saved = ckpt_config.get(key)
        if saved is not None and saved != current:
            log(f"[train] ERROR: checkpoint has {key}={saved} but config has {key}={current}. "
                f"Net size / board size mismatch — can't resume. "
                f"Either restore the matching config or move checkpoints/ aside for a fresh start.")
            return False
    return True


# ── Training ─────────────────────────────────────────────────────────────────

def _train_steps(net, optimizer, buffer: ReplayBuffer, cfg: Config, rng: np.random.Generator,
                 device: str, generation: int):
    net.train()
    policy_losses, value_losses, total_losses = [], [], []
    steps = cfg.train.train_steps_per_generation
    grad_clip = cfg.train.grad_clip
    step_start = time.time()
    # Timer-based progress: log every TRAIN_LOG_INTERVAL seconds (default 120),
    # plus always at the final step.  Step-count-based intervals are misleading
    # when training is fast (1000 steps in 3 min); timer is consistent.
    log_every = float(os.environ.get("TRAIN_LOG_INTERVAL", "120"))
    last_log = step_start

    lr_step_fn = _build_lr_scheduler(optimizer, cfg, generation)

    # AMP: mixed precision on CUDA gives ~1.5-2× training throughput for free.
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

    for step in range(steps):
        lr_step_fn(step, steps)

        planes, pi, z = buffer.sample(cfg.train.batch_size, rng)
        x = torch.from_numpy(planes).to(device, non_blocking=True)
        target_pi = torch.from_numpy(pi).to(device, non_blocking=True)
        target_v = torch.from_numpy(z).to(device, non_blocking=True)

        with amp_ctx:
            logits, value = net(x)
            logp = F.log_softmax(logits, dim=1)
            policy_loss = -(target_pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_v)
            loss = policy_loss + cfg.train.value_loss_weight * value_loss

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        total_losses.append(float(loss.item()))

        now = time.time()
        if now - last_log >= log_every or (step + 1) == steps:
            rate = (step + 1) / max(1e-9, now - step_start)
            avg_ploss = float(np.mean(policy_losses[-min(250, step + 1):]))
            avg_vloss = float(np.mean(value_losses[-min(250, step + 1):]))
            lr = _get_lr(optimizer)
            log(f"[train] step {step + 1}/{steps} ({rate:.0f}/s) "
                f"lr={lr:.2e} ploss={avg_ploss:.3f} vloss={avg_vloss:.3f}")
            last_log = now

    # Weight stats computed once at the end, not per logging interval.
    ws = _weight_stats(net)
    log(f"[train] |w|={ws.get('w_mean', 0):.4f}±{ws.get('w_std', 0):.4f} "
        f"|g|={ws.get('g_norm', 0):.2f}")

    return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(total_losses))


# ── Main loop ────────────────────────────────────────────────────────────────

def _raise_fd_limit() -> None:
    """Bump the open-file soft limit to the hard limit. Many self-play workers (and, with the GPU
    inference server, one response queue per worker) each consume file descriptors; the default
    1024 soft limit is easy to exhaust on a high-vCPU box. Best-effort, POSIX-only."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass


def main() -> None:
    _setup_cuda()
    _raise_fd_limit()

    cfg = load()
    _setup_log_file(cfg)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    device = cfg.device
    _assert_device_or_die(device)
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    set_start()

    # Thread the logging config through to subprocesses
    # (env vars are inherited by mp.Process).
    os.environ["INFERENCE_LOG_EVERY"] = str(cfg.logging.infer_heartbeat)
    os.environ["SELFPLAY_PROGRESS_INTERVAL"] = str(cfg.logging.selfplay_progress_interval)
    os.environ["TRAIN_LOG_INTERVAL"] = str(cfg.logging.train_log_interval)
    os.environ["TRAIN_START_EPOCH"] = str(time.time())

    # Background hardware-utilisation logger (GPU%, CPU%, RAM, VRAM).
    hw_monitor = None
    if cfg.logging.util_interval > 0:
        hw_monitor = HardwareMonitor(cfg.logging.util_interval)
        hw_monitor.start()

    # Log the full config at startup so the user can see what's active
    budget_s = cfg.train.hours * 3600
    log("[train] " + "=" * 60)
    log(f"[train] NeuralEngine training starting")
    log(f"[train]   device    = {device}")
    if device == "cuda":
        _log_gpu_device()
    log(f"[train]   board     = {cfg.game.board_size}×{cfg.game.board_size}  swap={cfg.game.swap_rule}")
    log(f"[train]   net       = {cfg.net.channels} ch × {cfg.net.blocks} blocks  value_hidden={cfg.net.value_hidden}  se={cfg.net.use_se}")
    log(f"[train]   mcts      = {cfg.mcts.simulations} sims  cpuct={cfg.mcts.c_puct}  "
        f"dirichlet=({cfg.mcts.dirichlet_alpha},{cfg.mcts.dirichlet_epsilon})  "
        f"solver≤{cfg.mcts.solver_empty_threshold}  vc={cfg.mcts.use_virtual_connection}")
    log(f"[train]   selfplay  = {cfg.selfplay.parallel_games} parallel  temp={cfg.selfplay.temperature}({cfg.selfplay.temperature_moves} plies)  "
        f"pipeline={cfg.mcts.pipeline_shards}  no-resign")
    log(f"[train]   train     = {cfg.train.hours}h budget  {cfg.train.games_per_generation} games/gen  "
        f"{cfg.train.train_steps_per_generation} steps  batch={cfg.train.batch_size}  "
        f"buffer={cfg.train.replay_buffer_size}")
    log(f"[train]   optimizer = lr={cfg.train.learning_rate}  wd={cfg.train.weight_decay}  "
        f"clip={cfg.train.grad_clip}  schedule={cfg.train.lr_schedule}  "
        f"lr_min={cfg.train.lr_min}  warmup={cfg.train.lr_warmup_steps}")
    log(f"[train]   arena     = {cfg.train.arena_games} games @ {cfg.train.arena_simulations} sims  "
        f"threshold={cfg.train.arena_win_rate:.0%}")
    _selfplay_eval = f"gpu-server({device})" if cfg.use_inference_server() else cfg.worker_eval_device()
    log(f"[train]   actors    = {cfg.resolve_actors()} (of {available_cpus()} usable cpus)  "
        f"seed={cfg.train.seed}  selfplay_eval={_selfplay_eval}")
    log(f"[train]   checkpoint_dir = {cfg.train.checkpoint_dir}")
    log(f"[train]   save_every_ckpt = {cfg.train.save_every_checkpoint}")
    log(f"[train]   log_dir   = {cfg.train.log_dir or '(stdout only)'}")
    log(f"[train]   amp       = {'on' if device == 'cuda' else 'off'}  compile={'on' if device == 'cuda' else 'off'}")
    if device == "cuda":
        mem = _gpu_memory_str()
        if mem:
            log(f"[train]   {mem}")
    log("[train] " + "=" * 60)

    net = build_net(cfg).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    log(f"[train]   model parameters: {total_params:,}")

    # torch.compile on CUDA: JIT-compiles the network into fused kernels.
    # Uses "reduce-overhead" mode for training (amortizes compilation overhead
    # across many forward/backward passes).  MPS / CPU stay eager.
    #
    # compile() is lazy — it doesn't fail until the first forward.  We force a
    # tiny warmup forward NOW so a missing C compiler (Triton's JIT
    # needs gcc) or an unsupported-op crashes here with a clear message and a
    # fallback to eager mode, instead of silently succeeding at build time then
    # crashing 37 minutes into a paid run.
    use_compile = (device == "cuda")
    compiled_ok = False
    if use_compile:
        try:
            net = torch.compile(net, mode="reduce-overhead")
            # Force compile: one tiny forward through the real net.  A forward
            # alone triggers Triton JIT; no backward needed for the check.
            dummy = torch.zeros(1, cfg.net.in_planes, cfg.game.board_size, cfg.game.board_size,
                                device=device)
            _ = net(dummy)
            compiled_ok = True
            log(f"[train]   torch.compile: enabled (reduce-overhead) — warmup OK")
        except Exception as e:
            net = build_net(cfg).to(device)  # fresh uncompiled net
            log(f"[train]   torch.compile: skipped — warmup failed ({e})")
            log(f"[train]   (install gcc in the container image to enable torch.compile)")

    # Bare (uncompiled) view of the model: its state_dict has no '_orig_mod.' prefix,
    # so checkpoints and weights shipped to workers load into plain nets.
    bare = BareModule(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay)
    buffer = ReplayBuffer(cfg.train.replay_buffer_size, cfg.game.board_size)
    generation = 0
    best_state = copy.deepcopy(bare.state_dict())

    if device == "cuda":
        mem = _gpu_memory_str()
        if mem:
            log(f"[train]   after model+optim: {mem}")

    latest_path = os.path.join(cfg.train.checkpoint_dir, LATEST)
    best_path = os.path.join(cfg.train.checkpoint_dir, BEST)

    # ─── RESUME ──────────────────────────────────────────────────────────
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        if not _validate_checkpoint_config(ckpt.get("config"), cfg):
            sys.exit(1)

        bare.load_state_dict(CleanStateDict(ckpt["model"]))
        best_state = CleanStateDict(ckpt.get("best", copy.deepcopy(bare.state_dict())))
        optimizer.load_state_dict(ckpt["optimizer"])
        generation = ckpt["generation"]

        if "buffer" in ckpt:
            buffer.load_state_dict(ckpt["buffer"])
            log(f"[train] resumed from generation {generation} with buffer ({len(buffer)} samples)")
        else:
            log(f"[train] WARNING: checkpoint has no buffer — starting with empty buffer (old format)")

        if "best" not in ckpt:
            log(f"[train] WARNING: checkpoint has no 'best' key (old format) — using current model as best")

        log(f"[train] resumed from generation {generation} ({latest_path})")
    else:
        _save(best_path, {"model": best_state, "config": _config_summary(cfg), "generation": 0})
        log(f"[train] fresh start; wrote initial best -> {best_path}")

    start = time.time()
    gen_times: list[float] = []
    max_gens = cfg.train.max_generations  # 0 = unlimited; a cap lets a test run a fixed few generations
    try:
        while time.time() - start < budget_s and (max_gens <= 0 or generation < max_gens):
            generation += 1
            set_gen(generation)
            os.environ["TRAIN_GENERATION"] = str(generation)  # for infer server _ts()
            gen_start = time.time()

            # ── Self-play ────────────────────────────────────────────────
            log(f"[selfplay] {cfg.train.games_per_generation} games "
                f"@ {cfg.mcts.simulations} sims on {cfg.resolve_actors()} actor(s)…")
            sp_start = time.time()
            samples = selfplay.generate(
                cfg, best_state, cfg.train.games_per_generation,
                base_seed=cfg.train.seed + generation * 1000,
                progress=_throttled("selfplay", cfg.train.games_per_generation),
            )
            buffer.extend(samples)
            sp_dt = time.time() - sp_start
            mem = _gpu_memory_str()
            log(f"[selfplay] done in {offset_str(sp_dt)} "
                f"({len(samples)} samples, {len(samples) / max(1e-9, sp_dt):.0f} samples/s, "
                f"buffer {len(buffer)}/{buffer.capacity}"
                + (f", {mem}" if mem else "") + ")")
            if len(buffer) < cfg.train.batch_size:
                log(f"[train] buffer warming ({len(buffer)}/{cfg.train.batch_size}) "
                    f"— skipping train/arena")
                continue

            # ── Training ─────────────────────────────────────────────────
            log(f"[train] training {cfg.train.train_steps_per_generation} steps "
                f"(batch {cfg.train.batch_size}, clip={cfg.train.grad_clip}, "
                f"schedule={cfg.train.lr_schedule})…")
            tr_start = time.time()
            policy_loss, value_loss, total_loss = _train_steps(
                net, optimizer, buffer, cfg, rng, device, generation)
            tr_dt = time.time() - tr_start
            mem = _gpu_memory_str()
            log(f"[train] training done in {offset_str(tr_dt)} "
                f"(ploss={policy_loss:.3f} vloss={value_loss:.3f} total={total_loss:.3f}"
                + (f", {mem}" if mem else "") + ")")

            # ── Arena ────────────────────────────────────────────────────
            log(f"[arena] {cfg.train.arena_games} games "
                f"@ {cfg.train.arena_simulations} sims on {cfg.resolve_actors()} actor(s) "
                f"vs current best…")
            ar_start = time.time()
            win_rate = arena.play_match_parallel(
                cfg, bare.state_dict(), best_state,
                cfg.train.arena_games, cfg.train.arena_simulations,
                base_seed=cfg.train.seed + generation * 1000 + 500,
                progress=_throttled("arena", cfg.train.arena_games),
            )
            ar_dt = time.time() - ar_start
            mem = _gpu_memory_str()
            log(f"[arena] done in {offset_str(ar_dt)} "
                f"(candidate win rate {win_rate:.0%}, threshold {cfg.train.arena_win_rate:.0%}"
                + (f", {mem}" if mem else "") + ")")

            # ── Promotion ────────────────────────────────────────────────
            promoted = win_rate >= cfg.train.arena_win_rate
            if promoted:
                best_state = copy.deepcopy(bare.state_dict())
                _save(best_path, {"model": best_state, "config": _config_summary(cfg),
                                  "generation": generation})
                log(f"[arena] PROMOTED — new best.pt at generation {generation}")
            else:
                log(f"[arena] kept current best (win rate {win_rate:.0%} < {cfg.train.arena_win_rate:.0%})")

            # ── Checkpoint ───────────────────────────────────────────────
            _save(latest_path, {
                "model": bare.state_dict(),
                "best": best_state,
                "optimizer": optimizer.state_dict(),
                "generation": generation,
                "config": _config_summary(cfg),
                "buffer": buffer.state_dict(),
            })

            # Save a generation snapshot if configured.
            if cfg.train.save_every_checkpoint:
                gen_path = os.path.join(cfg.train.checkpoint_dir, f"gen_{generation:04d}.pt")
                _save(gen_path, {
                    "model": bare.state_dict(),
                    "config": _config_summary(cfg),
                    "generation": generation,
                })

            # ── Generation summary ───────────────────────────────────────
            gen_dt = time.time() - gen_start
            gen_times.append(gen_dt)
            elapsed = time.time() - start
            remaining = max(0.0, budget_s - elapsed)
            avg_gen = sum(gen_times) / len(gen_times)
            eta_gens = int(remaining // avg_gen) if avg_gen > 0 else 0
            log(
                f"[summary] DONE samples={len(samples)} buffer={len(buffer)} "
                f"ploss={policy_loss:.3f} vloss={value_loss:.3f} arena={win_rate:.0%} "
                f"{'PROMOTED' if promoted else 'kept'} gen_time={offset_str(gen_dt)} "
                f"elapsed={offset_str(elapsed)} remaining={offset_str(remaining)} "
                f"(~{eta_gens} more gens)"
            )
    except KeyboardInterrupt:
        log("[train] interrupted — saving latest checkpoint")
        _save(latest_path, {
            "model": bare.state_dict(), "best": best_state, "optimizer": optimizer.state_dict(),
            "generation": generation, "config": _config_summary(cfg),
            "buffer": buffer.state_dict(),
        })
    finally:
        if hw_monitor is not None:
            hw_monitor.stop()
        set_gen(None)  # clear generation context

    log(f"[train] done at generation {generation} after {offset_str(time.time() - start)}; "
        f"best -> {best_path}")


def _config_summary(cfg: Config) -> dict:
    return {
        "board_size": cfg.game.board_size,
        "swap_rule": cfg.game.swap_rule,
        "in_planes": cfg.net.in_planes,
        "channels": cfg.net.channels,
        "blocks": cfg.net.blocks,
        "value_hidden": cfg.net.value_hidden,
        "use_se": cfg.net.use_se,
    }


if __name__ == "__main__":
    main()
