"""Self-play reinforcement-learning loop (AlphaZero-style) for Hex.

Each generation:
  1. the *best* network generates self-play games (exploration via Dirichlet noise + temperature),
  2. the *current* network trains on a replay buffer of recent games (policy cross-entropy + value MSE),
  3. an arena gates promotion: the current network only becomes the new best if it beats the incumbent,
  4. checkpoints are written (latest.pt always; best.pt on promotion).

The loop runs until the wall-clock budget (TRAIN_HOURS) elapses and can be stopped/resumed at any time;
the deployed engine simply loads checkpoints/best.pt. Run via run_training.sh or `python -m train.train`.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Allow `python -m train.train` and `python train/train.py` alike.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load, Config
from net.model import build_net
from train.replay_buffer import ReplayBuffer
from train import selfplay, arena
from train.clock import log, set_start, offset_str

LATEST = "latest.pt"
BEST = "best.pt"


def _save(path: str, payload: dict) -> None:
    """Atomic checkpoint save (model + buffer + metadata)."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _throttled(label: str, total: int, every_seconds: float = 15.0):
    """A progress(done, total) callback that logs at most once every `every_seconds` (and at 100%)."""
    state = {"last": 0.0}
    phase_start = time.time()

    def cb(done: int, _total: int) -> None:
        now = time.time()
        if done < total and (now - state["last"]) < every_seconds:
            return
        state["last"] = now
        rate = done / max(1e-9, now - phase_start)
        log(f"    {label}: {done}/{total} ({done / total:.0%}, {rate:.1f}/s)")

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


# ── Weight statistics (extra logging) ────────────────────────────────────────

def _weight_stats(net: torch.nn.Module) -> dict:
    """Collect mean/std of parameters and their gradients for observability."""
    params = []
    grads = []
    for p in net.parameters():
        if p.requires_grad:
            params.append(p.detach().float())
            if p.grad is not None:
                grads.append(p.grad.detach().float())
    stats = {}
    if params:
        all_p = torch.cat([p.flatten() for p in params])
        stats["w_mean"] = float(all_p.mean().item())
        stats["w_std"] = float(all_p.std().item())
    if grads:
        all_g = torch.cat([g.flatten() for g in grads])
        stats["g_mean"] = float(all_g.mean().item())
        stats["g_std"] = float(all_g.std().item())
        stats["g_norm"] = float(torch.norm(all_g).item())
    return stats


# ── Config validation on resume ──────────────────────────────────────────────

def _validate_checkpoint_config(ckpt_config: dict | None, cfg: Config) -> bool:
    """Return True if the checkpoint is compatible with the current config."""
    if ckpt_config is None:
        return True
    checks = [
        ("board_size", cfg.game.board_size),
        ("channels", cfg.net.channels),
        ("blocks", cfg.net.blocks),
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
    report_every = max(1, steps // 4)
    step_start = time.time()

    lr_step_fn = _build_lr_scheduler(optimizer, cfg, generation)

    for step in range(steps):
        lr_step_fn(step, steps)

        planes, pi, z = buffer.sample(cfg.train.batch_size, rng)
        x = torch.from_numpy(planes).to(device)
        target_pi = torch.from_numpy(pi).to(device)
        target_v = torch.from_numpy(z).to(device)

        logits, value = net(x)
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(target_pi * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(value, target_v)
        loss = policy_loss + cfg.train.value_loss_weight * value_loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

        optimizer.step()

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        total_losses.append(float(loss.item()))

        if (step + 1) % report_every == 0 or (step + 1) == steps:
            rate = (step + 1) / max(1e-9, time.time() - step_start)
            recent = report_every if (step + 1) >= report_every else (step + 1)
            avg_ploss = float(np.mean(policy_losses[-recent:]))
            avg_vloss = float(np.mean(value_losses[-recent:]))
            lr = _get_lr(optimizer)
            ws = _weight_stats(net)
            log(f"    train: step {step + 1}/{steps} ({rate:.0f}/s) "
                f"lr={lr:.2e} ploss={avg_ploss:.3f} vloss={avg_vloss:.3f} "
                f"|w|={ws.get('w_mean', 0):.4f}±{ws.get('w_std', 0):.4f} "
                f"|g|={ws.get('g_norm', 0):.2f}")

    return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(total_losses))


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load()
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    device = cfg.device
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    set_start()

    # Log the full config at startup so the user can see what's active
    budget_s = cfg.train.hours * 3600
    log("=" * 60)
    log(f"NeuralEngine training starting")
    log(f"  device    = {device}")
    log(f"  board     = {cfg.game.board_size}×{cfg.game.board_size}  swap={cfg.game.swap_rule}")
    log(f"  net       = {cfg.net.channels} ch × {cfg.net.blocks} blocks  value_hidden={cfg.net.value_hidden}")
    log(f"  mcts      = {cfg.mcts.simulations} sims  cpuct={cfg.mcts.c_puct}  "
        f"dirichlet=({cfg.mcts.dirichlet_alpha},{cfg.mcts.dirichlet_epsilon})  "
        f"solver≤{cfg.mcts.solver_empty_threshold}  vc={cfg.mcts.use_virtual_connection}")
    log(f"  selfplay  = {cfg.selfplay.parallel_games} parallel  temp={cfg.selfplay.temperature}({cfg.selfplay.temperature_moves} plies)  "
        f"resign={cfg.selfplay.resign_threshold}@{cfg.selfplay.resign_min_ply}ply")
    log(f"  train     = {cfg.train.hours}h budget  {cfg.train.games_per_generation} games/gen  "
        f"{cfg.train.train_steps_per_generation} steps  batch={cfg.train.batch_size}  "
        f"buffer={cfg.train.replay_buffer_size}")
    log(f"  optimizer = lr={cfg.train.learning_rate}  wd={cfg.train.weight_decay}  "
        f"clip={cfg.train.grad_clip}  schedule={cfg.train.lr_schedule}  "
        f"lr_min={cfg.train.lr_min}  warmup={cfg.train.lr_warmup_steps}")
    log(f"  arena     = {cfg.train.arena_games} games @ {cfg.train.arena_simulations} sims  "
        f"threshold={cfg.train.arena_win_rate:.0%}")
    log(f"  actors    = {cfg.resolve_actors()}  seed={cfg.train.seed}")
    log(f"  checkpoint_dir = {cfg.train.checkpoint_dir}")
    log("=" * 60)

    net = build_net(cfg).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    log(f"  model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay)
    buffer = ReplayBuffer(cfg.train.replay_buffer_size, cfg.game.board_size)
    generation = 0
    best_state = copy.deepcopy(net.state_dict())

    latest_path = os.path.join(cfg.train.checkpoint_dir, LATEST)
    best_path = os.path.join(cfg.train.checkpoint_dir, BEST)

    # ─── RESUME ──────────────────────────────────────────────────────────
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        if not _validate_checkpoint_config(ckpt.get("config"), cfg):
            sys.exit(1)

        net.load_state_dict(ckpt["model"])
        best_state = ckpt.get("best", copy.deepcopy(net.state_dict()))
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
    try:
        while time.time() - start < budget_s:
            generation += 1
            gen_start = time.time()

            # ── Self-play ────────────────────────────────────────────────
            log(f"[gen {generation}] self-play: {cfg.train.games_per_generation} games "
                f"@ {cfg.mcts.simulations} sims on {cfg.resolve_actors()} actor(s)…")
            sp_start = time.time()
            samples = selfplay.generate(
                cfg, best_state, cfg.train.games_per_generation,
                base_seed=cfg.train.seed + generation * 1000,
                progress=_throttled("self-play", cfg.train.games_per_generation),
            )
            buffer.extend(samples)
            sp_dt = time.time() - sp_start
            log(f"[gen {generation}] self-play done in {offset_str(sp_dt)} "
                f"({len(samples)} samples, {len(samples) / max(1e-9, sp_dt):.0f} samples/s, "
                f"buffer {len(buffer)}/{buffer.capacity})")
            if len(buffer) < cfg.train.batch_size:
                log(f"[gen {generation}] buffer warming ({len(buffer)}/{cfg.train.batch_size}) "
                    f"— skipping train/arena")
                continue

            # ── Training ─────────────────────────────────────────────────
            log(f"[gen {generation}] training {cfg.train.train_steps_per_generation} steps "
                f"(batch {cfg.train.batch_size}, clip={cfg.train.grad_clip}, "
                f"schedule={cfg.train.lr_schedule})…")
            tr_start = time.time()
            policy_loss, value_loss, total_loss = _train_steps(
                net, optimizer, buffer, cfg, rng, device, generation)
            tr_dt = time.time() - tr_start
            log(f"[gen {generation}] training done in {offset_str(tr_dt)} "
                f"(ploss={policy_loss:.3f} vloss={value_loss:.3f} total={total_loss:.3f})")

            # ── Arena ────────────────────────────────────────────────────
            log(f"[gen {generation}] arena: {cfg.train.arena_games} games "
                f"@ {cfg.train.arena_simulations} sims on {cfg.resolve_actors()} actor(s) "
                f"vs current best…")
            ar_start = time.time()
            win_rate = arena.play_match_parallel(
                cfg, net.state_dict(), best_state,
                cfg.train.arena_games, cfg.train.arena_simulations,
                base_seed=cfg.train.seed + generation * 1000 + 500,
                progress=_throttled("arena", cfg.train.arena_games),
            )
            ar_dt = time.time() - ar_start
            log(f"[gen {generation}] arena done in {offset_str(ar_dt)} "
                f"(candidate win rate {win_rate:.0%}, threshold {cfg.train.arena_win_rate:.0%})")

            # ── Promotion ────────────────────────────────────────────────
            promoted = win_rate >= cfg.train.arena_win_rate
            if promoted:
                best_state = copy.deepcopy(net.state_dict())
                _save(best_path, {"model": best_state, "config": _config_summary(cfg),
                                  "generation": generation})
                log(f"[gen {generation}] PROMOTED — new best.pt at generation {generation}")
            else:
                log(f"[gen {generation}] kept current best (win rate {win_rate:.0%} < {cfg.train.arena_win_rate:.0%})")

            # ── Checkpoint ───────────────────────────────────────────────
            _save(latest_path, {
                "model": net.state_dict(),
                "best": best_state,
                "optimizer": optimizer.state_dict(),
                "generation": generation,
                "config": _config_summary(cfg),
                "buffer": buffer.state_dict(),
            })

            # ── Generation summary ───────────────────────────────────────
            gen_dt = time.time() - gen_start
            gen_times.append(gen_dt)
            elapsed = time.time() - start
            remaining = max(0.0, budget_s - elapsed)
            avg_gen = sum(gen_times) / len(gen_times)
            eta_gens = int(remaining // avg_gen) if avg_gen > 0 else 0
            log(
                f"[gen {generation}] DONE samples={len(samples)} buffer={len(buffer)} "
                f"ploss={policy_loss:.3f} vloss={value_loss:.3f} arena={win_rate:.0%} "
                f"{'PROMOTED' if promoted else 'kept'} gen_time={offset_str(gen_dt)} "
                f"elapsed={offset_str(elapsed)} remaining={offset_str(remaining)} "
                f"(~{eta_gens} more gens)"
            )
    except KeyboardInterrupt:
        log("[train] interrupted — saving latest checkpoint")
        _save(latest_path, {
            "model": net.state_dict(), "best": best_state, "optimizer": optimizer.state_dict(),
            "generation": generation, "config": _config_summary(cfg),
            "buffer": buffer.state_dict(),
        })

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
    }


if __name__ == "__main__":
    main()
