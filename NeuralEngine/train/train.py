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
from net.evaluator import Evaluator
from train.replay_buffer import ReplayBuffer
from train import selfplay, arena
from train.clock import log, set_start, offset_str

LATEST = "latest.pt"
BEST = "best.pt"


def _save(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic — never leave a half-written checkpoint


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


def _train_steps(net, optimizer, buffer: ReplayBuffer, cfg: Config, rng: np.random.Generator, device: str):
    net.train()
    policy_losses, value_losses = [], []
    steps = cfg.train.train_steps_per_generation
    report_every = max(1, steps // 4)
    step_start = time.time()
    for step in range(steps):
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
        optimizer.step()
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

        if (step + 1) % report_every == 0 or (step + 1) == steps:
            rate = (step + 1) / max(1e-9, time.time() - step_start)
            log(f"    train: step {step + 1}/{steps} ({rate:.0f}/s) "
                f"ploss={np.mean(policy_losses[-report_every:]):.3f} "
                f"vloss={np.mean(value_losses[-report_every:]):.3f}")
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


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
    budget_s = cfg.train.hours * 3600
    log(f"[train] device={device} actors={cfg.resolve_actors()} board={cfg.game.board_size} "
        f"net={cfg.net.channels}x{cfg.net.blocks} sims={cfg.mcts.simulations} "
        f"games/gen={cfg.train.games_per_generation} budget={cfg.train.hours}h")

    net = build_net(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    buffer = ReplayBuffer(cfg.train.replay_buffer_size, cfg.game.board_size)
    generation = 0
    best_state = copy.deepcopy(net.state_dict())

    latest_path = os.path.join(cfg.train.checkpoint_dir, LATEST)
    best_path = os.path.join(cfg.train.checkpoint_dir, BEST)
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        net.load_state_dict(ckpt["model"])
        best_state = ckpt["best"]
        optimizer.load_state_dict(ckpt["optimizer"])
        generation = ckpt["generation"]
        log(f"[train] resumed from generation {generation} ({latest_path})")
    else:
        # Save an initial best so the engine has something to load immediately (weak, but valid).
        _save(best_path, {"model": best_state, "config": _config_summary(cfg), "generation": 0})
        log(f"[train] fresh start; wrote initial best -> {best_path}")

    start = time.time()
    gen_times: list[float] = []
    try:
        while time.time() - start < budget_s:
            generation += 1
            gen_start = time.time()

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
            log(f"[gen {generation}] self-play done in {sp_dt:.0f}s "
                f"({len(samples)} samples, {len(samples) / max(1e-9, sp_dt):.0f} samples/s, buffer {len(buffer)})")
            if len(buffer) < cfg.train.batch_size:
                log(f"[gen {generation}] buffer warming ({len(buffer)}/{cfg.train.batch_size}) — skipping train/arena")
                continue

            log(f"[gen {generation}] training {cfg.train.train_steps_per_generation} steps "
                f"(batch {cfg.train.batch_size})…")
            tr_start = time.time()
            policy_loss, value_loss = _train_steps(net, optimizer, buffer, cfg, rng, device)
            log(f"[gen {generation}] training done in {time.time() - tr_start:.0f}s "
                f"(ploss={policy_loss:.3f} vloss={value_loss:.3f})")

            log(f"[gen {generation}] arena: {cfg.train.arena_games} games "
                f"@ {cfg.train.arena_simulations} sims vs current best…")
            ar_start = time.time()
            cand_net = build_net(cfg).to(device)
            cand_net.load_state_dict(net.state_dict())
            cand_net.eval()
            best_net = build_net(cfg).to(device)
            best_net.load_state_dict(best_state)
            best_net.eval()
            win_rate = arena.play_match(
                cfg, Evaluator(cand_net, device), Evaluator(best_net, device),
                cfg.train.arena_games, cfg.train.arena_simulations, rng,
                progress=_throttled("arena", cfg.train.arena_games),
            )
            log(f"[gen {generation}] arena done in {time.time() - ar_start:.0f}s "
                f"(candidate win rate {win_rate:.0%}, threshold {cfg.train.arena_win_rate:.0%})")

            promoted = win_rate >= cfg.train.arena_win_rate
            if promoted:
                best_state = copy.deepcopy(net.state_dict())
                _save(best_path, {"model": best_state, "config": _config_summary(cfg), "generation": generation})
                log(f"[gen {generation}] PROMOTED — new best.pt at generation {generation}")

            _save(latest_path, {
                "model": net.state_dict(),
                "best": best_state,
                "optimizer": optimizer.state_dict(),
                "generation": generation,
                "config": _config_summary(cfg),
            })

            gen_dt = time.time() - gen_start
            gen_times.append(gen_dt)
            elapsed = time.time() - start
            remaining = max(0.0, budget_s - elapsed)
            avg_gen = sum(gen_times) / len(gen_times)
            eta_gens = int(remaining // avg_gen) if avg_gen > 0 else 0
            log(
                f"[gen {generation}] DONE samples={len(samples)} buffer={len(buffer)} "
                f"ploss={policy_loss:.3f} vloss={value_loss:.3f} arena={win_rate:.0%} "
                f"{'PROMOTED' if promoted else 'kept'} gen_time={gen_dt:.0f}s "
                f"elapsed={offset_str(elapsed)} remaining={offset_str(remaining)} (~{eta_gens} more gens)"
            )
    except KeyboardInterrupt:
        log("[train] interrupted — saving latest checkpoint")
        _save(latest_path, {
            "model": net.state_dict(), "best": best_state, "optimizer": optimizer.state_dict(),
            "generation": generation, "config": _config_summary(cfg),
        })

    log(f"[train] done at generation {generation} after {offset_str(time.time() - start)}; best -> {best_path}")


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
