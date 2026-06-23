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

LATEST = "latest.pt"
BEST = "best.pt"


def _save(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic — never leave a half-written checkpoint


def _train_steps(net, optimizer, buffer: ReplayBuffer, cfg: Config, rng: np.random.Generator, device: str):
    net.train()
    policy_losses, value_losses = [], []
    for _ in range(cfg.train.train_steps_per_generation):
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

    print(f"[train] device={device} actors={cfg.resolve_actors()} board={cfg.game.board_size} "
          f"net={cfg.net.channels}x{cfg.net.blocks} sims={cfg.mcts.simulations} budget={cfg.train.hours}h",
          flush=True)

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
        print(f"[train] resumed from generation {generation}", flush=True)
    else:
        # Save an initial best so the engine has something to load immediately (weak, but valid).
        _save(best_path, {"model": best_state, "config": _config_summary(cfg), "generation": 0})

    start = time.time()
    try:
        while time.time() - start < cfg.train.hours * 3600:
            generation += 1
            gen_start = time.time()

            print(f"[gen {generation}] self-play: {cfg.train.games_per_generation} games "
                  f"@ {cfg.mcts.simulations} sims on {cfg.resolve_actors()} actor(s)…", flush=True)
            samples = selfplay.generate(cfg, best_state, cfg.train.games_per_generation, base_seed=cfg.train.seed + generation * 1000)
            buffer.extend(samples)
            print(f"[gen {generation}] self-play done in {time.time()-gen_start:.0f}s "
                  f"({len(samples)} samples, buffer {len(buffer)})", flush=True)
            if len(buffer) < cfg.train.batch_size:
                print(f"[gen {generation}] buffer warming ({len(buffer)} samples)", flush=True)
                continue

            print(f"[gen {generation}] training {cfg.train.train_steps_per_generation} steps…", flush=True)
            policy_loss, value_loss = _train_steps(net, optimizer, buffer, cfg, rng, device)
            print(f"[gen {generation}] arena: {cfg.train.arena_games} games vs current best…", flush=True)

            cand_net = build_net(cfg).to(device)
            cand_net.load_state_dict(net.state_dict())
            cand_net.eval()
            best_net = build_net(cfg).to(device)
            best_net.load_state_dict(best_state)
            best_net.eval()
            win_rate = arena.play_match(
                cfg, Evaluator(cand_net, device), Evaluator(best_net, device),
                cfg.train.arena_games, cfg.train.arena_simulations, rng,
            )

            promoted = win_rate >= cfg.train.arena_win_rate
            if promoted:
                best_state = copy.deepcopy(net.state_dict())
                _save(best_path, {"model": best_state, "config": _config_summary(cfg), "generation": generation})

            _save(latest_path, {
                "model": net.state_dict(),
                "best": best_state,
                "optimizer": optimizer.state_dict(),
                "generation": generation,
                "config": _config_summary(cfg),
            })

            elapsed = time.time() - start
            print(
                f"[gen {generation}] samples={len(samples)} buffer={len(buffer)} "
                f"ploss={policy_loss:.3f} vloss={value_loss:.3f} arena={win_rate:.0%} "
                f"{'PROMOTED' if promoted else 'kept'} gen_time={time.time()-gen_start:.0f}s "
                f"elapsed={elapsed/3600:.2f}h",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n[train] interrupted — saving latest checkpoint", flush=True)
        _save(latest_path, {
            "model": net.state_dict(), "best": best_state, "optimizer": optimizer.state_dict(),
            "generation": generation, "config": _config_summary(cfg),
        })

    print(f"[train] done at generation {generation}; best -> {best_path}", flush=True)


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
