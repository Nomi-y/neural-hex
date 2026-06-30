"""Fast end-to-end sanity check — runs every component on a tiny board/net in seconds.

Use it after any change, before committing:
    python smoke_test.py

It exercises: the rules engine, bridges + endgame solver, encoding/canonicalisation, the network,
batched MCTS, self-play sample generation, a few optimiser steps, and an arena match — on CPU.

If a CUDA GPU is present it ALSO runs the GPU-only paths that CPU smoke can't reach (device-property
logging, the arch/kernel guard, a real GPU forward, and self-play + arena through the GPU inference
server) — run it on a CUDA box before committing. With no GPU (e.g. the Docker build host) the GPU
section is skipped, so it stays a valid build-time guard.
"""

import os

# Shrink everything to laptop scale BEFORE importing config (defaults read env at import time).
os.environ.setdefault("BOARD_SIZE", "5")
os.environ.setdefault("NET_CHANNELS", "16")
os.environ.setdefault("NET_BLOCKS", "2")
os.environ.setdefault("MCTS_SIMS", "16")
os.environ.setdefault("PARALLEL_GAMES", "6")
os.environ.setdefault("DEVICE", "cpu")

import numpy as np
import torch

from config import load
from hex.board import HexState, RED, detect_win
from hex.bridges import virtual_connection, bridges_table
from hex.solver import solve
from net.model import build_net
from net.evaluator import Evaluator
from net.encoding import encode, canon_to_real_action, real_to_canon_action
from search import mcts
from train.replay_buffer import ReplayBuffer
from train import selfplay, arena


def main() -> None:
    cfg = load()
    rng = np.random.default_rng(0)
    N = cfg.game.board_size
    print(f"smoke: board={N} net={cfg.net.channels}x{cfg.net.blocks} sims={cfg.mcts.simulations}")

    # Rules + win detection: a full Red column connects top-bottom.
    s = HexState.initial(N, swap_rule=True)
    for r in range(N):
        s = s.play(r * N)  # Red plays column 0
        if not s.is_terminal():
            s = s.play(r * N + 1)  # Blue plays column 1 (harmless)
    assert s.winner == RED, "Red should have connected its column"
    assert detect_win(s.cells, N, RED) is not None

    # Encoding round-trips actions through canonicalisation.
    st = HexState.initial(N, True).play(N // 2 * N + N // 2)  # one Red stone, Blue to move (transposed)
    for a in st.legal_actions():
        assert canon_to_real_action(st.to_move, N, real_to_canon_action(st.to_move, N, a)) == a
    _ = bridges_table(N), virtual_connection(st.cells, N, RED)

    # Solver resolves a near-full board exactly.
    val, action = solve(s.play(s.legal_actions()[0]) if s.legal_actions() else s, node_budget=50_000)

    # Network + batched MCTS.
    net = build_net(cfg)
    net.eval()
    evaluator = Evaluator(net, "cpu")
    root = mcts.make_root(HexState.initial(N, True))
    mcts.run_batched([root], evaluator, cfg, simulations=cfg.mcts.simulations, add_noise=True, rng=rng)
    assert root.sum_n > 0
    print("  mcts root visits:", root.sum_n, "top:", mcts.ranked_moves(root, cfg.game.num_actions)[:3])

    # Self-play produces labelled samples.
    samples = selfplay.play_games(evaluator, cfg, num_games=4, add_noise=True, rng=rng)
    assert samples and all(planes.shape == (cfg.net.in_planes, N, N) for planes, _, _ in samples)
    print(f"  self-play samples: {len(samples)}")

    # A few optimiser steps run without error.
    buffer = ReplayBuffer(10_000, N)
    buffer.extend(samples)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(3):
        planes, pi, z = buffer.sample(16, rng)
        logits, value = net(torch.from_numpy(planes))
        logp = torch.log_softmax(logits, dim=1)
        loss = -(torch.from_numpy(pi) * logp).sum(1).mean() + torch.nn.functional.mse_loss(value, torch.from_numpy(z))
        optim.zero_grad(); loss.backward(); optim.step()
    print(f"  trained 3 steps, last loss={loss.item():.3f}")

    # Arena: a network playing itself should land near 50%.
    net.eval()
    wr = arena.play_match(cfg, Evaluator(net, "cpu"), Evaluator(net, "cpu"), num_games=2, simulations=8, rng=rng)
    print(f"  arena win rate (self vs self): {wr:.0%}")

    # Parallel plumbing: exercise the multi-process self-play + arena path (spawn workers, per-worker
    # net rebuild, chunking, progress) on CPU, so the same code that fans out onto a GPU is validated.
    cfg.train.num_actors = 2
    par_samples = selfplay.generate(cfg, net.state_dict(), num_games=4, base_seed=1)
    assert par_samples, "parallel self-play produced no samples"
    wr2 = arena.play_match_parallel(cfg, net.state_dict(), net.state_dict(), num_games=4, simulations=8, base_seed=2)
    print(f"  parallel: {len(par_samples)} self-play samples, arena win rate {wr2:.0%}")

    # Watchdog: a microscopic timeout guarantees no result arrives in time, forcing the pool-watchdog
    # branch — proves a dead/deadlocked worker can't hang the loop forever (it returns instead).
    cfg.train.arena_timeout = 1e-9
    wr3 = arena.play_match_parallel(cfg, net.state_dict(), net.state_dict(), num_games=4, simulations=8, base_seed=3)
    assert 0.0 <= wr3 <= 1.0, "watchdog path should still return a valid win rate"
    cfg.train.arena_timeout = 900.0
    print(f"  watchdog path returned {wr3:.0%} without hanging")

    print(f"  solver on a near-terminal position returned value={val} action={action}")

    cuda_checks()
    print("SMOKE TEST PASSED")


def cuda_checks() -> None:
    """Exercise the GPU-only paths that never run under DEVICE=cpu, so bugs that only surface on a
    real device are caught here — e.g. the `total_mem`/`total_memory` crash in _log_gpu_device, or a
    torch wheel lacking the GPU's arch (cu121 on a Blackwell sm_120 5090, where cuda.is_available() is
    True but every kernel fails). No-op (skips) when there's no GPU, so the Docker build guard is fine."""
    if not torch.cuda.is_available():
        print("  cuda: not available — GPU checks skipped")
        return
    from train.train import _assert_device_or_die, _log_gpu_device, _gpu_memory_str

    _assert_device_or_die("cuda")   # arch/kernel usability — aborts loudly on an incompatible wheel
    _log_gpu_device()               # the call that crashed in prod (total_mem -> total_memory)
    mem = _gpu_memory_str()
    if mem:
        print(f"  cuda mem: {mem}")

    cfg = load()
    cfg.device = "cuda"
    cfg.train.num_actors = 2
    N = cfg.game.board_size

    net = build_net(cfg).to("cuda").eval()
    pol, val = Evaluator(net, "cuda").evaluate([HexState.initial(N, True)])
    assert pol.shape[0] == 1 and val.shape[0] == 1, "cuda evaluator returned wrong shape"
    print(f"  cuda evaluator ok (policy{tuple(pol.shape)} value{tuple(val.shape)})")

    # Self-play + arena through the GPU inference server (on by default for cuda) — the deploy path.
    sd = net.state_dict()
    assert cfg.use_inference_server(), "inference server should be on for cuda by default"
    par = selfplay.generate(cfg, sd, num_games=2, base_seed=10)
    assert par, "cuda self-play via the inference server produced no samples"
    wr = arena.play_match_parallel(cfg, sd, sd, num_games=2, simulations=8, base_seed=11)
    assert 0.0 <= wr <= 1.0, "cuda arena via the inference server returned an invalid win rate"
    print(f"  cuda inference-server: {len(par)} self-play samples, arena {wr:.0%}")
    print("  CUDA CHECKS PASSED")


if __name__ == "__main__":
    main()
