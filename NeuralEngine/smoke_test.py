"""Fast end-to-end sanity check — runs every component on a tiny board/net in seconds.

Use it after any change, before committing:
    python smoke_test.py

It exercises: the rules engine, bridges + endgame solver, encoding/canonicalisation, the network,
batched MCTS, self-play sample generation, a few optimiser steps, and an arena match — on CPU.

If a CUDA GPU is present it ALSO runs the GPU-only paths that CPU smoke can't reach (device-property
logging, the arch/kernel guard, a real GPU forward, self-play + arena through the GPU inference
server, and a bounded few-generation run of the real training entrypoint that exercises the logging
format and the pipelined self-play) — run it on a CUDA box before committing. With no GPU (e.g. the
Docker build host) the GPU section is skipped, so it stays a valid build-time guard.
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
    cuda_training_loops()
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

    # torch.compile warmup: a lazy compile() doesn't fail until first forward.
    # The real training net goes through this; verify Triton can JIT a kernel
    # (needs gcc in the image — missing = crash 37 min into a paid run).
    try:
        net2 = build_net(cfg).to("cuda")
        net2 = torch.compile(net2, mode="reduce-overhead")
        dummy = torch.zeros(1, cfg.net.in_planes, N, N, device="cuda")
        _ = net2(dummy)
        print("  cuda torch.compile warmup ok")
    except Exception as e:
        print(f"  cuda torch.compile warmup FAILED: {e}")
        print("  (install gcc in the container image — Triton JIT needs a C compiler)")

    # Self-play + arena through the GPU inference server (on by default for cuda) — the deploy path.
    sd = net.state_dict()
    assert cfg.use_inference_server(), "inference server should be on for cuda by default"
    par = selfplay.generate(cfg, sd, num_games=2, base_seed=10)
    assert par, "cuda self-play via the inference server produced no samples"
    wr = arena.play_match_parallel(cfg, sd, sd, num_games=2, simulations=8, base_seed=11)
    assert 0.0 <= wr <= 1.0, "cuda arena via the inference server returned an invalid win rate"
    print(f"  cuda inference-server: {len(par)} self-play samples, arena {wr:.0%}")
    print("  CUDA CHECKS PASSED")


# Every training log line must look like `[HH:MM:SS +H:MM:SS] [gen N] [tag] …`, with the [gen N]
# segment present once inside a generation and omitted for startup/shutdown lines.
_LOG_LINE = __import__("re").compile(
    r"^\[\d{2}:\d{2}:\d{2} \+\d+:\d{2}:\d{2}\]( \[gen \d+\])? \[[\w-]+\] ")


def cuda_training_loops(generations: int = 2) -> None:
    """Run the REAL training entrypoint (`python -m train.train`) for a bounded few generations on
    the GPU — the closest thing to a live run — so the logging changes and the self-play pipelining
    are exercised end-to-end, not just their units.  Skips cleanly without CUDA (build-guard safe).

    Asserts: the run completes the requested generations; the self-play fan-out advertises the
    pipeline; and EVERY emitted log line carries the `[time +elapsed] [gen] [tag]` prefix."""
    if not torch.cuda.is_available():
        print("  cuda training loop: no GPU — skipped")
        return

    import subprocess
    import sys
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(prefix="hex_smoke_") as tmp:
        # Tiny, fast, CUDA — env overrides win over hyperparams.toml (read at config import in the
        # child). Short logging intervals so [hw]/[infer]/[selfplay] progress lines actually appear
        # in a seconds-long run and get format-checked. PIPELINE_SHARDS>1 exercises the optimization.
        env = {
            **os.environ,
            "DEVICE": "cuda",
            "BOARD_SIZE": "5", "NET_CHANNELS": "16", "NET_BLOCKS": "2",
            "NET_SE": "false", "NET_VALUE_HIDDEN": "32",
            "MCTS_SIMS": "12", "PARALLEL_GAMES": "8", "PIPELINE_SHARDS": "2",
            "NUM_ACTORS": "2", "GAMES_PER_GEN": "8", "BATCH_SIZE": "32",
            "TRAIN_STEPS": "5", "REPLAY_BUFFER": "5000",
            "ARENA_GAMES": "2", "ARENA_SIMS": "8",
            "TRAIN_HOURS": "1", "TRAIN_MAX_GENS": str(generations), "SEED": "0",
            "CHECKPOINT_DIR": os.path.join(tmp, "ckpt"), "LOG_DIR": os.path.join(tmp, "logs"),
            "HW_LOG_INTERVAL": "1", "INFERENCE_LOG_EVERY": "1",
            "SELFPLAY_PROGRESS_INTERVAL": "1", "TRAIN_LOG_INTERVAL": "1",
        }
        print(f"  cuda training loop: running {generations} generations via `python -m train.train`…")
        proc = subprocess.run([sys.executable, "-m", "train.train"], cwd=here, env=env,
                              capture_output=True, text=True, timeout=600)

    out = proc.stdout
    if proc.returncode != 0:
        print(out)
        print(proc.stderr[-2000:])
        raise AssertionError(f"training run exited {proc.returncode}")

    log_lines = [ln for ln in out.splitlines() if ln.startswith("[")]
    bad = [ln for ln in log_lines if not _LOG_LINE.match(ln)]
    assert not bad, f"{len(bad)} log line(s) missing the [time +elapsed] [gen] [tag] prefix, e.g.:\n    " \
                    + "\n    ".join(bad[:5])

    done = [ln for ln in log_lines if "[summary] DONE" in ln]
    assert len(done) >= generations, f"expected {generations} generation summaries, got {len(done)}"
    assert any("pipeline" in ln for ln in log_lines), "self-play fan-out never reported the pipeline"

    # Progress dedupe: at most one [selfplay] N/total progress line per second (interval=1s here),
    # not one per worker — a decent proxy for the fan-out no longer being actors× too chatty.
    print(f"  ✓ cuda training loop: {len(done)} generations, {len(log_lines)} tagged log lines, "
          f"pipelined self-play OK")
    print("  CUDA TRAINING LOOP PASSED")


if __name__ == "__main__":
    main()
