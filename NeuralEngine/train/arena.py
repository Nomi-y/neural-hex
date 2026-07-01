"""Arena: pit a freshly trained network against the current best to decide promotion.

This is the "reward-based evolution" gate — a new network only becomes the self-play generator if it
actually beats the incumbent over a set of games (alternating colours for fairness), played greedily
with a modest search. Both networks evaluate in batches across the games that are currently on their
turn.

Like self-play, the match fans out across many worker processes (each holding its own copy of both
nets on the shared GPU, or single-threaded on CPU) so a fast card / many cores stay saturated rather
than bottlenecked on one Python actor. MPS runs in-process.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, List, Optional

import numpy as np

from config import Config
from hex.board import HexState, RED, BLUE
from net.evaluator import Evaluator
from search import mcts
from train.selfplay import build_eval_net, to_numpy_state, chunk_sizes, drain_pool, _claim_worker_index
from train.clock import log


class _Game:
    __slots__ = ("state", "candidate_color", "done", "winner")

    def __init__(self, state: HexState, candidate_color: int) -> None:
        self.state = state
        self.candidate_color = candidate_color
        self.done = False
        self.winner = 0


def _run_games(cfg: Config, candidate: Evaluator, best: Evaluator, num_games: int, simulations: int,
               rng: np.random.Generator, color_offset: int = 0,
               progress: Optional[Callable[[int, int], None]] = None) -> int:
    """Play `num_games` and return how many the candidate won. `color_offset` keeps the global RED/BLUE
    alternation consistent when games are split into chunks (chunk starting at global index s passes
    color_offset=s), so exactly half the whole match has the candidate as RED."""
    num_actions = cfg.game.num_actions
    games = [
        _Game(HexState.initial(cfg.game.board_size, cfg.game.swap_rule),
              RED if (color_offset + i) % 2 == 0 else BLUE)
        for i in range(num_games)
    ]

    finished = 0
    while True:
        active = [g for g in games if not g.done]
        if not active:
            break
        # Split by whose turn it is *before* anyone moves, so each game is played exactly once this
        # round (recomputing after a move would flip turns and double-move some games).
        cand_group = [g for g in active if g.state.to_move == g.candidate_color]
        best_group = [g for g in active if g.state.to_move != g.candidate_color]
        for evaluator, group in ((candidate, cand_group), (best, best_group)):
            if not group:
                continue
            roots = [mcts.make_root(g.state) for g in group]
            mcts.run_batched(roots, evaluator, cfg, simulations, add_noise=False, rng=rng)
            for g, root in zip(group, roots):
                action = mcts.select_action(root, num_actions, 0.0, rng)
                g.state = g.state.play(action)
                if g.state.is_terminal():
                    g.done = True
                    g.winner = g.state.winner
        newly_done = sum(1 for g in games if g.done)
        if progress and newly_done != finished:
            finished = newly_done
            progress(finished, num_games)

    return sum(1 for g in games if g.winner == g.candidate_color)


def play_match(cfg: Config, candidate: Evaluator, best: Evaluator, num_games: int, simulations: int,
               rng: np.random.Generator, progress: Optional[Callable[[int, int], None]] = None) -> float:
    """In-process match (used by the smoke test). Returns the candidate's win rate."""
    return _run_games(cfg, candidate, best, num_games, simulations, rng, 0, progress) / num_games


# ---- process-level parallelism (mirrors selfplay) ----

_ARENA: dict = {}


def _init_arena_worker(cfg: Config, cand_np, best_np, device: str) -> None:
    _ARENA["cfg"] = cfg
    _ARENA["candidate"] = build_eval_net(cfg, cand_np, device)
    _ARENA["best"] = build_eval_net(cfg, best_np, device)


def _init_arena_worker_remote(cfg: Config, servers_data, counter, lock) -> None:
    """Server mode (single or multi-GPU): both nets live on the GPU server
    (net_id 0=candidate, 1=best); the worker searches on CPU and evaluates
    each through its assigned GPU's RemoteEvaluator."""
    from train.inference_server import RemoteEvaluator
    idx = _claim_worker_index(counter, lock, sum(len(sd[1]) for sd in servers_data))
    gpu = idx % len(servers_data)
    req_q, resp_qs = servers_data[gpu]
    worker_slot = (idx // len(servers_data)) % len(resp_qs)
    _ARENA["cfg"] = cfg
    _ARENA["candidate"] = RemoteEvaluator(0, worker_slot, req_q, resp_qs[worker_slot])
    _ARENA["best"] = RemoteEvaluator(1, worker_slot, req_q, resp_qs[worker_slot])


def _play_arena_chunk(args):
    num_games, color_offset, simulations, seed = args
    rng = np.random.default_rng(seed)
    wins = _run_games(_ARENA["cfg"], _ARENA["candidate"], _ARENA["best"], num_games, simulations, rng,
                      color_offset)
    return num_games, wins


def play_match_parallel(cfg: Config, cand_state, best_state, num_games: int, simulations: int,
                        base_seed: int, progress: Optional[Callable[[int, int], None]] = None) -> float:
    """Fan the gating match out across worker processes; return the candidate's win rate. Weights are
    passed as torch state dicts (converted to numpy for transfer; workers rebuild on cfg.device)."""
    actors = cfg.resolve_actors()
    cand_np = to_numpy_state(cand_state)
    best_np = to_numpy_state(best_state)

    if cfg.device == "mps" or actors <= 1:
        candidate = build_eval_net(cfg, cand_np)
        best = build_eval_net(cfg, best_np)
        rng = np.random.default_rng(base_seed)
        return _run_games(cfg, candidate, best, num_games, simulations, rng, 0, progress) / num_games

    use_server = cfg.use_inference_server()
    ctx = mp.get_context("spawn")
    total_wins = 0
    done = 0

    def _on(item) -> None:
        nonlocal total_wins, done
        n, wins = item
        total_wins += wins
        done += n
        if progress:
            progress(done, num_games)

    servers: list = []
    if use_server:
        from train.inference_server import InferenceServer
        ngpus = cfg.num_gpus()
        workers_per_gpu = actors // ngpus
        alloc = [workers_per_gpu] * ngpus
        for i in range(actors - workers_per_gpu * ngpus):
            alloc[i] += 1
        alloc = [a for a in alloc if a > 0]
        ngpus = len(alloc)

        for gpu_id, nw in enumerate(alloc):
            srv = InferenceServer(cfg, [cand_np, best_np], cfg.device, nw, ctx, gpu_id=gpu_id)
            srv.start()
            servers.append(srv)

        servers_data = [(s.req_q, s.resp_qs) for s in servers]
        counter, lock = servers[0].counter, servers[0].lock
        initializer, initargs = _init_arena_worker_remote, (cfg, servers_data, counter, lock)
        sizes = chunk_sizes(cfg, num_games, actors, cfg.device, streaming=True)
        gpu_label = f"gpu-server({cfg.device})" if ngpus == 1 else f"gpu-server({ngpus}×{cfg.device})"
    else:
        chunk_dev = cfg.worker_eval_device()
        sizes = chunk_sizes(cfg, num_games, actors, chunk_dev)
        initializer, initargs = _init_arena_worker, (cfg, cand_np, best_np, chunk_dev)
        gpu_label = chunk_dev

    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    tasks = [(size, int(off), simulations, base_seed + i)
             for i, (size, off) in enumerate(zip(sizes, offsets))]
    log(f"[arena] fanning {num_games} games across {actors} actors (eval on {gpu_label}), "
        f"{len(sizes)} chunks (avg {num_games // max(1, len(sizes))} games/chunk, "
        f"max parallel per chunk={min(cfg.selfplay.parallel_games, max(sizes) if sizes else 0)})")

    try:
        with ctx.Pool(processes=actors, initializer=initializer, initargs=initargs) as pool:
            it = pool.imap_unordered(_play_arena_chunk, tasks)
            ok = drain_pool(pool, it, len(tasks), cfg.train.arena_timeout, _on)
    finally:
        for srv in servers:
            srv.stop()
    if not ok:
        log(f"[arena] WARNING: worker watchdog fired after {cfg.train.arena_timeout:.0f}s with no result "
            f"— terminated pool. Scoring {done}/{num_games} completed games; the rest count as losses, "
            f"so a wedged arena won't falsely promote.")
    return total_wins / num_games
