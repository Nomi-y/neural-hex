"""Self-play game generation — the source of training data.

Plays many games in parallel with batched MCTS, recording (canonical planes, canonical MCTS policy,
side-to-move) at every position; once a game ends, each record is labelled with z = +1/-1 for whether
that side went on to win.  Early plies are sampled with temperature for exploration, later plies played
greedily.  Every game plays to the last stone — no early resignation.

`generate()` runs in-process on a GPU/MPS device (one big batched actor saturates it) or fans out across
all CPU cores (one process per core) when there is no accelerator — the "use all resources" requirement.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import Config
from hex.board import HexState
from net.model import build_net, CleanStateDict
from net.evaluator import Evaluator
from net.encoding import encode, real_to_canon_action
from search import mcts
from train.clock import log

Sample = Tuple[np.ndarray, np.ndarray, float]


class _Game:
    __slots__ = ("state", "history", "ply", "done", "winner")

    def __init__(self, state: HexState) -> None:
        self.state = state
        self.history: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (canonical planes, canonical pi, to_move)
        self.ply = 0
        self.done = False
        self.winner = 0


def _canonical_pi(state: HexState, pi_real: np.ndarray) -> np.ndarray:
    canon = np.zeros_like(pi_real)
    size = state.size
    nz = np.nonzero(pi_real)[0]
    for a in nz:
        canon[real_to_canon_action(state.to_move, size, int(a))] = pi_real[a]
    return canon


def play_games(evaluator: Evaluator, cfg: Config, num_games: int, add_noise: bool, rng: np.random.Generator,
               simulations: int | None = None) -> List[Sample]:
    sims = simulations if simulations is not None else cfg.mcts.simulations
    num_actions = cfg.game.num_actions
    games = [_Game(HexState.initial(cfg.game.board_size, cfg.game.swap_rule)) for _ in range(num_games)]
    samples: List[Sample] = []

    while True:
        active = [g for g in games if not g.done]
        if not active:
            break
        roots = [mcts.make_root(g.state) for g in active]
        mcts.run_batched(roots, evaluator, cfg, sims, add_noise, rng)

        for g, root in zip(active, roots):
            pi_real = mcts.policy_distribution(root, num_actions)
            g.history.append((encode(g.state), _canonical_pi(g.state, pi_real), g.state.to_move))

            temperature = cfg.selfplay.temperature if g.ply < cfg.selfplay.temperature_moves else 0.0
            action = mcts.select_action(root, num_actions, temperature, rng)

            g.state = g.state.play(action)
            g.ply += 1
            if g.state.is_terminal():
                g.done = True
                g.winner = g.state.winner

    for g in games:
        for planes, pi, to_move in g.history:
            z = 1.0 if g.winner == to_move else -1.0
            samples.append((planes, pi, z))
    return samples


# ---- process-level parallelism for CPU boxes ----

# Per-worker state, built once in the Pool initializer so the weights are pickled once per process
# (not once per chunk). Lets us hand out many small chunks for load-balancing without re-sending the net.
_WORKER: dict = {}


def build_eval_net(cfg: Config, np_state) -> Evaluator:
    """Rebuild a net on cfg.device from numpy weights and wrap it in an Evaluator.

    Weights arrive as numpy (not torch tensors) so the inter-process transfer avoids torch's
    shared-memory tensor reducer, which would hold an FD per tensor per worker in the parent and
    exhaust the open-file limit (EMFILE) once many workers run.
    """
    import torch

    if cfg.device == "cpu":
        torch.set_num_threads(1)  # each CPU worker is single-threaded; parallelism is across workers
    net = build_net(cfg).to(cfg.device)
    # np_state may carry torch.compile's '_orig_mod.' prefix; strip it so it loads into a plain net.
    net.load_state_dict(CleanStateDict({k: torch.from_numpy(v).to(cfg.device) for k, v in np_state.items()}))
    net.eval()
    return Evaluator(net, cfg.device)


def to_numpy_state(state_dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def _init_worker(cfg: Config, np_state) -> None:
    _WORKER["cfg"] = cfg
    _WORKER["evaluator"] = build_eval_net(cfg, np_state)


def _play_chunk(args) -> Tuple[int, List[Sample]]:
    num_games, seed = args
    rng = np.random.default_rng(seed)
    samples = play_games(_WORKER["evaluator"], _WORKER["cfg"], num_games, add_noise=True, rng=rng)
    return num_games, samples  # report game count so the parent can track progress in completion order


def split_evenly(total: int, parts: int) -> List[int]:
    """Split `total` into `parts` near-equal sizes, each >= 1."""
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def chunk_sizes(cfg: Config, num_games: int, actors: int) -> List[int]:
    """Per-task game counts, tuned per device.

    GPU: chunks sized to parallel_games (the optimal batch size for the GPU forward pass),
    with 2-3× more chunks than workers so imap_unordered naturally balances — fast workers
    grab extra chunks instead of idling behind a slow one.  CPU: many small chunks so fast
    cores keep grabbing work; batch size is irrelevant there (the net runs one sample at a
    time on CPU regardless)."""
    if cfg.device == "cuda":
        chunk = max(1, cfg.selfplay.parallel_games)
        parts = max(actors * 2, (num_games + chunk - 1) // chunk)
        return split_evenly(num_games, parts)
    return split_evenly(num_games, actors * 4)


def drain_pool(pool, result_iter, num_tasks: int, timeout: float, on_result: Callable[[object], None]) -> bool:
    """Consume `num_tasks` results from an `imap_unordered` iterator, calling `on_result(item)` for each.

    If no result arrives within `timeout` seconds (<=0 disables the watchdog), assume a worker has died
    or deadlocked: terminate the pool and stop early. Returns True if every result was collected, False
    if it timed out. This is the guard against the imap_unordered hang where one dead worker would
    otherwise block the parent forever (no per-result deadline of its own)."""
    to = timeout if timeout and timeout > 0 else None
    for _ in range(num_tasks):
        try:
            item = result_iter.next(to)
        except mp.TimeoutError:
            pool.terminate()
            return False
        on_result(item)
    return True


ProgressFn = Callable[[int, int], None]


def generate(cfg: Config, state_dict, num_games: int, base_seed: int,
             progress: Optional[ProgressFn] = None) -> List[Sample]:
    """Generate `num_games` self-play games using the given network weights.

    `progress(done, total)` (optional) is called from this process as chunks complete (in completion
    order), so the caller can log incremental progress.

    MPS runs a single in-process actor (multi-process CUDA-style sharing isn't reliable on Apple).
    Both CUDA and CPU fan out across many worker processes — on CUDA so the GPU stays fed while many
    cores do tree work in parallel (each worker holds its own copy of the net on the shared card); on
    CPU so every core generates games. The net is loaded once per worker via the Pool initializer.
    """
    actors = cfg.resolve_actors()

    if cfg.device == "mps" or actors <= 1:
        evaluator = build_eval_net(cfg, to_numpy_state(state_dict))
        rng = np.random.default_rng(base_seed)
        out: List[Sample] = []
        remaining = num_games
        done = 0
        while remaining > 0:
            chunk = min(cfg.selfplay.parallel_games, remaining)
            out.extend(play_games(evaluator, cfg, chunk, add_noise=True, rng=rng))
            remaining -= chunk
            done += chunk
            if progress:
                progress(done, num_games)
        return out

    np_state = to_numpy_state(state_dict)
    sizes = chunk_sizes(cfg, num_games, actors)
    tasks = [(size, base_seed + i) for i, size in enumerate(sizes)]
    ctx = mp.get_context("spawn")
    results: List[List[Sample]] = []
    done = 0

    def _on(item) -> None:
        nonlocal done
        n, samples = item
        results.append(samples)
        done += n
        if progress:
            progress(done, num_games)

    with ctx.Pool(processes=actors, initializer=_init_worker, initargs=(cfg, np_state)) as pool:
        it = pool.imap_unordered(_play_chunk, tasks)
        ok = drain_pool(pool, it, len(tasks), cfg.train.selfplay_timeout, _on)
    if not ok:
        log(f"[self-play] WARNING: worker watchdog fired after {cfg.train.selfplay_timeout:.0f}s with no "
            f"result — terminated pool, continuing with {done}/{num_games} games "
            f"({sum(len(r) for r in results)} samples).")
    return [s for r in results for s in r]
