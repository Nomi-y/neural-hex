"""Self-play game generation — the source of training data.

Plays many games in parallel with batched MCTS, recording (canonical planes, canonical MCTS policy,
side-to-move) at every position; once a game ends, each record is labelled with z = +1/-1 for whether
that side went on to win. Early plies are sampled with temperature for exploration, later plies played
greedily. A clearly lost game may resign early to save compute.

`generate()` runs in-process on a GPU/MPS device (one big batched actor saturates it) or fans out across
all CPU cores (one process per core) when there is no accelerator — the "use all resources" requirement.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List, Tuple

import numpy as np

from config import Config
from hex.board import HexState
from net.model import build_net
from net.evaluator import Evaluator
from net.encoding import encode, real_to_canon_action
from search import mcts

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

            # Optional early resignation: if the search thinks the side to move is lost, concede.
            root_value = (sum(root.child_w.values()) / root.sum_n) if root.sum_n > 0 else 0.0
            if (cfg.selfplay.resign_threshold < 0 and g.ply >= cfg.selfplay.resign_min_ply
                    and root.solved_value is None and root_value < cfg.selfplay.resign_threshold):
                g.done = True
                g.winner = 3 - g.state.to_move  # the other colour (RED=1/BLUE=2)
                continue

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

def _worker(args) -> List[Sample]:
    cfg, state_dict, num_games, seed = args
    import torch

    torch.set_num_threads(1)  # each actor is single-threaded; parallelism comes from many actors
    net = build_net(cfg)
    net.load_state_dict(state_dict)
    net.eval()
    evaluator = Evaluator(net, "cpu")
    rng = np.random.default_rng(seed)
    return play_games(evaluator, cfg, num_games, add_noise=True, rng=rng)


def generate(cfg: Config, state_dict, num_games: int, base_seed: int) -> List[Sample]:
    """Generate `num_games` self-play games using the given network weights."""
    if cfg.device in ("cuda", "mps"):
        import torch

        net = build_net(cfg).to(cfg.device)
        net.load_state_dict(state_dict)
        net.eval()
        evaluator = Evaluator(net, cfg.device)
        rng = np.random.default_rng(base_seed)
        # One actor, but games_per_generation are batched in chunks of parallel_games for big NN batches.
        out: List[Sample] = []
        remaining = num_games
        while remaining > 0:
            chunk = min(cfg.selfplay.parallel_games, remaining)
            out.extend(play_games(evaluator, cfg, chunk, add_noise=True, rng=rng))
            remaining -= chunk
        return out

    # CPU: fan out across cores.
    actors = cfg.resolve_actors()
    per_actor = max(1, num_games // actors)
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    tasks = [(cfg, cpu_state, per_actor, base_seed + i) for i in range(actors)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=actors) as pool:
        results = pool.map(_worker, tasks)
    return [s for r in results for s in r]
