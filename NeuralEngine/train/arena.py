"""Arena: pit a freshly trained network against the current best to decide promotion.

This is the "reward-based evolution" gate — a new network only becomes the self-play generator if it
actually beats the incumbent over a set of games (alternating colours for fairness), played greedily
with a modest search. Both networks evaluate in batches across the games that are currently on their
turn.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from config import Config
from hex.board import HexState, RED, BLUE
from net.evaluator import Evaluator
from search import mcts


class _Game:
    __slots__ = ("state", "candidate_color", "done", "winner")

    def __init__(self, state: HexState, candidate_color: int) -> None:
        self.state = state
        self.candidate_color = candidate_color
        self.done = False
        self.winner = 0


def play_match(cfg: Config, candidate: Evaluator, best: Evaluator, num_games: int, simulations: int,
               rng: np.random.Generator, progress: Optional[Callable[[int, int], None]] = None) -> float:
    """Return the candidate's win rate over `num_games` (it plays RED in half, BLUE in the other half).

    `progress(done, total)` (optional) is called as games finish so a caller can log arena progress.
    """
    num_actions = cfg.game.num_actions
    games = [
        _Game(HexState.initial(cfg.game.board_size, cfg.game.swap_rule), RED if i % 2 == 0 else BLUE)
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

    wins = sum(1 for g in games if g.winner == g.candidate_color)
    return wins / num_games
