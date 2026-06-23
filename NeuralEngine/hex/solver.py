"""Bounded exact endgame solver — "solve endgame board states and find winning paths".

A negamax alpha-beta search that resolves a position to a *proven* win or loss (Hex has no draws) by
playing the game out over the remaining empty cells. It is only invoked when few cells remain
(`solver_empty_threshold` in config) so the O(b^n) blow-up stays tiny; a node budget bails out to None
("unknown") on the rare wide endgame. A transposition table memoises solved positions within a search
(cheap precomputation of winning sub-positions), and bridge/centre move ordering makes alpha-beta cut
hard. Results feed MCTS (a solved node becomes an exact terminal) and the engine (it can finish won
games precisely instead of trusting the net near the end).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .board import HexState, EMPTY, RED, BLUE
from .bridges import forced_response, bridge_carrier_moves


class _BudgetExceeded(Exception):
    pass


def _ordered_actions(state: HexState, last_move: Optional[int]) -> List[int]:
    """Order moves to maximise alpha-beta cutoffs: a forced bridge response first (it is almost always
    correct), then bridge-completing moves, then centre-out."""
    legal = [a for a in range(state.num_cells) if state.cells[a] == EMPTY]
    size = state.size
    mid = (size - 1) / 2.0
    priority: Dict[int, float] = {}

    forced = forced_response(state.cells, size, state.to_move, last_move) if last_move is not None else None
    bridge_moves = set(bridge_carrier_moves(state.cells, size, state.to_move))
    for a in legal:
        row, col = divmod(a, size)
        score = -((row - mid) ** 2 + (col - mid) ** 2)  # centre-out
        if a in bridge_moves:
            score += 1000.0
        if forced is not None and a == forced:
            score += 1_000_000.0
        priority[a] = score
    legal.sort(key=lambda a: priority[a], reverse=True)
    return legal


def solve(state: HexState, node_budget: int, last_move: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """Solve `state` for the side to move. Returns (value, best_action):
      value = +1 (side to move wins with perfect play), -1 (loses), or None if the node budget was hit.
      best_action is the proven winning/best move (None when value is None)."""
    tt: Dict[bytes, int] = {}
    nodes = [0]

    def negamax(s: HexState, alpha: int, beta: int, prev_move: Optional[int]) -> int:
        if s.is_terminal():
            return -1  # side to move has already lost (opponent connected on the prior ply)
        key = s.key()
        cached = tt.get(key)
        if cached is not None:
            return cached
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise _BudgetExceeded()
        best = -2
        for a in _ordered_actions(s, prev_move):
            value = -negamax(s.play(a), -beta, -alpha, a)
            if value > best:
                best = value
            if value > alpha:
                alpha = value
            if alpha >= beta or best == 1:
                break
        tt[key] = best
        return best

    try:
        best_value = -2
        best_action: Optional[int] = None
        alpha = -1
        for a in _ordered_actions(state, last_move):
            value = -negamax(state.play(a), -1, -alpha, a)
            if value > best_value:
                best_value, best_action = value, a
            if value > alpha:
                alpha = value
            if best_value == 1:
                break
        return best_value, best_action
    except _BudgetExceeded:
        return None, None


def maybe_solve(state: HexState, empty_threshold: int, node_budget: int, last_move: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """Solve only when the endgame is small enough to stay cheap; otherwise (None, None)."""
    empties = int(np.count_nonzero(state.cells == EMPTY))
    if empties > empty_threshold:
        return None, None
    return solve(state, node_budget, last_move)
