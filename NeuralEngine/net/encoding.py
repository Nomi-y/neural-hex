"""Board <-> network tensor encoding, with canonicalisation.

Hex is symmetric under transpose + colour-swap: a "Blue to move connecting left-right" position is the
same game as the transposed board with "Red to move connecting top-bottom". We exploit this by always
presenting the network a *canonical* view in which the side to move is connecting top<->bottom. The net
therefore only ever has to understand one orientation, which roughly halves what it must learn and makes
its value/policy directly comparable across both colours.

Planes (all from the side-to-move's canonical perspective), shape (5, N, N):
  0: my stones
  1: opponent stones
  2: my edges   (constant: top & bottom rows — the edges I am trying to join)
  3: opponent edges (constant: left & right columns)
  4: ones (bias / lets the net detect the board frame)

Action space is N*N cell placements plus one swap action (index N*N). Canonical<->real action mapping is
the same transpose used for the planes (an involution); the swap action is orientation-independent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from hex.board import HexState, RED, BLUE


@lru_cache(maxsize=None)
def _edge_masks(size: int):
    my_edge = np.zeros((size, size), dtype=np.float32)
    my_edge[0, :] = 1.0
    my_edge[size - 1, :] = 1.0
    opp_edge = np.zeros((size, size), dtype=np.float32)
    opp_edge[:, 0] = 1.0
    opp_edge[:, size - 1] = 1.0
    return my_edge, opp_edge


def is_transposed(to_move: int) -> bool:
    return to_move == BLUE


def encode(state: HexState) -> np.ndarray:
    """Canonical planes for a single state -> float32 array (5, N, N)."""
    size = state.size
    board = state.cells.reshape(size, size)
    if state.to_move == RED:
        me = (board == RED).astype(np.float32)
        opp = (board == BLUE).astype(np.float32)
    else:
        me = (board == BLUE).astype(np.float32).T.copy()
        opp = (board == RED).astype(np.float32).T.copy()
    my_edge, opp_edge = _edge_masks(size)
    ones = np.ones((size, size), dtype=np.float32)
    return np.stack([me, opp, my_edge, opp_edge, ones], axis=0)


def encode_batch(states: List[HexState]) -> np.ndarray:
    return np.stack([encode(s) for s in states], axis=0)


def canon_to_real_action(to_move: int, size: int, canon_action: int) -> int:
    swap_action = size * size
    if canon_action == swap_action:
        return swap_action
    if not is_transposed(to_move):
        return canon_action
    row, col = divmod(canon_action, size)
    return col * size + row


def real_to_canon_action(to_move: int, size: int, real_action: int) -> int:
    # The transpose is its own inverse, so this is identical to canon_to_real.
    return canon_to_real_action(to_move, size, real_action)


def canonical_legal_mask(state: HexState) -> np.ndarray:
    """Legal-move mask in *canonical* action order (so it lines up with the network's policy head)."""
    size = state.size
    real_mask = state.legal_mask()
    canon = np.zeros_like(real_mask)
    for real_a in np.nonzero(real_mask)[0]:
        canon[real_to_canon_action(state.to_move, size, int(real_a))] = True
    return canon
