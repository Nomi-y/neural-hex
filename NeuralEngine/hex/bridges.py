"""Bridge / virtual-connection patterns — the "recognise and play patterns" requirement.

A *bridge* is the fundamental Hex pattern: two same-colour stones two steps apart that share exactly
two empty common neighbours (the *carriers*). The connection is safe — if the opponent plays one
carrier you reply on the other — so a bridge behaves like a solid link the opponent cannot cut in one
move. Chaining bridges (and bridges to a player's own edge) gives a *virtual connection*: a position
that is won with correct (often forced) responses even though the stones are not yet solidly joined.

This module provides:
  - the six bridge offsets and their carrier pairs (precomputed per board size),
  - `forced_response`: maintain a bridge when the opponent intrudes on a carrier (used by the solver and
    by the engine to play patterns soundly),
  - `virtual_connection`: an optimistic union-find VC check (adjacency + intact bridges + edges) used as
    a fast value/endgame signal. It is heuristic (it ignores carrier overlap / double-threats), so it is
    never treated as an authoritative win — that remains board.detect_win.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from .board import EMPTY, RED, BLUE, other, neighbours_table

# Each entry: ((drow, dcol) to the bridged cell, ((dr,dc), (dr,dc)) of the two carrier cells).
_BRIDGE_PATTERNS: Tuple[Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]], ...] = (
    ((-1, -1), ((-1, 0), (0, -1))),
    ((-2, 1), ((-1, 0), (-1, 1))),
    ((-1, 2), ((-1, 1), (0, 1))),
    ((1, 1), ((0, 1), (1, 0))),
    ((2, -1), ((1, 0), (1, -1))),
    ((1, -2), ((1, -1), (0, -1))),
)


@lru_cache(maxsize=None)
def bridges_table(size: int) -> Tuple[Tuple[Tuple[int, int, int], ...], ...]:
    """For each cell: tuples (other_cell, carrier_a, carrier_b) for every in-bounds bridge."""
    table: List[Tuple[Tuple[int, int, int], ...]] = []
    for index in range(size * size):
        row, col = divmod(index, size)
        out: List[Tuple[int, int, int]] = []
        for (dr, dc), ((c1r, c1c), (c2r, c2c)) in _BRIDGE_PATTERNS:
            br, bc = row + dr, col + dc
            a_r, a_c = row + c1r, col + c1c
            b_r, b_c = row + c2r, col + c2c
            if not (0 <= br < size and 0 <= bc < size):
                continue
            if not (0 <= a_r < size and 0 <= a_c < size and 0 <= b_r < size and 0 <= b_c < size):
                continue
            out.append((br * size + bc, a_r * size + a_c, b_r * size + b_c))
        table.append(tuple(out))
    return tuple(table)


def forced_response(cells: np.ndarray, size: int, color: int, opponent_move: int) -> Optional[int]:
    """If the opponent's last move intruded on a carrier of one of `color`'s intact bridges (both
    endpoints `color`, the other carrier still empty), return the carrier that restores the bridge."""
    for end_a, end_b, carrier_a, carrier_b in _affected_bridges(size, opponent_move):
        if cells[end_a] != color or cells[end_b] != color:
            continue  # only a real same-colour bridge is worth defending
        if opponent_move == carrier_a:
            other_carrier = carrier_b
        elif opponent_move == carrier_b:
            other_carrier = carrier_a
        else:
            continue
        if cells[other_carrier] == EMPTY:
            return int(other_carrier)
    return None


@lru_cache(maxsize=None)
def _carrier_index(size: int):
    """Reverse index: carrier cell -> bridges (end_a, end_b, carrier_a, carrier_b) that use it as a
    carrier. Both endpoints are kept so a bridge is only defended when both ends are actually ours."""
    table = bridges_table(size)
    rev: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(size * size)]
    for start in range(size * size):
        for (endpoint, ca, cb) in table[start]:
            rev[ca].append((start, endpoint, ca, cb))
            rev[cb].append((start, endpoint, ca, cb))
    return tuple(tuple(x) for x in rev)


def _affected_bridges(size: int, carrier: int):
    return _carrier_index(size)[carrier]


@lru_cache(maxsize=None)
def _edge_cells(size: int):
    """(red_start, red_end, blue_start, blue_end) cell indices — the edges each colour must join."""
    n = size * size
    return (
        tuple(range(size)),            # RED start: top row
        tuple(range(n - size, n)),     # RED end:   bottom row
        tuple(range(0, n, size)),      # BLUE start: left column
        tuple(range(size - 1, n, size))  # BLUE end:  right column
    )


def virtual_connection(cells: np.ndarray, size: int, color: int) -> bool:
    """Optimistic VC: does `color` connect its edges through solid stones, intact bridges, and edge
    contact? Heuristic only (see module docstring).

    Runs over a Python list (cells.tolist()) with precomputed edge-cell tuples — avoids per-cell numpy
    scalar reads and divmod in this per-leaf hot loop."""
    n = size * size
    # Disjoint-set over cells plus two virtual edge nodes.
    start_node, end_node = n, n + 1
    parent = list(range(n + 2))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cl = cells.tolist()
    nbrs = neighbours_table(size)
    bridges = bridges_table(size)
    red_start, red_end, blue_start, blue_end = _edge_cells(size)
    starts, ends = (red_start, red_end) if color == RED else (blue_start, blue_end)

    # Edge contact for this colour's direction.
    for index in starts:
        if cl[index] == color:
            union(index, start_node)
    for index in ends:
        if cl[index] == color:
            union(index, end_node)
    for index in range(n):
        if cl[index] != color:
            continue
        # Solid links.
        for nb in nbrs[index]:
            if cl[nb] == color:
                union(index, nb)
        # Intact bridges (both carriers empty).
        for (endpoint, ca, cb) in bridges[index]:
            if cl[endpoint] == color and cl[ca] == EMPTY and cl[cb] == EMPTY:
                union(index, endpoint)
    return find(start_node) == find(end_node)


def bridge_carrier_moves(cells: np.ndarray, size: int, color: int) -> List[int]:
    """Empty cells that would *complete* a bridge for `color` (one endpoint placed, the other empty
    with both carriers free) — strong candidate moves used to order the endgame solver."""
    moves: set[int] = set()
    bridges = bridges_table(size)
    for index in range(size * size):
        if cells[index] != color:
            continue
        for (endpoint, ca, cb) in bridges[index]:
            if cells[endpoint] == EMPTY and cells[ca] == EMPTY and cells[cb] == EMPTY:
                moves.add(int(endpoint))
    return sorted(moves)
