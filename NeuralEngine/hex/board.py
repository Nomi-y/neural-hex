"""Hex game rules — a faithful Python port of the backend's authoritative implementation.

Mirrors:
  - Backend/Source/Game/HexBoard.ts  (row-major indexing, the six-neighbour adjacency)
  - Backend/Source/Game/WinDetection.ts (BFS connection check + winning path)
  - Backend/Source/Game/GameInstance.ts (turn order, the swap/pie rule)

Conventions (identical to the backend):
  - Board is N×N, row-major: index = row * N + col.
  - Stones: EMPTY=0, RED=1, BLUE=2. RED moves first.
  - RED connects top row (0) to bottom row (N-1); BLUE connects left col (0) to right col (N-1).
  - Six-neighbour adjacency via the offsets below.

The swap rule is implemented as the transpose+colour-swap symmetry of Hex rather than the backend's
seat reassignment: taking the opening is equivalent to relabelling the board under that symmetry, which
keeps each colour's fixed connection direction intact — exactly what a colour-orientation-aware learner
needs. The two are game-theoretically identical (see README).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

EMPTY = 0
RED = 1
BLUE = 2

# (drow, dcol) — identical to Backend HexBoard.NeighbourOffsets.
_NEIGHBOUR_OFFSETS: Tuple[Tuple[int, int], ...] = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


def other(color: int) -> int:
    return BLUE if color == RED else RED


@lru_cache(maxsize=None)
def neighbours_table(size: int) -> Tuple[Tuple[int, ...], ...]:
    """For each cell index, the flat indices of its up-to-six neighbours (precomputed per board size)."""
    table: List[Tuple[int, ...]] = []
    for index in range(size * size):
        row, col = divmod(index, size)
        out: List[int] = []
        for drow, dcol in _NEIGHBOUR_OFFSETS:
            r, c = row + drow, col + dcol
            if 0 <= r < size and 0 <= c < size:
                out.append(r * size + c)
        table.append(tuple(out))
    return tuple(table)


@lru_cache(maxsize=None)
def transpose_map(size: int) -> np.ndarray:
    """index -> transposed index (row/col swapped). Used for the swap rule and data augmentation."""
    m = np.empty(size * size, dtype=np.int64)
    for index in range(size * size):
        row, col = divmod(index, size)
        m[index] = col * size + row
    return m


def detect_win(cells: np.ndarray, size: int, color: int) -> Optional[List[int]]:
    """BFS from the colour's start edge to its end edge through its own stones; returns the winning
    path (cell indices) or None. Mirrors Backend WinDetection.DetectWin.

    Works on a plain Python list (cells.tolist()) with a list `parent`/queue: element-by-element numpy
    scalar indexing in this per-`play()` hot loop is far slower than native Python ints."""
    nbrs = neighbours_table(size)
    cl = cells.tolist()
    n = size * size
    parent = [-2] * n
    queue: List[int] = []
    is_red = color == RED
    if is_red:
        for index in range(size):                 # start edge = top row
            if cl[index] == color:
                parent[index] = -1
                queue.append(index)
        end_lo = n - size                         # end edge = bottom row (index >= end_lo)
    else:
        for index in range(0, n, size):           # start edge = left column
            if cl[index] == color:
                parent[index] = -1
                queue.append(index)
        end_col = size - 1                         # end edge = right column (index % size == end_col)
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        on_end = current >= end_lo if is_red else current % size == end_col
        if on_end:
            path: List[int] = []
            node = current
            while node != -1:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        for nxt in nbrs[current]:
            if cl[nxt] == color and parent[nxt] == -2:
                parent[nxt] = current
                queue.append(nxt)
    return None


class HexState:
    """An immutable-by-convention game position. `play()` returns a fresh state; callers never mutate
    `cells` in place (MCTS/solver rely on this)."""

    __slots__ = ("size", "swap_rule", "cells", "to_move", "move_count", "winner", "winning_path")

    def __init__(
        self,
        size: int,
        swap_rule: bool,
        cells: Optional[np.ndarray] = None,
        to_move: int = RED,
        move_count: int = 0,
        winner: int = EMPTY,
        winning_path: Optional[List[int]] = None,
    ) -> None:
        self.size = size
        self.swap_rule = swap_rule
        self.cells = cells if cells is not None else np.zeros(size * size, dtype=np.int8)
        self.to_move = to_move
        self.move_count = move_count
        self.winner = winner
        self.winning_path = winning_path

    # ---- construction ----

    @staticmethod
    def initial(size: int, swap_rule: bool) -> "HexState":
        return HexState(size, swap_rule)

    @staticmethod
    def from_cells(size: int, swap_rule: bool, cells: np.ndarray, to_move: int, move_count: int, swap_available: bool = False) -> "HexState":
        """Rebuild a position from an external snapshot (e.g. a backend GameView). `move_count` only
        needs to be accurate enough to gate swap availability; pass it through when known."""
        state = HexState(size, swap_rule, cells.astype(np.int8).copy(), to_move, move_count)
        # The caller may force swap availability (the backend tells us via CanSwap); otherwise infer it.
        if swap_available:
            state.move_count = 1  # the one ply where swap is offered
        state._refresh_winner()
        return state

    def copy(self) -> "HexState":
        return HexState(self.size, self.swap_rule, self.cells.copy(), self.to_move, self.move_count, self.winner,
                        None if self.winning_path is None else list(self.winning_path))

    # ---- queries ----

    @property
    def num_cells(self) -> int:
        return self.size * self.size

    @property
    def swap_action(self) -> int:
        return self.size * self.size

    def is_terminal(self) -> bool:
        return self.winner != EMPTY

    def swap_available(self) -> bool:
        # The pie rule: the second player, on their first turn only, may steal the opening move.
        return self.swap_rule and self.move_count == 1

    def legal_actions(self) -> List[int]:
        if self.is_terminal():
            return []
        actions = [i for i in range(self.num_cells) if self.cells[i] == EMPTY]
        if self.swap_available():
            actions.append(self.swap_action)
        return actions

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_cells + 1, dtype=bool)
        if self.is_terminal():
            return mask
        mask[: self.num_cells] = self.cells == EMPTY
        if self.swap_available():
            mask[self.swap_action] = True
        return mask

    # ---- transitions ----

    def play(self, action: int) -> "HexState":
        if action == self.swap_action:
            return self._play_swap()
        return self._play_place(action)

    def _play_place(self, index: int) -> "HexState":
        cells = self.cells.copy()
        cells[index] = self.to_move
        win = detect_win(cells, self.size, self.to_move)
        return HexState(
            self.size,
            self.swap_rule,
            cells,
            other(self.to_move),
            self.move_count + 1,
            self.to_move if win is not None else EMPTY,
            win,
        )

    def _play_swap(self) -> "HexState":
        # Steal the opening via the transpose+colour-swap symmetry: every stone moves to its transposed
        # cell with its colour flipped, and the opener replies next. The lone opening RED stone becomes a
        # BLUE stone on the transposed cell, with RED to move — strategically identical to the backend's
        # seat swap, but with colour orientations preserved so the network sees a consistent world.
        tmap = transpose_map(self.size)
        swapped = np.zeros_like(self.cells)
        for index in range(self.num_cells):
            value = self.cells[index]
            if value != EMPTY:
                swapped[tmap[index]] = other(value)
        return HexState(self.size, self.swap_rule, swapped, RED, self.move_count + 1)

    def _refresh_winner(self) -> None:
        for color in (RED, BLUE):
            path = detect_win(self.cells, self.size, color)
            if path is not None:
                self.winner = color
                self.winning_path = path
                return
        self.winner = EMPTY
        self.winning_path = None

    # ---- hashing (transposition tables, dedup) ----

    def key(self) -> bytes:
        return self.cells.tobytes() + bytes((self.to_move, 1 if self.swap_available() else 0))

    def __repr__(self) -> str:
        glyph = {EMPTY: ".", RED: "R", BLUE: "B"}
        rows = []
        for r in range(self.size):
            indent = " " * r
            row = " ".join(glyph[int(self.cells[r * self.size + c])] for c in range(self.size))
            rows.append(indent + row)
        turn = "RED" if self.to_move == RED else "BLUE"
        return f"<HexState {self.size}x{self.size} move={self.move_count} turn={turn} winner={self.winner}>\n" + "\n".join(rows)
