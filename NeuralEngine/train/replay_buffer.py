"""Replay buffer of self-play samples, organised into per-generation blocks.

Each sample is (canonical planes, canonical policy target, value target). Samples are grouped by the
generation that produced them, which a flat FIFO can't express and which enables two things:

  * recency-weighted sampling — newer generations are drawn more often (weight halves every
    `recency_halflife` generations of age), so training chases the net's own improvements instead of
    waiting for stale positions to physically evict, WHILE a large capacity is still kept for value-
    target diversity (the anti-overfit lever).  A half-life ≤ 0 disables it → plain uniform sampling.
  * generation-granular eviction — whole oldest generations drop once the sample capacity is exceeded.

Sampling optionally applies Hex's 180° board rotation — a symmetry that preserves the canonical
orientation (both of the side-to-move's edges map to each other, as do the opponent's), a free data
multiplier.
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

Sample = Tuple[np.ndarray, np.ndarray, float]


class _Block:
    """One generation's worth of samples, tagged with the generation that produced them."""
    __slots__ = ("gen", "samples")

    def __init__(self, gen: int, samples: List[Sample]) -> None:
        self.gen = gen
        self.samples = samples


class ReplayBuffer:
    def __init__(self, capacity: int, board_size: int, recency_halflife: float = 0.0) -> None:
        self.capacity = capacity          # max samples kept across all blocks (RAM ceiling)
        self.board_size = board_size
        self.recency_halflife = float(recency_halflife)  # generations; ≤0 → uniform sampling
        self.blocks: deque[_Block] = deque()
        self._size = 0
        self._next_gen = 0                # auto-increment for extend() calls without an explicit gen

    def __len__(self) -> int:
        return self._size

    def extend(self, samples: List[Sample], generation: int | None = None) -> None:
        """Append one generation's samples as a new block.  `generation` tags the block for recency
        weighting; if omitted it auto-increments (keeps old callers / smoke tests working)."""
        if generation is None:
            generation = self._next_gen
        self._next_gen = max(self._next_gen, generation + 1)
        stored = [(planes.astype(np.float32), pi.astype(np.float32), float(z))
                  for planes, pi, z in samples]
        if not stored:
            return
        self.blocks.append(_Block(generation, stored))
        self._size += len(stored)
        self._evict()

    def _evict(self) -> None:
        # Drop whole oldest generations once over capacity (but never the only block).
        while self._size > self.capacity and len(self.blocks) > 1:
            old = self.blocks.popleft()
            self._size -= len(old.samples)

    def _rotate180(self, planes: np.ndarray, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.board_size
        rot_planes = planes[:, ::-1, ::-1].copy()
        cells = pi[: n * n].reshape(n, n)[::-1, ::-1].reshape(-1)
        rot_pi = np.concatenate([cells, pi[n * n:]]).astype(np.float32)
        return rot_planes, rot_pi

    def _block_probs(self) -> np.ndarray:
        """Per-block sampling probability.  Uniform: ∝ block size (every sample equally likely).
        Recency-weighted: ∝ size × 0.5^(age/halflife), so within a block the size factor cancels and
        each sample's draw probability ∝ its block's recency decay — recent positions dominate without
        old ones being evicted."""
        sizes = np.array([len(b.samples) for b in self.blocks], dtype=np.float64)
        if self.recency_halflife > 0 and len(self.blocks) > 1:
            newest = self.blocks[-1].gen
            ages = np.array([newest - b.gen for b in self.blocks], dtype=np.float64)
            weights = sizes * (0.5 ** (ages / self.recency_halflife))
        else:
            weights = sizes
        total = weights.sum()
        if total <= 0:                    # degenerate (e.g. all-decayed) → fall back to uniform
            weights = sizes
            total = weights.sum()
        return weights / total

    def sample(self, batch_size: int, rng: np.random.Generator):
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        blocks = list(self.blocks)        # O(1) indexing for the per-draw fetch below
        n = min(batch_size, self._size)
        chosen = rng.choice(len(blocks), size=n, p=self._block_probs())
        planes_list, pi_list, z_list = [], [], []
        for bi in chosen:
            block = blocks[int(bi)].samples
            planes, pi, z = block[int(rng.integers(0, len(block)))]
            if rng.random() < 0.5:
                planes, pi = self._rotate180(planes, pi)
            planes_list.append(planes)
            pi_list.append(pi)
            z_list.append(z)
        return (
            np.stack(planes_list).astype(np.float32),
            np.stack(pi_list).astype(np.float32),
            np.asarray(z_list, dtype=np.float32),
        )

    # ─── persistence ─────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        """Serialisable snapshot of every block (numpy arrays are handled by torch.save)."""
        return {
            "capacity": self.capacity,
            "board_size": self.board_size,
            "next_gen": self._next_gen,
            "blocks": [(b.gen, b.samples) for b in self.blocks],
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from a snapshot.  `recency_halflife` is intentionally NOT restored — it's a runtime
        knob, so the value passed to __init__ (from the live config) wins on resume."""
        self.capacity = d["capacity"]
        self.board_size = d["board_size"]
        self.blocks = deque()
        self._size = 0
        if "blocks" in d:
            self._next_gen = d.get("next_gen", 0)
            for gen, samples in d["blocks"]:
                stored = [(p.astype(np.float32), pi.astype(np.float32), float(z))
                          for p, pi, z in samples]
                self.blocks.append(_Block(gen, stored))
                self._size += len(stored)
        elif "buffer" in d:               # legacy flat format → collapse into one gen-0 block
            stored = [(p.astype(np.float32), pi.astype(np.float32), float(z))
                      for p, pi, z in d["buffer"]]
            if stored:
                self.blocks.append(_Block(0, stored))
                self._size = len(stored)
            self._next_gen = 1
        self._evict()
