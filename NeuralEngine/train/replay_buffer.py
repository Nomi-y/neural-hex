"""Replay buffer of self-play samples, with Hex's 180° symmetry augmentation.

Each sample is (canonical planes, canonical policy target, value target). Sampling optionally applies the
180° board rotation — a symmetry that preserves the canonical orientation (both of the side-to-move's
edges map to each other, as do the opponent's), so it is a free data multiplier.
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

Sample = Tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, capacity: int, board_size: int) -> None:
        self.buffer: deque[Sample] = deque(maxlen=capacity)
        self.board_size = board_size

    def add(self, planes: np.ndarray, pi: np.ndarray, z: float) -> None:
        self.buffer.append((planes.astype(np.float32), pi.astype(np.float32), float(z)))

    def extend(self, samples: List[Sample]) -> None:
        for planes, pi, z in samples:
            self.add(planes, pi, z)

    def __len__(self) -> int:
        return len(self.buffer)

    def _rotate180(self, planes: np.ndarray, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.board_size
        rot_planes = planes[:, ::-1, ::-1].copy()
        cells = pi[: n * n].reshape(n, n)[::-1, ::-1].reshape(-1)
        rot_pi = np.concatenate([cells, pi[n * n:]]).astype(np.float32)
        return rot_planes, rot_pi

    def sample(self, batch_size: int, rng: np.random.Generator):
        count = len(self.buffer)
        idx = rng.integers(0, count, size=min(batch_size, count))
        planes_list, pi_list, z_list = [], [], []
        for i in idx:
            planes, pi, z = self.buffer[int(i)]
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
