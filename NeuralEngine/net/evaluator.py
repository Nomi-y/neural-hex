"""Batched network inference for MCTS.

Turns a list of positions into (policy over *real* actions, value) by encoding canonical planes, running
one batched forward pass, masking illegal moves, and mapping the canonical policy back to real action
indices.  Softmax stays on GPU; only the final policy array moves to CPU.  Batching across many positions
(many self-play games, or one engine search's leaves) is what keeps a GPU busy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch

from hex.board import HexState, RED, BLUE
from net.encoding import encode_batch, canonical_legal_mask, canon_to_real_action


@lru_cache(maxsize=2)
def _canon_to_real_map(size: int, to_move: int) -> np.ndarray:
    """Precomputed index array: canonical_action -> real_action for `to_move` on a board of `size`."""
    num_actions = size * size + 1
    m = np.empty(num_actions, dtype=np.int32)
    for ca in range(num_actions):
        m[ca] = canon_to_real_action(to_move, size, ca)
    return m


class Evaluator:
    def __init__(self, net, device: str) -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def evaluate(self, states: List[HexState]) -> Tuple[np.ndarray, np.ndarray]:
        was_training = self.net.training
        self.net.eval()
        planes = encode_batch(states)
        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        logits, values = self.net(x)

        num_actions = logits.shape[1]
        batch = len(states)
        size = states[0].size

        # Build canonical legal mask batch on GPU and softmax there — keeps
        # the heavy float math on the accelerator instead of a Python loop.
        mask = torch.zeros(batch, num_actions, dtype=torch.bool, device=self.device)
        for i, state in enumerate(states):
            cmask = canonical_legal_mask(state)
            mask[i] = torch.from_numpy(cmask).to(self.device, non_blocking=True)

        masked = logits.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(masked, dim=1).float().cpu().numpy()
        values_np = values.float().cpu().numpy().reshape(-1)

        if was_training:
            self.net.train()

        # Map canonical probs → real-action-indexed policies.
        # Precomputed index arrays avoid recomputing canon_to_real_action per action.
        policies = np.zeros((batch, num_actions), dtype=np.float32)
        mask_np = mask.cpu().numpy()
        for i, state in enumerate(states):
            cmap = _canon_to_real_map(size, state.to_move)
            for ca in np.nonzero(mask_np[i])[0]:
                policies[i, cmap[ca]] = probs[i, ca]
        return policies, values_np
