"""Batched network inference for MCTS.

Turns a list of positions into (policy over *real* actions, value) by encoding canonical planes, running
one batched forward pass, masking illegal moves, and mapping the canonical policy back to real action
indices. Batching across many positions (many self-play games, or one engine search's leaves) is what
keeps a GPU busy.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from hex.board import HexState
from net.encoding import encode_batch, canonical_legal_mask, canon_to_real_action


class Evaluator:
    def __init__(self, net, device: str) -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def evaluate(self, states: List[HexState]) -> Tuple[np.ndarray, np.ndarray]:
        was_training = self.net.training
        self.net.eval()
        planes = encode_batch(states)
        x = torch.from_numpy(planes).to(self.device)
        logits, values = self.net(x)
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy().reshape(-1)
        if was_training:
            self.net.train()

        num_actions = logits.shape[1]
        size = states[0].size
        policies = np.zeros((len(states), num_actions), dtype=np.float32)
        for i, state in enumerate(states):
            cmask = canonical_legal_mask(state)
            masked = np.where(cmask, logits[i], -1e30)
            masked -= masked.max()
            exp = np.exp(masked) * cmask
            total = exp.sum()
            if total <= 0:
                # Degenerate (shouldn't happen): fall back to uniform over legal moves.
                exp = cmask.astype(np.float32)
                total = exp.sum()
            canon_probs = exp / total
            for ca in np.nonzero(cmask)[0]:
                policies[i, canon_to_real_action(state.to_move, size, int(ca))] = canon_probs[ca]
        return policies, values
