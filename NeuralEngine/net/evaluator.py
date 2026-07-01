"""Batched network inference for MCTS.

Turns a list of positions into (policy over *real* actions, value) by encoding canonical planes, running
one batched forward pass, masking illegal moves, and mapping the canonical policy back to real action
indices.  Softmax stays on GPU; only the final policy array moves to CPU.  Batching across many positions
(many self-play games, or one engine search's leaves) is what keeps a GPU busy.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from hex.board import HexState, RED, BLUE
from net.encoding import encode_batch, canonical_legal_mask, action_transpose


class Evaluator:
    def __init__(self, net, device: str) -> None:
        self.net = net
        self.device = device
        self.net.eval()  # evaluator nets are inference-only; avoid toggling per call

    @torch.no_grad()
    def evaluate(self, states: List[HexState]) -> Tuple[np.ndarray, np.ndarray]:
        planes = encode_batch(states)
        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        logits, values = self.net(x)

        num_actions = logits.shape[1]
        size = states[0].size

        # Canonical legal masks: built vectorised on CPU, transferred in one shot, softmaxed on GPU.
        mask_np = np.stack([canonical_legal_mask(s) for s in states])
        mask = torch.from_numpy(mask_np).to(self.device, non_blocking=True)

        masked = logits.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(masked, dim=1).float().cpu().numpy()
        values_np = values.float().cpu().numpy().reshape(-1)

        # Map canonical probs → real-action order: a transpose for BLUE, identity for RED.
        perm = action_transpose(size)
        policies = np.zeros((len(states), num_actions), dtype=np.float32)
        for i, state in enumerate(states):
            policies[i] = probs[i] if state.to_move == RED else probs[i][perm]
        return policies, values_np

    # Synchronous stand-in for the async RemoteEvaluator API, so mcts.run_batched_streaming can
    # drive any evaluator uniformly: submit() computes immediately and hands back the result as
    # the handle; receive() just returns it.  (No CPU/GPU overlap here — that's the server's job.)
    def submit(self, states: List[HexState]):
        return self.evaluate(states)

    def receive(self, handle):
        return handle
