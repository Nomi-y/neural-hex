"""PUCT Monte-Carlo Tree Search — the lookahead, guided by the policy/value network.

This is the "look a couple of moves into the future" engine: an AlphaZero-style search where the policy
net provides move priors, the value net replaces random rollouts, and selection follows PUCT. Two extra
features sharpen the endgame:
  - terminal positions back up an exact ±1,
  - near the end (few empty cells) the exact solver (hex.solver) proves the result and the node becomes
    an exact terminal — so won/lost endgames are played perfectly rather than estimated.

Searches for many positions are batched: `run_batched` advances one root per game in lockstep and
evaluates all their leaves in a single network call, which is what saturates a GPU during self-play.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from hex.board import HexState, EMPTY
from hex.bridges import virtual_connection
from hex.solver import maybe_solve


class Node:
    __slots__ = ("state", "to_play", "expanded", "terminal", "solved_value", "solved_action",
                 "priors", "child_n", "child_w", "children", "sum_n")

    def __init__(self, state: HexState) -> None:
        self.state = state
        self.to_play = state.to_move
        self.expanded = False
        self.terminal = state.is_terminal()
        self.solved_value: Optional[int] = None
        self.solved_action: Optional[int] = None
        self.priors: Dict[int, float] = {}
        self.child_n: Dict[int, int] = {}
        self.child_w: Dict[int, float] = {}
        self.children: Dict[int, "Node"] = {}
        self.sum_n = 0

    def resolved(self) -> bool:
        """A node we never search below: a real terminal or an exactly solved position."""
        return self.terminal or self.solved_value is not None


def make_root(state: HexState) -> Node:
    return Node(state)


def _select(node: Node, c_puct: float) -> int:
    sqrt_total = math.sqrt(node.sum_n + 1)
    best_score = -1e30
    best_action = -1
    for action, prior in node.priors.items():
        n = node.child_n.get(action, 0)
        q = (node.child_w[action] / n) if n > 0 else 0.0
        u = c_puct * prior * sqrt_total / (1 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _expand(node: Node, policy_real: np.ndarray) -> None:
    """Attach priors over legal moves (renormalised) and mark the node expanded."""
    legal = node.state.legal_actions()
    total = float(sum(policy_real[a] for a in legal))
    if total <= 0:
        for a in legal:
            node.priors[a] = 1.0 / len(legal)
    else:
        for a in legal:
            node.priors[a] = float(policy_real[a]) / total
    node.expanded = True


def _add_dirichlet(node: Node, alpha: float, epsilon: float, rng: np.random.Generator) -> None:
    actions = list(node.priors.keys())
    if not actions:
        return
    noise = rng.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        node.priors[a] = (1 - epsilon) * node.priors[a] + epsilon * float(n)


def _backup(path: List[tuple], leaf_value: float) -> None:
    """leaf_value is from the leaf's to_play perspective; flip it at each level going up."""
    value = leaf_value
    for node, action in reversed(path):
        value = -value
        node.child_n[action] = node.child_n.get(action, 0) + 1
        node.child_w[action] = node.child_w.get(action, 0.0) + value
        node.sum_n += 1


def run_batched(roots: List[Node], evaluator, cfg, simulations: int, add_noise: bool, rng: np.random.Generator) -> None:
    """Run `simulations` PUCT simulations on each root, batching leaf network evaluations across roots."""
    # Expand every root first (one batched evaluation), so PUCT has priors from move one.
    _ensure_expanded(roots, evaluator, cfg)
    if add_noise:
        for root in roots:
            if root.expanded:
                _add_dirichlet(root, cfg.mcts.dirichlet_alpha, cfg.mcts.dirichlet_epsilon, rng)

    for _ in range(simulations):
        pending_leaves: List[Node] = []
        pending_paths: List[List[tuple]] = []
        for root in roots:
            if root.resolved():
                continue
            node = root
            path: List[tuple] = []
            while node.expanded and not node.resolved():
                action = _select(node, cfg.mcts.c_puct)
                if action not in node.children:
                    node.children[action] = Node(node.state.play(action))
                path.append((node, action))
                node = node.children[action]
            # `node` is a leaf: terminal, solved, or not-yet-expanded.
            value = _resolve_leaf(node, cfg)
            if value is not None:
                _backup(path, value)
            else:
                pending_leaves.append(node)
                pending_paths.append(path)

        if pending_leaves:
            policies, values = evaluator.evaluate([n.state for n in pending_leaves])
            for i, leaf in enumerate(pending_leaves):
                _expand(leaf, policies[i])
                _backup(pending_paths[i], float(values[i]))


def _ensure_expanded(roots: List[Node], evaluator, cfg) -> None:
    to_eval: List[Node] = []
    for root in roots:
        if root.resolved():
            continue
        if _resolve_leaf(root, cfg) is None and not root.expanded:
            to_eval.append(root)
    if to_eval:
        policies, _ = evaluator.evaluate([n.state for n in to_eval])
        for i, root in enumerate(to_eval):
            _expand(root, policies[i])


def _resolve_leaf(node: Node, cfg) -> Optional[float]:
    """Return a value (from node.to_play perspective) if the leaf is terminal or exactly solvable, and
    mark the node resolved. Returns None when the network must evaluate it."""
    if node.terminal:
        return -1.0  # the side to move has already lost (opponent connected on the prior ply)
    if node.solved_value is not None:
        return float(node.solved_value)

    # Virtual-connection fast path: an optimistic bridge-chain check that is
    # nearly always correct when few cells remain.  It is a heuristic, so it
    # never overrides the solver — but when the solver is out of reach (too many
    # empties) it still gives a useful early value signal.
    if cfg.mcts.use_virtual_connection:
        if virtual_connection(node.state.cells, node.state.size, node.to_play):
            return 1.0  # side to move has a winning bridge-chain
        other_color = 3 - node.to_play  # RED=1, BLUE=2
        if virtual_connection(node.state.cells, node.state.size, other_color):
            return -1.0  # opponent already has a winning bridge-chain

    if cfg.mcts.solver_empty_threshold > 0:
        empties = int(np.count_nonzero(node.state.cells == EMPTY))
        if empties <= cfg.mcts.solver_empty_threshold:
            value, action = maybe_solve(node.state, cfg.mcts.solver_empty_threshold, cfg.mcts.solver_node_budget)
            if value is not None:
                node.solved_value = value
                node.solved_action = action
                return float(value)
    return None


def visit_counts(node: Node, num_actions: int) -> np.ndarray:
    counts = np.zeros(num_actions, dtype=np.float32)
    if node.solved_value is not None and node.solved_action is not None:
        # A solved root: put all weight on the proven best move.
        counts[node.solved_action] = 1.0
        return counts
    for action, n in node.child_n.items():
        counts[action] = n
    return counts


def policy_distribution(node: Node, num_actions: int) -> np.ndarray:
    counts = visit_counts(node, num_actions)
    total = counts.sum()
    return counts / total if total > 0 else counts


def select_action(node: Node, num_actions: int, temperature: float, rng: np.random.Generator) -> int:
    """Pick a move from the visit counts. temperature==0 => most-visited; else sample N^(1/temp)."""
    if node.solved_value is not None and node.solved_action is not None:
        return int(node.solved_action)
    counts = visit_counts(node, num_actions)
    if counts.sum() <= 0:
        # No search performed (e.g. an immediately resolved root with no solved_action): fall back to a
        # uniform legal choice.
        legal = node.state.legal_actions()
        return int(rng.choice(legal))
    if temperature <= 1e-6:
        return int(counts.argmax())
    logits = np.power(counts, 1.0 / temperature)
    probs = logits / logits.sum()
    return int(rng.choice(len(counts), p=probs))


def ranked_moves(node: Node, num_actions: int):
    """(action, visit_probability, mean_value) sorted best-first — the engine's move list output."""
    counts = visit_counts(node, num_actions)
    total = counts.sum()
    moves = []
    for action in node.priors.keys() if not node.child_n else node.child_n.keys():
        n = node.child_n.get(action, 0)
        q = (node.child_w[action] / n) if n > 0 else 0.0
        prob = (counts[action] / total) if total > 0 else 0.0
        moves.append((action, float(prob), float(q)))
    moves.sort(key=lambda m: m[1], reverse=True)
    return moves
