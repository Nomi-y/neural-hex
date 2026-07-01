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
                 "num_actions", "priors", "child_n", "child_w", "children", "sum_n", "sum_w")

    def __init__(self, state: HexState, num_actions: int) -> None:
        self.state = state
        self.to_play = state.to_move
        self.expanded = False
        self.terminal = state.is_terminal()
        self.solved_value: Optional[int] = None
        self.solved_action: Optional[int] = None
        self.num_actions = num_actions
        self.priors: Dict[int, float] = {}
        # Fixed-size arrays indexed by action (faster than dicts for lookup/update)
        self.child_n = [0] * num_actions
        self.child_w = [0.0] * num_actions
        self.children: Dict[int, "Node"] = {}
        self.sum_n = 0
        self.sum_w = 0.0   # sum of backed-up values (node.to_play perspective); for FPU

    def resolved(self) -> bool:
        """A node we never search below: a real terminal or an exactly solved position."""
        return self.terminal or self.solved_value is not None


def make_root(state: HexState) -> Node:
    return Node(state, state.size * state.size + 1)


def _select(node: Node, c_puct: float, fpu: float) -> int:
    sqrt_total = math.sqrt(node.sum_n + 1)
    best_score = -1e30
    best_action = -1
    cn = node.child_n
    cw = node.child_w
    # First-Play-Urgency: value to assume for not-yet-visited children. fpu<=0 keeps the legacy 0;
    # fpu>0 estimates them from this node's mean value minus a reduction, so search exploits the
    # policy's best moves before fanning out to unproven ones.
    fpu_q = (node.sum_w / node.sum_n - fpu) if (fpu > 0.0 and node.sum_n > 0) else 0.0
    for action, prior in node.priors.items():
        n = cn[action]
        q = (cw[action] / n) if n > 0 else fpu_q
        u = c_puct * prior * sqrt_total / (1 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _expand(node: Node, policy_real: np.ndarray) -> None:
    """Attach priors over legal moves (renormalised) and mark the node expanded."""
    legal = node.state.legal_actions()
    pr = policy_real[legal]                       # one fancy-index instead of per-action numpy scalar reads
    total = float(pr.sum())
    priors = node.priors
    if total <= 0:
        inv = 1.0 / len(legal)
        for a in legal:
            priors[a] = inv
    else:
        for a, p in zip(legal, (pr / total).tolist()):
            priors[a] = p
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
        node.child_n[action] += 1
        node.child_w[action] += value
        node.sum_n += 1
        node.sum_w += value


def _collect_pending(roots: List[Node], cfg) -> tuple:
    """One PUCT descent per root: walk to a leaf, and for leaves that resolve without the
    network (terminal / solved / virtual-connection) back the value up immediately.  Returns
    the leaves that still need a network evaluation and their root→leaf paths."""
    num_actions = cfg.game.num_actions
    c_puct = cfg.mcts.c_puct
    fpu = cfg.mcts.fpu_reduction
    pending_leaves: List[Node] = []
    pending_paths: List[List[tuple]] = []
    for root in roots:
        if root.resolved():
            continue
        node = root
        path: List[tuple] = []
        while node.expanded and not node.resolved():
            action = _select(node, c_puct, fpu)
            if action not in node.children:
                node.children[action] = Node(node.state.play(action), num_actions)
            path.append((node, action))
            node = node.children[action]
        # `node` is a leaf: terminal, solved, or not-yet-expanded.
        value = _resolve_leaf(node, cfg)
        if value is not None:
            _backup(path, value)
        else:
            pending_leaves.append(node)
            pending_paths.append(path)
    return pending_leaves, pending_paths


def _apply_pending(pending_leaves: List[Node], pending_paths: List[List[tuple]],
                   policies, values) -> None:
    """Expand each evaluated leaf with its network priors and back its value up the tree."""
    for i, leaf in enumerate(pending_leaves):
        _expand(leaf, policies[i])
        _backup(pending_paths[i], float(values[i]))


def run_batched(roots: List[Node], evaluator, cfg, simulations: int, add_noise: bool, rng: np.random.Generator) -> None:
    """Run `simulations` PUCT simulations on each root, batching leaf network evaluations across roots."""
    # Expand every root first (one batched evaluation), so PUCT has priors from move one.
    _ensure_expanded(roots, evaluator, cfg)
    if add_noise:
        for root in roots:
            if root.expanded:
                _add_dirichlet(root, cfg.mcts.dirichlet_alpha, cfg.mcts.dirichlet_epsilon, rng)

    for _ in range(simulations):
        pending_leaves, pending_paths = _collect_pending(roots, cfg)
        if pending_leaves:
            policies, values = evaluator.evaluate([n.state for n in pending_leaves])
            _apply_pending(pending_leaves, pending_paths, policies, values)


def _split_list(items: List, parts: int) -> List[List]:
    """Split `items` into `parts` near-even contiguous groups (each non-empty; fewer groups
    than `parts` if there aren't enough items)."""
    parts = max(1, min(parts, len(items)))
    base, extra = divmod(len(items), parts)
    out, i = [], 0
    for p in range(parts):
        n = base + (1 if p < extra else 0)
        out.append(items[i:i + n])
        i += n
    return out


def run_batched_streaming(roots: List[Node], evaluator, cfg, simulations: int, add_noise: bool,
                          rng: np.random.Generator, shards: int) -> None:
    """Pipelined variant of `run_batched` for the GPU inference server.

    The serial loop blocks each worker on every leaf batch: build request → wait on the GPU →
    expand/backup → repeat, so the worker's CPU tree work and the GPU forward never overlap.
    Here the worker's games are split into `shards` groups and their evaluations are double-buffered:
    while one group's leaves are on the GPU, the next group's tree walk runs on the CPU.  The
    inference server's queue stays full (no CPU/GPU ping-pong bubble) and the GPU stays pinned.

    Requires an evaluator exposing submit()/receive() (RemoteEvaluator, or the Evaluator shim).
    The search is numerically identical to run_batched — each game's tree is independent, so
    regrouping only changes which leaves share a GPU batch, not any per-leaf result.  shards<=1
    (or a single root) falls back to the serial path."""
    if shards <= 1 or len(roots) < 2:
        run_batched(roots, evaluator, cfg, simulations, add_noise, rng)
        return

    _ensure_expanded(roots, evaluator, cfg)
    if add_noise:
        for root in roots:
            if root.expanded:
                _add_dirichlet(root, cfg.mcts.dirichlet_alpha, cfg.mcts.dirichlet_epsilon, rng)

    groups = _split_list(roots, shards)
    inflight: List[Optional[tuple]] = [None] * len(groups)  # (handle, leaves, paths) per group

    def _submit(gi: int) -> None:
        leaves, paths = _collect_pending(groups[gi], cfg)
        handle = evaluator.submit([n.state for n in leaves]) if leaves else None
        inflight[gi] = (handle, leaves, paths)

    def _complete(gi: int) -> None:
        handle, leaves, paths = inflight[gi]
        if leaves:
            policies, values = evaluator.receive(handle)
            _apply_pending(leaves, paths, policies, values)
        inflight[gi] = None

    # Prime one simulation for every group, then keep exactly one batch per group in flight:
    # complete a group's outstanding batch and immediately submit its next — so while we process
    # group i on the CPU, groups i+1… stay queued on the GPU.
    for gi in range(len(groups)):
        _submit(gi)
    for _ in range(simulations - 1):
        for gi in range(len(groups)):
            _complete(gi)
            _submit(gi)
    for gi in range(len(groups)):
        _complete(gi)


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
    counts = np.array(node.child_n, dtype=np.float32)
    if node.solved_value is not None and node.solved_action is not None:
        # A solved root: put all weight on the proven best move.
        out = np.zeros(num_actions, dtype=np.float32)
        out[node.solved_action] = 1.0
        return out
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
    cn = node.child_n
    cw = node.child_w
    for action in node.priors:
        n = cn[action]
        q = (cw[action] / n) if n > 0 else 0.0
        prob = (counts[action] / total) if total > 0 else 0.0
        moves.append((action, float(prob), float(q)))
    moves.sort(key=lambda m: m[1], reverse=True)
    return moves
