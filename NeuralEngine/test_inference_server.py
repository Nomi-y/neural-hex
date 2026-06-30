"""Equivalence + plumbing test for the GPU inference server (runs on CPU here).

Confirms a RemoteEvaluator driving the real InferenceServer process (through the actual
request/response queues) returns the same policies/values as a local net.evaluator.Evaluator
for the same weights and states — for one net (self-play) and two nets (arena), and that a
single request batching many leaves splits back correctly.
"""

import os

os.environ.setdefault("BOARD_SIZE", "7")
os.environ.setdefault("NET_CHANNELS", "16")
os.environ.setdefault("NET_BLOCKS", "2")
os.environ.setdefault("DEVICE", "cpu")

import multiprocessing as mp

import numpy as np

from config import load
from hex.board import HexState
from net.model import build_net
from net.evaluator import Evaluator
from train.selfplay import to_numpy_state
from train.inference_server import InferenceServer, RemoteEvaluator


def _random_states(cfg, n, seed=0):
    """A spread of non-terminal positions (some RED to move, some BLUE) by playing random moves."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        s = HexState.initial(cfg.game.board_size, cfg.game.swap_rule)
        for _ in range(k % 5):  # vary ply -> vary side-to-move and legal sets
            if s.is_terminal():
                break
            legal = np.flatnonzero(s.legal_mask())
            s = s.play(int(rng.choice(legal)))
        if not s.is_terminal():
            out.append(s)
    return out


def main():
    cfg = load()
    ctx = mp.get_context("spawn")

    # Two independent nets so the arena path (net_id routing) is actually exercised.
    net_a = build_net(cfg).eval()
    net_b = build_net(cfg).eval()
    states = _random_states(cfg, 24, seed=1)

    ref_a = Evaluator(net_a, "cpu").evaluate(states)
    ref_b = Evaluator(net_b, "cpu").evaluate(states)

    server = InferenceServer(cfg, [to_numpy_state(net_a.state_dict()),
                                   to_numpy_state(net_b.state_dict())], "cpu", 2, ctx)
    server.start()
    try:
        ev_a = RemoteEvaluator(0, 0, server.req_q, server.resp_qs[0])
        ev_b = RemoteEvaluator(1, 1, server.req_q, server.resp_qs[1])

        # Full batch through net 0 and net 1.
        got_a = ev_a.evaluate(states)
        got_b = ev_b.evaluate(states)

        # Many small requests (1–3 leaves) to exercise per-request split/scatter.
        pieces_p, pieces_v = [], []
        i = 0
        while i < len(states):
            chunk = states[i:i + 3]
            p, v = ev_a.evaluate(chunk)
            pieces_p.append(p)
            pieces_v.append(v)
            i += 3
        split_p = np.concatenate(pieces_p, axis=0)
        split_v = np.concatenate(pieces_v, axis=0)
    finally:
        server.stop()

    checks = [
        ("net0 policy", got_a[0], ref_a[0]),
        ("net0 value", got_a[1], ref_a[1]),
        ("net1 policy", got_b[0], ref_b[0]),
        ("net1 value", got_b[1], ref_b[1]),
        ("net0 policy (split reqs)", split_p, ref_a[0]),
        ("net0 value (split reqs)", split_v, ref_a[1]),
    ]
    ok = True
    for name, got, ref in checks:
        max_err = float(np.max(np.abs(np.asarray(got) - np.asarray(ref))))
        good = max_err < 1e-5
        ok = ok and good
        print(f"  {name:28} max|Δ|={max_err:.2e} {'✓' if good else '✗ MISMATCH'}")

    print("\nINFERENCE SERVER TEST PASSED" if ok else "\nFAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
