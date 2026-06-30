"""Single-GPU batched inference server for self-play / arena.

Many CPU search workers do the MCTS tree work and ship leaf positions to ONE GPU process that
batches forward passes across all of them. Two wins over "the net on the GPU in every worker":

  * one CUDA context instead of one-per-worker — no per-worker context OOM on high-vCPU boxes;
  * the network forward runs on the (otherwise idle) GPU, so CPU cores spend their time on search.

Each worker holds a `RemoteEvaluator` with the SAME `.evaluate(states) -> (policies, values)`
contract as `net.evaluator.Evaluator`, so `search.mcts` / `train.selfplay.play_games` are unchanged.

Work split (keeps the server doing only GPU work, fans the CPU cost across workers):
  worker:  encode planes + canonical legal masks  ->  send numpy  ->  map canonical->real policy
  server:  stack -> H2D -> forward -> mask+softmax on GPU -> D2H -> scatter results back

Arena evaluates two nets (candidate, best); the server hosts a list of nets and every request
carries a `net_id`, so a single server serves both. Requests are bucketed by net_id per cycle
(one forward per net), since a batch can't mix two networks.
"""

from __future__ import annotations

import os
import queue
import time
from typing import List, Tuple

import numpy as np

from hex.board import HexState, RED
from net.encoding import encode_batch, canonical_legal_mask, action_transpose

_STOP = "__STOP__"  # sentinel pushed onto the request queue to end the server loop

# How long a worker waits for one inference result before assuming the server died. Generous: a
# big first batch plus torch.compile warmup can take a while. Exceeding it raises so the worker
# fails loudly (and the pool watchdog tears the run down) instead of hanging forever.
_RESULT_TIMEOUT_S = float(os.environ.get("INFERENCE_RESULT_TIMEOUT", "300"))


class RemoteEvaluator:
    """Worker-side stand-in for net.evaluator.Evaluator, bound to one net_id on the server.

    Synchronous: each worker has at most one outstanding request (MCTS calls evaluate and blocks),
    so a single per-worker response queue with request-id matching is all the routing we need."""

    def __init__(self, net_id: int, worker_idx: int, req_q, resp_q) -> None:
        self.net_id = net_id
        self.worker_idx = worker_idx
        self.req_q = req_q
        self.resp_q = resp_q
        self._rid = 0

    def evaluate(self, states: List[HexState]) -> Tuple[np.ndarray, np.ndarray]:
        planes = encode_batch(states)
        masks = np.stack([canonical_legal_mask(s) for s in states])
        self._rid += 1
        rid = self._rid
        self.req_q.put((self.net_id, self.worker_idx, rid, planes, masks))
        while True:
            try:
                r_rid, probs, values = self.resp_q.get(timeout=_RESULT_TIMEOUT_S)
            except queue.Empty:
                raise RuntimeError(
                    f"inference server returned no result in {_RESULT_TIMEOUT_S:.0f}s "
                    f"(worker {self.worker_idx}) — the GPU server likely died")
            if r_rid == rid:
                break  # stale id can't normally happen (synchronous), but never trust a mismatch

        # Map canonical probs -> real-action order: transpose for BLUE, identity for RED.
        perm = action_transpose(states[0].size)
        policies = np.zeros((len(states), probs.shape[1]), dtype=np.float32)
        for i, state in enumerate(states):
            policies[i] = probs[i] if state.to_move == RED else probs[i][perm]
        return policies, values


def _server_loop(cfg, np_states, device, req_q, resp_qs, stop_evt, max_batch) -> None:
    """Own the GPU: load the net(s), then batch leaf requests across workers into one forward each."""
    import torch
    from net.model import build_net, CleanStateDict

    torch.set_grad_enabled(False)
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    nets = []
    for st in np_states:
        net = build_net(cfg).to(device)
        net.load_state_dict(CleanStateDict({k: torch.from_numpy(v).to(device) for k, v in st.items()}))
        net.eval()
        nets.append(net)

    report_every = float(os.environ.get("INFERENCE_LOG_EVERY", "10"))
    print(f"[infer] server ready on {device}: {len(nets)} net(s), max_batch={max_batch}, "
          f"heartbeat every {report_every:.0f}s", flush=True)
    n_batches = n_leaves = max_seen = 0
    last_report = time.time()

    while not stop_evt.is_set():
        try:
            first = req_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if first == _STOP:
            break

        # Greedily drain whatever is already queued so a single forward serves many workers,
        # capped at max_batch leaves to bound server-side memory.
        batch = [first]
        total = first[3].shape[0]
        stop = False
        while total < max_batch:
            try:
                item = req_q.get_nowait()
            except queue.Empty:
                break
            if item == _STOP:
                stop = True
                break
            batch.append(item)
            total += item[3].shape[0]

        # Bucket by net_id — a forward can't mix two networks (arena has candidate + best).
        buckets: dict = {}
        for it in batch:
            buckets.setdefault(it[0], []).append(it)

        for net_id, items in buckets.items():
            planes = np.concatenate([it[3] for it in items], axis=0)
            masks = np.concatenate([it[4] for it in items], axis=0)
            x = torch.from_numpy(planes).to(device, non_blocking=True)
            logits, values = nets[net_id](x)
            m = torch.from_numpy(masks).to(device, non_blocking=True)
            probs = torch.softmax(logits.masked_fill(~m, float("-inf")), dim=1).float().cpu().numpy()
            values_np = values.float().cpu().numpy().reshape(-1)
            off = 0
            for net_id_, worker_idx, rid, p, _mask in items:
                n = p.shape[0]
                resp_qs[worker_idx].put((rid, probs[off:off + n], values_np[off:off + n]))
                off += n
            n_batches += 1
            n_leaves += planes.shape[0]
            max_seen = max(max_seen, planes.shape[0])

        # Heartbeat: avg/peak batch size + leaves/s show directly whether the GPU is being fed.
        now = time.time()
        if now - last_report >= report_every:
            dt = now - last_report
            print(f"[infer] {n_batches} batches, {n_leaves / max(1, n_batches):.0f} avg leaves/batch, "
                  f"{n_leaves / dt:.0f} leaves/s, peak batch {max_seen}", flush=True)
            n_batches = n_leaves = max_seen = 0
            last_report = now

        if stop:
            break


class InferenceServer:
    """Lifecycle owner: spins up the GPU server process and the queues, hands out RemoteEvaluators.

    Queues/primitives come from the caller's spawn context so they're shareable with the Pool
    workers. resp_qs has one queue per worker; a worker claims an index via next_worker_index()
    in the Pool initializer (shared counter under a lock)."""

    def __init__(self, cfg, np_states: List[dict], device: str, num_workers: int, ctx) -> None:
        self.num_nets = len(np_states)
        self.num_workers = num_workers
        self.req_q = ctx.Queue()
        self.resp_qs = [ctx.Queue() for _ in range(num_workers)]
        self.counter = ctx.Value("i", 0)
        self.lock = ctx.Lock()
        self.stop_evt = ctx.Event()
        env_mb = int(os.environ.get("INFERENCE_MAX_BATCH", "0"))
        self.max_batch = env_mb if env_mb > 0 else 2048
        self._proc = ctx.Process(
            target=_server_loop,
            args=(cfg, np_states, device, self.req_q, self.resp_qs, self.stop_evt, self.max_batch),
            daemon=True,
        )

    def start(self) -> None:
        self._proc.start()

    def stop(self) -> None:
        self.stop_evt.set()
        try:
            self.req_q.put(_STOP)
        except Exception:
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
