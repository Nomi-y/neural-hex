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


def _ts() -> str:
    """Timestamp prefix matching train.clock.log() format: [HH:MM:SS +offset] [gen N].

    Reads TRAIN_START_EPOCH and TRAIN_GENERATION from the environment (set by
    train.py) so the offset and generation are consistent with all other log
    lines.  Falls back to bare [HH:MM:SS] when env vars are unavailable
    (standalone tests)."""
    now = time.strftime("%H:%M:%S")
    prefix = f"[{now}]"
    start_raw = os.environ.get("TRAIN_START_EPOCH")
    if start_raw:
        try:
            start = float(start_raw)
            elapsed = int(max(0, time.time() - start))
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            prefix = f"[{now} +{h}:{m:02d}:{s:02d}]"
        except (ValueError, TypeError):
            pass
    gen_raw = os.environ.get("TRAIN_GENERATION")
    if gen_raw:
        prefix += f" [gen {gen_raw}]"
    return prefix


class RemoteEvaluator:
    """Worker-side stand-in for net.evaluator.Evaluator, bound to one net_id on the server.

    submit()/receive() split the request from the wait so a worker can keep several leaf batches
    in flight at once — pipelined MCTS (mcts.run_batched_streaming) submits the next group's leaves
    while the previous group's forward is still running on the GPU, overlapping CPU search with GPU
    inference.  evaluate() is the synchronous submit-then-receive used by the serial path.

    Replies land on the one per-worker response queue in request order, but receive() buffers any
    result that isn't the one being awaited, so out-of-order arrivals (more than one request
    outstanding) are matched by request id rather than assumed away."""

    def __init__(self, net_id: int, worker_idx: int, req_q, resp_q) -> None:
        self.net_id = net_id
        self.worker_idx = worker_idx
        self.req_q = req_q
        self.resp_q = resp_q
        self._rid = 0
        self._states: dict = {}    # rid -> states, kept until receive() maps its policy back
        self._buffered: dict = {}  # rid -> (probs, values) that arrived before their receive()

    def submit(self, states: List[HexState]) -> int:
        """Queue a leaf batch on the GPU server and return its request id WITHOUT blocking."""
        planes = encode_batch(states)
        masks = np.stack([canonical_legal_mask(s) for s in states])
        self._rid += 1
        rid = self._rid
        self.req_q.put((self.net_id, self.worker_idx, rid, planes, masks))
        self._states[rid] = states
        return rid

    def receive(self, rid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Block for the result of request `rid`, buffering any other results that arrive first."""
        states = self._states.pop(rid)
        if rid in self._buffered:
            probs, values = self._buffered.pop(rid)
        else:
            while True:
                try:
                    r_rid, r_probs, r_values = self.resp_q.get(timeout=_RESULT_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError(
                        f"inference server returned no result in {_RESULT_TIMEOUT_S:.0f}s "
                        f"(worker {self.worker_idx}) — the GPU server likely died")
                if r_rid == rid:
                    probs, values = r_probs, r_values
                    break
                self._buffered[r_rid] = (r_probs, r_values)

        # Map canonical probs -> real-action order: transpose for BLUE, identity for RED.
        perm = action_transpose(states[0].size)
        policies = np.zeros((len(states), probs.shape[1]), dtype=np.float32)
        for i, state in enumerate(states):
            policies[i] = probs[i] if state.to_move == RED else probs[i][perm]
        return policies, values

    def evaluate(self, states: List[HexState]) -> Tuple[np.ndarray, np.ndarray]:
        return self.receive(self.submit(states))


def _server_loop(cfg, np_states, device, req_q, resp_qs, stop_evt, max_batch) -> None:
    """Own the GPU: load the net(s), then batch leaf requests across workers into one forward each."""
    import torch
    from net.model import build_net, CleanStateDict

    torch.set_grad_enabled(False)
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # The forward is the self-play bottleneck: a big conv net on FP32 is compute-bound even at modest
    # batch (so coalescing bigger batches doesn't help — measured). Two forward speedups on CUDA, both
    # safe for leaf eval (it only guides MCTS, needs no FP32 precision) and matching how the net was
    # trained: AMP (tensor-core half precision, ~1.6× on Turing, more on Blackwell) and channels_last
    # (NHWC — the tensor cores' native conv layout, another ~1.15×). INFERENCE_AMP=0 forces FP32.
    amp = device == "cuda" and os.environ.get("INFERENCE_AMP", "1") != "0"
    channels_last = device == "cuda"

    nets = []
    for st in np_states:
        net = build_net(cfg).to(device)
        net.load_state_dict(CleanStateDict({k: torch.from_numpy(v).to(device) for k, v in st.items()}))
        net.eval()
        if channels_last:
            net = net.to(memory_format=torch.channels_last)
        nets.append(net)

    report_every = float(os.environ.get("INFERENCE_LOG_EVERY", "30"))
    # Coalescing window (ms): after draining the queued burst, linger this long for more requests.
    # DEFAULT 0 (off): with synchronous workers (each blocks on its one outstanding request) the
    # server is starved, not saturated — lingering just delays results the workers are waiting on,
    # so measured leaves/s FELL on the 5090 (12.5k at 0 → 11k at 2ms → 9k at 4ms+pipeline). The knob
    # stays for a future async/virtual-loss worker that could actually keep the queue deep.
    coalesce_s = max(0.0, float(os.environ.get("INFERENCE_COALESCE_MS", "0"))) / 1000.0
    print(f"{_ts()} [infer] server ready on {device}: {len(nets)} net(s), max_batch={max_batch}, "
          f"amp={'fp16' if amp else 'off'}, coalesce={coalesce_s * 1000:.0f}ms, "
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

        # Coalesce requests into one forward, capped at max_batch to bound server memory: drain the
        # burst already queued, then (if a window is set) briefly wait for more arrivals. A quiet gap
        # of coalesce_s — or the cap — ends the batch, so the server adapts to load: it fills big
        # batches under pressure and still fires promptly when the trickle stops.
        batch = [first]
        total = first[3].shape[0]
        stop = False
        while total < max_batch:
            try:
                item = req_q.get_nowait()  # fast path: take whatever is already waiting
            except queue.Empty:
                if coalesce_s <= 0:
                    break
                try:
                    item = req_q.get(timeout=coalesce_s)  # linger for the next arrival
                except queue.Empty:
                    break  # quiet gap → fire what we have
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
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast("cuda", enabled=amp):
                logits, values = nets[net_id](x)
            m = torch.from_numpy(masks).to(device, non_blocking=True)
            # Mask + softmax in FP32 (upcast first): -inf masking is exact and the distribution is clean.
            probs = torch.softmax(logits.float().masked_fill(~m, float("-inf")), dim=1).cpu().numpy()
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
            print(f"{_ts()} [infer] {n_batches} batches, "
                  f"{n_leaves / max(1, n_batches):.0f} avg leaves/batch, "
                  f"{n_leaves / dt:.0f} leaves/s, peak batch {max_seen}", flush=True)
            n_batches = n_leaves = max_seen = 0
            last_report = now

        if stop:
            break


class InferenceServer:
    """Lifecycle owner: spins up the GPU server process and the queues, hands out RemoteEvaluators.

    Queues/primitives come from the caller's spawn context so they're shareable with the Pool
    workers. resp_qs has one queue per worker; a worker claims an index via next_worker_index()
    in the Pool initializer (shared counter under a lock).

    Pass `gpu_id` to bind to a specific GPU (cuda:N).  When None, uses the `device` arg directly
    (single-GPU or CPU fallback)."""

    def __init__(self, cfg, np_states: List[dict], device: str, num_workers: int, ctx,
                 gpu_id: int | None = None) -> None:
        self.num_nets = len(np_states)
        self.num_workers = num_workers
        self.req_q = ctx.Queue()
        self.resp_qs = [ctx.Queue() for _ in range(num_workers)]
        self.counter = ctx.Value("i", 0)
        self.lock = ctx.Lock()
        self.stop_evt = ctx.Event()
        env_mb = int(os.environ.get("INFERENCE_MAX_BATCH", "0"))
        self.max_batch = env_mb if env_mb > 0 else 2048
        dev = f"cuda:{gpu_id}" if gpu_id is not None else device
        self._proc = ctx.Process(
            target=_server_loop,
            args=(cfg, np_states, dev, self.req_q, self.resp_qs, self.stop_evt, self.max_batch),
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
