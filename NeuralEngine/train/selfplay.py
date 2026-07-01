"""Self-play game generation — the source of training data.

Plays many games in parallel with batched MCTS, recording (canonical planes, canonical MCTS policy,
side-to-move) at every position; once a game ends, each record is labelled with z = +1/-1 for whether
that side went on to win.  Early plies are sampled with temperature for exploration, later plies played
greedily.  Every game plays to the last stone — no early resignation.

`generate()` runs in-process on a GPU/MPS device (one big batched actor saturates it) or fans out across
all CPU cores (one process per core) when there is no accelerator — the "use all resources" requirement.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import Config
from hex.board import HexState, RED
from net.model import build_net, CleanStateDict
from net.evaluator import Evaluator
from net.encoding import encode, action_transpose
from search import mcts
from train.clock import log

Sample = Tuple[np.ndarray, np.ndarray, float]


class _Game:
    __slots__ = ("state", "history", "ply", "done", "winner", "root", "no_resign", "resigned", "would_resign")

    def __init__(self, state: HexState, no_resign: bool = False) -> None:
        self.state = state
        self.history: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (canonical planes, canonical pi, to_move)
        self.ply = 0
        self.done = False
        self.winner = 0
        self.root = None        # carried MCTS subtree for the next move when cfg.mcts.reuse_tree
        self.no_resign = no_resign  # playthrough game: never resign (used to measure false positives)
        self.resigned = False       # ended by resignation rather than a real terminal
        self.would_resign = 0       # side that first crossed the resign threshold (0 = none yet)


def _canonical_pi(state: HexState, pi_real: np.ndarray) -> np.ndarray:
    if state.to_move == RED:
        return pi_real
    return pi_real[action_transpose(state.size)]


def _new_game(cfg: Config, board: int, swap: bool, rng: np.random.Generator) -> _Game:
    """Fresh game; when resignation is on, a `resign_playthrough` fraction are flagged no-resign so
    they play to the end and let us measure the resignation false-positive rate."""
    no_resign = cfg.selfplay.resign_enabled and rng.random() < cfg.selfplay.resign_playthrough
    return _Game(HexState.initial(board, swap), no_resign=no_resign)


def _advance_game(g: _Game, root, cfg: Config, rng: np.random.Generator, resign: bool) -> bool:
    """Record the searched position for `g`, then resign or play one selected move; set g.done/winner.
    Returns True when the game finishes.  Shared by both self-play loops so the move/resign logic lives
    in one place.  With `resign` off this is exactly the plain record-then-play step.

    Resignation: once past `resign_min_ply`, if the post-search root value (side-to-move perspective)
    is below `resign_threshold`, that side is almost surely lost and gives up.  A `no_resign`
    playthrough game never resigns but records the side that WOULD have (g.would_resign), so the
    caller can check afterwards whether it actually went on to win (a false positive)."""
    num_actions = cfg.game.num_actions
    pi_real = mcts.policy_distribution(root, num_actions)
    g.history.append((encode(g.state), _canonical_pi(g.state, pi_real), g.state.to_move))

    if resign and g.ply >= cfg.selfplay.resign_min_ply and mcts.root_value(root) < cfg.selfplay.resign_threshold:
        if g.would_resign == 0:
            g.would_resign = g.state.to_move
        if not g.no_resign:
            g.done = True
            g.winner = 3 - g.state.to_move  # side to move resigns → opponent wins (RED=1, BLUE=2)
            g.resigned = True
            g.root = None
            return True

    temperature = cfg.selfplay.temperature if g.ply < cfg.selfplay.temperature_moves else 0.0
    action = mcts.select_action(root, num_actions, temperature, rng)
    child = root.children.get(action) if cfg.mcts.reuse_tree else None
    g.state = g.state.play(action)
    g.ply += 1
    if g.state.is_terminal():
        g.done = True
        g.winner = g.state.winner
    g.root = child if (cfg.mcts.reuse_tree and not g.done) else None
    return g.done


def play_games(evaluator: Evaluator, cfg: Config, num_games: int, add_noise: bool, rng: np.random.Generator,
               simulations: int | None = None, max_concurrent: int | None = None) -> List[Sample]:
    """Play `num_games` self-play games.  If `max_concurrent` is set (and < num_games),
    maintains that many active games at all times — when a game finishes, a new one takes
    its slot immediately.  This keeps the inference server fed with a constant-size batch
    of leaves, eliminating the GPU-utilisation decline as games finish.

    Default (max_concurrent=None) plays all games concurrently — the original behaviour,
    used by the single-process path and smoke tests."""
    sims = simulations if simulations is not None else cfg.mcts.simulations
    num_actions = cfg.game.num_actions
    reuse = cfg.mcts.reuse_tree
    max_conc = max_concurrent or num_games
    max_conc = min(max_conc, num_games)
    samples: List[Sample] = []
    board = cfg.game.board_size
    swap = cfg.game.swap_rule

    resign = cfg.selfplay.resign_enabled
    # Active game slots (None = empty)
    slots: list[object] = [None] * max_conc
    completed = 0
    started = 0
    # Fill initial slots
    for i in range(max_conc):
        slots[i] = _new_game(cfg, board, swap, rng)
        started += 1

    while completed < num_games:
        active = [(i, g) for i, g in enumerate(slots) if g is not None and not g.done]
        if not active:
            break
        indices, games_list = zip(*active)
        roots = [(g.root if reuse and g.root is not None else mcts.make_root(g.state))
                 for g in games_list]
        mcts.run_batched(roots, evaluator, cfg, sims, add_noise, rng)

        for idx, g, root in zip(indices, games_list, roots):
            if _advance_game(g, root, cfg, rng, resign):
                completed += 1
                # Extract samples from the finished game
                for planes, pi, to_move in g.history:
                    z = 1.0 if g.winner == to_move else -1.0
                    samples.append((planes, pi, z))
                # Replace with a new game if we haven't reached the target
                if started < num_games:
                    slots[idx] = _new_game(cfg, board, swap, rng)
                    started += 1
                else:
                    slots[idx] = None

    return samples


# ---- process-level parallelism for CPU boxes ----

# Per-worker state, built once in the Pool initializer so the weights are pickled once per process
# (not once per chunk). Lets us hand out many small chunks for load-balancing without re-sending the net.
_WORKER: dict = {}


def build_eval_net(cfg: Config, np_state, device: str = None) -> Evaluator:
    """Rebuild a net on `device` (default cfg.device) from numpy weights and wrap it in an Evaluator.

    Fan-out workers pass cfg.worker_eval_device() (CPU on a CUDA box) so hundreds of workers don't
    each spin up a CUDA context and OOM the GPU; the in-process single-actor path uses cfg.device.

    Weights arrive as numpy (not torch tensors) so the inter-process transfer avoids torch's
    shared-memory tensor reducer, which would hold an FD per tensor per worker in the parent and
    exhaust the open-file limit (EMFILE) once many workers run.
    """
    import torch

    device = device or cfg.device
    if device == "cpu":
        torch.set_num_threads(1)  # each CPU worker is single-threaded; parallelism is across workers
    net = build_net(cfg).to(device)
    # np_state may carry torch.compile's '_orig_mod.' prefix; strip it so it loads into a plain net.
    net.load_state_dict(CleanStateDict({k: torch.from_numpy(v).to(device) for k, v in np_state.items()}))
    net.eval()
    return Evaluator(net, device)


def to_numpy_state(state_dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def _init_worker(cfg: Config, np_state, device: str) -> None:
    _WORKER["cfg"] = cfg
    _WORKER["evaluator"] = build_eval_net(cfg, np_state, device)


def _claim_worker_index(counter, lock, num_workers: int) -> int:
    """Each Pool worker claims a distinct index (its own response queue) once, in the initializer."""
    with lock:
        idx = counter.value
        counter.value += 1
    return idx % num_workers


def _init_worker_remote(cfg: Config, servers_data, counter, lock,
                        stream_args=None) -> None:
    """Server mode (single or multi-GPU): the worker does CPU search and evaluates
    leaves via its assigned GPU's inference server.

    servers_data = [(req_q, resp_qs), ...] — one entry per GPU.  The worker claims
    a global index, picks its GPU via index % num_gpus, and then its per-GPU worker
    slot via index // num_gpus.

    If stream_args=(game_counter, seed_counter, stream_lock, total_games) is
    provided, the worker runs in streaming mode — continuously replacing finished
    games from a shared seed pool."""
    from train.inference_server import RemoteEvaluator
    idx = _claim_worker_index(counter, lock, sum(len(sd[1]) for sd in servers_data))
    gpu = idx % len(servers_data)
    req_q, resp_qs = servers_data[gpu]
    worker_slot = (idx // len(servers_data)) % len(resp_qs)
    _WORKER["cfg"] = cfg
    _WORKER["evaluator"] = RemoteEvaluator(0, worker_slot, req_q, resp_qs[worker_slot])
    if stream_args is not None:
        _WORKER["stream"] = stream_args


def _play_chunk(args) -> Tuple[int, List[Sample]]:
    num_games, seed = args
    rng = np.random.default_rng(seed)
    cfg = _WORKER["cfg"]
    samples = play_games(_WORKER["evaluator"], cfg, num_games, add_noise=True, rng=rng,
                         max_concurrent=cfg.selfplay.parallel_games)
    return num_games, samples  # report game count so the parent can track progress in completion order


def _play_worker_stream(_args=None) -> Tuple[int, List[Sample]]:
    """Server-mode worker: maintain parallel_games concurrent games, replacing
    finished ones from a shared seed pool, until total_games are completed across
    all workers.  Keeps the inference server fed with a constant-size batch.

    Reads shared counters from _WORKER[\"stream\"] (set by the initializer)."""
    game_counter, seed_counter, lock, total_games, max_conc, progress_log, resign_stats = _WORKER["stream"]
    resign_ct, pt_ct, fp_ct = resign_stats  # resigned games, playthrough triggers, false positives
    cfg = _WORKER["cfg"]
    evaluator = _WORKER["evaluator"]
    num_actions = cfg.game.num_actions
    reuse = cfg.mcts.reuse_tree
    sims = cfg.mcts.simulations
    shards = max(1, cfg.mcts.pipeline_shards)
    resign = cfg.selfplay.resign_enabled
    board = cfg.game.board_size
    swap = cfg.game.swap_rule
    rng = np.random.default_rng()  # per-worker; determinism across workers doesn't matter
    samples: List[Sample] = []
    completed = 0  # by this worker
    progress_every = float(os.environ.get("SELFPLAY_PROGRESS_INTERVAL", "30"))

    def _maybe_log_progress() -> None:
        # All workers share one last-log timestamp, so the whole generation emits at most one
        # progress line per interval (not one per worker — that was ~actors× too chatty).
        now = time.time()
        should_log = False
        with lock:
            gd = game_counter.value
            if now - progress_log.value >= progress_every:
                progress_log.value = now
                should_log = True
        if should_log:
            from train.clock import log
            log(f"[selfplay] {gd}/{total_games} ({gd / max(1, total_games):.0%})")

    # Active slots
    slots: list = [None] * max_conc
    started = 0
    for i in range(max_conc):
        slots[i] = _new_game(cfg, board, swap, rng)
        started += 1

    def _claim_seed() -> int | None:
        with lock:
            if seed_counter.value >= total_games:
                return None
            s = seed_counter.value
            seed_counter.value += 1
        return s

    while True:
        with lock:
            global_done = game_counter.value
        if global_done >= total_games:
            break
        active = [(i, g) for i, g in enumerate(slots) if g is not None and not g.done]
        if not active:
            break
        indices, games_list = zip(*active)
        roots = [(g.root if reuse and g.root is not None else mcts.make_root(g.state))
                 for g in games_list]
        mcts.run_batched_streaming(roots, evaluator, cfg, sims, True, rng, shards)

        for idx, g, root in zip(indices, games_list, roots):
            if not _advance_game(g, root, cfg, rng, resign):
                continue
            completed += 1
            with lock:
                game_counter.value += 1
                if g.resigned:
                    resign_ct.value += 1
                if g.no_resign and g.would_resign:               # a playthrough that hit the threshold
                    pt_ct.value += 1
                    if g.winner == g.would_resign:               # …and the would-resign side still won
                        fp_ct.value += 1
            _maybe_log_progress()
            for planes, pi, to_move in g.history:
                z = 1.0 if g.winner == to_move else -1.0
                samples.append((planes, pi, z))
            slots[idx] = _new_game(cfg, board, swap, rng) if _claim_seed() is not None else None

    return completed, samples


def split_evenly(total: int, parts: int) -> List[int]:
    """Split `total` into `parts` near-equal sizes, each >= 1."""
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def chunk_sizes(cfg: Config, num_games: int, actors: int, device: str = None) -> List[int]:
    """Per-task game counts, tuned to the worker inference `device` (default cfg.device).

    Each chunk contains enough concurrent games that MCTS produces many leaves per step,
    which the inference server coalesces into large batches.  (1-game chunks destroy
    batching: one leaf per worker per step.)
    """
    device = device or cfg.device
    if device == "cuda":
        chunk = max(1, cfg.selfplay.parallel_games)
        parts = max(actors * 2, (num_games + chunk - 1) // chunk)
        return split_evenly(num_games, parts)
    return split_evenly(num_games, actors * 4)


def drain_pool(pool, result_iter, num_tasks: int, timeout: float, on_result: Callable[[object], None],
               read_progress: Optional[Callable[[], int]] = None) -> bool:
    """Consume `num_tasks` results from an `imap_unordered` iterator, calling `on_result(item)` for each.

    Watchdog (terminate the pool and return False if it fires; `timeout` <= 0 disables it):

      * default (`read_progress` is None) — fire when no RESULT arrives within `timeout`. Right for the
        chunked paths, where results stream back as chunks finish, so a silent gap means a dead worker.

      * streaming self-play (`read_progress` given) — fire only when that counter (completed games)
        hasn't advanced for `timeout`. Streaming workers return just ONCE, at the very end of the whole
        generation, so a return-based deadline would kill a slow-but-healthy generation and discard
        every completed game (a random-net gen 1 can run well past 1800s before the first worker
        returns). Progress-based detection still catches a true hang — the game count simply stalls."""
    to = timeout if timeout and timeout > 0 else None
    if read_progress is None:
        for _ in range(num_tasks):
            try:
                item = result_iter.next(to)
            except mp.TimeoutError:
                pool.terminate()
                return False
            on_result(item)
        return True

    # Progress-aware (stall) watchdog: poll for results; on each gap check that games are still
    # completing, and only give up when the count has been frozen for `timeout`.
    poll = min(30.0, to) if to else 30.0
    last_count = read_progress()
    last_advance = time.time()
    collected = 0
    while collected < num_tasks:
        try:
            item = result_iter.next(poll)
        except mp.TimeoutError:
            count = read_progress()
            if count != last_count:
                last_count, last_advance = count, time.time()
            elif to is not None and time.time() - last_advance >= to:
                pool.terminate()
                return False
            continue
        on_result(item)
        collected += 1
        last_count, last_advance = read_progress(), time.time()  # a return is progress too
    return True


ProgressFn = Callable[[int, int], None]


def generate(cfg: Config, state_dict, num_games: int, base_seed: int,
             progress: Optional[ProgressFn] = None) -> List[Sample]:
    """Generate `num_games` self-play games using the given network weights.

    `progress(done, total)` (optional) is called from this process as chunks complete (in completion
    order), so the caller can log incremental progress.

    MPS runs a single in-process actor (multi-process CUDA-style sharing isn't reliable on Apple).
    Both CUDA and CPU fan out across many worker processes — on CUDA so the GPU stays fed while many
    cores do tree work in parallel (each worker holds its own copy of the net on the shared card); on
    CPU so every core generates games. The net is loaded once per worker via the Pool initializer.
    """
    actors = cfg.resolve_actors()

    if cfg.device == "mps" or actors <= 1:
        evaluator = build_eval_net(cfg, to_numpy_state(state_dict))
        rng = np.random.default_rng(base_seed)
        out: List[Sample] = []
        remaining = num_games
        done = 0
        while remaining > 0:
            chunk = min(cfg.selfplay.parallel_games, remaining)
            out.extend(play_games(evaluator, cfg, chunk, add_noise=True, rng=rng))
            remaining -= chunk
            done += chunk
            if progress:
                progress(done, num_games)
        return out

    np_state = to_numpy_state(state_dict)
    use_server = cfg.use_inference_server()
    ctx = mp.get_context("spawn")
    results: List[List[Sample]] = []
    done = 0

    # Streaming workers report their own incremental progress (deduped across workers); the parent
    # only sees them return at the very end, so its callback would just add a redundant, overshooting
    # line there. Keep the parent progress for the chunked path, where it IS the incremental signal.
    emit_progress = None if use_server else progress
    read_progress: Optional[Callable[[], int]] = None  # streaming sets this → stall-based watchdog
    resign_stats = None  # streaming sets this → (resigned, playthrough-triggers, false-positives)

    def _on(item) -> None:
        nonlocal done
        n, samples = item
        results.append(samples)
        done += n
        if emit_progress:
            emit_progress(done, num_games)

    servers: list = []
    if use_server:
        from train.inference_server import InferenceServer
        ngpus = cfg.num_gpus()
        workers_per_gpu = actors // ngpus
        alloc = [workers_per_gpu] * ngpus
        for i in range(actors - workers_per_gpu * ngpus):
            alloc[i] += 1
        alloc = [a for a in alloc if a > 0]
        ngpus = len(alloc)

        for gpu_id, nw in enumerate(alloc):
            srv = InferenceServer(cfg, [np_state], cfg.device, nw, ctx, gpu_id=gpu_id)
            srv.start()
            servers.append(srv)

        servers_data = [(s.req_q, s.resp_qs) for s in servers]
        counter, lock = servers[0].counter, servers[0].lock

        # Streaming: shared game_counter + seed_counter.  Each worker maintains
        # parallel_games active games, atomically claiming new seeds from the
        # pool when a game finishes.  The GPU sees a constant-size batch of
        # leaves — no decline as games complete.
        game_counter = ctx.Value("i", 0)
        seed_counter = ctx.Value("i", 0)  # game counter, not RNG seed
        progress_log = ctx.Value("d", 0.0)  # shared last-progress-log time → one line/interval, not per worker
        resign_stats = (ctx.Value("i", 0), ctx.Value("i", 0), ctx.Value("i", 0))
        stream_lock = ctx.Lock()
        max_conc = max(1, cfg.selfplay.parallel_games // max(1, actors))
        stream_args = (game_counter, seed_counter, stream_lock, num_games, max_conc, progress_log, resign_stats)
        read_progress = lambda: game_counter.value  # completed-game count → watchdog measures progress
        tasks = [None] * actors  # dummy — real work is driven by shared counters
        fn = _play_worker_stream
        initializer, initargs = _init_worker_remote, (cfg, servers_data, counter, lock, stream_args)
        gpu_label = (f"gpu-server({cfg.device})" if ngpus == 1
                     else f"gpu-server({ngpus}×{cfg.device})")
        shards = max(1, cfg.mcts.pipeline_shards)
        log(f"[selfplay] fanning {num_games} games across {actors} actors "
            f"(eval on {gpu_label}, streaming {max_conc} concurrent per worker"
            + (f", {shards}-way pipeline" if shards > 1 else "") + ")")
    else:
        chunk_dev = cfg.worker_eval_device()
        sizes = chunk_sizes(cfg, num_games, actors, chunk_dev)
        tasks = [(size, base_seed + i) for i, size in enumerate(sizes)]
        fn = _play_chunk
        initializer, initargs = _init_worker, (cfg, np_state, chunk_dev)
        gpu_label = chunk_dev
        log(f"[selfplay] fanning {num_games} games across {actors} actors (eval on {gpu_label}), "
            f"{len(sizes)} chunks (avg {num_games // max(1, len(sizes))} games/chunk, "
            f"max parallel per chunk={min(cfg.selfplay.parallel_games, max(sizes) if sizes else 0)})")

    try:
        with ctx.Pool(processes=actors, initializer=initializer, initargs=initargs) as pool:
            it = pool.imap_unordered(fn, tasks)
            ok = drain_pool(pool, it, len(tasks), cfg.train.selfplay_timeout, _on, read_progress)
    finally:
        for srv in servers:
            srv.stop()
    if not ok:
        stalled = " with no games completing" if read_progress is not None else " with no result"
        log(f"[selfplay] WARNING: worker watchdog fired after {cfg.train.selfplay_timeout:.0f}s"
            f"{stalled} — terminated pool, continuing with {done}/{num_games} games "
            f"({sum(len(r) for r in results)} samples).")
    if resign_stats is not None and cfg.selfplay.resign_enabled:
        n_resign, n_pt, n_fp = (v.value for v in resign_stats)
        fp_rate = n_fp / n_pt if n_pt else 0.0
        log(f"[selfplay] resign: {n_resign}/{num_games} games ended early ({n_resign / max(1, num_games):.0%}); "
            f"false-positive {n_fp}/{n_pt} playthroughs ({fp_rate:.0%}) — raise RESIGN_THRESHOLD if high")
    return [s for r in results for s in r]
