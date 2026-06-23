"""The deployed external engine: plays and analyses on the live backend.

It connects to the backend exactly as documented in Backend/Engines.md (one durable WebSocket with its
(EngineId, Token) credential), loads checkpoints/best.pt, and:
  - on its turn, runs MCTS and plays a move chosen from the visit-count *move list* by temperature
    (temperature 0 = the most-visited move; >0 = sample the list — the directive's behaviour),
  - on an AnalyzeRequest, evaluates the finished match position-by-position and returns an
    AnalysisResult (win probability + best moves per ply) for the replay viewer.

Heavy search runs in a thread so the socket stays responsive. Env: ENGINE_WS, ENGINE_ID, ENGINE_TOKEN,
MODEL_PATH (see config.EngineConfig).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets

from config import load
from hex.board import HexState, RED, BLUE, EMPTY
from net.model import HexNet
from net.evaluator import Evaluator
from search import mcts

CFG = load()
RNG = np.random.default_rng()
EXECUTOR = ThreadPoolExecutor(max_workers=1)  # serialise heavy compute; keep the event loop free


def _color_to_stone(color: str) -> int:
    return RED if color == "Red" else BLUE


def load_engine(model_path: str, device: str):
    ckpt = torch.load(model_path, map_location=device)
    c = ckpt["config"]
    net = HexNet(c["board_size"], c["in_planes"], c["channels"], c["blocks"], c["value_hidden"])
    net.load_state_dict(ckpt["model"])
    net.eval().to(device)
    print(f"[engine] loaded {model_path}: board={c['board_size']} net={c['channels']}x{c['blocks']} "
          f"generation={ckpt.get('generation', '?')} device={device}", flush=True)
    return net, c["board_size"], Evaluator(net, device)


class HexEngine:
    def __init__(self) -> None:
        self.device = CFG.device
        self.net, self.board_size, self.evaluator = load_engine(CFG.engine.model_path, self.device)
        self.num_actions = self.board_size * self.board_size + 1

    # ---- play ----

    def decide(self, game: dict) -> dict:
        """Return the message to send for one game where it is our turn."""
        size = game["BoardSize"]
        if size != self.board_size:
            # The engine is trained for one board size; play a safe legal move on anything else.
            cells = np.asarray(game["Cells"], dtype=np.int8)
            empties = np.nonzero(cells == EMPTY)[0]
            idx = int(empties[0])
            return {"Type": "MakeMove", "Row": idx // size, "Col": idx % size, "GameId": game["GameId"]}

        state = HexState.from_cells(
            size, CFG.game.swap_rule,
            np.asarray(game["Cells"], dtype=np.int8),
            _color_to_stone(game["Turn"]),
            move_count=game.get("MoveCount", 0),
            swap_available=bool(game.get("CanSwap", False)),
        )
        root = mcts.make_root(state)
        mcts.run_batched([root], self.evaluator, CFG, CFG.engine.simulations, add_noise=False, rng=RNG)
        action = mcts.select_action(root, self.num_actions, CFG.engine.temperature, RNG)
        if action == size * size:
            return {"Type": "Swap", "GameId": game["GameId"]}
        return {"Type": "MakeMove", "Row": action // size, "Col": action % size, "GameId": game["GameId"]}

    # ---- analysis ----

    def analyze(self, request: dict) -> List[dict]:
        size = request["BoardSize"]
        moves = request["Moves"]
        if size != self.board_size:
            # Can't analyse a board the network wasn't trained on; return neutral evaluations.
            return [{"Ply": ply, "WinProb": 0.5, "BestMoves": []} for ply in range(len(moves) + 1)]

        states: List[Optional[HexState]] = []
        cells = np.zeros(size * size, dtype=np.int8)
        for ply in range(len(moves) + 1):
            # side to move at this position
            if ply < len(moves):
                to_move = _color_to_stone(moves[ply]["Color"])
            else:
                last = moves[-1] if moves else None
                to_move = RED if last is None else (BLUE if last["Color"] == "Red" else RED)
            states.append(HexState.from_cells(size, CFG.game.swap_rule, cells, to_move, move_count=ply))
            # apply this ply's placement for the next position (swaps move no stone — mirrors the backend)
            if ply < len(moves):
                mv = moves[ply]
                if mv["Kind"] == "Place" and mv["Row"] is not None and mv["Col"] is not None:
                    cells = cells.copy()
                    cells[mv["Row"] * size + mv["Col"]] = _color_to_stone(mv["Color"])

        roots = [mcts.make_root(s) for s in states]
        non_terminal = [r for r in roots if not r.resolved()]
        if non_terminal:
            sims = max(16, CFG.engine.simulations // 4)
            mcts.run_batched(non_terminal, self.evaluator, CFG, sims, add_noise=False, rng=RNG)

        out: List[dict] = []
        for ply, root in enumerate(roots):
            if root.terminal:
                # The side to move has lost (opponent connected); 0 win probability for them.
                out.append({"Ply": ply, "WinProb": 0.0, "BestMoves": []})
                continue
            value = self._root_value(root)
            win_prob = (value + 1.0) / 2.0
            best = []
            for action, _prob, q in mcts.ranked_moves(root, self.num_actions)[:4]:
                move_win = (q + 1.0) / 2.0
                if action == size * size:
                    best.append({"Row": 0, "Col": "Swap", "WinProb": move_win})
                else:
                    best.append({"Row": action // size, "Col": action % size, "WinProb": move_win})
            out.append({"Ply": ply, "WinProb": float(win_prob), "BestMoves": best, "DepthOrNodes": int(root.sum_n)})
        return out

    @staticmethod
    def _root_value(root: mcts.Node) -> float:
        if root.solved_value is not None:
            return float(root.solved_value)
        if root.sum_n > 0:
            return float(sum(root.child_w.values()) / root.sum_n)
        return 0.0


async def run() -> None:
    if not CFG.engine.engine_id or not CFG.engine.token:
        print("ENGINE_ID and ENGINE_TOKEN are required", file=sys.stderr)
        sys.exit(2)
    engine = HexEngine()
    url = f"{CFG.engine.backend_ws}/Ws?EngineId={CFG.engine.engine_id}&Token={CFG.engine.token}"
    loop = asyncio.get_event_loop()
    backoff = 0.5

    while True:
        try:
            async with websockets.connect(url, max_size=None) as ws:
                print(f"[engine] connected to {CFG.engine.backend_ws}", flush=True)
                backoff = 0.5
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = msg.get("Type")
                    if kind == "Error":
                        print("[engine] server error:", msg.get("Message"), flush=True)
                    elif kind == "EngineState":
                        for game in msg.get("Games", []):
                            if game.get("Status") != "Active" or game.get("Turn") != game.get("YourColor"):
                                continue
                            reply = await loop.run_in_executor(EXECUTOR, engine.decide, game)
                            await ws.send(json.dumps(reply))
                    elif kind == "AnalyzeRequest":
                        analysis = await loop.run_in_executor(EXECUTOR, engine.analyze, msg)
                        await ws.send(json.dumps({"Type": "AnalysisResult", "RequestId": msg["RequestId"], "Analysis": analysis}))
        except Exception as exc:  # reconnect on any drop
            print(f"[engine] disconnected ({exc}); reconnecting in {backoff:.1f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)


if __name__ == "__main__":
    asyncio.run(run())
