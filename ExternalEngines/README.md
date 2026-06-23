# ExternalEngines

Two real, self-hosted Hex engines that connect to the backend over WebSocket — one that **plays** and
one that **analyses** finished matches — plus a tool to study the server's push behaviour. Both engines
are built on a single shared evaluator (`Hex.ts`) and need no training: they reason about the classic
**shortest-connection-distance** heuristic (own stones free, empty cells cost a move, opponent stones
are walls; the side closer to joining its two edges is winning). See `../engine-analysis.md` for the
analysis model and `../Backend/Engines.md` for the protocol.

## Files

| File | Role |
|---|---|
| `Hex.ts` | Shared evaluator: connection distance → win probability, ranked candidate moves, whole-match analysis, and the play decision (incl. swap). Standalone copy of the server's `Source/Engines/Evaluation.ts`. |
| `PlayEngine.ts` | Playing engine. Connects, watches for its turn, plays the best move by `Hex.ts`, steals a strong opening via the swap rule. One connection handles every concurrent match. |
| `AnalysisEngine.ts` | Analysis engine. Answers the server's `AnalyzeRequest` (a finished match's move list) with an `AnalysisResult` — one win-probability + top moves per ply. Register it with **analysis** enabled. |
| `PerfCheck.ts` | Verification of the external-engine network-performance fix: spins up N concurrent matches and reports `max GameViews in a push` (PASS when it's 1). |

## Registering an engine (admin, one time)

Each engine needs a durable `(EngineId, Token)` credential. Register via the admin Engines tab (tick
**analysis** for the analysis engine) or the API:

```bash
# Playing engine
curl -X POST http://<backend>/Admin/Api/Engines/External \
  -H "X-Admin-Session: <admin-session-token>" -H "Content-Type: application/json" \
  -d '{ "Name": "Hex Heuristic", "MinBoardSize": 5, "MaxBoardSize": 19, "Description": "Shortest-path play" }'

# Analysis engine — SupportsAnalysis routes Analyse requests to it
curl -X POST http://<backend>/Admin/Api/Engines/External \
  -H "X-Admin-Session: <admin-session-token>" -H "Content-Type: application/json" \
  -d '{ "Name": "Hex Analyser", "MinBoardSize": 5, "MaxBoardSize": 19, "Description": "Match analysis", "SupportsAnalysis": true }'
# -> { "EngineId": "…", "Token": "…", "Name": "…" }
```

> The bundled **managed** `Heuristic` engine already plays and analyses in-process, so analysis works
> with nothing self-hosted. These external engines are the pluggable, crash-isolated alternative behind
> the same interface — and the place to grow a stronger evaluator later.

## Running

Against the local stack (`docker compose up` from the repo root → backend on `:3001`):

```bash
# Playing engine — leave it running, then challenge it from the web client.
ENGINE_WS=ws://localhost:3001 ENGINE_ID=<EngineId> ENGINE_TOKEN=<Token> bun run PlayEngine.ts

# Analysis engine — leave it running, then open a finished match → replay → Analyse (pick this provider).
ENGINE_WS=ws://localhost:3001 ENGINE_ID=<EngineId> ENGINE_TOKEN=<Token> bun run AnalysisEngine.ts

# Network-performance check — drives N concurrent matches and prints the verdict.
ENGINE_WS=ws://localhost:3001 ENGINE_ID=<EngineId> ENGINE_TOKEN=<Token> GAMES=10 BOARD_SIZE=11 bun run PerfCheck.ts
```

## How analysis flows (server ↔ engine)

1. A user clicks **Analyse** on a finished match's replay.
2. The server checks its cache `(GameId, ProviderId)`; on a miss it sends the analysis engine
   `{ Type: "AnalyzeRequest", RequestId, GameId, BoardSize, SwapRule, Moves }`.
3. The engine replies `{ Type: "AnalysisResult", RequestId, Analysis: PositionAnalysis[] }`.
4. The server persists the result (paid once) and returns it; the client overlays the eval graph,
   per-move quality badges, and the best-move ring on the replay timeline.
