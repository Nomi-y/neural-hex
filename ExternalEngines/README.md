# ExternalEngines

A real, self-hosted external Hex engine plus a tool to study the server's push behaviour.
This directory exists to develop and verify the **external-engine network-performance** fix:
the server pushes an external engine **only the game that changed** on each change (and the
full set of active games only right after it connects), instead of a snapshot of every game
on every move. See `Backend/Engines.md` for the protocol.

## Files

| File | Role |
|---|---|
| `Engine.ts` | Standalone reference external engine. Connects, plays the first empty cell, declines swaps, and logs how many `GameView`s each push carries. |
| `PerfCheck.ts` | Verification: connects an instrumented engine, then spins up N human clients that each challenge it, and reports `max GameViews in a push`. PASS when it's 1. |

## Running

Both take the engine's durable credential via env vars. Against the local stack
(`docker compose up` from the repo root → backend on `:3001`):

```bash
# The reference engine — leave it running, then challenge "Claude Test" from the web client.
ENGINE_WS=ws://localhost:3001 \
ENGINE_ID=<EngineId> ENGINE_TOKEN=<Token> \
bun run Engine.ts

# The performance check — drives N concurrent matches and prints the verdict.
ENGINE_WS=ws://localhost:3001 \
ENGINE_ID=<EngineId> ENGINE_TOKEN=<Token> GAMES=10 BOARD_SIZE=11 \
bun run PerfCheck.ts
```

An admin registers an external engine (`POST /Admin/Api/Engines/External`) to obtain the
`EngineId`/`Token` pair — see the external-engine section of `Backend/Engines.md`.

Expected `PerfCheck.ts` output: `avg GameViews / push ≈ 1.00`, `max GameViews in a push = 1`
regardless of `GAMES` — i.e. concurrency no longer multiplies per-move bandwidth.
