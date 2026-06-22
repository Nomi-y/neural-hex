# Engine analysis — feasibility study

*Authored by Claude. This document is an **analysis**, not an implementation: it was produced while
building the match-replay viewer (`match-history.md`) to scope what adding engine-driven move
analysis to Hex would take. Nothing here is wired up yet; it is the groundwork for a future
directive. Engine analysis is explicitly **out of scope** for the replay work — the viewer was built
to leave room for it, and this note records where that room is.*

---

## 1. The question: how does an engine decide a move is "good" or "bad"?

A playing engine and an *analysis* engine are not the same thing, and the gap matters here.

- A **playing engine** answers one question: *given this position, what is the single best move?* It
  returns that move and throws everything else away.
- An **analysis engine** answers a richer question: *how good is this position, and how much did each
  move help or hurt?* It returns a **scalar evaluation** of a position — for Hex the natural unit is a
  **win probability** for the side to move (0–1), since Hex has no draws and every position is a
  theoretical win for exactly one player.

Move quality is then a **delta of evaluations**, exactly as chess GUIs compute centipawn loss:

```
QualityOf(move) = Eval(position before move, from mover's view)
               −  Eval(position after  move, from mover's view, sign-flipped to the mover)
```

In practice you evaluate the position before the move, find the engine's best move and its
evaluation, then compare the *played* move's resulting evaluation against the best. A small drop is a
good/normal move; a large drop is an inaccuracy, mistake, or blunder. The familiar buckets
(`!!`, `!`, `?!`, `?`, `??`) are just thresholds on that drop. Because Hex evaluations are win
probabilities, the thresholds are intuitive: "this move handed the opponent ~30% win probability."

So "good vs bad" needs **two** things our current engines do not expose:
1. a position **evaluation** (a number), and
2. ideally the engine's **preferred move(s)** at that position, to measure the played move against.

## 2. Are the current playing engines sufficient? No — but the seam is already the right shape.

The managed-engine contract is a single method (`Backend/Source/Engines/EngineDriver.ts`):

```ts
ChooseMove(View: EngineView): Awaitable<EngineChoice | null>;
```

It returns *a move*, not *an evaluation*. The built-in engines (`FirstFree`, `RandomEngine`) have no
internal value function at all — `FirstFree` plays the first empty cell, `RandomEngine` plays a
random one. Asking them "how good was move 14?" is meaningless; they have no notion of good.

Three things are, however, already in our favour:

- **The position wrapper is analysis-ready.** `EngineView` hands an engine the full board, adjacency,
  legal moves, and side-to-move with named accessors. An analysis method would consume the *same*
  view — no new position plumbing.
- **Async is already supported.** `ChooseMove` may return a `Promise`, and the orchestrator drives it
  off the event loop and drops stale results (see iteration 12's async-managed-engines work). A real
  analysis engine *will* be heavyweight (search, a subprocess, a service call); the infrastructure to
  run it without stalling the server already exists.
- **External engines already speak over a socket.** The external-engine path means a strong
  third-party Hex bot (e.g. a MoHex-style MCTS engine) can live out of process and connect in. That is
  almost certainly where real evaluation strength would come from.

The conclusion: we do **not** need to discard the engine abstraction, we need to **extend** it with an
optional analysis capability that engines opt into.

## 3. Proposed shape (for a future directive)

### 3.1 An optional `Analyze` capability on the driver

```ts
export type MoveEval = {
  Row: number; Col: number | "Swap";
  WinProb: number;        // 0–1 for the side to move
};

export type PositionAnalysis = {
  Ply: number;            // which half-move this evaluates (0 = start)
  WinProb: number;        // evaluation of the position for the side to move
  BestMoves: MoveEval[];  // top-k candidate moves, best first (optional, k small)
  DepthOrNodes?: number;  // how hard the engine looked (for display / caching key)
};

export interface AnalysisEngine {
  AnalyzeMatch(Moves: ReplayMove[], BoardSize: number, SwapRule: boolean): Awaitable<PositionAnalysis[]>;
}
```

Keep it **optional**: an engine advertises `SupportsAnalysis` (or simply implements the interface).
Engines that don't are still fully playable; the UI just won't offer "Analyse" for them. Analysing the
*whole match in one call* (rather than ply-by-ply round-trips) lets an external engine reuse its search
tree between positions, which is a large speed-up for tree-search engines.

### 3.2 Server flow (matches the directive's sketch)

1. User clicks **Analyse** in the replay viewer.
2. Server checks the DB for a stored analysis of this `GameId` (+ engine + strength key). Cache hit →
   return it immediately.
3. Cache miss → load the move list (already available via `GetMatchReplay`), hand it to a chosen
   analysis engine over the existing managed/external seam, `await` the result.
4. **Persist** the `PositionAnalysis[]` keyed by `(GameId, EngineId, Strength)` so the compute is paid
   once. A new `MatchAnalysis` table (or a JSON column alongside the match) fits the existing Drizzle
   store cleanly.
5. Return it to the client, which overlays it on the replay.

This is a natural fit for the replay viewer that now exists: the replay is **purely client-side after
one fetch**, and an analysis overlay is just a *second* optional fetch (`/Api/Matches/:id/Analysis`)
that decorates the same timeline — eval bar, per-move badges, suggested-move arrows drawn with the
planning-overlay machinery the board already has.

### 3.3 Client display (incremental, all reusing existing machinery)

- **Eval graph** under the timeline — one win-probability point per ply.
- **Per-move badge** in the timeline list (`?!`, `??`, …) from the eval delta.
- **Best-move arrow** drawn via the board's existing arrow overlay when paused on a ply.
- All of it lazy: no analysis is computed or shown until the user asks.

## 4. Difficulty assessment

| Piece | Effort | Notes |
|-------|--------|-------|
| `AnalysisEngine` interface + capability flag | **Low** | Mirrors the existing `EngineDriver` seam. |
| Server orchestration (request → engine → respond) | **Low–Med** | Reuses async driving + external-socket transport. |
| DB persistence + cache key | **Low** | One table or JSON column; same Drizzle patterns. |
| `/Api/Matches/:id/Analysis` endpoint | **Low** | Sibling of the new `/Replay` endpoint. |
| Client overlay (eval bar, badges, arrows) | **Medium** | UI only; board/timeline already exist. |
| **A genuinely strong evaluation function** | **High** | The real cost. See below. |

The plumbing is **cheap** — days, not weeks — because every seam it needs already exists. The
expensive, open-ended part is the **evaluation quality itself**: a trustworthy Hex evaluator is a
serious undertaking (Hex is PSPACE-complete; strong engines use MCTS with a trained neural net, à la
MoHex / NeuroHex / AlphaZero-style self-play). I would **not** build that in-repo. The right move is:

- Define the analysis interface and the whole request/store/display pipeline now (cheap, high value).
- Treat the *strong evaluator* as a pluggable **external** engine — connect an existing open-source
  Hex AI rather than writing one. A weak built-in analyser (e.g. a short MCTS or even a heuristic
  bridge/edge evaluator) can ship first as a placeholder so the pipeline is exercised end-to-end.

## 5. Summary and recommendation

- "Good/bad" = **win-probability delta** between the played move and the engine's best. This requires
  an engine that emits **evaluations**, which our current play-only engines do not.
- The existing architecture is **well-shaped** for this: the `EngineView` position wrapper, async
  non-blocking engine driving, the external-engine socket transport, the Drizzle store, and the new
  client-side replay viewer are all reusable with no rework.
- **Recommended path:** add an optional `AnalyzeMatch` capability to the engine contract; route
  *Analyse* requests through the existing managed/external seam; cache results in the DB keyed by game
  + engine; overlay them on the replay timeline. Ship the pipeline against a **weak placeholder**
  analyser, then plug a **strong external** Hex engine in behind the same interface.
- The replay work was deliberately built to accommodate this: one-fetch client-side replay, a clean
  `/Api/Matches/:id/Replay` endpoint with an obvious `/Analysis` sibling, and a board that already
  knows how to draw highlights and arrows.

When a directive for this lands, the first concrete step is the `AnalysisEngine` interface plus a
throwaway placeholder analyser, so the full request → store → display loop can be proven before any
effort goes into evaluation strength.
