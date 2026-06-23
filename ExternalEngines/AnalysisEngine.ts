// A real, self-hosted external Hex *analysis* engine. It connects like any external engine, but
// instead of playing it answers AnalyzeRequest messages: the server sends a finished match's move
// list, this engine evaluates it ply by ply with the shared evaluator in ./Hex.ts and replies with an
// AnalysisResult. The result is what the replay viewer overlays (eval graph, per-move quality badges,
// best-move ring). Register it with SupportsAnalysis = true so the server routes analysis to it.
//
// One connection handles every request; reconnects are automatic. Heavy work runs off the socket
// callback via a microtask so a long analysis never blocks the message loop.
//
// Env: ENGINE_WS (default ws://localhost:3001), ENGINE_ID, ENGINE_TOKEN.

import { AnalyzeMatch, type ReplayMove } from "./Hex";

type AnalyzeRequest = {
  Type: "AnalyzeRequest";
  RequestId: string;
  GameId: string;
  BoardSize: number;
  SwapRule: boolean;
  Moves: ReplayMove[];
};

const Backend = process.env.ENGINE_WS ?? "ws://localhost:3001";
const EngineId = process.env.ENGINE_ID ?? "";
const Token = process.env.ENGINE_TOKEN ?? "";

if (!EngineId || !Token) {
  console.error("ENGINE_ID and ENGINE_TOKEN are required");
  process.exit(2);
}

const Url = `${Backend}/Ws?EngineId=${encodeURIComponent(EngineId)}&Token=${encodeURIComponent(Token)}`;

async function Handle(Socket: WebSocket, Request: AnalyzeRequest): Promise<void> {
  console.log(`Analysing ${Request.GameId} (${Request.Moves.length} moves, ${Request.BoardSize}×${Request.BoardSize})…`);
  // Yield to the event loop first so a large board doesn't stall other messages on the socket.
  await Promise.resolve();
  const Analysis = AnalyzeMatch(Request.Moves, Request.BoardSize, Request.SwapRule);
  Socket.send(JSON.stringify({ Type: "AnalysisResult", RequestId: Request.RequestId, Analysis }));
}

let Sock: WebSocket | null = null;
let Backoff = 500;

function Connect(): void {
  console.log(`AnalysisEngine connecting to ${Backend} as ${EngineId.slice(0, 8)}…`);
  const S = new WebSocket(Url);
  Sock = S;

  S.addEventListener("open", () => {
    Backoff = 500;
    console.log("WebSocket open.");
  });

  S.addEventListener("message", (Event: MessageEvent) => {
    let Message: { Type: string; Message?: string } & Partial<AnalyzeRequest>;
    try {
      Message = JSON.parse(String(Event.data));
    } catch {
      return;
    }
    if (Message.Type === "Error") {
      console.error("server error:", Message.Message);
      return;
    }
    // Active games are still pushed as EngineState (an analysis engine simply never plays); ignore
    // everything except analysis requests.
    if (Message.Type === "AnalyzeRequest" && Message.RequestId !== undefined) {
      void Handle(S, Message as AnalyzeRequest);
    }
  });

  S.addEventListener("close", () => {
    Sock = null;
    console.log(`WebSocket closed; reconnecting in ${Backoff}ms…`);
    setTimeout(Connect, Backoff);
    Backoff = Math.min(Backoff * 2, 5000);
  });

  S.addEventListener("error", () => console.error("WebSocket error."));
}

process.on("SIGINT", () => {
  console.log("\nShutting down.");
  Sock?.close();
  process.exit(0);
});

Connect();
