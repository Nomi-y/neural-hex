// A real, self-hosted external Hex *playing* engine. It opens ONE long-lived WebSocket to the backend
// with its durable (EngineId, Token) credential, watches for its turn in every game pushed to it, and
// plays by the shared shortest-connection-distance evaluation in ./Hex.ts — a genuine, if modest,
// opponent (no neural net, no training). One connection handles every concurrent match; reconnects
// are automatic and idempotent: it reads the position each time rather than tracking what it played.
//
// Env: ENGINE_WS (default ws://localhost:3001), ENGINE_ID, ENGINE_TOKEN.

import { ChoosePlay, type Color } from "./Hex";

type GameView = {
  GameId: string;
  BoardSize: number;
  Cells: number[];
  YourColor: Color;
  Turn: Color;
  CanSwap: boolean;
  Status: "Active" | "RedWon" | "BlueWon";
};

const Backend = process.env.ENGINE_WS ?? "ws://localhost:3001";
const EngineId = process.env.ENGINE_ID ?? "";
const Token = process.env.ENGINE_TOKEN ?? "";

if (!EngineId || !Token) {
  console.error("ENGINE_ID and ENGINE_TOKEN are required");
  process.exit(2);
}

const Url = `${Backend}/Ws?EngineId=${encodeURIComponent(EngineId)}&Token=${encodeURIComponent(Token)}`;

function Act(Socket: WebSocket, Game: GameView): void {
  if (Game.Status !== "Active" || Game.Turn !== Game.YourColor) return;
  const Choice = ChoosePlay(Game.Cells, Game.BoardSize, Game.YourColor, Game.CanSwap);
  if (Choice === null) return;
  if (Choice.Kind === "Swap") {
    Socket.send(JSON.stringify({ Type: "Swap", GameId: Game.GameId }));
  } else {
    Socket.send(JSON.stringify({ Type: "MakeMove", Row: Choice.Row, Col: Choice.Col, GameId: Game.GameId }));
  }
}

let Sock: WebSocket | null = null;
let Backoff = 500;

function Connect(): void {
  console.log(`PlayEngine connecting to ${Backend} as ${EngineId.slice(0, 8)}…`);
  const S = new WebSocket(Url);
  Sock = S;

  S.addEventListener("open", () => {
    Backoff = 500;
    console.log("WebSocket open.");
  });

  S.addEventListener("message", (Event: MessageEvent) => {
    let Message: { Type: string; Message?: string; Games?: GameView[] };
    try {
      Message = JSON.parse(String(Event.data));
    } catch {
      return;
    }
    if (Message.Type === "Error") {
      console.error("server error:", Message.Message);
      return;
    }
    if (Message.Type !== "EngineState") return;
    // Treat each push as "here is some game state, act on whatever is your turn" — never assume Games
    // is the complete list (after the first push it carries only the one game that changed).
    for (const Game of Message.Games ?? []) Act(S, Game);
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
