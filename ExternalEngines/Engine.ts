// A real, standalone external Hex engine — the reference for ./ExternalEngines.
//
// It opens ONE long-lived WebSocket to the backend with its durable (EngineId, Token)
// credential, watches for its turn in every game pushed to it, and plays the first empty
// cell (a trivial but legal strategy — swap is always declined). One connection handles
// every concurrent match; reconnects are automatic and idempotent.
//
// It also instruments the one thing this directory exists to study: how much the server
// pushes. For every `EngineState` it records how many GameViews the message carried, and
// prints a running summary. After the network-performance fix a *change* to one game pushes
// exactly that one game (not a snapshot of every active game), so once steady state is
// reached the per-message game count is 1 regardless of how many matches are in flight.
//
// Env: ENGINE_WS (default ws://localhost:3001), ENGINE_ID, ENGINE_TOKEN.

type Cell = number; // 0 empty, 1 Red, 2 Blue
type GameView = {
  GameId: string;
  BoardSize: number;
  Cells: Cell[];
  YourColor: "Red" | "Blue";
  Turn: "Red" | "Blue";
  CanSwap: boolean;
  MoveCount: number;
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

// ---- Instrumentation: how big are the pushes the server sends us? ----
const Stats = { Messages: 0, GameViews: 0, MaxGamesInMessage: 0, Moves: 0 };

function RecordPush(Count: number): void {
  Stats.Messages++;
  Stats.GameViews += Count;
  if (Count > Stats.MaxGamesInMessage) Stats.MaxGamesInMessage = Count;
}

function PrintStats(): void {
  const Avg = Stats.Messages === 0 ? 0 : (Stats.GameViews / Stats.Messages).toFixed(2);
  console.log(
    `[stats] messages=${Stats.Messages} gameViews=${Stats.GameViews} ` +
      `avgGamesPerMessage=${Avg} maxGamesInMessage=${Stats.MaxGamesInMessage} movesSent=${Stats.Moves}`,
  );
}

// ---- Decision logic: play the first empty cell, decline swaps ----
function ChooseMove(Game: GameView): { Row: number; Col: number } | null {
  const Index = Game.Cells.findIndex((Value) => Value === 0);
  if (Index < 0) return null;
  return { Row: Math.floor(Index / Game.BoardSize), Col: Index % Game.BoardSize };
}

let Sock: WebSocket | null = null;
let Backoff = 500;

function Connect(): void {
  console.log(`Connecting to ${Backend} as engine ${EngineId.slice(0, 8)}…`);
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

    const Games = Message.Games ?? [];
    RecordPush(Games.length);

    for (const Game of Games) {
      if (Game.Status !== "Active") continue;
      if (Game.Turn !== Game.YourColor) continue; // not our move in this game
      const Move = ChooseMove(Game);
      if (Move === null) continue;
      S.send(JSON.stringify({ Type: "MakeMove", Row: Move.Row, Col: Move.Col, GameId: Game.GameId }));
      Stats.Moves++;
    }
    PrintStats();
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
  PrintStats();
  Sock?.close();
  process.exit(0);
});

Connect();
