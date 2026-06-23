// Self-contained verification of the external-engine network-performance fix.
//
// It opens an instrumented engine socket (which only counts what the server pushes) and then
// spins up N human clients, each of which challenges the same external engine — giving the
// engine N concurrent matches. Both sides play the first empty cell, so moves keep flowing.
//
// The metric that matters is MaxGamesInMessage: how many GameViews the server packs into a
// single push. The engine connects before any games exist, so it never receives a multi-game
// resume snapshot; every push is driven by a single game changing. Before the fix each such
// push carried a snapshot of ALL N active games (so MaxGamesInMessage ≈ N and total bandwidth
// is O(N) per move); after the fix each push carries exactly the one game that changed
// (MaxGamesInMessage === 1, O(1) per move).
//
// Env: ENGINE_WS, BACKEND_WS (human socket base), ENGINE_ID, ENGINE_TOKEN, GAMES, BOARD_SIZE.

const EngineBase = process.env.ENGINE_WS ?? "ws://localhost:3001";
const HumanBase = process.env.BACKEND_WS ?? EngineBase;
const EngineId = process.env.ENGINE_ID ?? "";
const Token = process.env.ENGINE_TOKEN ?? "";
const GameCount = Number(process.env.GAMES ?? "10");
const BoardSize = Number(process.env.BOARD_SIZE ?? "11");

if (!EngineId || !Token) {
  console.error("ENGINE_ID and ENGINE_TOKEN are required");
  process.exit(2);
}

type GameView = {
  GameId: string;
  BoardSize: number;
  Cells: number[];
  YourColor: "Red" | "Blue";
  Turn: "Red" | "Blue";
  Status: "Active" | "RedWon" | "BlueWon";
};

const Stats = { Messages: 0, GameViews: 0, MaxGamesInMessage: 0, EngineMoves: 0 };

function FirstFree(Cells: number[], Size: number): { Row: number; Col: number } | null {
  const Index = Cells.findIndex((Value) => Value === 0);
  if (Index < 0) return null;
  return { Row: Math.floor(Index / Size), Col: Index % Size };
}

function Delay(Ms: number): Promise<void> {
  return new Promise((Resolve) => setTimeout(Resolve, Ms));
}

// ---- The instrumented engine ----
function StartEngine(): WebSocket {
  const Url = `${EngineBase}/Ws?EngineId=${encodeURIComponent(EngineId)}&Token=${encodeURIComponent(Token)}`;
  const Socket = new WebSocket(Url);
  Socket.addEventListener("message", (Event: MessageEvent) => {
    let Message: { Type: string; Games?: GameView[] };
    try {
      Message = JSON.parse(String(Event.data));
    } catch {
      return;
    }
    if (Message.Type !== "EngineState") return;
    const Games = Message.Games ?? [];
    Stats.Messages++;
    Stats.GameViews += Games.length;
    Stats.MaxGamesInMessage = Math.max(Stats.MaxGamesInMessage, Games.length);
    for (const Game of Games) {
      if (Game.Status !== "Active" || Game.Turn !== Game.YourColor) continue;
      const Move = FirstFree(Game.Cells, Game.BoardSize);
      if (Move === null) continue;
      Socket.send(JSON.stringify({ Type: "MakeMove", Row: Move.Row, Col: Move.Col, GameId: Game.GameId }));
      Stats.EngineMoves++;
    }
  });
  return Socket;
}

// ---- One human who challenges the engine and plays first-free ----
function StartHuman(): WebSocket {
  const Socket = new WebSocket(`${HumanBase}/Ws`);
  let Challenged = false;
  Socket.addEventListener("message", (Event: MessageEvent) => {
    let Message: { Type: string; State?: { Phase: string; Game?: GameView } };
    try {
      Message = JSON.parse(String(Event.data));
    } catch {
      return;
    }
    if (Message.Type === "Session" && !Challenged) {
      Challenged = true;
      // Human plays Blue → the engine is Red and moves first, so every game starts producing pushes.
      Socket.send(
        JSON.stringify({
          Type: "ChallengeEngine",
          EngineId,
          Settings: { BoardSize, SwapRule: false, Color: "Blue", Clock: { Mode: "Unlimited" } },
        }),
      );
      return;
    }
    if (Message.Type !== "State" || Message.State?.Phase !== "InGame") return;
    const Game = Message.State.Game!;
    if (Game.Status !== "Active" || Game.Turn !== Game.YourColor) return;
    const Move = FirstFree(Game.Cells, Game.BoardSize);
    if (Move === null) return;
    Socket.send(JSON.stringify({ Type: "MakeMove", Row: Move.Row, Col: Move.Col, GameId: Game.GameId }));
  });
  return Socket;
}

async function Main(): Promise<void> {
  const Engine = StartEngine();
  await Delay(500); // let the engine connect before any games exist

  const Humans: WebSocket[] = [];
  for (let Index = 0; Index < GameCount; Index++) {
    Humans.push(StartHuman());
    await Delay(80); // stagger so each challenge is its own game-creation push
  }

  await Delay(6000); // let the matches play out

  console.log(
    `\nResult over ${GameCount} concurrent matches (board ${BoardSize}×${BoardSize}):\n` +
      `  engine pushes received : ${Stats.Messages}\n` +
      `  total GameViews sent    : ${Stats.GameViews}\n` +
      `  avg GameViews / push    : ${(Stats.GameViews / Stats.Messages).toFixed(2)}\n` +
      `  max GameViews in a push : ${Stats.MaxGamesInMessage}\n` +
      `  engine moves played     : ${Stats.EngineMoves}`,
  );
  if (Stats.MaxGamesInMessage <= 1) {
    console.log("\nPASS — every change pushed exactly one game (no all-games snapshot per move).");
  } else {
    console.log(`\nFAIL — a single change pushed ${Stats.MaxGamesInMessage} games (the quadratic blow-up).`);
  }

  Engine.close();
  for (const Human of Humans) Human.close();
  process.exit(Stats.MaxGamesInMessage <= 1 ? 0 : 1);
}

void Main();
