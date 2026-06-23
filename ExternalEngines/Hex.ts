// Self-contained Hex evaluation shared by the two external engines (PlayEngine, AnalysisEngine).
// It is a standalone copy of the server's Source/Engines/Evaluation.ts: the external engines are
// separate deployments and must not import the backend, so the logic is duplicated deliberately.
//
// The idea (no neural net, no training): for a colour, the cheapest way to join its two edges costs
// one move per *empty* cell the connecting path still needs (own stones are free, opponent stones are
// walls). Red joins top↔bottom, Blue joins left↔right. The side whose connection is closer to complete
// is winning, and that distance gap maps to a win probability — the unit both playing and analysing
// rest on (see engine-analysis.md).

export type Color = "Red" | "Blue";
export type ReplayMove = { Ordinal: number; Color: Color; Kind: "Place" | "Swap"; Row: number | null; Col: number | null };
export type MoveEval = { Row: number; Col: number | "Swap"; WinProb: number };
export type PositionAnalysis = { Ply: number; WinProb: number; BestMoves: MoveEval[]; DepthOrNodes?: number };

const Empty = 0;
const Red = 1;
const Blue = 2;
const Unreachable = Number.POSITIVE_INFINITY;
const Sharpness = 0.7;
const Tempo = 0.5;

function Other(Color: Color): Color {
  return Color === "Red" ? "Blue" : "Red";
}
function StoneOf(Color: Color): number {
  return Color === "Red" ? Red : Blue;
}

// The up-to-six hex neighbours of a flat index on an N×N row-major board.
const Offsets: ReadonlyArray<readonly [number, number]> = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];
function Neighbours(Size: number, Index: number): number[] {
  const Row = Math.floor(Index / Size);
  const Col = Index % Size;
  const Out: number[] = [];
  for (const [DRow, DCol] of Offsets) {
    const R = Row + DRow;
    const C = Col + DCol;
    if (R >= 0 && R < Size && C >= 0 && C < Size) Out.push(R * Size + C);
  }
  return Out;
}

/** Fewest additional moves the colour needs to complete an edge-to-edge connection (0-1 BFS). */
export function ConnectionDistance(Cells: number[], Size: number, Color: Color): number {
  const Mine = StoneOf(Color);
  const Theirs = StoneOf(Other(Color));
  const Cost = (Index: number): number => (Cells[Index] === Mine ? 0 : 1);
  const OnStartEdge = (Index: number): boolean => (Color === "Red" ? Math.floor(Index / Size) === 0 : Index % Size === 0);
  const OnEndEdge = (Index: number): boolean => (Color === "Red" ? Math.floor(Index / Size) === Size - 1 : Index % Size === Size - 1);

  const Dist = new Array<number>(Size * Size).fill(Unreachable);
  const Deque: number[] = [];
  for (let Index = 0; Index < Cells.length; Index++) {
    if (Cells[Index] === Theirs || !OnStartEdge(Index)) continue;
    const Entry = Cost(Index);
    if (Entry < Dist[Index]!) {
      Dist[Index] = Entry;
      if (Entry === 0) Deque.unshift(Index);
      else Deque.push(Index);
    }
  }

  let Best = Unreachable;
  while (Deque.length > 0) {
    const Current = Deque.shift()!;
    const Here = Dist[Current]!;
    if (Here >= Best) continue;
    if (OnEndEdge(Current)) Best = Math.min(Best, Here);
    for (const Next of Neighbours(Size, Current)) {
      if (Cells[Next] === Theirs) continue;
      const Candidate = Here + Cost(Next);
      if (Candidate < Dist[Next]!) {
        Dist[Next] = Candidate;
        if (Cost(Next) === 0) Deque.unshift(Next);
        else Deque.push(Next);
      }
    }
  }
  return Best;
}

/** Win probability for the side to move, from the connection-distance gap. */
export function EvaluatePosition(Cells: number[], Size: number, SideToMove: Color): number {
  const Mine = ConnectionDistance(Cells, Size, SideToMove);
  const Theirs = ConnectionDistance(Cells, Size, Other(SideToMove));
  if (Mine === 0) return 1;
  if (Theirs === 0) return 0;
  if (Mine === Unreachable && Theirs === Unreachable) return 0.5;
  if (Mine === Unreachable) return 0.01;
  if (Theirs === Unreachable) return 0.99;
  const Gap = Theirs - (Mine - Tempo);
  return Math.min(0.99, Math.max(0.01, 1 / (1 + Math.exp(-Sharpness * Gap))));
}

function CentreBias(Row: number, Col: number, Size: number): number {
  const Mid = (Size - 1) / 2;
  const Span = Mid === 0 ? 1 : Mid * Mid * 2;
  return 1 - ((Row - Mid) ** 2 + (Col - Mid) ** 2) / Span;
}

/** Rank every legal placement by the win probability it leaves the mover with (one-ply search). */
export function RankMoves(Cells: number[], Size: number, SideToMove: Color, Top = 4): MoveEval[] {
  const Mine = StoneOf(SideToMove);
  const Opp = Other(SideToMove);
  const Scored: { Eval: MoveEval; Bias: number }[] = [];
  for (let Index = 0; Index < Cells.length; Index++) {
    if (Cells[Index] !== Empty) continue;
    Cells[Index] = Mine;
    const MoverWinProb = 1 - EvaluatePosition(Cells, Size, Opp);
    Cells[Index] = Empty;
    const Row = Math.floor(Index / Size);
    const Col = Index % Size;
    Scored.push({ Eval: { Row, Col, WinProb: MoverWinProb }, Bias: CentreBias(Row, Col, Size) });
  }
  Scored.sort((A, B) => B.Eval.WinProb - A.Eval.WinProb || B.Bias - A.Bias);
  return Scored.slice(0, Top).map((S) => S.Eval);
}

function CellsAfter(Moves: ReplayMove[], Size: number, Ply: number): number[] {
  const Cells = new Array<number>(Size * Size).fill(Empty);
  for (let Index = 0; Index < Ply && Index < Moves.length; Index++) {
    const Move = Moves[Index]!;
    if (Move.Kind === "Place" && Move.Row !== null && Move.Col !== null) {
      Cells[Move.Row * Size + Move.Col] = StoneOf(Move.Color);
    }
  }
  return Cells;
}

/** Evaluate a finished match ply by ply (0 = empty board, k = after the k-th move). */
export function AnalyzeMatch(Moves: ReplayMove[], BoardSize: number, _SwapRule: boolean, Top = 4): PositionAnalysis[] {
  const Out: PositionAnalysis[] = [];
  for (let Ply = 0; Ply <= Moves.length; Ply++) {
    const Cells = CellsAfter(Moves, BoardSize, Ply);
    const EmptyCount = Cells.reduce((Sum, Value) => Sum + (Value === Empty ? 1 : 0), 0);
    const Next = Moves[Ply];
    if (Next !== undefined) {
      const Best = RankMoves(Cells, BoardSize, Next.Color, Top);
      const WinProb = Best.length > 0 ? Best[0]!.WinProb : EvaluatePosition(Cells, BoardSize, Next.Color);
      Out.push({ Ply, WinProb, BestMoves: Best, DepthOrNodes: EmptyCount });
    } else {
      const Last = Moves[Moves.length - 1];
      const SideToMove = Last === undefined ? "Red" : Other(Last.Color);
      Out.push({ Ply, WinProb: EvaluatePosition(Cells, BoardSize, SideToMove), BestMoves: [] });
    }
  }
  return Out;
}

/** The play engine's choice for a live position: swap a strong opening, else the best placement. */
export function ChoosePlay(
  Cells: number[],
  Size: number,
  Me: Color,
  CanSwap: boolean,
): { Kind: "Swap" } | { Kind: "Place"; Row: number; Col: number } | null {
  const Best = RankMoves(Cells, Size, Me, 1)[0];
  if (CanSwap) {
    // After a swap we hold the opener's colour with the board unchanged and the opponent to move,
    // so our value is 1 − (the opponent's value to move). Steal only when it beats our best reply.
    const WinIfSwap = 1 - EvaluatePosition(Cells, Size, Me);
    if (WinIfSwap > (Best?.WinProb ?? 0)) return { Kind: "Swap" };
  }
  if (Best === undefined || Best.Col === "Swap") return null;
  return { Kind: "Place", Row: Best.Row, Col: Best.Col };
}
