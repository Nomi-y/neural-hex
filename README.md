# Hex

An online application to play the two-player board game **Hex**.

Two independent projects, each its own git repository and Docker image, plus a PostgreSQL
database:

- **[`Backend/`](./Backend)** — authoritative game server (Bun + TypeScript). Matchmaking,
  rules, swap rule, clocks, engines, and move history; persisted to PostgreSQL via Drizzle.
- **[`Frontend/`](./Frontend)** — React + TypeScript client (Vite), served by nginx.

This root is its own git repository tracking the orchestration glue (compose file, docs). The
`Backend/` and `Frontend/` directories are independent git repositories and are intentionally
git-ignored here, so each component is versioned on its own.

## Game

Standard Hex: players alternately place a stone; **Red** connects top↔bottom, **Blue**
connects left↔right, and the first to bridge their two sides wins (Hex cannot draw).

Customisation when creating a match:

- **Board size** — 5×5 up to 19×19 (default 11).
- **Colour** — play as Red, Blue, or Random.
- **Swap (pie) rule** — toggle on to neutralise the first-move advantage.
- **Clock** — Unlimited, or Fischer `X` minutes + `Y` seconds increment (1–60 min, 0–60 s).

Features: random matchmaking, private invite links, **play against an engine**, **register
your own self-hosted engine**, refresh-and-rejoin, full **move history** per match, and a
one-match-per-player limit (engines are unlimited). No accounts yet — a locally stored
`PlayerId` is the identity, behind a `Participant` model ready for accounts later.

## Run it

### Docker (whole stack)

```bash
docker compose up --build
# open http://localhost:8080
```

nginx serves the client and reverse-proxies `/Ws`, `/Api` and `/Health` to the backend, so
everything is same-origin.

### Local development

```bash
# Postgres for development
docker run -d --name hexpg -e POSTGRES_PASSWORD=hex -e POSTGRES_USER=hex -e POSTGRES_DB=hex \
  -p 5433:5432 postgres:16-alpine

# terminal 1 — backend (DATABASE_URL defaults to the dev Postgres above)
cd Backend && bun install && bun run Dev      # http://localhost:3001

# terminal 2 — frontend
cd Frontend && bun install && bun run Dev     # http://localhost:5173
```

Open two browser tabs (or one normal + one private window so they get distinct `PlayerId`s)
to play both sides — or share the invite link.

## Tests

```bash
cd Backend && DATABASE_URL=postgres://hex:hex@localhost:5433/hex bun test
```

Covers win detection, clocks (incl. restart rebasing), move/turn rules, the swap rule, move
history, matchmaking, the participant model, the engine orchestrator, and a full WebSocket
end-to-end game including reconnection. Tests use the in-memory store, so they do not require
Postgres — the `DATABASE_URL` above only matters if you also boot the server.

## Design notes

See each project's `README.md`. In short: the backend is the authoritative source of truth,
holding active games in memory for low latency and **writing through to PostgreSQL** (via
Drizzle) so games — and full move histories — survive refreshes and server restarts. The
server pushes full state snapshots over one WebSocket, keeping the client a thin renderer and
making rejoining trivial. Engines are modelled as a kind of `Participant` and connect either
in-process (orchestrator-driven) or as external self-hosted WebSocket clients. The `Store`
and `ConnectionRegistry` seams are where horizontal scaling (Redis, pub/sub fan-out,
game-sticky routing) would later plug in.
