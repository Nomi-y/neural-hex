# Hex — Project Context

Online two-player **Hex** game. This root repo is orchestration glue; the two apps are
**git submodules** pinned to exact commits.

```
hex/                      root repo (compose, docs, planning notes)
├── Backend/   submodule  authoritative game server — Bun + TypeScript, PostgreSQL + Drizzle
├── Frontend/  submodule  React + TypeScript client (Vite), served by nginx
└── docker-compose.yml    full stack: postgres → backend → frontend (:8080)
```

A SessionStart hook (`.claude/Context.sh`) prints live git/submodule state and the current
planning doc at the start of every session — read it before assuming repo state.

## Directive: keep `summary.txt` current

After making a meaningful change to the app (a feature, behaviour change, or notable fix),
**append** it to `summary.txt`. Follow the existing format: group work under the current
iteration heading (start a new `Hex — Iteration N summary` block when an iteration begins),
and record not just *what* changed but *why*. This is the durable history every future session
reads first — never overwrite past iterations, only add to them.

## Where context lives — read these, don't re-derive

| File | What it holds |
|------|---------------|
| `summary.txt` | **Iteration history** (1–4). What was built and *why*. Read the last iteration first. |
| `coding-conventions.md` | **Authoritative** conventions + the permissions granted for this project. |
| `additions.md`, `todos*.md`, `prompt.md`, `questions.md` | Planning notes / requested work, oldest→newest. The newest is the active spec. |
| `Backend/README.md`, `Frontend/README.md` | Per-project design notes. |

## Conventions (see `coding-conventions.md` for the full list)

- **PascalCase** for file names, directory names, functions, classes, methods, members.
- Comments only for genuinely confusing code. Prefer simple, line-by-line-readable logic.
- DRY only when sensible — bloat is the greater sin. Test flawed-prone *logic*, not coverage.
- Everything should be easy to extend, scale, or rework (seams over hardcoding).

## Permissions (granted — see `coding-conventions.md`)

- Install Bun packages as needed (keep dependencies lean).
- Run git freely; commit on feature/checkpoint. Separate branches not required.

## Architecture (one paragraph)

The backend is the single source of truth: active games live in memory for low latency and are
**written through to PostgreSQL** (via Drizzle, no raw SQL) so games and full move histories
survive refresh/restart. The server pushes full state snapshots over one WebSocket; the client
is a thin renderer, making rejoin trivial. Engines are a kind of `Participant` (Human = 1
concurrent match, Engine = unlimited) and connect either in-process (orchestrator-driven) or as
external self-hosted WebSocket clients. The `Store` and `ConnectionRegistry` interfaces are the
seams for later horizontal scaling. Tests use `InMemoryStore` — no database required.

## Commands

```bash
# Whole stack
docker compose up --build                 # http://localhost:8080

# Dev (needs the dev postgres container — see README.md)
cd Backend  && bun install && bun run Dev  # :3001
cd Frontend && bun install && bun run Dev  # :5173

# Tests (in-memory store; DATABASE_URL only needed if also booting the server)
cd Backend && bun run Test
```

Advance a submodule pin: commit inside `Backend/` (or `Frontend/`), then
`git add Backend && git commit` in the root.

## Key env vars

- Backend: `PORT` (3001), `DATABASE_URL`, `ADMIN_TOKEN`, `ADMIN_DEV_LOGIN` (default true),
  `ADMIN_PUBLIC_KEYS`, `ABANDON_TIMEOUT_MS` (initial value; live value is admin-editable).
- Frontend: `VITE_BACKEND_HTTP` / `VITE_BACKEND_WS` (default page origin).
