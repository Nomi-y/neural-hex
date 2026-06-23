# Hex — Project Rules, Conventions & Permissions

**This file is the single source of truth** for how to work in this repository:
rules, conventions, permissions, workflow, orientation. If anything here ever
conflicts with another doc, **this file wins** — update it here. `CLAUDE.md` imports
this file so every session loads it automatically; `coding-conventions.md` is a thin
pointer back here.

---

## 1. What this repository is

Online two-player **Hex** game. The root repo is orchestration glue; the two apps
are **git submodules** pinned to exact commits.

```
hex/                      root repo (compose, docs, planning notes)
├── Backend/   submodule  authoritative game server — Bun + TypeScript, PostgreSQL + Drizzle
├── Frontend/  submodule  React + TypeScript client (Vite), served by nginx
└── docker-compose.yml    full stack: postgres → backend → frontend (:8080)
```

**Architecture in one paragraph.** The backend is the single source of truth:
active games live in memory for low latency and are **written through to
PostgreSQL** (via Drizzle, no raw SQL) so games and full move histories survive
refresh/restart. The server pushes full state snapshots over one WebSocket; the
client is a thin renderer, making rejoin trivial. Engines are a kind of
`Participant` (Human = 1 concurrent match, Engine = unlimited) and connect either
in-process (orchestrator-driven) or as external self-hosted WebSocket clients. The
`Store` and `ConnectionRegistry` interfaces are the seams for later horizontal
scaling. Tests use `InMemoryStore` — no database required.

---

## 2. Coding conventions

- **PascalCase** for: file names, directory names, functions, classes, methods,
  class members.
- **Comments at an absolute minimum** — only explain genuinely confusing parts.
  Prefer simple, line-by-line-readable logic.
- **Maintainability/extensibility first.** Any feature should be easy to expand,
  add on to, scale up, or rework. Favour seams over hardcoding.
- **Tests:** write sensible unit tests for features that need them. The point is
  testing *logic that might be flawed*, **not** code coverage. (Note: the Backend
  has a test suite via `bun run Test`; the Frontend currently has no test harness.)
- **DRY only when sensible.** Bloating code is the greater sin.
- **Keep logic simple.** Advanced TypeScript features are fine, but the actual
  logic should be understandable just by reading it line by line.

---

## 3. Permissions granted for this project

- **Install Bun packages** as needed — keep dependencies lean / on the lower end.
- **Run git freely** — set up local repositories and commit on feature/checkpoint.
- **Commit is authorized** for this project (this is the durable per-project grant;
  it overrides the default "commit only when asked").

---

## 4. Standing workflow rules

### 4.1 Branch-per-feature, merged to `master`
- Do work on a `feature/<name>` branch **inside the relevant submodule**.
- After verifying, merge it into the mainline branch — which is **`master`**, not
  `main` — typically with `--no-ff`.
- Separate long-lived branches are not otherwise required.

### 4.2 Advancing a submodule pin
The root repo pins each submodule to an exact commit. After committing inside a
submodule:
```bash
# inside Backend/ or Frontend/: commit (and merge to master) first, then in the root:
git add Backend            # or: git add Frontend
git commit -m "Advance <X> pin: <what changed>"
```

### 4.3 Always rebuild the stack after changes  ← operational rule
After making changes, rebuild and bring the full stack back up, detached:
```bash
docker compose down
docker compose up --build -d        # http://localhost:8080
```
(Run from the repo root. Use this to validate changes against the real running
stack, not just tests.)

### 4.4 Keep `summary.txt` current  ← required directive
After a meaningful change (a feature, behaviour change, or notable fix),
**append** it to `summary.txt`:
- Group work under the current iteration heading; start a new
  `Hex — Iteration N summary` block when an iteration begins.
- Record not just *what* changed but *why*.
- **Never overwrite past iterations — only add.** This is the durable history every
  future session reads first.

---

## 5. Where context lives (read these; don't re-derive)

| File | What it holds |
|------|---------------|
| `project-rules.md` | **This file — the single source of truth** for rules, conventions, permissions, workflow. |
| `summary.txt` | **Iteration history.** What was built and *why*. Read the last iteration first. |
| `CLAUDE.md` | Thin session loader — imports this file. |
| `coding-conventions.md` | Thin pointer back to this file. |
| `additions.md`, `todos*.md`, `prompt.md`, `questions.md` | Planning notes / requested work, oldest→newest. **The newest is the active spec.** |
| `Backend/README.md`, `Frontend/README.md` | Per-project design notes. |
| `.claude/Context.sh` | SessionStart hook — prints live git/submodule state and the active planning doc. Read its output before assuming repo state. |

---

## 6. Commands

```bash
# Whole stack (preferred — and the standing rule, §4.3)
docker compose down
docker compose up --build -d              # http://localhost:8080

# Dev (needs the dev postgres container — see README.md)
cd Backend  && bun install && bun run Dev  # :3001
cd Frontend && bun install && bun run Dev  # :5173

# Backend tests (in-memory store; DATABASE_URL only needed if also booting the server)
cd Backend && bun run Test

# Frontend typecheck / build
cd Frontend && npx tsc --noEmit
cd Frontend && bun run Build
```

---

## 7. Key environment variables

**Backend:** `PORT` (3001), `DATABASE_URL`, `ADMIN_TOKEN`,
`ADMIN_DEV_LOGIN` (default true), `ADMIN_PUBLIC_KEYS`,
`ABANDON_TIMEOUT_MS` (initial value; live value is admin-editable).

**Frontend:** `VITE_BACKEND_HTTP` / `VITE_BACKEND_WS` (default page origin).
