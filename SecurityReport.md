# Hex — Security Review (2026-06-22)

Scope: the authoritative game server (`Backend/`) and admin panel, plus the client's
input-handling surface (`Frontend/`). Method: manual source review of every externally
reachable entry point (HTTP routes, WebSocket messages, admin routes, the DB layer), with
each *suspected* issue reproduced against the running dev build (`docker compose`) before
being accepted or dismissed.

## Summary

One genuinely exploitable vulnerability was found and **fixed**: an unauthenticated endpoint
that disclosed any participant's private live state — including their open invite code, which
is enough to hijack their match. Everything else reviewed was either already well-defended or
is a low-severity / by-design item recorded below as a hardening recommendation.

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | **High** | `/Api/State` returned any participant's live state with no token (IDOR) | **Fixed** |
| 2 | Low | No rate limiting on `/Api/Login` and `/Api/Register` | Noted |
| 3 | Low | `/Api/Matches/:id/Moves` is world-readable, incl. active games | Noted |
| 4 | Info | Board-size ceiling is unbounded; a large admin-set max enables a memory DoS | Noted |
| 5 | Info | Session / engine tokens travel in WebSocket URL query strings | Noted |

---

## 1. Unauthenticated live-state disclosure / invite-code leak — High (FIXED)

**Where:** `Backend/Source/Server/HttpRouter.ts` — `GET /Api/State?PlayerId=`.

**What:** the endpoint returned `Service.ComputeState(ParticipantId)` for any id, with **no
token check**. A participant id is *not* a secret — it keys the public profile pages
(`/Api/Participants/:id`) and is embedded in client-side profile links — so anyone could read
another player's private state: their in-game board and clocks, and, while they have a game
open, their **invite `Code`**. Because `JoinInvite` needs only that code, a stranger could
take the code leaked here and join the victim's game as their opponent (match hijack), or
simply surveil any player's live game.

**Proof (against the running build):** a victim socket created an invite; an attacker who knew
only the victim's public id fetched `/Api/State` with no token and received the live state with
the matching invite code:

```
victim ParticipantId : abc6090d-9d1c-4bb5-b6af-caf52a75fb91
true invite Code (WS): 3WZ1LC
/Api/State (no token): {"Phase":"Inviting","Code":"3WZ1LC", ...}
>>> LEAKED invite code without auth: YES — VULNERABLE
```

**Fix:** `/Api/State` now requires a valid `(PlayerId, Token)` pair and verifies it with
`IsSessionValid` before returning anything — the same proof of ownership the WebSocket already
demands. Missing token → 400; wrong token → 401; only the owner's own token → 200. The official
client never used this endpoint (state arrives over the WebSocket), so nothing legitimate
breaks. Covered by `Tests/HttpAuth.test.ts` and re-verified live:

```
attacker no-token   : HTTP 400
attacker wrong-token: HTTP 401
owner valid-token   : HTTP 200, sees own code YES
>>> IDOR fixed: YES
```

---

## 2. No rate limiting on login / register — Low (noted)

`POST /Api/Login` and `POST /Api/Register` accept unlimited attempts. Password verification uses
`Bun.password.verify` (bcrypt), whose cost makes online brute-forcing slow and CPU-expensive,
which is the main mitigation today. **Recommendation:** add a simple per-IP / per-username
attempt throttle (and consider it for the admin SSHSIG challenge too) to blunt credential
stuffing and to bound the CPU a flood of verifies can consume.

## 3. `/Api/Matches/:id/Moves` world-readable — Low (noted)

Move history for any game id is public, with no participant check, including games still in
progress. Game ids are random UUIDs and an active game's id is only handed to its own
participants, so a third party cannot readily enumerate live games; finished-game replay is an
intended public feature. **Recommendation:** if move privacy for in-progress games is ever
wanted, gate the active-game case behind participant/session auth.

## 4. Unbounded board-size ceiling → memory DoS — Info (noted)

`Limits.BoardSizeHardMax` is `Number.MAX_SAFE_INTEGER` (an intentional "no upper limit" for the
admin board-range control). Game creation validates board size against the *live* admin limits,
so with the default max (19) this is harmless. But an admin who sets a very large max lets any
player create an N×N board that allocates ~N² cells, which can exhaust memory. The trigger is
admin-gated (a trusted role), so this is informational. **Recommendation:** keep a sane
practical hard cap, or allocate the board lazily / sparsely, so a misconfiguration can't be
turned into a crash by an ordinary player.

## 5. Tokens in WebSocket URL query strings — Info (noted)

`/Ws?PlayerId=&Token=` and `/Ws?EngineId=&Token=` carry secrets in the URL, which can surface in
proxy/access logs. Low impact (tokens rotate on each human connect; this is the standard browser
WebSocket constraint, which can't send custom auth headers). **Recommendation:** prefer an
in-band auth message after upgrade, or a short-lived single-use ticket in the query, if the
deployment logs full request URLs.

---

## Controls reviewed and found sound (no action needed)

These were checked specifically and are in good shape — recorded so future reviews need not
re-derive them:

- **SQL injection:** all DB access goes through Drizzle's query builder; the one `sql` template
  (`GetParticipantByUsername`) interpolates the username as a **bound parameter**, not string
  concatenation. No raw SQL.
- **Password storage:** bcrypt via `Bun.password.hash` / `.verify`. Session and engine tokens are
  stored only as SHA-256 hashes; raw tokens are never persisted.
- **Move authorization:** WebSocket actions act on the socket's authenticated `ParticipantId`, not
  on any id in the message body, so a player cannot move, swap, or resign for someone else.
  Presenting a wrong session token mints a *fresh anonymous* identity rather than granting access
  to the claimed one — no impersonation path.
- **Admin auth:** SSHSIG challenge/response over trusted keys; `ssh-keygen` is invoked with an
  **argument array** (no shell), the signature and allowed-keys are passed as files and the message
  over stdin, so attacker-controlled content can't break out into a command. Sessions are 256-bit
  random tokens in an **HttpOnly, SameSite=Strict** cookie (CSRF- and script-resistant), with a
  sliding inactivity expiry. (Dev open-login is env-gated and documented to be `false` in prod.)
- **Stored XSS in the admin page:** all user-controlled fields (usernames, external-engine names
  and descriptions) are HTML-escaped via `Esc()` before insertion; ids/fingerprints used in
  attribute context are server-generated. Theme values are server-side **sanitized to a strict CSS
  colour grammar** (`ColorPattern`), so the colour editor can't smuggle markup.
- **Client XSS:** the React client renders names/descriptions as text (default escaping) and uses
  no `dangerouslySetInnerHTML` / `innerHTML`.
- **Secret disclosure:** profile, user-list, and engine endpoints never return password hashes,
  session-token hashes, or engine-token hashes.
