# Hex — Project Context

**All project rules, conventions, permissions, workflow, architecture, commands and
env vars live in one place — the single source of truth:**

@project-rules.md

Read that file before assuming anything. Do not duplicate its content back into this
file; if a rule changes, change it in `project-rules.md`.

A SessionStart hook (`.claude/Context.sh`) also prints live git/submodule state and the
active planning doc at the start of every session — read its output before assuming repo state.
