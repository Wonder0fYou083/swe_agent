# Phase 00 — Skeleton and a simple tool call · `v0.0.0`

**Dates:** 2026-10-17 · **PRs:** #1, #3

## Built

- Repo skeleton: uv project with a `src/` layout, a `swe-agent --version` CLI (typer) and its test.
- Checks: ruff and gitleaks as pre-commit hooks; GitHub Actions CI running lint (ruff), types (pyright) and tests (pytest); `main` protected.
- `scratch/tool_call.py`: one tool-use round trip with the Anthropic Messages API over raw HTTP (httpx), using a `bash` tool. The command's output is entered by hand.
- `.env.example` listing `ANTHROPIC_API_KEY`.

## Issues and fixes

- pyright: `tool_call` possibly unbound → set to `None` before the loop and exit if no `tool_use` block is found.

## Next

- Phase 1: the agent loop. Repeat model calls and tool runs until a stop condition is met.
