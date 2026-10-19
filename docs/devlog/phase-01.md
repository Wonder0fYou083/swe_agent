# Phase 00 — Skeleton and a simple tool call · `v0.1.0`

**Dates:** 2026-10-18 · **PRs:** #8

## Built

- collect all tool calls returned by the model, reply to each one with matching results and tool_use_id.
- catch time out exception.

## Issues and fixes

- first used -1 as return code to let the model recognize time out result, which will be mistaken with other returns
- Wrong key: "id" was used instead of "tool_use_id", which would have made the second API call fail.
- Binary output crashed the script: strict decoding failed on non-UTF-8 bytes; fixed with errors="replace".
- Commands ran in the wrong directory: they ran wherever the script was launched, not the repo root the tool description promises. Fixed with cwd=REPO_ROOT.

## Next

- Phase 1: the agent loop. Repeat model calls and tool runs until a stop condition is met.
