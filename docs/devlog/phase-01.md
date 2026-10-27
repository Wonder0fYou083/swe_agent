# Phase 01 — The agent loop · `v0.1.0`

**Dates:** 2026-10-18 · **PRs:** #8, #10, #11, #13

## Built

- `scratch/tool_call.py`: reply to every tool call in a response, each with a matching `tool_use_id`, and catch command timeouts.

## Issues and fixes

- Timeouts were first reported only as return code -1, which the model could mistake for a real exit code → added an explicit `timeout` field to the tool result.
- Wrong key: `id` was used instead of `tool_use_id`, which would have made the second API call fail.
- Binary output crashed the script: strict decoding failed on non-UTF-8 bytes → decode with `errors="replace"`.
- Commands ran wherever the script was launched, not the repo root the tool description promises → `cwd=REPO_ROOT`.

## Next

- The bash tool, then the loop itself: repeat model calls and tool runs until a stop condition is met.
