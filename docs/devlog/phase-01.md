# Phase 01 — The agent loop · `v0.1.0`

**Dates:** 2026-11-09 · **PRs:** #8, #10, #11, #13, #15, #17, #18, #19, #20, #23, #28

## Built

- `src/swe_agent/tools/bash.py`: run bash commands on macos using python subprocess, timeout set. Warning: no sandboxing for now.
- `src/swe_agent/telemetry/logger.py`: record every run, including messages between user and agent, agent and LLM, and token usage. Logs live in `~/.swe-agent/runs/`.
- `src/swe_agent/core/agent.py`: the main agent ReAct loop. Stop until answer's ready/max_step reached/max_cost reached.
- `src/swe_agent/models/pricing.py`: Look up `model` in PRICES. Raise UnknownModelError if it is missing.
- `src/swe_agent/models/http.py`: build http request to Anthropic's API. Uniting format.
- `src/swe_agent/cli/main.py`: `swe-agent run "<task>"` with step, cost and time budgets. Installable and runnable from any folder; the key can live in `~/.swe-agent/env`.
- `src/swe_agent/models/openai.py`: a second adapter, for OpenAI's Chat Completions format. With `--provider openai` and `--base-url`, one adapter covers OpenAI, OpenRouter and local servers. Its tests use a recorded `gpt-6-luna` response.
- Live run with `gpt-6-luna`: the toy repo bug fixed in 4 steps for $0.0004. On step 4, 1,358 of the 1,526 prompt tokens came from OpenAI's automatic cache.
- Live run on the toy repo: bug fixed in 4 steps, 5.4 s, $0.00074; prompt caching made it 31% cheaper. Steps, trace and cache numbers: [docs/live-run](../live-run/README.md).

## Issues and fixes

- Timeouts were first reported only as return code -1, which the model could mistake for a real exit code → added an explicit `timeout` field to the tool result.
- Wrong key: `id` was used instead of `tool_use_id`, which would have made the second API call fail.
- Binary output crashed the script: strict decoding failed on non-UTF-8 bytes → decode with `errors="replace"`.
- Commands ran wherever the script was launched, not the repo root the tool description promises → `cwd=REPO_ROOT`.
- killpg is needed rather than kill, otherwise the child process might still be running after the parent process is killed.
- OpenAI's `prompt_tokens` includes cached tokens, unlike Anthropic's `input_tokens` → subtract `cached_tokens`, or they are billed twice.
- `gpt-5.6-luna` rejected tools on Chat Completions while reasoning was on (HTTP 400) → added `--reasoning-effort`; the live run used `none`.
- A malformed 200 response (an error body, a tool call with no id) crashed the run with a traceback → it now raises `ModelError`, and the loop stops cleanly.

## Next

- Sandbox: run the bash tool in a container with no network, resource limits and only the workspace writable.
- Eval harness: 10–20 fixed tasks with automated pass/fail and a pass rate. Then the project is frozen.
