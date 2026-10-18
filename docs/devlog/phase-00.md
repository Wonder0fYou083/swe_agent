# Phase 0 — Foundations · `v0.0.0`

## notes

- **Goal:** build an industry-level SWE agent from scratch, learn agent engineering along the way (harness, context and loop engineering, RAG, MCP, ACP), and make it one of my main GitHub projects.
- **Decisions so far:**
    - Name: `swe_agent`
    - Python 3.12+, uv, ruff, pyright, pytest.
    - 12 phases, one tagged release each: `v0.0.0` → `v1.0.0`.
    - No agent frameworks in the core: write the loop, tools, context manager, retrieval, MCP/ACP code and eval runner myself.
    - Baseline: mini-swe-agent with the same model. Dev signal: a 50-task SWE-bench Verified subset. Headline number: a SWE-bench Pro public subset.
- **Mentor:**
    - Claude, in the "AI Agent" Project: design discussions before each phase, code reviews against "Done when", concept explanations.
    - Claude Code: plumbing only (config, CI, fixtures) plus a reference harness to study. The learning rule in `AGENTS.md` keeps it out of the core.

## Checklist

- [ ] Create the GitHub repo with LICENSE (MIT or Apache-2.0) and a README stub
- [ ] Commit `AGENTS.md`, `CLAUDE.md`, `ROADMAP.md` and `docs/`
- [ ] `pyproject.toml` with uv; ruff, pyright and pytest configured
- [ ] pre-commit with ruff and gitleaks
- [ ] GitHub Actions CI (lint, types, tests) and branch protection on `main`
- [ ] `swe-agent --version` entry point with one passing test
- [ ] `scratch/` script: one request with one tool definition to each of two providers; print the raw tool call
- [ ] Tag `v0.0.0` and write the GitHub Release note

## What I built

- uv basics, pre-commit hooks, CI checks

## What failed or surprised me

- 

## Concepts I can now explain

- How each provider's tool-use request and response look on the wire:
- How the providers bill tokens and rate-limit:

## Open questions for the next mentor session

-

## Predictions vs. answers: the tool-use wire format (Anthropic Messages API)

**1. Who executes the tool?**

- My prediction: my code executes the tool. ✅

- Answer: My code. The model only *outputs a request*: a `tool_use` block with a tool `name` and an `input` object. It never runs anything. My program decides whether and how to run it, then sends back the result. (Exception: Anthropic's own *server tools*, like web search, run on Anthropic's servers. Tools I define, like `bash`, always run in my code.)
- [ ] Verified in my run

**2. Text, a tool call, or both? Several at once?**

- My prediction: a tool call; yes.
- Answer: Both are possible. `content` is a **list of blocks**: often a `text` block ("I'll count the files…") followed by one or more `tool_use` blocks. Yes, the model can request several tools in one response (parallel tool use), and I must return a result for each one.
- [ ] Verified in my run

**3. How does my code know if the model wants a tool or is finished?**

- My prediction: by seeing whether it produced the final answer and achieved the task.
- Answer: The `stop_reason` field. `"tool_use"` means "run my tools and send back the results". `"end_turn"` means the model finished its turn. Others to handle: `"max_tokens"` (cut off mid-reply), `"refusal"`. Important: `end_turn` only means the model *stopped*, not that the task *succeeded*. Checking success is the harness's job (evals in Phase 4, sensors in Phase 7).
- [ ] Verified in my run

**4. How does the model know which call a result answers?**

- My prediction: I don't know.
- Answer: An ID. Each `tool_use` block has a unique `id` (like `toolu_01…`). My `tool_result` block carries the same value in its `tool_use_id` field. Results go in a `user`-role message placed right after the assistant message that made the calls.
- [ ] Verified in my run

**5. Does the API remember the conversation?**

- My prediction: No. (I wasn't sure what an API is.)
- Answer: No. The Messages API is stateless, like HTTP itself: each request is a separate, self-contained letter. My second request must contain the **whole** conversation: the original user message, the assistant's message with its `tool_use` block(s) exactly as returned, and my `tool_result`. So every turn resends everything, and requests grow longer and more expensive. That's the problem Phase 5 (context engineering) solves.
- [ ] Verified in my run

**6. Do tool definitions cost tokens? Where do I check?**

- My prediction: Yes; I don't know where.
- Answer: Yes, on **every** request. The tool name, description and schema are turned into text the model reads, so they count as input tokens. When tools are present, the API also adds a hidden tool-use system prompt: 286 extra tokens for Haiku 5.5 with the default tool choice. Check `usage.input_tokens` in the response. With vs. without `tools`, the difference ≈ my definition's tokens + 286.
- [ ] Verified in my run (difference I measured: ___ tokens)

## Next

Phase 1: the minimal agent loop.
