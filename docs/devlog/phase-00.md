# Phase 0 — Foundations · `v0.0.0`

**Planned:** 2026-10-19 → 2026-10-25 · **Actual:** not started

## Kickoff notes (2026-10-17)

- **Goal of the project:** build an industry-level SWE agent from scratch, learn agent engineering along the way (harness, context and loop engineering, RAG, MCP, ACP), and make it one of my main GitHub projects.
- **Decisions so far:**
    - Name: `swe_agent` (repo and Python package), CLI command `swe-agent`. Note: Princeton's SWE-agent already uses the name and owns `sweagent` on PyPI, so pick a different PyPI distribution name before Phase 11.
    - Python 3.12+, uv, ruff, pyright, pytest.
    - 12 phases, one tagged release each: `v0.0.0` → `v1.0.0`, about 33 weeks at 10–12 hours a week.
    - No agent frameworks in the core; I write the loop, tools, context manager, retrieval, MCP/ACP code and eval runner myself.
    - Baseline: mini-swe-agent with the same model. Dev signal: a 50-task SWE-bench Verified subset. Headline number: a SWE-bench Pro public subset.
- **Mentor setup:**
    - Claude chat, in the "AI Agent" Project: design discussions before each phase, code reviews against "Done when", concept explanations.
    - Claude Code: plumbing only (config, CI, fixtures) plus a reference harness to study. The learning rule in `AGENTS.md` keeps it out of the core.
- **Living roadmap:** https://claude.ai/code/artifact/4d23e51b-49d3-40e2-b86e-00cff928406f (private to me). `ROADMAP.md` in the repo is a snapshot.

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

- 

## What failed or surprised me

- 

## Concepts I can now explain

- How each provider's tool-use request and response look on the wire:
- How the providers bill tokens and rate-limit:

## Open questions for the next mentor session

- 

## Next

Phase 1: the minimal agent loop.
