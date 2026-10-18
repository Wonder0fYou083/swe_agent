# swe_agent

A model-agnostic, terminal-first software-engineering agent, built from scratch to learn agent engineering.

> **Status:** Phase 0 — Foundations (`v0.0.0`). The repo skeleton exists; the agent does not yet. See [ROADMAP.md](ROADMAP.md) for the plan.
>
> Not affiliated with Princeton's [SWE-agent](https://github.com/SWE-agent/SWE-agent).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a clone of this repo:

```bash
uv sync
uv run swe-agent --version
```

## Development

```bash
uv run pre-commit install    # once: run ruff and gitleaks on every commit
uv run pytest                # tests
uv run ruff check . && uv run ruff format --check .
uv run pyright               # types
```

CI runs the same lint, type and test commands on every pull request.

## Project docs

- [ROADMAP.md](ROADMAP.md): phases, deliverables and "Done when" criteria
- [AGENTS.md](AGENTS.md): instructions for coding agents working on this repo
- [docs/devlog/](docs/devlog/): one entry per phase

## License

[MIT](LICENSE)
