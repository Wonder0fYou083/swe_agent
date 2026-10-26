# swe_agent

A SWE agent with KV Cache built from scratch .

> **Status:** Phase 0 — developed the repo's skeleton. Real development of the agent has not started yet :(
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

- [AGENTS.md](AGENTS.md): instructions for coding agents working on this repo
- [docs/devlog/](docs/devlog/): what was built in each phase

## License

[MIT](LICENSE)
