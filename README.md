# swe_agent

A SWE agent with KV Cache built from scratch .

> Not affiliated with Princeton's [SWE-agent](https://github.com/SWE-agent/SWE-agent).

## Current Status

A MVP swe agent is finished. I first built a bash tool from scratch,
which uses python subprocess to run bash command. It starts a new session every time (`start_new_session=True`)
and there's no sandbox yet. I built a logger to record each session from start to end, which also
includes every step, it's type (tool_call, text, etc), and it's token cost. I built raw http request from scratch
utilizing python httpx (I know it's kinda stupid since I can just use Anthropic(), but it's just for the love of the game).
I unite the message (from and to the Anthropic API) format to my customized types for the agent to read.
I asked claude to build the cli and then I conducted a live run: the agent is asked to fix a small bug in a toy repo.
There're two runs: cache and no cache. Each run's log is in docs/live-run.

## Next move

Add more tools, harness engineering, agent's KV cache, etc. Happy to get prs.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a clone of this repo:

```bash
uv sync
uv run swe-agent --version
```

To run it from any folder, install it and put your key in `~/.swe-agent/env` as `ANTHROPIC_API_KEY=...`:

```bash
uv tool install --editable .
swe-agent run "make the tests pass"    # works on the current folder; no sandbox yet
```

Each run's log is written to `~/.swe-agent/runs/<session_id>.jsonl`.

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
