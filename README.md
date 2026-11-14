# swe_agent

A SWE agent built from scratch, with no agent framework.

> Not affiliated with Princeton's [SWE-agent](https://github.com/SWE-agent/SWE-agent).

## Current Status (updated on 11/13/2026)

**Update (11/13/2026):** I added an interactive GUI. Check it out by running

```bash
swe-agent
```

from the project folder you wanna work in.

**Update (11/13/2026):** I ran a harness evaluation on my agent by letting claude create 10 tests and record the results.
[Evaluation results](evals/results); [Evaluation tests](evals/tests), and live tests can be run with

```bash
python3 evals/run.py
```

**Update (11/13/2026):** I added a docker container for each bash command run: every bash command now runs in a fresh Docker container with no network, CPU, memory, process and time limits, and only the repository writable. Tests show that escape attempts fail cleanly: writing outside the workspace, reaching the network, a fork bomb, using too much memory, reading host secrets.

**Update (11/12/2026):** I added an OpenAi adapter so that my agent
now supports other models (gpt-6-luna). I also added streaming (`sse.py`) which records TTFT of each step which can be
tracked in the session's log.

**update (11/09/2026):** A MVP swe agent is finished. I first built a bash tool from scratch,
which uses python subprocess to run bash command. It starts a new session every time (`start_new_session=True`)
and there's no sandbox yet. I built a logger to record each session from start to end, which also
includes every step, it's type (tool_call, text, etc), and it's token cost. I built raw http request from scratch
utilizing python httpx (I know it's kinda stupid since I can just use Anthropic(), but it's just for the love of the game).
I unite the message (from and to the Anthropic API) format to my customized types for the agent to read.
I asked claude to build the cli and then I conducted a live run: the agent is asked to fix a small bug in a toy repo.
There're two runs: cache and no cache. Each run's log is in docs/live-run. 

## Quickstart

Requires Python 3.12+. Install the `swe-agent` command with [pipx](https://pipx.pypa.io/) (or plain `pip`):

```bash
pipx install git+https://github.com/Wonder0fYou083/swe_agent.git
```

Commands run in a Docker container, so you also need [Docker](https://docs.docker.com/get-docker/) running, and the sandbox image built once. The build needs the network; the containers it makes have none:

```bash
curl -fsSL https://raw.githubusercontent.com/Wonder0fYou083/swe_agent/main/docker/sandbox.Dockerfile | docker build -t swe-agent-sandbox -
```

Put your key in `~/.swe-agent/env` as `ANTHROPIC_API_KEY=...`, then run it from any folder:

```bash
swe-agent run "make the tests pass"    # works on the current folder, mounted at /workspace
```

Only that folder (or the one given with `--cwd`) is visible and writable inside the container. `--no-sandbox` runs the commands directly on your machine instead, without any of these limits.

To work on the code, clone the repo and use [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run swe-agent --version
uv tool install --editable .    # optional: puts your clone's swe-agent on PATH
docker build -t swe-agent-sandbox -f docker/sandbox.Dockerfile docker/    # the sandbox image
```

To use an OpenAI model instead, put `OPENAI_API_KEY=...` in the same file. Any other server that speaks the same Chat Completions format works with `--base-url`:

```bash
swe-agent run "make the tests pass" --provider openai --model gpt-6-luna --reasoning-effort none
```

Each run's log is written to `~/.swe-agent/runs/<session_id>.jsonl`. Add `--stream` (Anthropic only) to stream responses, which records each step's TTFT as `model_first_token_ns`.

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
- [docs/live-run/](docs/live-run/README.md): one real run, start to finish, with its traces

## License

[MIT](LICENSE)
