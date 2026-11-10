"""The `swe-agent` command. Subcommands such as `run` get added here as phases land."""

import os
from importlib.resources import files
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from swe_agent import __version__
from swe_agent.core.agent import Agent, RunResult
from swe_agent.models.anthropic import AnthropicModel
from swe_agent.models.pricing import UnknownModelError
from swe_agent.telemetry.trace import Budgets, StepRecord

# Prompts live inside the package (src/swe_agent/prompts/), so an installed copy has them
# too. The file name is also the prompt version recorded in the trace, so the two cannot
# drift apart.
SYSTEM_PROMPT = "system_v1"

# The per-user folder: an optional `env` file with the API key, and `runs/` for trajectories.
HOME_ENV = "SWE_AGENT_HOME"
API_KEY = "ANTHROPIC_API_KEY"

OBSERVATION_LINES = 5  # lines of each command's output shown per step

app = typer.Typer(
    help="A model-agnostic, terminal-first software-engineering agent.",
    no_args_is_help=True,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"swe-agent {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """A model-agnostic, terminal-first software-engineering agent."""


@app.command()
def run(
    task: Annotated[str, typer.Argument(help='What to do, e.g. "make the tests pass".')],
    cwd: Annotated[
        Path,
        typer.Option(help="Repository to work in.", exists=True, file_okay=False),
    ] = Path("."),
    model: Annotated[str, typer.Option(help="Model name.")] = "claude-haiku-5-5",
    max_steps: Annotated[int, typer.Option(help="Stop after this many model calls.")] = 30,
    max_cost: Annotated[
        float, typer.Option(help="Stop once the run has cost this, in USD.")
    ] = 0.25,
    max_time: Annotated[float, typer.Option(help="Stop after this many seconds.")] = 600.0,
    runs_dir: Annotated[
        Path | None,
        typer.Option(help="Where trajectories are written. Default: ~/.swe-agent/runs."),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Turn off automatic prompt caching.")
    ] = False,
) -> None:
    """Run the agent on the task until it finishes or a budget runs out."""
    # Read the key here, never at import time: CI has no key, and --help must still work.
    api_key = read_api_key()
    if not api_key:
        _fail(f"{API_KEY} is not set. Export it, or put {API_KEY}=... in {agent_home() / 'env'}.")

    try:
        system_prompt = load_system_prompt()
    except OSError as e:
        _fail(f"cannot read the system prompt: {e}")

    try:
        llm = AnthropicModel(model, api_key, cache=not no_cache)
    except UnknownModelError:
        _fail(f"no price for model {model!r}; add it to PRICES in models/pricing.py")

    cost_so_far = 0.0

    def print_step(step: StepRecord) -> None:
        nonlocal cost_so_far
        cost_so_far += step.cost_usd
        typer.echo(format_step(step, cost_so_far))

    agent = Agent(
        llm,
        cwd=cwd.resolve(),
        system_prompt=system_prompt,
        prompt_version=SYSTEM_PROMPT,
        budgets=Budgets(max_steps=max_steps, max_cost_usd=max_cost, max_wall_time_s=max_time),
        # The default is under the user's home, so trajectories never land inside the
        # repository the agent is editing, whichever folder the command starts in.
        trajectory_dir=(runs_dir or agent_home() / "runs").resolve(),
        on_step=print_step,
    )
    result = agent.run(task)
    typer.echo(format_summary(result))
    raise typer.Exit(code=0 if result.stop == "submitted" else 1)


def load_system_prompt(name: str = SYSTEM_PROMPT) -> str:
    """The text of the packaged prompts/<name>.md."""
    return files("swe_agent").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


def agent_home() -> Path:
    """The per-user folder: $SWE_AGENT_HOME if set, else ~/.swe-agent."""
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".swe-agent")


def read_env_file(path: Path) -> dict[str, str]:
    """The KEY=VALUE lines of `path`, or {} if it cannot be read.

    Blank lines and # comments are skipped. A leading `export` and quotes around the
    value are dropped.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def read_api_key() -> str:
    """The API key from the environment, else from <home>/env. "" if neither has it.

    pop, not get: every bash command the agent runs inherits this process's environment,
    so an `env` call would otherwise copy the key into the history and the trajectory.
    """
    key = os.environ.pop(API_KEY, "")
    return key or read_env_file(agent_home() / "env").get(API_KEY, "")


def format_step(step: StepRecord, cost_so_far: float) -> str:
    """One step for the terminal: header, the model's text, each command and its output."""
    lines = [f"[step {step.step}] ${cost_so_far:.4f} so far"]
    if step.assistant.text.strip():
        lines.append(step.assistant.text.strip())
    if not step.appended:
        lines.append(f"(response dropped from history: stop_reason={step.stop_reason})")
    for tool in step.tools:
        lines.append(f"$ {tool.command}" if tool.command else f"$ ({tool.tool_name})")
        if tool.timed_out:
            lines.append("  timed out")
        elif tool.exit_code is not None:
            lines.append(f"  exit {tool.exit_code}")
        output = _output_lines(tool.observation)
        lines += [f"  {line}" for line in output[:OBSERVATION_LINES]]
        if len(output) > OBSERVATION_LINES:
            lines.append(f"  ... {len(output) - OBSERVATION_LINES} more lines")
    return "\n".join(lines) + "\n"


def format_summary(result: RunResult) -> str:
    """The last lines of a run: how it stopped, how long it took, what it cost, where it went."""
    lines = [f"stopped: {result.stop} after {result.steps} steps, ${result.cost_usd:.4f}"]
    if result.error:
        lines.append(f"error: {result.error}")
    if result.trajectory is not None:
        lines.append(f"trajectory: {result.trajectory}")
    return "\n".join(lines)


def _output_lines(observation: str) -> list[str]:
    """The command's output from an observation, without the status line and <output> tags."""
    body = observation.split("<output>\n", 1)[-1].rsplit("</output>", 1)[0]
    return body.rstrip("\n").splitlines()


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)
