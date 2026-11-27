"""The `swe-agent` command: `run` for one task, `chat` for many in one conversation, or bare
`swe-agent` for the start page and then chat."""

import os
import subprocess
import sys
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from swe_agent import __version__
from swe_agent.cli.splash import PosixTerminal, run_splash
from swe_agent.core.agent import Agent, RunResult
from swe_agent.models.anthropic import AnthropicModel
from swe_agent.models.base import Model
from swe_agent.models.openai import DEFAULT_BASE_URL, OpenAIModel
from swe_agent.models.pricing import UnknownModelError
from swe_agent.telemetry.trace import Budgets, StepRecord
from swe_agent.tools.bash import SANDBOX_IMAGE, run_bash, run_bash_in_sandbox

# Prompts live inside the package (src/swe_agent/prompts/), so an installed copy has them
# too. The file name is also the prompt version recorded in the trace, so the two cannot
# drift apart.
SYSTEM_PROMPT = "system_v2"  # the sandbox: Linux, GNU tools, no network, /workspace
HOST_SYSTEM_PROMPT = "system_v1"  # --no-sandbox: commands run on this Mac, with BSD tools
# `chat`: the same two, except that a reply ends the turn and the person answers next.
CHAT_SYSTEM_PROMPT = "chat_v1"
CHAT_HOST_SYSTEM_PROMPT = "chat_host_v1"

SANDBOX_BUILD = f"docker build -t {SANDBOX_IMAGE} -f docker/sandbox.Dockerfile docker/"

# The per-user folder: an optional `env` file with the API key, and `runs/` for trajectories.
HOME_ENV = "SWE_AGENT_HOME"
API_KEY = "ANTHROPIC_API_KEY"


class Provider(StrEnum):
    anthropic = "anthropic"
    openai = "openai"  # anything that speaks OpenAI's /chat/completions


# Every key is read, and so removed from the environment, on every run (see read_api_key).
API_KEYS = {Provider.anthropic: API_KEY, Provider.openai: "OPENAI_API_KEY"}
DEFAULT_MODELS = {Provider.anthropic: "claude-haiku-5-5"}

OBSERVATION_LINES = 5  # lines of each command's output shown per step

# Each model's context window, in tokens: only sizes checked against the provider's docs. A
# model not listed shows its token count with no percentage and never warns.
CONTEXT_WINDOWS = {
    "claude-haiku-5-5": 1_000_000,
    "claude-sonnet-5-5": 1_000_000,
    "claude-opus-5-5": 1_000_000,
    "claude-fable-5-2": 1_000_000,
}

CHAT_HINT = "Type a task and press Enter. exit, quit or Ctrl-D ends the session."
CHAT_EXIT = ("exit", "quit")

app = typer.Typer(help="A model-agnostic, terminal-first software-engineering agent.")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"swe-agent {__version__}")
        raise typer.Exit


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    """A model-agnostic, terminal-first software-engineering agent.

    With no command, it opens the start page and then a chat session.
    """
    if ctx.invoked_subcommand is None:
        start(ctx)


def start(ctx: typer.Context) -> None:
    """Bare `swe-agent`: the start page, then a chat session with `chat`'s defaults.

    Anywhere but an interactive terminal it prints the help and exits 2, as it always has.
    """
    if not is_interactive():
        typer.echo(ctx.get_help())
        raise typer.Exit(code=2)
    preflight()
    try:
        go = run_splash(PosixTerminal(), title=f"swe-agent {__version__}")
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    if not go:
        raise typer.Exit(code=0)
    chat()


def is_interactive() -> bool:
    """True when a person is at the keyboard: stdin and stdout are a real terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def preflight() -> None:
    """The checks `run` makes with its defaults, made before the start page, not after it.

    The key is looked up without being removed from the environment: `run` reads it later.
    """
    if not (os.environ.get(API_KEY) or read_env_file(agent_home() / "env").get(API_KEY)):
        _fail(_missing_key(API_KEY))
    if problem := sandbox_problem():
        _fail(f"{problem}. Build the image with `{SANDBOX_BUILD}`, or use `run --no-sandbox`.")


# Options `run` and `chat` share. Typer reads each command's signature, so the two commands
# name these types instead of repeating them. The budgets differ in their help text only.
CwdOption = Annotated[
    Path, typer.Option(help="Repository to work in.", exists=True, file_okay=False)
]
ProviderOption = Annotated[
    Provider, typer.Option(help="Which API to call. openai covers any compatible server.")
]
ModelOption = Annotated[
    str | None,
    typer.Option(help="Model name. Default: claude-haiku-5-5 for anthropic; required for openai."),
]
BaseUrlOption = Annotated[
    str | None,
    typer.Option(help="API base URL, e.g. http://localhost:11434/v1 for a local server."),
]
ReasoningEffortOption = Annotated[
    str | None,
    typer.Option(
        help="openai: sent as reasoning_effort. Some OpenAI models need none to use tools."
    ),
]
RunsDirOption = Annotated[
    Path | None,
    typer.Option(help="Where trajectories are written. Default: ~/.swe-agent/runs."),
]
NoCacheOption = Annotated[
    bool, typer.Option("--no-cache", help="Turn off automatic prompt caching (anthropic).")
]
StreamOption = Annotated[
    bool,
    typer.Option(
        "--stream", help="Stream responses and record the time to first token (anthropic)."
    ),
]
SandboxOption = Annotated[
    bool,
    typer.Option(
        help="Run each command in a fresh Docker container: no network, and only --cwd "
        "is writable. --no-sandbox runs commands directly on this machine."
    ),
]


@app.command()
def run(
    task: Annotated[str, typer.Argument(help='What to do, e.g. "make the tests pass".')],
    cwd: CwdOption = Path("."),
    provider: ProviderOption = Provider.anthropic,
    model: ModelOption = None,
    base_url: BaseUrlOption = None,
    reasoning_effort: ReasoningEffortOption = None,
    max_steps: Annotated[int, typer.Option(help="Stop after this many model calls.")] = 30,
    max_cost: Annotated[
        float, typer.Option(help="Stop once the run has cost this, in USD.")
    ] = 0.25,
    max_time: Annotated[float, typer.Option(help="Stop after this many seconds.")] = 600.0,
    runs_dir: RunsDirOption = None,
    no_cache: NoCacheOption = False,
    stream: StreamOption = False,
    sandbox: SandboxOption = True,
) -> None:
    """Run the agent on the task until it finishes or a budget runs out."""
    agent = build_agent(
        cwd=cwd,
        provider=provider,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        budgets=Budgets(max_steps=max_steps, max_cost_usd=max_cost, max_wall_time_s=max_time),
        runs_dir=runs_dir,
        no_cache=no_cache,
        stream=stream,
        sandbox=sandbox,
        chat=False,
    )
    result = agent.run(task)
    typer.echo(format_summary(result))
    raise typer.Exit(code=0 if result.stop == "submitted" else 1)


@app.command()
def chat(
    cwd: CwdOption = Path("."),
    provider: ProviderOption = Provider.anthropic,
    model: ModelOption = None,
    base_url: BaseUrlOption = None,
    reasoning_effort: ReasoningEffortOption = None,
    max_steps: Annotated[int, typer.Option(help="Stop a turn after this many model calls.")] = 30,
    max_cost: Annotated[
        float, typer.Option(help="Stop a turn once it has cost this, in USD.")
    ] = 0.25,
    max_time: Annotated[float, typer.Option(help="Stop a turn after this many seconds.")] = 600.0,
    runs_dir: RunsDirOption = None,
    no_cache: NoCacheOption = False,
    stream: StreamOption = False,
    sandbox: SandboxOption = True,
) -> None:
    """Keep one conversation open: each task you type sees every task before it.

    The budgets count each turn on its own. exit, quit or Ctrl-D ends the session.
    """
    agent = build_agent(
        cwd=cwd,
        provider=provider,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        budgets=Budgets(max_steps=max_steps, max_cost_usd=max_cost, max_wall_time_s=max_time),
        runs_dir=runs_dir,
        no_cache=no_cache,
        stream=stream,
        sandbox=sandbox,
        chat=True,
    )
    typer.echo(CHAT_HINT)
    code = 0
    with agent.session() as session:
        try:
            while (text := read_task()) is not None:
                result = session.send(text)
                typer.echo(format_turn(result, turn=session.turns, session_cost=session.cost_usd))
        except KeyboardInterrupt:  # Ctrl-C ends the whole session; the with block closes it
            typer.echo()
            code = 130
    if session.turns and session.trajectory is not None:
        typer.echo(f"trajectory: {session.trajectory}")
    raise typer.Exit(code=code)


def read_task() -> str | None:
    """The next task typed at the `> ` prompt, or None for exit, quit or Ctrl-D.

    Plain input(), not typer.prompt: Click turns Ctrl-D and Ctrl-C into "Aborted!".
    """
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            typer.echo()
            return None
        if text in CHAT_EXIT:
            return None
        if text:
            return text


def build_agent(
    *,
    cwd: Path,
    provider: Provider,
    model: str | None,
    base_url: str | None,
    reasoning_effort: str | None,
    budgets: Budgets,
    runs_dir: Path | None,
    no_cache: bool,
    stream: bool,
    sandbox: bool,
    chat: bool,
) -> Agent:
    """Everything `run` and `chat` check and build before the first task.

    Exits 1 with a message on a missing key, an unknown model, an unreadable prompt or a
    sandbox that cannot start. Each step is printed as it ends.
    """
    # Read the keys here, never at import time: CI has no key, and --help must still work.
    keys = {name: read_api_key(name) for name in API_KEYS.values()}
    key_name = API_KEYS[provider]
    api_key = keys[key_name]
    # A local server given by --base-url usually needs no key.
    if not api_key and not (provider is Provider.openai and base_url):
        _fail(_missing_key(key_name))

    model = model or DEFAULT_MODELS.get(provider)
    if model is None:
        _fail(f"pass --model: there is no default model for --provider {provider}")

    prompt_version = _prompt_version(sandbox=sandbox, chat=chat)
    try:
        system_prompt = load_system_prompt(prompt_version)
    except OSError as e:
        _fail(f"cannot read the system prompt: {e}")

    try:
        llm = build_model(
            provider,
            model,
            api_key,
            base_url=base_url,
            cache=not no_cache,
            reasoning_effort=reasoning_effort,
            stream=stream,
        )
    except UnknownModelError:
        _fail(f"no price for model {model!r}; add it to PRICES in models/pricing.py")

    if sandbox and (problem := sandbox_problem()):
        _fail(f"{problem}. Build the image with `{SANDBOX_BUILD}`, or pass --no-sandbox.")

    context_window = CONTEXT_WINDOWS.get(model)
    cost_so_far = 0.0

    def print_step(step: StepRecord) -> None:
        nonlocal cost_so_far
        cost_so_far += step.cost_usd
        typer.echo(format_step(step, cost_so_far, context_window=context_window))

    return Agent(
        llm,
        cwd=cwd.resolve(),
        system_prompt=system_prompt,
        prompt_version=prompt_version,
        budgets=budgets,
        # The default is under the user's home, so trajectories never land inside the
        # repository the agent is editing, whichever folder the command starts in.
        trajectory_dir=(runs_dir or agent_home() / "runs").resolve(),
        on_step=print_step,
        run_command=run_bash_in_sandbox if sandbox else run_bash,
        context_window=context_window,
    )


def _prompt_version(*, sandbox: bool, chat: bool) -> str:
    if chat:
        return CHAT_SYSTEM_PROMPT if sandbox else CHAT_HOST_SYSTEM_PROMPT
    return SYSTEM_PROMPT if sandbox else HOST_SYSTEM_PROMPT


def build_model(
    provider: Provider,
    model: str,
    api_key: str,
    *,
    base_url: str | None,
    cache: bool,
    reasoning_effort: str | None = None,
    stream: bool = False,
) -> Model:
    """The adapter for `provider`. Raises UnknownModelError for a model with no price."""
    if provider is Provider.openai:
        return OpenAIModel(
            model,
            api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            reasoning_effort=reasoning_effort,
        )
    if base_url is None:
        return AnthropicModel(model, api_key, cache=cache, stream=stream)
    return AnthropicModel(model, api_key, base_url=base_url, cache=cache, stream=stream)


def sandbox_problem() -> str | None:
    """Why the sandbox cannot start on this machine, or None when it can.

    One `docker image inspect` answers both questions: it fails when the daemon is down and
    when the image was never built. Checked once before the first model call, because the
    model would otherwise see exit code 125 on every command and could fix neither.
    """
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return "the sandbox needs Docker, and the docker command is not installed"
    except subprocess.TimeoutExpired:
        return "the sandbox needs Docker, and the Docker daemon did not answer within 15 s"
    if probe.returncode != 0:
        lines = probe.stderr.strip().splitlines()
        detail = lines[0] if lines else f"exit code {probe.returncode}"
        return f"the sandbox needs Docker and the {SANDBOX_IMAGE} image ({detail})"
    return None


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


def read_api_key(name: str = API_KEY) -> str:
    """The key `name` from the environment, else from <home>/env. "" if neither has it.

    pop, not get: every bash command the agent runs inherits this process's environment,
    so an `env` call would otherwise copy the key into the history and the trajectory.
    """
    key = os.environ.pop(name, "")
    return key or read_env_file(agent_home() / "env").get(name, "")


def format_step(step: StepRecord, cost_so_far: float, *, context_window: int | None = None) -> str:
    """One step for the terminal: header, the model's text, each command and its output.

    The header also says how full the context is, and a warning follows it at 90% or more.
    """
    lines = [f"[step {step.step}] ${cost_so_far:.4f} so far{_context(step, context_window)}"]
    if (warning := context_warning(step.context_tokens, context_window)) is not None:
        lines.append(warning)
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


def format_turn(result: RunResult, *, turn: int, session_cost: float) -> str:
    """The line after each chat turn: how it stopped, its steps, its cost and the session's."""
    steps = f"{result.steps} step" + ("" if result.steps == 1 else "s")
    lines = [
        f"[turn {turn}] {result.stop} · {steps} · ${result.cost_usd:.4f} this turn"
        f" · ${session_cost:.4f} session"
    ]
    if result.error:
        lines.append(f"error: {result.error}")
    return "\n".join(lines) + "\n"


def context_warning(tokens: int | None, window: int | None) -> str | None:
    """The warning for a context window 90% full or more; None below that or if unknown."""
    if tokens is None or window is None or tokens * 10 < window * 9:
        return None
    return (
        f"warning: context window {_percent(tokens, window)}% full "
        f"({format_tokens(tokens)} of {format_tokens(window)} tokens). "
        "Start a new session before it fills up."
    )


def format_tokens(n: int) -> str:
    """A token count, short: 850, 45.2K, 912.0K, 1M, 1.2M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".removesuffix(".0") + "M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _context(step: StepRecord, window: int | None) -> str:
    """The header's context part: ` · context 45.2K/1M (4%)`, or no window: ` · context 45.2K`."""
    if step.context_tokens is None:
        return ""
    if window is None:
        return f" · context {format_tokens(step.context_tokens)}"
    used = f"{format_tokens(step.context_tokens)}/{format_tokens(window)}"
    return f" · context {used} ({_percent(step.context_tokens, window)}%)"


def _percent(tokens: int, window: int) -> int:
    """Rounded down, so the header never reads 90% while the warning stays quiet."""
    return tokens * 100 // window


def _output_lines(observation: str) -> list[str]:
    """The command's output from an observation, without the status line and <output> tags."""
    body = observation.split("<output>\n", 1)[-1].rsplit("</output>", 1)[0]
    return body.rstrip("\n").splitlines()


def _missing_key(name: str) -> str:
    return f"{name} is not set. Export it, or put {name}=... in {agent_home() / 'env'}."


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)
