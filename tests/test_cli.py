import os
import re
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner, Result

from swe_agent.cli.main import (
    CHAT_HINT,
    CHAT_HOST_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    CONTEXT_WINDOWS,
    HOST_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    Provider,
    agent_home,
    app,
    build_model,
    context_warning,
    format_step,
    format_tokens,
    format_turn,
    load_system_prompt,
    read_api_key,
    read_env_file,
    sandbox_problem,
)
from swe_agent.core.agent import RunResult
from swe_agent.models.anthropic import AnthropicModel
from swe_agent.models.fake import ScriptedModel, text_reply
from swe_agent.models.openai import OpenAIModel
from swe_agent.models.pricing import PRICES, Price, UnknownModelError
from swe_agent.models.types import Endpoint, Message, ModelResponse, TextBlock, Usage
from swe_agent.telemetry.logger import read_trajectory
from swe_agent.telemetry.trace import SessionEnd, StepRecord, ToolRecord
from swe_agent.tools.bash import run_bash, run_bash_in_sandbox

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_version_flag_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"swe-agent {version('swe-agent')}"


@pytest.mark.parametrize("key", [None, ""])
def test_run_without_an_api_key_exits_1_before_doing_anything(
    key: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if key is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    runs = tmp_path / "runs"

    result = CliRunner().invoke(
        app, ["run", "make the tests pass", "--cwd", str(tmp_path), "--runs-dir", str(runs)]
    )

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert not runs.exists()


def test_run_with_an_unpriced_model_exits_1_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = CliRunner().invoke(
        app, ["run", "x", "--cwd", str(tmp_path), "--model", "no-such-model"]
    )

    assert result.exit_code == 1
    assert "no price for model 'no-such-model'" in result.output


def test_run_removes_the_api_key_from_the_environment_bash_inherits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # An unpriced model stops the run right after the key is read, before any network use.
    CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path), "--model", "no-such-model"])

    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_run_removes_every_api_key_whichever_provider_is_used(
    provider: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    CliRunner().invoke(
        app,
        ["run", "x", "--cwd", str(tmp_path), "--provider", provider, "--model", "no-such-model"],
    )

    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_openai_without_a_key_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = CliRunner().invoke(
        app, ["run", "x", "--cwd", str(tmp_path), "--provider", "openai", "--model", "m"]
    )

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output


def test_openai_needs_no_key_for_a_local_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Reaching the price check means the missing key was accepted.
    result = CliRunner().invoke(
        app,
        [
            "run",
            "x",
            "--cwd",
            str(tmp_path),
            "--provider",
            "openai",
            "--base-url",
            "http://localhost:11434/v1",
            "--model",
            "no-such-model",
        ],
    )

    assert result.exit_code == 1
    assert "no price for model 'no-such-model'" in result.output


def test_openai_without_a_model_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path), "--provider", "openai"])

    assert result.exit_code == 1
    assert "pass --model" in result.output


def test_build_model_picks_the_adapter_and_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PRICES,
        "test-model",
        Price(input_per_mtok=0, output_per_mtok=0, cache_read_per_mtok=0, cache_write_per_mtok=0),
    )

    local = build_model(
        Provider.openai, "test-model", "", base_url="http://localhost:11434/v1", cache=True
    )
    default = build_model(Provider.openai, "test-model", "k", base_url=None, cache=True)
    anthropic = build_model(Provider.anthropic, "test-model", "k", base_url=None, cache=False)

    assert isinstance(local, OpenAIModel)
    assert local.endpoint == Endpoint(
        provider="openai", base_url="http://localhost:11434/v1", cache_requested=False
    )
    assert default.endpoint.base_url == "https://api.openai.com/v1"
    assert isinstance(anthropic, AnthropicModel)
    assert anthropic.endpoint == Endpoint(
        provider="anthropic", base_url="https://api.anthropic.com", cache_requested=False
    )


def test_agent_home_is_the_env_var_or_a_folder_in_the_user_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SWE_AGENT_HOME", str(tmp_path / "elsewhere"))
    assert agent_home() == tmp_path / "elsewhere"

    monkeypatch.delenv("SWE_AGENT_HOME")
    assert agent_home() == Path.home() / ".swe-agent"


def test_read_env_file_parses_key_value_lines(tmp_path: Path) -> None:
    path = tmp_path / "env"
    path.write_text('# a comment\n\nexport A=1\nB = "two"\nnot a pair\n', encoding="utf-8")

    assert read_env_file(path) == {"A": "1", "B": "two"}
    assert read_env_file(tmp_path / "missing") == {}


def test_api_key_comes_from_the_home_env_file_when_the_environment_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent_home().mkdir(parents=True)
    (agent_home() / "env").write_text("ANTHROPIC_API_KEY=file-key\n", encoding="utf-8")

    assert read_api_key() == "file-key"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert read_api_key() == "env-key"  # the environment wins


def test_run_from_another_folder_finds_the_key_file_and_the_packaged_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent_home().mkdir(parents=True)
    (agent_home() / "env").write_text("ANTHROPIC_API_KEY=file-key\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # An unpriced model stops the run after the key and the prompt are read, before any
    # network use. Reaching that message means both were found.
    result = CliRunner().invoke(app, ["run", "x", "--model", "no-such-model"])

    assert result.exit_code == 1
    assert "no price for model 'no-such-model'" in result.output
    assert list(elsewhere.iterdir()) == []  # nothing is written into the working folder


def test_run_help_lists_every_option() -> None:
    result = CliRunner().invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
    text = ANSI.sub("", result.output)

    assert result.exit_code == 0
    for option in (
        "--cwd",
        "--provider",
        "--model",
        "--base-url",
        "--reasoning-effort",
        "--max-steps",
        "--max-cost",
        "--max-time",
        "--runs-dir",
        "--no-cache",
        "--stream",
        "--sandbox",
        "--no-sandbox",
    ):
        assert option in text


@pytest.mark.parametrize(("flags", "stream"), [([], False), (["--stream"], True)])
def test_stream_flag_reaches_the_anthropic_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str], stream: bool
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def record_and_stop(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise UnknownModelError(args[0])  # ends the run before any network use

    monkeypatch.setattr("swe_agent.cli.main.AnthropicModel", record_and_stop)
    result = CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path), *flags])

    assert result.exit_code == 1
    assert seen["stream"] is stream


@pytest.mark.parametrize("name", [SYSTEM_PROMPT, HOST_SYSTEM_PROMPT])
def test_system_prompt_loads_and_states_how_a_run_ends(name: str) -> None:
    # The loop stops on a reply without a tool call, so the prompt must say so.
    assert "A reply without a tool call ends the run." in load_system_prompt(name)


def test_the_sandbox_prompt_describes_the_container() -> None:
    prompt = load_system_prompt(SYSTEM_PROMPT)

    assert "/workspace" in prompt
    assert "no network" in prompt
    assert "macOS" not in prompt


class _FakeSession:
    """Stands in for Session: records each text sent and submits at once."""

    def __init__(self) -> None:
        self.turns = 0
        self.cost_usd = 0.0
        self.trajectory: Path | None = None

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def send(self, text: str) -> RunResult:
        _FakeAgent.tasks.append(text)
        self.turns += 1
        return RunResult(
            session_id="s", stop="submitted", steps=1, cost_usd=0.0, final_text="", trajectory=None
        )


class _FakeAgent:
    """Stands in for Agent: records how the CLI built it and submits at once."""

    built: ClassVar[list[dict[str, object]]] = []
    tasks: ClassVar[list[str]] = []

    def __init__(self, model: object, **kwargs: object) -> None:
        _FakeAgent.built.append(kwargs)

    def run(self, task: str) -> RunResult:
        _FakeAgent.tasks.append(task)
        return RunResult(
            session_id="s", stop="submitted", steps=1, cost_usd=0.0, final_text="", trajectory=None
        )

    def session(self) -> _FakeSession:
        return _FakeSession()


@pytest.fixture
def fake_agent(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAgent]:
    """Run the CLI with no model and no agent loop: nothing reaches the network."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("swe_agent.cli.main.build_model", lambda *a, **k: ScriptedModel([]))
    monkeypatch.setattr("swe_agent.cli.main.Agent", _FakeAgent)
    _FakeAgent.built = []
    _FakeAgent.tasks = []
    return _FakeAgent


@pytest.mark.parametrize(
    ("flags", "run_command", "prompt"),
    [
        ([], run_bash_in_sandbox, SYSTEM_PROMPT),
        (["--sandbox"], run_bash_in_sandbox, SYSTEM_PROMPT),
        (["--no-sandbox"], run_bash, HOST_SYSTEM_PROMPT),
    ],
)
def test_run_uses_the_sandbox_and_its_prompt_unless_told_not_to(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_agent: type[_FakeAgent],
    flags: list[str],
    run_command: object,
    prompt: str,
) -> None:
    monkeypatch.setattr("swe_agent.cli.main.sandbox_problem", lambda: None)

    result = CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path), *flags])

    assert result.exit_code == 0, result.output
    [kwargs] = fake_agent.built
    assert kwargs["run_command"] is run_command
    assert kwargs["prompt_version"] == prompt
    assert kwargs["system_prompt"] == load_system_prompt(prompt)


def test_run_stops_before_the_model_when_the_sandbox_cannot_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_agent: type[_FakeAgent]
) -> None:
    monkeypatch.setattr(
        "swe_agent.cli.main.sandbox_problem", lambda: "the sandbox needs Docker (no daemon)"
    )

    result = CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path)])

    assert result.exit_code == 1
    assert "the sandbox needs Docker (no daemon)" in result.output
    assert "docker build -t swe-agent-sandbox" in result.output
    assert "--no-sandbox" in result.output
    assert fake_agent.built == []


def test_no_sandbox_never_checks_for_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_agent: type[_FakeAgent]
) -> None:
    def must_not_run() -> str | None:
        raise AssertionError("--no-sandbox must not look for Docker")

    monkeypatch.setattr("swe_agent.cli.main.sandbox_problem", must_not_run)

    result = CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path), "--no-sandbox"])

    assert result.exit_code == 0, result.output


def test_sandbox_problem_explains_a_missing_docker_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_docker(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr("swe_agent.cli.main.subprocess.run", no_docker)

    assert sandbox_problem() == "the sandbox needs Docker, and the docker command is not installed"


def test_format_step_shows_text_commands_and_the_first_lines_of_output() -> None:
    output = "\n".join(f"line {i}" for i in range(1, 9))
    step = StepRecord(
        session_id="s",
        step=2,
        parent_step=1,
        request_messages=3,
        assistant=Message(role="assistant", content=[TextBlock(text="Running the tests.")]),
        appended=True,
        stop_reason="tool_use",
        usage=Usage(),
        cost_usd=0.001,
        model_started_ns=0,
        model_first_token_ns=None,
        model_ended_ns=0,
        attempts=1,
        tools=[
            ToolRecord(
                tool_call_id="call_1",
                tool_name="bash",
                command="pytest -q",
                observation=f"exit code: 1\n<output>\n{output}\n\n</output>\n",
                is_error=False,
                exit_code=1,
                timed_out=False,
                truncated_chars=0,
                started_ns=0,
                ended_ns=0,
            )
        ],
    )

    text = format_step(step, 0.0123)

    assert text.startswith("[step 2] $0.0123 so far\nRunning the tests.\n$ pytest -q\n  exit 1\n")
    assert "  line 5\n" in text
    assert "line 6" not in text
    assert "  ... 3 more lines\n" in text
    assert "<output>" not in text


# --- bare `swe-agent`: the start page


def test_bare_command_off_a_terminal_prints_the_help_and_exits_2() -> None:
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 2
    assert "Usage" in result.output
    assert "run" in result.output


@pytest.fixture
def start_page(monkeypatch: pytest.MonkeyPatch, fake_agent: type[_FakeAgent]) -> list[str]:
    """Bare `swe-agent` as if in a terminal, with a stand-in start page that records calls."""
    calls: list[str] = []
    monkeypatch.setattr("swe_agent.cli.main.is_interactive", lambda: True)
    monkeypatch.setattr("swe_agent.cli.main.sandbox_problem", lambda: None)
    monkeypatch.setattr("swe_agent.cli.main.PosixTerminal", lambda: "terminal")

    def started(term: object, *, title: str) -> bool:
        calls.append(title)
        return True

    monkeypatch.setattr("swe_agent.cli.main.run_splash", started)
    return calls


def test_after_the_start_page_a_chat_session_starts(
    start_page: list[str], fake_agent: type[_FakeAgent]
) -> None:
    result = CliRunner().invoke(app, [], input="fix the failing test\nadd a test\nexit\n")

    assert result.exit_code == 0, result.output
    assert start_page == [f"swe-agent {version('swe-agent')}"]
    assert CHAT_HINT in result.output
    assert fake_agent.tasks == ["fix the failing test", "add a test"]
    [kwargs] = fake_agent.built  # one agent, so one session, for both tasks
    assert kwargs["run_command"] is run_bash_in_sandbox  # chat's defaults
    assert kwargs["prompt_version"] == CHAT_SYSTEM_PROMPT


def test_quitting_the_start_page_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str], fake_agent: type[_FakeAgent]
) -> None:
    monkeypatch.setattr("swe_agent.cli.main.run_splash", lambda term, *, title: False)

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert CHAT_HINT not in result.output
    assert fake_agent.tasks == []


def test_ctrl_c_on_the_start_page_exits_130(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str]
) -> None:
    def interrupted(term: object, *, title: str) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr("swe_agent.cli.main.run_splash", interrupted)

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 130
    assert "Aborted" not in result.output


def test_a_missing_key_fails_before_the_start_page(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert start_page == []


def test_a_broken_sandbox_fails_before_the_start_page(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str]
) -> None:
    monkeypatch.setattr("swe_agent.cli.main.sandbox_problem", lambda: "no docker")

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 1
    assert "no docker" in result.output
    assert start_page == []


def test_the_check_before_the_start_page_leaves_the_key_for_run(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str]
) -> None:
    monkeypatch.setattr("swe_agent.cli.main.run_splash", lambda term, *, title: False)

    CliRunner().invoke(app, [])

    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"


def test_the_key_file_counts_for_the_check_before_the_start_page(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    agent_home().mkdir(parents=True)
    (agent_home() / "env").write_text("ANTHROPIC_API_KEY=from-file\n")

    result = CliRunner().invoke(app, [], input="go\n")

    assert result.exit_code == 0, result.output
    assert start_page != []


# --- swe-agent chat ------------------------------------------------------------------------


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[ModelResponse]], ScriptedModel]:
    """The real Agent and Session on a scripted model: give it the replies, get the model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def use(responses: list[ModelResponse]) -> ScriptedModel:
        model = ScriptedModel(responses)
        monkeypatch.setattr("swe_agent.cli.main.build_model", lambda *a, **k: model)
        return model

    return use


def _chat(tmp_path: Path, typed: str, *flags: str) -> Result:
    args = ["chat", "--cwd", str(tmp_path), "--runs-dir", str(tmp_path / "runs"), "--no-sandbox"]
    return CliRunner().invoke(app, [*args, *flags], input=typed)


def test_chat_keeps_one_history_across_tasks(
    scripted: Callable[[list[ModelResponse]], ScriptedModel], tmp_path: Path
) -> None:
    model = scripted([text_reply("a", cost_usd=0.001), text_reply("b", cost_usd=0.002)])

    result = _chat(tmp_path, "first task\nsecond task\n")

    assert result.exit_code == 0, result.output
    second = model.requests[1].messages
    assert [m.text for m in second] == ["first task", "a", "second task"]
    assert "[turn 1] submitted · 1 step · $0.0010 this turn · $0.0010 session" in result.output
    assert "[turn 2] submitted · 1 step · $0.0020 this turn · $0.0030 session" in result.output
    [trajectory] = (tmp_path / "runs").iterdir()  # one file for the whole session
    assert result.output.rstrip().endswith(f"trajectory: {trajectory}")
    assert isinstance(read_trajectory(trajectory)[-1], SessionEnd)


@pytest.mark.parametrize("word", ["exit", "quit"])
def test_chat_skips_blank_lines_and_ends_on_exit_or_quit(
    scripted: Callable[[list[ModelResponse]], ScriptedModel], tmp_path: Path, word: str
) -> None:
    model = scripted([text_reply("a")])

    result = _chat(tmp_path, f"\n   \nthe task\n{word}\nnever sent\n")

    assert result.exit_code == 0, result.output
    assert len(model.requests) == 1
    assert [m.text for m in model.requests[0].messages] == ["the task"]


def test_ctrl_d_before_any_task_ends_with_no_trajectory(
    scripted: Callable[[list[ModelResponse]], ScriptedModel], tmp_path: Path
) -> None:
    scripted([])

    result = _chat(tmp_path, "")

    assert result.exit_code == 0, result.output
    assert CHAT_HINT in result.output
    assert "trajectory:" not in result.output
    assert not (tmp_path / "runs").exists()


def test_ctrl_c_during_a_turn_ends_the_session_and_closes_its_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class Interrupted(ScriptedModel):
        def complete(self, *args: object, **kwargs: object) -> ModelResponse:
            raise KeyboardInterrupt

    monkeypatch.setattr("swe_agent.cli.main.build_model", lambda *a, **k: Interrupted([]))

    result = _chat(tmp_path, "the task\nnever sent\n")

    assert result.exit_code == 130
    assert "Aborted" not in result.output
    [trajectory] = (tmp_path / "runs").iterdir()
    assert isinstance(read_trajectory(trajectory)[-1], SessionEnd)


def test_a_failed_turn_prints_its_error_and_the_session_goes_on(
    scripted: Callable[[list[ModelResponse]], ScriptedModel], tmp_path: Path
) -> None:
    scripted([])  # every call fails: the script is empty

    result = _chat(tmp_path, "one\ntwo\n")

    assert result.exit_code == 0, result.output
    assert "[turn 1] error · 0 steps" in result.output
    assert "[turn 2] error · 0 steps" in result.output
    assert "error: script has 0 responses" in result.output


def test_chat_help_lists_the_same_options_as_run() -> None:
    def options(command: str) -> set[str]:
        result = CliRunner().invoke(app, [command, "--help"], env={"COLUMNS": "200"})
        return set(re.findall(r"--[a-z][a-z-]+", ANSI.sub("", result.output)))

    assert options("chat") == options("run")


def test_chat_budgets_say_they_count_each_turn() -> None:
    result = CliRunner().invoke(app, ["chat", "--help"], env={"COLUMNS": "200"})

    assert "Stop a turn once it has cost this" in ANSI.sub("", result.output)


@pytest.mark.parametrize(
    ("flags", "prompt"),
    [([], CHAT_SYSTEM_PROMPT), (["--no-sandbox"], CHAT_HOST_SYSTEM_PROMPT)],
)
def test_chat_uses_the_chat_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_agent: type[_FakeAgent],
    flags: list[str],
    prompt: str,
) -> None:
    monkeypatch.setattr("swe_agent.cli.main.sandbox_problem", lambda: None)

    result = CliRunner().invoke(app, ["chat", "--cwd", str(tmp_path), *flags], input="x\n")

    assert result.exit_code == 0, result.output
    [kwargs] = fake_agent.built
    assert kwargs["prompt_version"] == prompt
    assert kwargs["system_prompt"] == load_system_prompt(prompt)


@pytest.mark.parametrize("name", [CHAT_SYSTEM_PROMPT, CHAT_HOST_SYSTEM_PROMPT])
def test_chat_prompts_say_a_reply_ends_the_turn_not_the_run(name: str) -> None:
    prompt = load_system_prompt(name)

    assert "A reply without a tool call ends your turn" in prompt
    assert "ends the run" not in prompt
    assert "Nobody will answer" not in prompt


def test_the_chat_sandbox_prompt_describes_the_container() -> None:
    prompt = load_system_prompt(CHAT_SYSTEM_PROMPT)

    assert "/workspace" in prompt
    assert "no network" in prompt
    assert "macOS" not in prompt


def test_format_turn_reports_the_turn_and_the_session() -> None:
    result = RunResult(
        session_id="s", stop="submitted", steps=3, cost_usd=0.0012, final_text="", trajectory=None
    )

    text = format_turn(result, turn=2, session_cost=0.0031)

    assert text == "[turn 2] submitted · 3 steps · $0.0012 this turn · $0.0031 session\n"


# --- the context window ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "text"),
    [(850, "850"), (45_200, "45.2K"), (912_000, "912.0K"), (1_000_000, "1M"), (1_250_000, "1.2M")],
)
def test_format_tokens(tokens: int, text: str) -> None:
    assert format_tokens(tokens) == text


def test_the_warning_starts_at_exactly_90_percent() -> None:
    assert context_warning(900_000, 1_000_000) == (
        "warning: context window 90% full (900.0K of 1M tokens). "
        "Start a new session before it fills up."
    )
    assert context_warning(899_999, 1_000_000) is None


def test_an_unknown_window_or_count_never_warns() -> None:
    assert context_warning(5_000_000, None) is None
    assert context_warning(None, 1_000_000) is None


def _step_with_context(tokens: int | None) -> StepRecord:
    return StepRecord(
        session_id="s",
        step=1,
        parent_step=None,
        request_messages=1,
        assistant=Message(role="assistant", content=[TextBlock(text="hi")]),
        appended=True,
        stop_reason="end_turn",
        usage=Usage(),
        context_tokens=tokens,
        cost_usd=0.0,
        model_started_ns=0,
        model_first_token_ns=None,
        model_ended_ns=0,
        attempts=1,
        tools=[],
    )


def test_the_step_header_shows_how_full_the_context_is() -> None:
    step = _step_with_context(45_200)

    known = format_step(step, 0.0, context_window=1_000_000)
    unknown = format_step(step, 0.0)

    assert known.startswith("[step 1] $0.0000 so far · context 45.2K/1M (4%)\nhi\n")
    assert unknown.startswith("[step 1] $0.0000 so far · context 45.2K\nhi\n")
    assert format_step(_step_with_context(None), 0.0).startswith("[step 1] $0.0000 so far\n")


def test_the_step_header_never_reads_90_percent_without_the_warning() -> None:
    text = format_step(_step_with_context(899_999), 0.0, context_window=1_000_000)

    assert "(89%)" in text
    assert "warning" not in text


def _nearly_full() -> ModelResponse:
    return text_reply("done").model_copy(update={"usage": Usage(input_tokens=950_000)})


@pytest.mark.parametrize("command", ["run", "chat"])
def test_run_and_chat_both_warn_when_the_context_is_nearly_full(
    scripted: Callable[[list[ModelResponse]], ScriptedModel], tmp_path: Path, command: str
) -> None:
    scripted([_nearly_full()])
    task = ["x"] if command == "run" else []

    result = CliRunner().invoke(
        app,
        [
            command,
            *task,
            "--cwd",
            str(tmp_path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--no-sandbox",
        ],
        input="x\n",
    )

    assert result.exit_code == 0, result.output
    assert "· context 950.0K/1M (95%)" in result.output
    assert "warning: context window 95% full (950.0K of 1M tokens)." in result.output


@pytest.mark.parametrize(
    ("flags", "window"),
    [
        ([], 1_000_000),  # the default claude-haiku-5-5
        (["--model", "claude-opus-5-5"], 1_000_000),
        (["--provider", "openai", "--model", "m", "--base-url", "http://localhost:1/v1"], None),
    ],
)
def test_the_agent_gets_the_models_context_window(
    tmp_path: Path, fake_agent: type[_FakeAgent], flags: list[str], window: int | None
) -> None:
    result = CliRunner().invoke(app, ["run", "x", "--cwd", str(tmp_path), "--no-sandbox", *flags])

    assert result.exit_code == 0, result.output
    [kwargs] = fake_agent.built
    assert kwargs["context_window"] == window


def test_every_model_with_a_window_has_a_price() -> None:
    assert set(CONTEXT_WINDOWS) <= set(PRICES)
