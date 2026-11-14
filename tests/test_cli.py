import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from swe_agent.cli.main import (
    HOST_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    Provider,
    agent_home,
    app,
    build_model,
    format_step,
    load_system_prompt,
    read_api_key,
    read_env_file,
    sandbox_problem,
)
from swe_agent.core.agent import RunResult
from swe_agent.models.anthropic import AnthropicModel
from swe_agent.models.fake import ScriptedModel
from swe_agent.models.openai import OpenAIModel
from swe_agent.models.pricing import PRICES, Price, UnknownModelError
from swe_agent.models.types import Endpoint, Message, TextBlock, Usage
from swe_agent.telemetry.trace import StepRecord, ToolRecord
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


def test_after_the_start_page_the_task_typed_in_is_run(
    start_page: list[str], fake_agent: type[_FakeAgent]
) -> None:
    result = CliRunner().invoke(app, [], input="fix the failing test\n")

    assert result.exit_code == 0, result.output
    assert start_page == [f"swe-agent {version('swe-agent')}"]
    assert "What should I do?" in result.output
    assert fake_agent.tasks == ["fix the failing test"]
    assert fake_agent.built[0]["run_command"] is run_bash_in_sandbox  # run's defaults


def test_quitting_the_start_page_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, start_page: list[str], fake_agent: type[_FakeAgent]
) -> None:
    monkeypatch.setattr("swe_agent.cli.main.run_splash", lambda term, *, title: False)

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert "What should I do?" not in result.output
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
