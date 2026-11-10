import os
import re
from importlib.metadata import version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_agent.cli.main import (
    agent_home,
    app,
    format_step,
    load_system_prompt,
    read_api_key,
    read_env_file,
)
from swe_agent.models.types import Message, TextBlock, Usage
from swe_agent.telemetry.trace import StepRecord, ToolRecord

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
        "--model",
        "--max-steps",
        "--max-cost",
        "--max-time",
        "--runs-dir",
        "--no-cache",
    ):
        assert option in text


def test_system_prompt_loads_and_states_how_a_run_ends() -> None:
    # The loop stops on a reply without a tool call, so the prompt must say so.
    assert "A reply without a tool call ends the run." in load_system_prompt()


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
