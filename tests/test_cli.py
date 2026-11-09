import os
import re
from importlib.metadata import version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_agent.cli.main import app, format_step, load_system_prompt
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
