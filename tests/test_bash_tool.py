import os
import time
from pathlib import Path

from swe_agent.tools.bash import BashResult, format_observation, run_bash, truncate_middle


def test_returns_output_and_exit_code(tmp_path: Path) -> None:
    result = run_bash("echo hello", tmp_path)

    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert not result.timed_out


def test_reports_non_zero_exit_code(tmp_path: Path) -> None:
    assert run_bash("exit 3", tmp_path).exit_code == 3


def test_merges_stderr_into_output(tmp_path: Path) -> None:
    result = run_bash("echo out; echo err >&2", tmp_path)

    assert result.output == "out\nerr\n"


def test_runs_in_the_given_directory(tmp_path: Path) -> None:
    result = run_bash("pwd", tmp_path)

    assert Path(result.output.strip()).resolve() == tmp_path.resolve()


def test_stdin_is_closed_so_commands_do_not_wait_for_input(tmp_path: Path) -> None:
    result = run_bash("cat", tmp_path, timeout_s=5)

    assert result.exit_code == 0
    assert not result.timed_out


def test_binary_output_does_not_crash(tmp_path: Path) -> None:
    result = run_bash(r"printf '\xff\xfe'", tmp_path)

    assert "�" in result.output


def test_timeout_kills_the_command(tmp_path: Path) -> None:
    result = run_bash("sleep 10", tmp_path, timeout_s=0.5)

    assert result.timed_out
    assert result.exit_code is None
    assert (result.ended_ns - result.started_ns) / 1e9 < 5


def test_timeout_also_kills_child_processes(tmp_path: Path) -> None:
    run_bash("sleep 30 & echo $! > pid; wait", tmp_path, timeout_s=0.5)
    pid = int((tmp_path / "pid").read_text())

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # signal 0 only checks that the process exists
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"child process {pid} is still running")


def test_long_output_is_truncated_in_the_middle(tmp_path: Path) -> None:
    result = run_bash("seq 1 100000", tmp_path, max_output_chars=1000)

    assert result.output.startswith("1\n2\n")
    assert result.output.endswith("99999\n100000\n")
    assert "characters omitted" in result.output
    assert result.truncated_chars > 0


def test_truncate_middle_leaves_short_text_alone() -> None:
    assert truncate_middle("short", 100) == ("short", 0)


def test_truncate_middle_keeps_head_and_tail() -> None:
    text, omitted = truncate_middle("a" * 50 + "b" * 50, 20)

    assert omitted == 80
    assert text.startswith("a" * 10)
    assert text.endswith("b" * 10)
    assert "[... 80 characters omitted ...]" in text


def _result(**overrides: object) -> BashResult:
    fields: dict[str, object] = {
        "command": "x",
        "output": "boom\n",
        "exit_code": 1,
        "timed_out": False,
        "truncated_chars": 0,
        "started_ns": 0,
        "ended_ns": 0,
    }
    return BashResult.model_validate(fields | overrides)


def test_observation_shows_exit_code_and_output() -> None:
    observation = format_observation(_result())

    assert "exit code: 1" in observation
    assert "boom" in observation


def test_observation_says_when_a_command_timed_out() -> None:
    observation = format_observation(_result(exit_code=None, timed_out=True))

    assert "timed out" in observation
