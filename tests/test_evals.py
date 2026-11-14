"""The eval harness in evals/: the task folders, and the runner with a scripted model.

Everything here runs on the host with `run_bash` and `ScriptedModel`, so it needs no API key
and no Docker. The one exception, marked `docker`, runs a task in the sandbox.
"""

import json
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from evals.run import TASKS_DIR, TaskResult, grade, main, run_all, run_task, task_dirs
from swe_agent.models.base import Model
from swe_agent.models.fake import ScriptedModel, bash_call, text_reply
from swe_agent.models.types import ModelResponse
from swe_agent.telemetry.trace import Budgets
from swe_agent.tools.bash import run_bash

PYTEST_INI = "[pytest]\ntestpaths = .\n"
CALC_TEST = "from calc import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n"
# Scripted commands are heredocs, not `sed -i`, so they behave the same on macOS and Linux.
FIX_CALC = "cat > calc.py <<'EOF'\ndef add(a, b):\n    return a + b\nEOF"


@pytest.fixture(autouse=True)
def _venv_python_first(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Grading runs `python -m pytest`. On the host, make that the python running these tests."""
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def calc_task(tmp_path: Path) -> Path:
    """A one-bug task in the same layout as evals/tasks: add() subtracts."""
    task = tmp_path / "tasks" / "calc"
    for part in ("repo", "check"):
        (task / part).mkdir(parents=True)
        (task / part / "pytest.ini").write_text(PYTEST_INI)
        (task / part / "test_calc.py").write_text(CALC_TEST)
    (task / "repo" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (task / "task.md").write_text("make the tests pass\n")
    return task


def run_scripted(task: Path, *responses: ModelResponse) -> TaskResult:
    return run_task(task, ScriptedModel(responses), run_command=run_bash)


# The task folders


@pytest.mark.parametrize("task", task_dirs(), ids=lambda path: path.name)
def test_every_task_fails_on_its_untouched_repo(task: Path, tmp_path: Path) -> None:
    """Otherwise the task passes by doing nothing."""
    assert (task / "task.md").read_text().strip()
    assert (task / "check" / "pytest.ini").is_file()
    assert list((task / "check").glob("test_*.py"))

    workspace = tmp_path / "workspace"
    shutil.copytree(task / "repo", workspace)

    assert not grade(workspace, task / "check", run_bash)


def test_there_are_ten_tasks_and_the_toy_repo_is_one() -> None:
    names = [path.name for path in task_dirs()]

    assert len(names) == 10
    assert "toy_median" in names


# The runner


def test_scripted_fix_passes(calc_task: Path) -> None:
    result = run_scripted(calc_task, bash_call(FIX_CALC), text_reply("add() now adds"))

    assert result.passed
    assert (result.stop, result.steps) == ("submitted", 2)


def test_doing_nothing_fails(calc_task: Path) -> None:
    result = run_scripted(calc_task, text_reply("nothing to change"))

    assert not result.passed
    assert result.stop == "submitted"


def test_editing_the_test_does_not_pass(calc_task: Path) -> None:
    edit = "cat > test_calc.py <<'EOF'\ndef test_add() -> None:\n    assert True\nEOF"

    result = run_scripted(calc_task, bash_call(edit), text_reply("done"))

    assert not result.passed


def test_a_conftest_that_skips_every_test_does_not_pass(calc_task: Path) -> None:
    """Plain `pytest` exits with 0 when every test is skipped."""
    conftest = (
        "cat > conftest.py <<'EOF'\n"
        "import pytest\n\n\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        item.add_marker(pytest.mark.skip)\n"
        "EOF"
    )

    result = run_scripted(calc_task, bash_call(conftest), text_reply("done"))

    assert not result.passed


def test_a_pytest_config_that_only_collects_does_not_pass(calc_task: Path) -> None:
    """pytest 9 reads a new pytest.toml before pytest.ini, and --collect-only exits with 0."""
    config = "printf '[pytest]\\naddopts = [\"--collect-only\"]\\n' > pytest.toml"

    result = run_scripted(calc_task, bash_call(config), text_reply("done"))

    assert not result.passed


def test_running_out_of_steps_is_recorded(calc_task: Path) -> None:
    result = run_task(
        calc_task,
        ScriptedModel([bash_call("ls")] * 5),
        budgets=Budgets(max_steps=2),
        run_command=run_bash,
    )

    assert not result.passed
    assert (result.stop, result.steps) == ("max_steps", 2)


def test_run_all_writes_one_line_per_task_and_prints_the_summary(
    calc_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripts: Iterator[list[ModelResponse]] = iter(
        [
            [bash_call(FIX_CALC, cost_usd=0.01), text_reply("fixed", cost_usd=0.02)],
            [text_reply("nothing to change", cost_usd=0.005)],
        ]
    )

    def make_model() -> Model:
        return ScriptedModel(next(scripts))

    results_path = tmp_path / "results" / "run.jsonl"
    runs = tmp_path / "runs"

    run_all(
        [calc_task, calc_task],
        make_model,
        results_path,
        trajectory_dir=runs,
        run_command=run_bash,
    )

    lines = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert [line["passed"] for line in lines] == [True, False]
    assert set(lines[0]) == {
        "task",
        "passed",
        "stop",
        "steps",
        "cost_usd",
        "wall_time_s",
        "trajectory",
    }
    assert all((runs / line["trajectory"]).is_file() for line in lines)
    out = capsys.readouterr().out
    assert "[1/2] calc: PASS (submitted, 2 steps, $0.0300" in out
    assert out.rstrip().endswith("1/2 passed, $0.0350")


def test_main_without_an_api_key_fails_before_any_task(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main(["--task", "toy_median"]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_main_rejects_an_unknown_task(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--task", "no_such_task"]) == 1
    assert "no such task: no_such_task" in capsys.readouterr().err


# In the sandbox


@pytest.mark.docker
def test_toy_task_passes_in_the_sandbox_with_a_scripted_fix() -> None:
    fix = (
        "cat > stats.py <<'EOF'\n"
        "def mean(values):\n"
        "    if not values:\n"
        "        raise ValueError('mean of an empty list')\n"
        "    return sum(values) / len(values)\n\n\n"
        "def median(values):\n"
        "    if not values:\n"
        "        raise ValueError('median of an empty list')\n"
        "    s = sorted(values)\n"
        "    m = len(s) // 2\n"
        "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n"
        "EOF"
    )

    result = run_task(TASKS_DIR / "toy_median", ScriptedModel([bash_call(fix), text_reply("ok")]))

    assert result.passed
