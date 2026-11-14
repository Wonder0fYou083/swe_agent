"""Run the agent on every task in evals/tasks and print the pass rate.

    uv run python -m evals.run                       # every task, claude-haiku-5-5
    uv run python -m evals.run --task toy_median     # one task; repeat --task for more

Each task folder has `repo/` (the code the agent starts from), `task.md` (the prompt) and
`check/` (hidden tests and the pytest.ini they run under). For every task the runner:

1. copies `repo/` to a temporary folder, then runs `git init` and commits it;
2. runs the agent there in the sandbox, with fixed budgets;
3. copies `check/` over the workspace, which restores any test the agent edited;
4. runs the check's tests in the sandbox; the task passes on exit code 0;
5. writes one JSON line to evals/results/<date>-<model>.jsonl.

Trajectories go to ~/.swe-agent/runs/, as for `swe-agent run`.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from swe_agent.cli.main import (
    API_KEY,
    SANDBOX_BUILD,
    Provider,
    agent_home,
    build_model,
    load_system_prompt,
    read_api_key,
    sandbox_problem,
)
from swe_agent.core.agent import Agent, RunCommand
from swe_agent.models.base import Model
from swe_agent.models.pricing import UnknownModelError
from swe_agent.telemetry.trace import Budgets, StopKind
from swe_agent.tools.bash import run_bash_in_sandbox

TASKS_DIR = Path(__file__).parent / "tasks"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "system_v2"  # the sandbox prompt, as in `swe-agent run`
BUDGETS = Budgets(max_steps=20, max_cost_usd=0.25, max_wall_time_s=300)

# The agent made no commits, so the identity only has to exist; CI runners have none.
GIT = ["git", "-c", "user.name=swe-agent-eval", "-c", "user.email=eval@localhost"]


class TaskResult(BaseModel):
    task: str
    passed: bool
    stop: StopKind
    steps: int
    cost_usd: float
    wall_time_s: float  # the agent's run, without setup and grading
    trajectory: str | None  # file name in ~/.swe-agent/runs/, not a full path


def task_dirs(names: Sequence[str] = ()) -> list[Path]:
    """The task folders, sorted by name; only `names` if given. Raises ValueError for a typo."""
    tasks = sorted(path for path in TASKS_DIR.iterdir() if (path / "task.md").is_file())
    if not names:
        return tasks
    by_name = {path.name: path for path in tasks}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"no such task: {', '.join(unknown)}")
    return [by_name[name] for name in names]


def run_task(
    task_dir: Path,
    model: Model,
    *,
    budgets: Budgets = BUDGETS,
    trajectory_dir: Path | None = None,
    run_command: RunCommand = run_bash_in_sandbox,
) -> TaskResult:
    """Steps 1 to 4 for one task. Tests pass `run_bash` to run on the host without Docker."""
    with tempfile.TemporaryDirectory(prefix="swe-eval-") as tmp:
        workspace = Path(tmp) / "workspace"
        shutil.copytree(task_dir / "repo", workspace)
        git_init(workspace)

        agent = Agent(
            model,
            cwd=workspace,
            system_prompt=load_system_prompt(PROMPT_VERSION),
            prompt_version=PROMPT_VERSION,
            budgets=budgets,
            trajectory_dir=trajectory_dir,
            run_command=run_command,
        )
        started = time.monotonic()
        result = agent.run((task_dir / "task.md").read_text(encoding="utf-8"))
        wall_time_s = time.monotonic() - started

        passed = grade(workspace, task_dir / "check", run_command)

    return TaskResult(
        task=task_dir.name,
        passed=passed,
        stop=result.stop,
        steps=result.steps,
        cost_usd=result.cost_usd,
        wall_time_s=round(wall_time_s, 1),
        trajectory=result.trajectory.name if result.trajectory else None,
    )


def git_init(workspace: Path) -> None:
    """Make `workspace` a git repo with one commit. Runs on the host: the sandbox has no git."""
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "--no-verify", "-m", "start"]):
        subprocess.run(
            [*GIT, "-c", "commit.gpgsign=false", *args],
            cwd=workspace,
            capture_output=True,
            check=True,
        )


def grade(workspace: Path, check_dir: Path, run_command: RunCommand) -> bool:
    """Copy `check_dir` over the workspace and run its tests. True when they all pass.

    The copy restores any test the agent edited. `--noconftest` and `-c pytest.ini` ignore
    any conftest.py or pytest config the agent added, which could skip every test and still
    exit with 0.
    """
    shutil.copytree(check_dir, workspace, dirs_exist_ok=True)
    tests = " ".join(sorted(path.name for path in check_dir.glob("test_*.py")))
    result = run_command(f"python -m pytest -q --noconftest -c pytest.ini {tests}", workspace)
    return result.exit_code == 0


def run_all(
    tasks: Sequence[Path],
    make_model: Callable[[], Model],
    results_path: Path,
    *,
    budgets: Budgets = BUDGETS,
    trajectory_dir: Path | None = None,
    run_command: RunCommand = run_bash_in_sandbox,
) -> list[TaskResult]:
    """Run every task with a fresh model, write one JSON line each, and print the summary."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[TaskResult] = []
    with results_path.open("w", encoding="utf-8") as out:
        for i, task_dir in enumerate(tasks, start=1):
            result = run_task(
                task_dir,
                make_model(),
                budgets=budgets,
                trajectory_dir=trajectory_dir,
                run_command=run_command,
            )
            out.write(result.model_dump_json() + "\n")
            out.flush()  # a crash later still leaves the finished tasks on disk
            print(f"[{i}/{len(tasks)}] {format_result(result)}", flush=True)
            results.append(result)
    print(format_summary(results))
    return results


def format_result(result: TaskResult) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    return (
        f"{result.task}: {verdict} ({result.stop}, {result.steps} steps, "
        f"${result.cost_usd:.4f}, {result.wall_time_s:.1f}s)"
    )


def format_summary(results: Sequence[TaskResult]) -> str:
    passed = sum(result.passed for result in results)
    cost = sum(result.cost_usd for result in results)
    return f"{passed}/{len(results)} passed, ${cost:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run", description="Run every eval task and print the pass rate."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--task", action="append", default=[], help="run only this task; repeat for more"
    )
    args = parser.parse_args(argv)

    try:
        tasks = task_dirs(args.task)
    except ValueError as e:
        return _fail(str(e))
    api_key = read_api_key(API_KEY)
    if not api_key:
        return _fail(f"{API_KEY} is not set. Export it, or put it in {agent_home() / 'env'}.")

    def make_model() -> Model:
        return build_model(Provider.anthropic, args.model, api_key, base_url=None, cache=True)

    try:
        make_model()  # fail on a model with no price before any task starts
    except UnknownModelError:
        return _fail(f"no price for model {args.model!r}; add it to PRICES in models/pricing.py")
    if problem := sandbox_problem():
        return _fail(f"{problem}. Build the image with `{SANDBOX_BUILD}`.")

    results_path = RESULTS_DIR / f"{date.today().isoformat()}-{args.model}.jsonl"
    run_all(tasks, make_model, results_path, trajectory_dir=agent_home() / "runs")
    print(f"results: {results_path}")
    return 0


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
