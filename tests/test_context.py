"""Phase 4, PR 1: how full the context window is. Copy to tests/test_context.py.

Every test runs on ScriptedModel. The scripted replies carry made-up token counts with all
four counters different, so a sum that drops, repeats or accumulates one of them fails.
"""

from pathlib import Path

from swe_agent.core.agent import Agent
from swe_agent.models.fake import ScriptedModel, bash_call, text_reply
from swe_agent.models.types import ModelResponse, Usage
from swe_agent.telemetry.logger import read_trajectory
from swe_agent.telemetry.trace import SessionStart, StepRecord

WINDOW = 1_000_000

# prompt_tokens = input + cache_read + cache_write; the context after the step adds output.
FIRST = Usage(input_tokens=1000, cache_read_tokens=200, cache_write_tokens=30, output_tokens=7)
SECOND = Usage(input_tokens=40, cache_read_tokens=1230, cache_write_tokens=5, output_tokens=3)
FIRST_CONTEXT = 1237
SECOND_CONTEXT = 1278  # not 1237 + 1278: the second request already resent the first


def _with(response: ModelResponse, usage: Usage) -> ModelResponse:
    return response.model_copy(update={"usage": usage})


def _agent(model: ScriptedModel, tmp_path: Path, *, context_window: int | None = None) -> Agent:
    return Agent(
        model,
        cwd=tmp_path,
        system_prompt="sys",
        trajectory_dir=tmp_path / "runs",
        context_window=context_window,
    )


def _steps(path: Path | None) -> list[StepRecord]:
    assert path is not None
    return [r for r in read_trajectory(path) if isinstance(r, StepRecord)]


def _start(path: Path | None) -> SessionStart:
    assert path is not None
    start = read_trajectory(path)[0]
    assert isinstance(start, SessionStart)
    return start


# --- StepRecord.context_tokens ---------------------------------------------------------------


def test_each_step_records_how_full_the_context_is(tmp_path: Path) -> None:
    model = ScriptedModel([_with(bash_call("true"), FIRST), _with(text_reply("done"), SECOND)])

    result = _agent(model, tmp_path).run("task")

    steps = _steps(result.trajectory)
    assert [s.context_tokens for s in steps] == [FIRST_CONTEXT, SECOND_CONTEXT]


def test_on_step_sees_the_context_tokens(tmp_path: Path) -> None:
    model = ScriptedModel([_with(text_reply("done"), FIRST)])
    seen: list[int | None] = []
    agent = Agent(
        model,
        cwd=tmp_path,
        system_prompt="sys",
        on_step=lambda step: seen.append(step.context_tokens),
    )

    agent.run("task")

    assert seen == [FIRST_CONTEXT]  # what the CLI prints, and warns on


def test_every_turn_of_a_session_records_the_context(tmp_path: Path) -> None:
    model = ScriptedModel([_with(text_reply("a"), FIRST), _with(text_reply("b"), SECOND)])

    with _agent(model, tmp_path).session() as session:
        first = session.send("one")
        session.send("two")

    steps = _steps(first.trajectory)
    assert [(s.turn, s.context_tokens) for s in steps] == [(1, FIRST_CONTEXT), (2, SECOND_CONTEXT)]


# --- SessionStart.context_window -------------------------------------------------------------


def test_a_run_records_the_context_window(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("done")])

    result = _agent(model, tmp_path, context_window=WINDOW).run("task")

    assert _start(result.trajectory).context_window == WINDOW


def test_a_session_records_the_context_window(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("done")])

    with _agent(model, tmp_path, context_window=WINDOW).session() as session:
        result = session.send("one")

    assert _start(result.trajectory).context_window == WINDOW


def test_an_unknown_context_window_is_recorded_as_none(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("done")])

    result = _agent(model, tmp_path).run("task")

    assert _start(result.trajectory).context_window is None


# --- Files from v0.3.0 -----------------------------------------------------------------------


def test_records_from_before_phase_4_read_with_no_context(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("done")])
    result = _agent(model, tmp_path, context_window=WINDOW).run("task")
    assert result.trajectory is not None
    start, step = read_trajectory(result.trajectory)[:2]
    assert isinstance(start, SessionStart)
    assert isinstance(step, StepRecord)

    old_start = start.model_dump(exclude={"context_window"})  # what a v0.3.0 line holds
    old_step = step.model_dump(exclude={"turn", "context_tokens"})

    assert SessionStart.model_validate(old_start).context_window is None
    assert StepRecord.model_validate(old_step).context_tokens is None
