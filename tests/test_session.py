"""Phase 4, PR 1: one conversation across turns. Copy to tests/test_session.py.

Every test runs on ScriptedModel: no network, no API key, no Docker.
"""

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from swe_agent.core.agent import Agent, add_user_text
from swe_agent.models.fake import ScriptedModel, bash_call, text_reply
from swe_agent.models.types import Message, TextBlock, ToolResult
from swe_agent.telemetry.logger import read_trajectory
from swe_agent.telemetry.trace import (
    Budgets,
    SessionEnd,
    SessionStart,
    StepRecord,
    TurnEnd,
    UserTurn,
)

SECOND = 10**9


def _agent(
    model: ScriptedModel,
    cwd: Path,
    *,
    budgets: Budgets | None = None,
    trajectory_dir: Path | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> Agent:
    return Agent(
        model,
        cwd=cwd,
        system_prompt="sys",
        budgets=budgets,
        trajectory_dir=trajectory_dir,
        clock=clock,
    )


def _user(*texts: str) -> Message:
    return Message(role="user", content=[TextBlock(text=t) for t in texts])


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)])


# --- add_user_text: the one rule for adding what the person typed ---------------------------


def test_add_user_text_starts_a_history() -> None:
    history: list[Message] = []

    add_user_text(history, "one")

    assert history == [_user("one")]


def test_add_user_text_after_an_answer_adds_a_message() -> None:
    history = [_user("one"), _assistant("a")]

    add_user_text(history, "two")

    assert history == [_user("one"), _assistant("a"), _user("two")]


def test_add_user_text_after_a_user_message_joins_it() -> None:
    results = ToolResult(tool_call_id="call_1", content="hi")
    history = [_user("one"), Message(role="user", content=[results])]

    add_user_text(history, "two")

    assert len(history) == 2
    assert history[-1] == Message(role="user", content=[results, TextBlock(text="two")])


# --- Session: turns share one history -------------------------------------------------------


def test_a_second_turn_sees_the_first_turn(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("a"), text_reply("b")])

    with _agent(model, tmp_path).session() as session:
        first = session.send("one")
        second = session.send("two")

    assert first.stop == "submitted"
    assert second.stop == "submitted"
    assert second.final_text == "b"
    assert model.requests[1].messages == [_user("one"), _assistant("a"), _user("two")]


def test_history_stays_append_only_across_turns(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo hi"), text_reply("done"), text_reply("ok")])

    with _agent(model, tmp_path).session() as session:
        session.send("one")
        session.send("two")

    second, third = model.requests[1].messages, model.requests[2].messages
    assert third[: len(second)] == second
    assert third[len(second)] == _assistant("done")
    assert third[-1] == _user("two")


def test_text_after_a_budget_stop_joins_the_tool_results(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo hi"), text_reply("done")])
    agent = _agent(model, tmp_path, budgets=Budgets(max_steps=1))

    with agent.session() as session:
        first = session.send("one")
        second = session.send("two")

    assert first.stop == "max_steps"
    assert second.stop == "submitted"
    sent = model.requests[1].messages
    roles = [m.role for m in sent]
    assert roles == ["user", "assistant", "user"]  # never two user messages in a row
    results, text = sent[-1].content
    assert isinstance(results, ToolResult)
    assert results.tool_call_id == "call_1"
    assert text == TextBlock(text="two")


def test_text_after_a_dropped_response_joins_the_unanswered_text(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("touch ran", stop_reason="max_tokens"), text_reply("done")])

    with _agent(model, tmp_path).session() as session:
        first = session.send("one")
        session.send("two")

    assert first.stop == "model_stopped"
    assert not (tmp_path / "ran").exists()
    assert model.requests[1].messages == [_user("one", "two")]


def test_a_failed_model_call_ends_the_turn_not_the_session(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("a")])  # every call after the first raises ModelError

    with _agent(model, tmp_path).session() as session:
        session.send("one")
        second = session.send("two")
        third = session.send("three")

    assert second.stop == "error"
    assert second.error is not None
    assert third.stop == "error"
    # "two" got no answer, so "three" joins it instead of starting a second user message.
    assert model.requests[2].messages == [_user("one"), _assistant("a"), _user("two", "three")]


# --- Budgets are per turn; totals are per session -------------------------------------------


def test_step_and_cost_budgets_apply_to_each_turn(tmp_path: Path) -> None:
    turn = [bash_call("true", cost_usd=0.02), text_reply("done", cost_usd=0.02)]
    model = ScriptedModel(turn * 2)
    agent = _agent(model, tmp_path, budgets=Budgets(max_steps=2, max_cost_usd=0.05))

    with agent.session() as session:
        first = session.send("one")
        second = session.send("two")

    assert (first.stop, first.steps) == ("submitted", 2)
    assert (second.stop, second.steps) == ("submitted", 2)
    assert first.cost_usd == pytest.approx(0.04)
    assert second.cost_usd == pytest.approx(0.04)
    assert session.steps == 4
    assert session.cost_usd == pytest.approx(0.08)


def test_the_wall_clock_restarts_each_turn(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("a"), text_reply("b"), text_reply("c")])

    def clock() -> int:  # 100 s pass for every model call
        return len(model.requests) * 100 * SECOND

    agent = _agent(model, tmp_path, budgets=Budgets(max_wall_time_s=150), clock=clock)

    with agent.session() as session:
        results = [session.send(text) for text in ("one", "two", "three")]

    # 200 s into the session, but the third turn has only just started.
    assert [r.stop for r in results] == ["submitted"] * 3


def test_every_turn_reports_the_same_session_and_file(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo hi"), text_reply("a"), text_reply("b")])
    agent = _agent(model, tmp_path, trajectory_dir=tmp_path / "runs")

    with agent.session() as session:
        first = session.send("one")
        second = session.send("two")

    assert first.session_id == second.session_id == session.session_id
    assert first.trajectory is not None
    assert first.trajectory == second.trajectory
    assert (first.steps, second.steps) == (2, 1)
    assert session.turns == 2
    assert session.steps == 3


# --- The trajectory: one file per session ---------------------------------------------------


def test_one_trajectory_holds_every_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            bash_call("echo hi", cost_usd=0.01),
            text_reply("a", cost_usd=0.02),
            text_reply("b", cost_usd=0.04),
        ]
    )
    runs = tmp_path / "runs"

    with _agent(model, tmp_path, trajectory_dir=runs).session() as session:
        first = session.send("one")
        session.send("two")

    assert list(runs.glob("*.jsonl")) == [first.trajectory]
    assert first.trajectory is not None
    records = read_trajectory(first.trajectory)
    assert [type(r) for r in records] == [
        SessionStart,
        StepRecord,
        StepRecord,
        TurnEnd,
        UserTurn,
        StepRecord,
        TurnEnd,
        SessionEnd,
    ]
    start, user_turn, end = records[0], records[4], records[-1]
    assert isinstance(start, SessionStart)
    assert start.task == "one"  # turn 1's text lives in session_start, as it does for `run`
    assert isinstance(user_turn, UserTurn)
    assert (user_turn.turn, user_turn.text) == (2, "two")

    steps = [r for r in records if isinstance(r, StepRecord)]
    assert [s.step for s in steps] == [1, 2, 3]  # step numbers carry on across turns
    assert [s.turn for s in steps] == [1, 1, 2]
    assert steps[2].parent_step == 2

    ends = [r for r in records if isinstance(r, TurnEnd)]
    assert [(t.turn, t.stop, t.steps) for t in ends] == [(1, "submitted", 2), (2, "submitted", 1)]
    assert ends[0].cost_usd == pytest.approx(0.03)

    assert isinstance(end, SessionEnd)
    assert (end.stop, end.steps) == ("submitted", 3)
    assert end.cost_usd == pytest.approx(0.07)


def test_the_trajectory_rebuilds_every_request_across_turns(tmp_path: Path) -> None:
    model = ScriptedModel(
        [bash_call("echo one"), bash_call("echo two", "exit 2"), text_reply("ok")]
    )
    agent = _agent(model, tmp_path, budgets=Budgets(max_steps=2), trajectory_dir=tmp_path / "r")

    with agent.session() as session:
        first = session.send("start")  # stops at max_steps, after the second tool results
        session.send("go on")

    assert first.trajectory is not None
    history: list[Message] = []
    request = 0
    for record in read_trajectory(first.trajectory):
        if isinstance(record, SessionStart):
            history = [_user(record.task)]
        elif isinstance(record, UserTurn):
            add_user_text(history, record.text)
        elif isinstance(record, StepRecord):
            assert model.requests[request].messages == history
            assert record.request_messages == len(history)
            request += 1
            if record.appended:
                history.append(record.assistant)
            if record.tools:
                results = [
                    ToolResult(
                        tool_call_id=t.tool_call_id, content=t.observation, is_error=t.is_error
                    )
                    for t in record.tools
                ]
                history.append(Message(role="user", content=list(results)))
    assert request == len(model.requests) == 3


def test_closing_before_any_turn_writes_nothing(tmp_path: Path) -> None:
    model = ScriptedModel([])

    session = _agent(model, tmp_path, trajectory_dir=tmp_path / "runs").session()
    session.close()

    assert not list(tmp_path.rglob("*.jsonl"))
    assert model.requests == []


def test_a_step_record_from_before_phase_4_reads_as_turn_1(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("done")])
    result = _agent(model, tmp_path, trajectory_dir=tmp_path / "runs").run("task")
    assert result.trajectory is not None
    step = read_trajectory(result.trajectory)[1]
    assert isinstance(step, StepRecord)

    old = step.model_dump(exclude={"turn"})  # what a v0.3.0 trajectory line holds

    assert StepRecord.model_validate(old).turn == 1
