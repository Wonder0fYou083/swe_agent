import time
from collections.abc import Callable
from pathlib import Path

from swe_agent.core.agent import Agent
from swe_agent.models.fake import ScriptedModel, bash_call, text_reply
from swe_agent.models.types import Message, TextBlock, ToolCall, ToolResult
from swe_agent.telemetry.logger import read_trajectory
from swe_agent.telemetry.trace import Budgets, SessionEnd, SessionStart, StepRecord
from swe_agent.tools.bash import BashResult


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


def _last_message(model: ScriptedModel, request: int) -> Message:
    return model.requests[request].messages[-1]


def test_stops_when_the_model_replies_without_a_tool_call(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("all done")])

    result = _agent(model, tmp_path).run("do the thing")

    assert result.stop == "submitted"
    assert result.steps == 1
    assert result.final_text == "all done"
    assert model.requests[0].messages == [
        Message(role="user", content=[TextBlock(text="do the thing")])
    ]
    assert [t.name for t in model.requests[0].tools] == ["bash"]


def test_runs_the_command_and_sends_the_result_back(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo hi"), text_reply("done")])

    _agent(model, tmp_path).run("task")

    result = _last_message(model, 1).content[0]
    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "call_1"
    assert "hi" in result.content
    assert not result.is_error


def test_history_is_append_only(tmp_path: Path) -> None:
    first = bash_call("echo hi")
    model = ScriptedModel([first, text_reply("done")])

    _agent(model, tmp_path).run("task")

    second_request = model.requests[1].messages
    assert second_request[:1] == model.requests[0].messages
    assert second_request[1] == first.message
    assert second_request[2].role == "user"


def test_parallel_calls_get_all_results_in_one_message_in_order(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo one", "echo two"), text_reply("done")])

    _agent(model, tmp_path).run("task")

    results = _last_message(model, 1).content
    assert [r.tool_call_id for r in results if isinstance(r, ToolResult)] == ["call_1", "call_2"]


def test_a_failing_command_is_not_a_tool_error(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("exit 1"), text_reply("done")])

    _agent(model, tmp_path).run("task")

    result = _last_message(model, 1).content[0]
    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert "exit code: 1" in result.content


def test_an_unknown_tool_gets_an_error_result_and_the_run_continues(tmp_path: Path) -> None:
    call = ToolCall(id="c1", name="python", arguments={})
    unknown = bash_call("ls").model_copy(
        update={"message": Message(role="assistant", content=[call])}
    )
    model = ScriptedModel([unknown, text_reply("done")])

    result = _agent(model, tmp_path).run("task")

    error = _last_message(model, 1).content[0]
    assert isinstance(error, ToolResult)
    assert error.is_error
    assert result.stop == "submitted"


def test_stops_at_max_steps(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("true")] * 5)

    result = _agent(model, tmp_path, budgets=Budgets(max_steps=2)).run("task")

    assert result.stop == "max_steps"
    assert result.steps == 2


def test_stops_at_max_cost(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("true", cost_usd=0.3)] * 5)

    result = _agent(model, tmp_path, budgets=Budgets(max_cost_usd=0.5)).run("task")

    assert result.stop == "max_cost"
    assert result.steps == 2
    assert result.cost_usd == 0.6


def test_stops_at_max_wall_time(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("true")] * 5)
    seconds = 10**9

    def clock() -> int:  # 100 s pass for every model call
        return len(model.requests) * 100 * seconds

    result = _agent(model, tmp_path, budgets=Budgets(max_wall_time_s=150), clock=clock).run("task")

    assert result.stop == "max_wall_time"
    assert result.steps == 2


def test_does_not_run_tools_from_a_truncated_response(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("touch ran", stop_reason="max_tokens")])

    result = _agent(model, tmp_path).run("task")

    assert result.stop == "model_stopped"
    assert not (tmp_path / "ran").exists()


def test_writes_a_complete_trajectory(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo hi", cost_usd=0.01), text_reply("done", cost_usd=0.02)])

    result = _agent(model, tmp_path, trajectory_dir=tmp_path / "runs").run("task")

    assert result.trajectory is not None
    records = read_trajectory(result.trajectory)
    start, steps, end = records[0], records[1:-1], records[-1]
    assert isinstance(start, SessionStart)
    assert isinstance(end, SessionEnd)
    assert all(isinstance(s, StepRecord) for s in steps)
    assert [s.step for s in steps if isinstance(s, StepRecord)] == [1, 2]
    assert end.stop == "submitted"
    assert end.cost_usd == 0.03
    tool = steps[0].tools[0] if isinstance(steps[0], StepRecord) else None
    assert tool is not None
    assert tool.command == "echo hi"
    assert tool.exit_code == 0
    assert tool.ended_ns >= tool.started_ns


def test_trajectory_holds_enough_to_rebuild_every_request(tmp_path: Path) -> None:
    model = ScriptedModel(
        [bash_call("echo one", "echo two"), bash_call("exit 2"), text_reply("done")]
    )

    result = _agent(model, tmp_path, trajectory_dir=tmp_path / "runs").run("task")

    assert result.trajectory is not None
    records = read_trajectory(result.trajectory)
    start = records[0]
    assert isinstance(start, SessionStart)
    history = [Message(role="user", content=[TextBlock(text=start.task)])]
    for k, step in enumerate(records[1:-1]):
        assert isinstance(step, StepRecord)
        assert model.requests[k].system == start.system_prompt
        assert model.requests[k].messages == history
        assert step.request_messages == len(history)
        history.append(step.assistant)
        if step.tools:
            history.append(
                Message(
                    role="user",
                    content=[
                        ToolResult(
                            tool_call_id=t.tool_call_id, content=t.observation, is_error=t.is_error
                        )
                        for t in step.tools
                    ],
                )
            )


def test_session_start_records_where_requests_went(tmp_path: Path) -> None:
    model = ScriptedModel([text_reply("done")])
    agent = Agent(
        model,
        cwd=tmp_path,
        system_prompt="sys",
        prompt_version="system_v1",
        trajectory_dir=tmp_path / "runs",
    )

    result = agent.run("task")

    assert result.trajectory is not None
    start = read_trajectory(result.trajectory)[0]
    assert isinstance(start, SessionStart)
    assert start.endpoint == model.endpoint
    assert start.prompt_version == "system_v1"
    assert start.model == model.name
    assert start.cwd == str(tmp_path)
    assert [t.name for t in start.tools] == ["bash"]


def test_no_trajectory_dir_writes_no_file(tmp_path: Path) -> None:
    model = ScriptedModel([bash_call("echo hi"), text_reply("done")])

    result = _agent(model, tmp_path).run("task")

    assert result.trajectory is None
    assert not list(tmp_path.glob("*.jsonl"))


def test_commands_go_to_the_run_command_it_was_given(tmp_path: Path) -> None:
    # The CLI passes run_bash_in_sandbox here; a fake one shows where commands go.
    calls: list[tuple[str, Path]] = []

    def fake_run(command: str, cwd: Path) -> BashResult:
        calls.append((command, cwd))
        return BashResult(
            command=command,
            output="from the fake\n",
            exit_code=0,
            timed_out=False,
            truncated_chars=0,
            started_ns=0,
            ended_ns=0,
        )

    model = ScriptedModel([bash_call("ls"), text_reply("done")])
    agent = Agent(model, cwd=tmp_path, system_prompt="sys", run_command=fake_run)

    agent.run("task")

    assert calls == [("ls", tmp_path)]
    result = _last_message(model, 1).content[0]
    assert isinstance(result, ToolResult)
    assert "from the fake" in result.content
