import json
from pathlib import Path

from swe_agent.models.types import Endpoint, Message, TextBlock, Usage
from swe_agent.telemetry.logger import TrajectoryLogger, read_trajectory
from swe_agent.telemetry.trace import Budgets, SessionEnd, SessionStart, StepRecord


def _records() -> list[SessionStart | StepRecord | SessionEnd]:
    start = SessionStart(
        session_id="s1",
        task="fix it",
        model="scripted",
        endpoint=Endpoint(provider="scripted", base_url=None, cache_requested=False),
        prompt_version="system_v1",
        system_prompt="sys",
        tools=[],
        cwd="/repo",
        budgets=Budgets(),
        swe_agent_version="0.0.0",
        started_at="2026-10-26T00:00:00+00:00",
        t0_ns=100,
    )
    step = StepRecord(
        session_id="s1",
        step=1,
        parent_step=None,
        request_messages=1,
        assistant=Message(role="assistant", content=[TextBlock(text="done")]),
        appended=True,
        stop_reason="end_turn",
        usage=Usage(input_tokens=10),
        cost_usd=0.001,
        model_started_ns=200,
        model_first_token_ns=None,
        model_ended_ns=300,
        attempts=1,
        tools=[],
    )
    end = SessionEnd(
        session_id="s1",
        stop="submitted",
        steps=1,
        cost_usd=0.001,
        usage=Usage(input_tokens=10),
        ended_ns=400,
    )
    return [start, step, end]


def test_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with TrajectoryLogger(path) as logger:
        for record in _records():
            logger.write(record)

    lines = path.read_text().splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["session_start", "step", "session_end"]


def test_each_line_is_on_disk_before_close(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    logger = TrajectoryLogger(path)
    logger.write(_records()[0])

    assert len(path.read_text().splitlines()) == 1
    logger.close()


def test_read_trajectory_returns_the_records_written(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run.jsonl"
    with TrajectoryLogger(path) as logger:
        for record in _records():
            logger.write(record)

    assert read_trajectory(path) == _records()
