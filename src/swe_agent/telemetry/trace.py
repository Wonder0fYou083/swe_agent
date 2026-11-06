"""Trace schema v1: one JSON object per line, told apart by `type`.

Every `*_ns` field is `time.monotonic_ns()` on the machine that ran the session.
`SessionStart.t0_ns` is the same clock at session start, and `started_at` is the only
wall-clock time in the file, so `started_at + (x_ns - t0_ns)` places any event in real time.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from swe_agent.models.types import Endpoint, Message, StopReason, ToolSpec, Usage

StopKind = Literal["submitted", "max_steps", "max_cost", "max_wall_time", "model_stopped", "error"]


class Budgets(BaseModel):
    max_steps: int = 30
    max_cost_usd: float = 0.50
    max_wall_time_s: float = 600.0


class SessionStart(BaseModel):
    type: Literal["session_start"] = "session_start"
    schema_version: Literal[1] = 1
    session_id: str
    task: str
    model: str
    endpoint: Endpoint
    prompt_version: str | None  # e.g. "system_v1"; None for an ad-hoc prompt
    system_prompt: str
    tools: list[ToolSpec]
    cwd: str
    budgets: Budgets
    swe_agent_version: str
    started_at: str  # ISO 8601, UTC
    t0_ns: int


class ToolRecord(BaseModel):
    tool_call_id: str
    tool_name: str
    command: str
    observation: str  # exactly the text sent back to the model
    is_error: bool
    exit_code: int | None
    timed_out: bool
    truncated_chars: int
    started_ns: int
    ended_ns: int


class StepRecord(BaseModel):
    type: Literal["step"] = "step"
    session_id: str
    step: int  # 1, 2, 3, ...
    parent_step: int | None  # step - 1 for now; branches later
    request_messages: int  # how many messages were sent with this request
    assistant: Message  # full text, tool calls and provider blocks
    appended: bool  # False if the response was dropped from history (max_tokens, refusal)
    stop_reason: StopReason
    usage: Usage
    cost_usd: float
    model_started_ns: int
    model_first_token_ns: int | None
    model_ended_ns: int
    attempts: int
    tools: list[ToolRecord]


class SessionEnd(BaseModel):
    type: Literal["session_end"] = "session_end"
    session_id: str
    stop: StopKind
    steps: int
    cost_usd: float
    usage: Usage
    ended_ns: int
    error: str | None = None


TraceRecord = Annotated[SessionStart | StepRecord | SessionEnd, Field(discriminator="type")]
