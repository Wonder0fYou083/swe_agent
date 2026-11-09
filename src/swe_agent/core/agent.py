"""The agent loop: call the model, run its tool calls, send the results back, repeat."""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from swe_agent import __version__
from swe_agent.models.base import Model, ModelError
from swe_agent.models.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from swe_agent.telemetry.logger import TrajectoryLogger
from swe_agent.telemetry.trace import (
    Budgets,
    SessionEnd,
    SessionStart,
    StepRecord,
    StopKind,
    ToolRecord,
)
from swe_agent.tools.bash import BASH_SPEC, format_observation, run_bash

TraceWriter = Callable[[SessionStart | StepRecord | SessionEnd], None]


class RunResult(BaseModel):
    session_id: str
    stop: StopKind
    steps: int
    cost_usd: float
    final_text: str  # the model's last text when it submitted, else ""
    trajectory: Path | None
    error: str | None = None


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        cwd: Path,
        system_prompt: str,
        prompt_version: str | None = None,  # e.g. "system_v1"; None for an ad-hoc prompt
        budgets: Budgets | None = None,
        trajectory_dir: Path | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.model = model  # also supplies the trace's model name and endpoint
        self.cwd = cwd
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.budgets = budgets or Budgets()
        self.trajectory_dir = trajectory_dir  # None: no trajectory file
        self.on_step = on_step  # the CLI uses this to print each step live
        self.clock = clock  # tests pass a fake clock to test max_wall_time
        self.tools = [BASH_SPEC]  # one list, so the request and SessionStart always agree

    def run(self, task: str) -> RunResult:
        """Call the model, run its tool calls, send the results back, until a stop condition."""
        session_id = uuid4().hex
        t0 = self.clock()
        path = self._trajectory_path(session_id)
        logger = TrajectoryLogger(path) if path is not None else None
        log: TraceWriter = logger.write if logger is not None else _discard
        try:
            log(self._session_start(session_id, task, t0))

            messages = [Message(role="user", content=[TextBlock(text=task)])]
            steps, cost_usd, usage = 0, 0.0, Usage()
            final_text, error = "", None
            stop: StopKind | None = None

            while stop is None:
                stop = self._check_budgets(steps, cost_usd, self.clock() - t0)  # before the call
                if stop is not None:
                    break

                sent = len(messages)
                try:
                    response = self.model.complete(self.system_prompt, messages, self.tools)
                except ModelError as e:
                    stop, error = "error", str(e)
                    break

                steps += 1
                cost_usd += response.cost_usd
                usage = usage + response.usage
                records: list[ToolRecord] = []

                appended = response.stop_reason not in ("max_tokens", "refusal")
                if not appended:
                    stop = "model_stopped"
                else:
                    messages.append(response.message)
                    calls = response.message.tool_calls
                    if not calls:
                        stop = "submitted"
                        final_text = response.message.text
                    else:
                        records = [self._execute(call) for call in calls]
                        messages.append(_tool_results(records))

                step = _step_record(session_id, steps, sent, response, appended, records)
                log(step)
                if self.on_step is not None:
                    self.on_step(step)

            log(
                SessionEnd(
                    session_id=session_id,
                    stop=stop,
                    steps=steps,
                    cost_usd=cost_usd,
                    usage=usage,
                    ended_ns=self.clock(),
                    error=error,
                )
            )
        finally:
            if logger is not None:
                logger.close()

        return RunResult(
            session_id=session_id,
            stop=stop,
            steps=steps,
            cost_usd=cost_usd,
            final_text=final_text,
            trajectory=path,
            error=error,
        )

    def _trajectory_path(self, session_id: str) -> Path | None:
        if self.trajectory_dir is None:
            return None
        return self.trajectory_dir / f"{session_id}.jsonl"

    def _session_start(self, session_id: str, task: str, t0: int) -> SessionStart:
        return SessionStart(
            session_id=session_id,
            task=task,
            model=self.model.name,
            endpoint=self.model.endpoint,
            prompt_version=self.prompt_version,
            system_prompt=self.system_prompt,
            tools=self.tools,
            cwd=str(self.cwd),
            budgets=self.budgets,
            swe_agent_version=__version__,
            started_at=datetime.now(UTC).isoformat(),
            t0_ns=t0,
        )

    def _check_budgets(self, steps: int, cost_usd: float, elapsed_ns: int) -> StopKind | None:
        """Return "max_steps", "max_cost" or "max_wall_time" if a budget is used up, else None."""
        if steps >= self.budgets.max_steps:
            return "max_steps"
        if cost_usd >= self.budgets.max_cost_usd:
            return "max_cost"
        if elapsed_ns >= self.budgets.max_wall_time_s * 1e9:
            return "max_wall_time"
        return None

    def _execute(self, call: ToolCall) -> ToolRecord:
        """Run one tool call and describe it as a ToolRecord.

        Unknown tool name, or no "command" string in the arguments -> a ToolRecord with
        is_error=True and an observation saying what was wrong. Do not raise: the model
        should read the error and try again.
        Otherwise run_bash in self.cwd, and observation = format_observation(result).
        is_error = result.timed_out. A non-zero exit code is a normal observation, not an error.
        """
        if call.name != "bash":
            return self._error_record(
                call, f"error: unknown tool {call.name!r}; the only tool is bash"
            )
        command = call.arguments.get("command")
        if not isinstance(command, str):
            return self._error_record(
                call, 'error: the bash tool needs a "command" string, for example {"command": "ls"}'
            )

        result = run_bash(command, self.cwd)
        return ToolRecord(
            tool_call_id=call.id,
            tool_name=call.name,
            command=command,
            observation=format_observation(result),
            is_error=result.timed_out,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            truncated_chars=result.truncated_chars,
            started_ns=result.started_ns,
            ended_ns=result.ended_ns,
        )

    def _error_record(self, call: ToolCall, message: str) -> ToolRecord:
        """A ToolRecord for a call that could not run. Nothing ran, so start and end match."""
        now = self.clock()
        return ToolRecord(
            tool_call_id=call.id,
            tool_name=call.name,
            command="",
            observation=message,
            is_error=True,
            exit_code=None,
            timed_out=False,
            truncated_chars=0,
            started_ns=now,
            ended_ns=now,
        )


def _discard(record: SessionStart | StepRecord | SessionEnd) -> None:
    """The trace writer when there is no trajectory file."""


def _tool_results(records: list[ToolRecord]) -> Message:
    """Every result goes back in one user message, in call order."""
    return Message(
        role="user",
        content=[
            ToolResult(tool_call_id=r.tool_call_id, content=r.observation, is_error=r.is_error)
            for r in records
        ],
    )


def _step_record(
    session_id: str,
    step: int,
    sent: int,
    response: ModelResponse,
    appended: bool,
    tools: list[ToolRecord],
) -> StepRecord:
    return StepRecord(
        session_id=session_id,
        step=step,
        parent_step=step - 1 if step > 1 else None,
        request_messages=sent,
        assistant=response.message,
        appended=appended,
        stop_reason=response.stop_reason,
        usage=response.usage,
        cost_usd=response.cost_usd,
        model_started_ns=response.started_ns,
        model_first_token_ns=response.first_token_ns,
        model_ended_ns=response.ended_ns,
        attempts=response.attempts,
        tools=tools,
    )
