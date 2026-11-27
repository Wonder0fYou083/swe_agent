"""The agent loop: call the model, run its tool calls, send the results back, repeat."""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import uuid4

from pydantic import BaseModel

from swe_agent import __version__
from swe_agent.models.base import Model, ModelError
from swe_agent.models.types import (
    Block,
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
    TurnEnd,
    UserTurn,
)
from swe_agent.tools.bash import BASH_SPEC, BashResult, format_observation, run_bash

TraceWriter = Callable[[SessionStart | UserTurn | StepRecord | TurnEnd | SessionEnd], None]
# run_bash (on the host) or run_bash_in_sandbox (in a container): same signature.
RunCommand = Callable[[str, Path], BashResult]


class RunResult(BaseModel):
    session_id: str
    stop: StopKind
    steps: int
    cost_usd: float
    final_text: str
    trajectory: Path | None
    error: str | None = None


class TurnResult(BaseModel):
    stop: StopKind
    steps: int
    cost_usd: float
    usage: Usage
    final_text: str
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
        run_command: RunCommand = run_bash,
        context_window: int | None = None,  # the model's limit in tokens; None: unknown
    ) -> None:
        self.model = model  # also supplies the trace's model name and endpoint
        self.cwd = cwd
        self.run_command = run_command  # the CLI passes run_bash_in_sandbox by default
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.budgets = budgets or Budgets()
        self.trajectory_dir = trajectory_dir  # None: no trajectory file
        self.on_step = on_step  # the CLI uses this to print each step live
        self.clock = clock  # tests pass a fake clock to test max_wall_time
        self.tools = [BASH_SPEC]  # one list, so the request and SessionStart always agree
        self.context_window = context_window

    def run(self, task: str) -> RunResult:
        """One task as a one-turn session: call the model, run its tool calls, send the
        results back, until a stop condition."""
        session_id = uuid4().hex
        path = self.trajectory_path(session_id)
        t0 = self.clock()
        logger = TrajectoryLogger(path) if path is not None else None
        log: TraceWriter = logger.write if logger is not None else _discard
        try:
            log(self.session_start(session_id, task, t0))

            messages = [Message(role="user", content=[TextBlock(text=task)])]
            out = self.run_turn(session_id, messages, log, turn=1, steps_before=0)

            log(
                SessionEnd(
                    session_id=session_id,
                    stop=out.stop,
                    steps=out.steps,
                    turns=1,
                    cost_usd=out.cost_usd,
                    usage=out.usage,
                    ended_ns=self.clock(),
                    error=out.error,
                )
            )
        finally:
            if logger is not None:
                logger.close()

        return RunResult(
            session_id=session_id,
            stop=out.stop,
            steps=out.steps,
            cost_usd=out.cost_usd,
            final_text=out.final_text,
            trajectory=path,
            error=out.error,
        )

    def session(self) -> "Session":
        """Phase 4: a conversation that keeps its history from one task to the next.

        with agent.session() as session:
            session.send("fix the failing test")
            session.send("now add a test for the empty list")
        """
        return Session(self)

    def run_turn(
        self,
        session_id: str,
        messages: list[Message],
        log: TraceWriter,
        *,
        turn: int,
        steps_before: int,
    ) -> TurnResult:
        """runs a turn"""
        stop: StopKind | None = None
        cost_usd = 0.0
        steps = 0
        t0 = self.clock()
        usage = Usage()
        final_text, error = "", None

        while stop is None:
            stop = self._check_budgets(steps, cost_usd, self.clock() - t0)
            if stop is not None:
                break

            sent = len(messages)
            try:
                response = self.model.complete(
                    self.system_prompt, messages=messages, tools=self.tools
                )
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

            step = _step_record(
                session_id, turn, steps_before + steps, sent, response, appended, records
            )
            log(step)
            if self.on_step is not None:
                self.on_step(step)

        return TurnResult(
            stop=stop,
            steps=steps,
            cost_usd=cost_usd,
            usage=usage,
            final_text=final_text,
            error=error,
        )

    def trajectory_path(self, session_id: str) -> Path | None:
        """Creates a path to write the trajectory for the current session"""
        if self.trajectory_dir is None:
            return None
        return self.trajectory_dir / f"{session_id}.jsonl"

    def session_start(self, session_id: str, task: str, t0: int) -> SessionStart:
        """Starts a session"""
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
            context_window=self.context_window,
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
        """Run one tool call and describe it as a ToolRecord."""
        if call.name != "bash":
            return self._error_record(
                call, f"error: unknown tool {call.name!r}; the only tool is bash"
            )
        command = call.arguments.get("command")
        if not isinstance(command, str):
            return self._error_record(
                call, 'error: the bash tool needs a "command" string, for example {"command": "ls"}'
            )

        result = self.run_command(command, self.cwd)
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


class Session:
    """
    One session:
        turn 1 (start_ns = session.start):
            step 1
            step 2
            ...
        turn 1 end

        turn 2:
        ...
        turn 2 end:

        ...
    session ends
    """

    session_id: str
    turns: int
    steps: int
    cost_usd: float  # per turn
    trajectory: Path | None

    def __init__(self, agent: Agent) -> None:
        self.session_id = uuid4().hex
        self.turns = 0
        self.steps = 0
        self.cost_usd = 0.0
        self.trajectory = agent.trajectory_path(self.session_id)

        self.agent = agent
        self.messages: list[Message] = []
        self.usage = Usage()
        self.logger: TrajectoryLogger | None = None
        self.last: TurnResult | None = None

    def send(self, text: str) -> RunResult:
        """Run one turn on `text` and return this turn's result."""
        self.turns += 1
        if self.turns == 1 and self.trajectory is not None:
            self.logger = TrajectoryLogger(self.trajectory)
        log: TraceWriter = self.logger.write if self.logger is not None else _discard
        t0 = self.agent.clock()

        if self.turns == 1:
            log(self.agent.session_start(self.session_id, text, t0))
        else:
            log(UserTurn(session_id=self.session_id, turn=self.turns, text=text, started_ns=t0))

        add_user_text(self.messages, text)

        out = self.agent.run_turn(
            self.session_id, self.messages, log, turn=self.turns, steps_before=self.steps
        )

        ended_ns = self.agent.clock()

        log(
            TurnEnd(
                session_id=self.session_id,
                turn=self.turns,
                stop=out.stop,
                steps=out.steps,
                usage=out.usage,
                error=out.error,
                ended_ns=ended_ns,
                cost_usd=out.cost_usd,
            )
        )

        self.steps += out.steps
        self.cost_usd += out.cost_usd
        self.usage += out.usage
        self.last = out

        return RunResult(
            session_id=self.session_id,
            stop=out.stop,
            steps=out.steps,
            cost_usd=out.cost_usd,
            final_text=out.final_text,
            trajectory=self.trajectory,
            error=out.error,
        )

    def close(self) -> None:
        """Finish the trajectory."""
        if self.logger is None:
            return

        ended_ns = self.agent.clock()
        stop: StopKind = self.last.stop if self.last is not None else "error"
        error = self.last.error if self.last is not None else "the first turn did not finish"

        self.logger.write(
            SessionEnd(
                session_id=self.session_id,
                turns=self.turns,
                stop=stop,
                steps=self.steps,
                cost_usd=self.cost_usd,
                usage=self.usage,
                ended_ns=ended_ns,
                error=error,
            )
        )

        self.logger.close()
        self.logger = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def add_user_text(messages: list[Message], text: str) -> None:
    """
    messages so far
    """
    if len(messages) == 0 or messages[-1].role == "assistant":
        content: list[Block] = []
        content.append(TextBlock(text=text))
        messages.append(Message(role="user", content=content))
    else:
        messages[-1].content.append(TextBlock(text=text))


def _discard(record: SessionStart | UserTurn | StepRecord | TurnEnd | SessionEnd) -> None:
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
    turn: int,
    step: int,
    sent: int,
    response: ModelResponse,
    appended: bool,
    tools: list[ToolRecord],
) -> StepRecord:
    """Records one step and returns a StepRecord"""
    return StepRecord(
        session_id=session_id,
        turn=turn,
        step=step,
        parent_step=step - 1 if step > 1 else None,
        request_messages=sent,
        assistant=response.message,
        appended=appended,
        stop_reason=response.stop_reason,
        usage=response.usage,
        context_tokens=response.usage.prompt_tokens + response.usage.output_tokens,
        cost_usd=response.cost_usd,
        model_started_ns=response.started_ns,
        model_first_token_ns=response.first_token_ns,
        model_ended_ns=response.ended_ns,
        attempts=response.attempts,
        tools=tools,
    )
