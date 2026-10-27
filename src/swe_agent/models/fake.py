"""A scripted model for tests: no network, fully deterministic."""

from collections.abc import Sequence

from pydantic import BaseModel

from swe_agent.models.base import ModelError
from swe_agent.models.types import (
    Message,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolSpec,
    Usage,
)


class ScriptedRequest(BaseModel):
    system: str
    messages: list[Message]
    tools: list[ToolSpec]


class ScriptedModel:
    def __init__(self, responses: Sequence[ModelResponse], name: str = "scripted") -> None:
        self.name = name
        self._responses = list(responses)
        self.requests: list[ScriptedRequest] = []

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        self.requests.append(
            ScriptedRequest(
                system=system,
                messages=[message.model_copy(deep=True) for message in messages],
                tools=list(tools),
            )
        )
        call = len(self.requests)
        if call > len(self._responses):
            raise ModelError(
                f"script has {len(self._responses)} responses, but this is call {call}"
            )
        return self._responses[call - 1]


def _response(message: Message, stop_reason: StopReason, cost_usd: float) -> ModelResponse:
    return ModelResponse(
        message=message,
        stop_reason=stop_reason,
        usage=Usage(),
        cost_usd=cost_usd,
        model="scripted",
        started_ns=0,
        ended_ns=0,
    )


def text_reply(text: str, *, cost_usd: float = 0.0) -> ModelResponse:
    """A final answer: text and no tool call."""
    return _response(
        Message(role="assistant", content=[TextBlock(text=text)]), "end_turn", cost_usd
    )


def bash_call(
    *commands: str,
    text: str = "",
    cost_usd: float = 0.0,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    """One assistant turn that calls `bash` once per command, with ids call_1, call_2, ..."""
    content: list[TextBlock | ToolCall] = [TextBlock(text=text)] if text else []
    content += [
        ToolCall(id=f"call_{i}", name="bash", arguments={"command": command})
        for i, command in enumerate(commands, start=1)
    ]
    return _response(Message(role="assistant", content=list(content)), stop_reason, cost_usd)
