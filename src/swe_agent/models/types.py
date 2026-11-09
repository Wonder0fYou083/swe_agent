"""Provider-neutral message types. Each adapter translates these to and from its wire format."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


class ProviderBlock(BaseModel):
    type: Literal["provider"] = "provider"
    provider: str
    data: dict[str, Any]


Block = Annotated[TextBlock | ToolCall | ToolResult | ProviderBlock, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[Block]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [block for block in self.content if isinstance(block, ToolCall)]

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.content if isinstance(block, TextBlock))


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0  # uncached input, billed at the full input price
    output_tokens: int = 0  # includes thinking tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Every input token the model saw, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


class ModelResponse(BaseModel):
    message: Message  # role == "assistant"
    stop_reason: StopReason
    usage: Usage
    cost_usd: float
    model: str
    started_ns: int  # time.monotonic_ns() just before the request was sent
    first_token_ns: int | None = None  # streaming only
    ended_ns: int  # time.monotonic_ns() once the full response was read
    attempts: int = 1  # 1 + number of retries


class Endpoint(BaseModel):
    """Where a model's requests go and how they are sent. Recorded once per session."""

    provider: str  # "anthropic", "openai" or "scripted"
    base_url: str | None  # None when nothing goes over the network
    cache_requested: bool  # what we asked for; the Usage cache counters show what happened
