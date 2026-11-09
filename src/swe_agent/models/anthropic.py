"""Anthropic Messages API adapter over raw HTTP."""

import time
from collections.abc import Iterable, Sequence
from typing import Any

import httpx

from swe_agent.models.http import RetryPolicy, post_json
from swe_agent.models.pricing import cost_usd, price_for
from swe_agent.models.types import (
    Block,
    Endpoint,
    Message,
    ModelResponse,
    ProviderBlock,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

API_VERSION = "2023-06-01"


def to_wire_block(block: Block) -> dict[str, Any]:
    ans: dict[str, Any] = {}
    match block:
        case TextBlock():
            ans["type"] = "text"
            ans["text"] = block.text
        case ToolCall():
            ans["type"] = "tool_use"
            ans["id"] = block.id
            ans["name"] = block.name
            ans["input"] = block.arguments
        case ToolResult():
            ans["type"] = "tool_result"
            ans["tool_use_id"] = block.tool_call_id
            ans["content"] = block.content
            ans["is_error"] = block.is_error
        case ProviderBlock():
            ans = block.data
    return ans


def to_wire_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """[{"role": m.role, "content": [to_wire_block(b) for b in m.content]}, ...]"""
    ans: list[dict[str, Any]] = []
    for m in messages:
        ans.append(
            {
                "role": m.role,
                "content": [to_wire_block(b) for b in m.content],
            }
        )
    return ans


def to_wire_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """[{"name": ..., "description": ..., "input_schema": ...}, ...]"""
    ans: list[dict[str, Any]] = []
    for t in tools:
        ans.append(
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
        )
    return ans


def from_wire_block(data: dict[str, Any]) -> Block:
    match data["type"]:
        case "text":
            return TextBlock(text=data["text"])
        case "tool_use":
            return ToolCall(id=data["id"], name=data["name"], arguments=data["input"])
        case _:
            return ProviderBlock(provider="anthropic", data=data)


def parse_usage(data: dict[str, Any]) -> Usage:
    """The response's "usage" object -> Usage.

    input_tokens                -> input_tokens (this EXCLUDES cached tokens)
    output_tokens               -> output_tokens (includes thinking)
    cache_read_input_tokens     -> cache_read_tokens
    cache_creation_input_tokens -> cache_write_tokens
    The cache fields can be missing or null: treat both as 0.
    """
    return Usage(
        input_tokens=int(data["input_tokens"]),
        output_tokens=int(data["output_tokens"]),
        cache_read_tokens=int(data.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(data.get("cache_creation_input_tokens") or 0),
    )


def parse_stop_reason(value: str | None) -> StopReason:
    """end_turn, tool_use, max_tokens and refusal map to themselves; anything else -> "other"."""
    match value:
        case "end_turn" | "tool_use" | "max_tokens" | "refusal":
            return value
        case _:
            return "other"


class AnthropicModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 16_000,
        cache: bool = True,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.name = model
        self.endpoint = Endpoint(provider="anthropic", base_url=base_url, cache_requested=cache)
        self._price = price_for(model)  # fails fast on a model with no price
        self._api_key = api_key
        # Tests pass a client built on httpx.MockTransport; real runs build their own.
        self._client = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))
        self._url = f"{base_url}/v1/messages"
        self._max_tokens = max_tokens
        self._cache = cache
        self._retry = retry or RetryPolicy()

    def build_body(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        """The JSON request body.

        Keys: model, max_tokens, system, messages (to_wire_messages), tools (to_wire_tools).
        If self._cache: add top-level "cache_control": {"type": "ephemeral"} (automatic
        prompt caching; without it the cache counters stay 0).
        Never send temperature, top_p or top_k: Haiku 5.5 rejects non-default values, and
        API version 2026-10-21 removes them.
        """
        return (
            {
                "cache_control": {"type": "ephemeral"},
                "model": self.name,
                "max_tokens": self._max_tokens,
                "system": system,
                "messages": to_wire_messages(messages),
                "tools": to_wire_tools(tools),
            }
            if self._cache
            else {
                "model": self.name,
                "max_tokens": self._max_tokens,
                "system": system,
                "messages": to_wire_messages(messages),
                "tools": to_wire_tools(tools),
            }
        )

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        """Pseudocode (you will import time, post_json and cost_usd):

        headers = x-api-key, anthropic-version: API_VERSION, content-type: application/json
        started = time.monotonic_ns()
        data, attempts = post_json(self._client, self._url, headers=..., body=build_body(...),
                                   policy=self._retry)
        ended = time.monotonic_ns()
        usage = parse_usage(data["usage"])
        return ModelResponse(
            message=assistant Message from [from_wire_block(b) for b in data["content"]],
            stop_reason=parse_stop_reason(data.get("stop_reason")),
            usage=usage, cost_usd=cost_usd(usage, self._price), model=data.get("model", ...),
            started_ns=started, ended_ns=ended, attempts=attempts)
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        body = self.build_body(system, messages, tools)
        started = time.monotonic_ns()
        data, attempts = post_json(
            self._client, self._url, headers=headers, body=body, policy=self._retry
        )
        ended = time.monotonic_ns()
        usage = parse_usage(data["usage"])
        return ModelResponse(
            message=Message(
                role="assistant",
                content=[from_wire_block(b) for b in data["content"]],
            ),
            stop_reason=parse_stop_reason(data.get("stop_reason")),
            usage=usage,
            cost_usd=cost_usd(usage, self._price),
            model=data.get("model", self.name),
            started_ns=started,
            ended_ns=ended,
            attempts=attempts,
        )


def accumulate_stream(
    events: Iterable[tuple[str | None, str]],
) -> tuple[dict[str, Any], int | None]:
    """Rebuild the non-streaming response body from SSE events.  [Issue 10]

    Returns (body, time.monotonic_ns() of the first content_block_delta).
    The event table and pseudocode are in 10-streaming.md.
    """
    raise NotImplementedError
