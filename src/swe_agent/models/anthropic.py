"""Anthropic Messages API adapter over raw HTTP."""

import json
import time
from collections.abc import Iterable, Sequence
from typing import Any, cast

import httpx

from swe_agent.models.base import ModelError
from swe_agent.models.http import RetryPolicy, post_json, post_stream
from swe_agent.models.pricing import cost_usd, price_for
from swe_agent.models.sse import iter_sse
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
        stream: bool = False,
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
        self._stream = stream  # streaming records the time to the first token
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
        first_token: int | None = None
        if self._stream:
            # Both paths end in the same body, so the parsing below is shared.
            (data, first_token), attempts = post_stream(
                self._client,
                self._url,
                headers=headers,
                body=body | {"stream": True},
                policy=self._retry,
                read=lambda response: accumulate_stream(iter_sse(response.iter_lines())),
            )
        else:
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
            first_token_ns=first_token,
            ended_ns=ended,
            attempts=attempts,
        )


def accumulate_stream(
    events: Iterable[tuple[str | None, str]],
) -> tuple[dict[str, Any], int | None]:
    """Rebuild the non-streaming response body from SSE events.

    Returns (body, time.monotonic_ns() of the first content_block_delta). Raises
    ModelError on an `error` event, or when the stream ends before `message_stop`.

    message_start        -> body, without content yet
    content_block_start  -> a new block at `index`
    content_block_delta  -> text, thinking, signature or a piece of the tool input's JSON
    content_block_stop   -> a tool_use block gets its input parsed
    message_delta        -> stop_reason, and running totals that replace the usage counts
    message_stop         -> done
    """
    body: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    json_parts: dict[int, list[str]] = {}
    first_token: int | None = None
    for _, data in events:
        event = cast(dict[str, Any], _loads(data))
        match event["type"]:
            case "message_start":
                body = event["message"]
            case "content_block_start":
                blocks[event["index"]] = dict(event["content_block"])
                json_parts[event["index"]] = []
            case "content_block_delta":
                if first_token is None:
                    first_token = time.monotonic_ns()
                index = event["index"]
                _apply_delta(blocks[index], json_parts[index], event["delta"])
            case "content_block_stop":
                block = blocks[event["index"]]
                if block["type"] == "tool_use":
                    block["input"] = _loads("".join(json_parts[event["index"]]) or "{}")
            case "message_delta":
                body.update(event["delta"])  # stop_reason, stop_sequence
                # Counts here are totals so far: they replace earlier values, never add to
                # them. A null count is skipped so it cannot erase one from message_start.
                usage = cast(dict[str, Any], event.get("usage") or {})
                body["usage"] = body.get("usage", {}) | {
                    k: v for k, v in usage.items() if v is not None
                }
            case "message_stop":
                body["content"] = [blocks[i] for i in sorted(blocks)]
                return body, first_token
            case "error":
                raise ModelError(f"error event in the stream: {event.get('error')}")
            case _:
                pass  # ping, and event types this adapter does not use
    raise ModelError("the stream ended before message_stop")


def _apply_delta(block: dict[str, Any], json_parts: list[str], delta: dict[str, Any]) -> None:
    match delta["type"]:
        case "text_delta":
            block["text"] = block.get("text", "") + delta["text"]
        case "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta["thinking"]
        case "signature_delta":
            block["signature"] = delta["signature"]
        case "input_json_delta":
            json_parts.append(delta["partial_json"])
        case _:
            pass  # e.g. citations_delta, which this adapter never asks for


def _loads(text: str) -> Any:
    """json.loads, but bad JSON raises ModelError: the agent loop stops cleanly on that."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ModelError(f"bad JSON in the stream: {e}") from e
