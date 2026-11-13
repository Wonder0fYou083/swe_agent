"""OpenAI-compatible Chat Completions adapter over raw HTTP.

One adapter for every server that speaks `POST {base_url}/chat/completions`: OpenAI,
OpenRouter and local servers. Only `base_url`, the key and the max-tokens field name differ.
"""

import json
import time
from collections.abc import Sequence
from typing import Any, cast

import httpx

from swe_agent.models.base import ModelError
from swe_agent.models.http import RetryPolicy, post_json
from swe_agent.models.pricing import cost_usd, price_for
from swe_agent.models.types import (
    Block,
    Endpoint,
    Message,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def to_wire_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "assistant":
            wire.append(_assistant_message(message.content))
            continue
        wire += [_tool_message(b) for b in message.content if isinstance(b, ToolResult)]
        text = "\n".join(b.text for b in message.content if isinstance(b, TextBlock))
        if text:
            wire.append({"role": "user", "content": text})
    return wire


def _assistant_message(content: Sequence[Block]) -> dict[str, Any]:
    text = "\n".join(b.text for b in content if isinstance(b, TextBlock))
    calls = [b for b in content if isinstance(b, ToolCall)]
    # content may be null only next to tool calls
    wire: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "")}
    if calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # The compact form OpenAI itself sends, so a reply goes back byte for byte
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            }
            for call in calls
        ]
    return wire


def _tool_message(result: ToolResult) -> dict[str, Any]:
    """There is no is_error field, so an error says so in its text."""
    content = result.content
    if result.is_error and not content.startswith("error"):
        content = f"error: {content}"
    return {"role": "tool", "tool_call_id": result.tool_call_id, "content": content}


def to_wire_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def parse_arguments(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    if not isinstance(raw, str):
        return {"_invalid_json": raw}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json": raw}
    if not isinstance(value, dict):
        return {"_invalid_json": raw}
    return cast(dict[str, Any], value)


def parse_message(data: dict[str, Any]) -> Message:
    """choices[0].message -> an assistant Message: its text, then its tool calls."""
    content: list[Block] = []
    if data.get("content"):
        content.append(TextBlock(text=data["content"]))
    calls: list[dict[str, Any]] = data.get("tool_calls") or []
    for call in calls:
        function: dict[str, Any] = call["function"]
        content.append(
            ToolCall(
                id=call["id"],
                name=function["name"],
                arguments=parse_arguments(function.get("arguments", "")),
            )
        )
    return Message(role="assistant", content=content)


def parse_usage(data: dict[str, Any] | None) -> Usage:
    """The response's "usage" object -> Usage.

    prompt_tokens INCLUDES cached tokens, unlike Anthropic's input_tokens, so the cached
    ones are subtracted. Without that they would be counted, and priced, twice.
    prompt_tokens_details.cache_write_tokens is reported too, but has no price of its own,
    so those tokens stay in input_tokens at the input price and cache_write_tokens is 0.
    completion_tokens includes reasoning tokens.
    """
    if not data:
        return Usage()
    details: dict[str, Any] = data.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    return Usage(
        input_tokens=int(data.get("prompt_tokens") or 0) - cached,
        output_tokens=int(data.get("completion_tokens") or 0),
        cache_read_tokens=cached,
    )


def parse_stop_reason(value: str | None) -> StopReason:
    match value:
        case "stop":
            return "end_turn"
        case "tool_calls":
            return "tool_use"
        case "length":
            return "max_tokens"
        case "content_filter":
            return "refusal"
        case _:
            return "other"


class OpenAIModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 16_000,
        # OpenAI wants "max_completion_tokens"; many compatible servers want "max_tokens".
        max_tokens_field: str = "max_completion_tokens",
        # Some OpenAI models (gpt-5.6-luna, gpt-6-luna) take tools on /chat/completions
        # only with "none".
        # None leaves the field out, for servers that don't know it.
        reasoning_effort: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        base_url = base_url.rstrip("/")
        self.name = model
        # Caching is automatic on OpenAI; nothing in the request asks for it.
        self.endpoint = Endpoint(provider="openai", base_url=base_url, cache_requested=False)
        self._price = price_for(model)  # fails fast on a model with no price
        self._api_key = api_key
        # Tests pass a client built on httpx.MockTransport; real runs build their own.
        self._client = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))
        self._url = f"{base_url}/chat/completions"
        self._max_tokens = max_tokens
        self._max_tokens_field = max_tokens_field
        self._reasoning_effort = reasoning_effort
        self._retry = retry or RetryPolicy()

    def build_body(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        """No temperature: the server default is the one every run gets."""
        body: dict[str, Any] = {
            "model": self.name,
            "messages": to_wire_messages(system, messages),
            self._max_tokens_field: self._max_tokens,
        }
        if tools:  # OpenAI rejects an empty "tools" list
            body["tools"] = to_wire_tools(tools)
        if self._reasoning_effort is not None:
            body["reasoning_effort"] = self._reasoning_effort
        return body

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        headers = {"content-type": "application/json"}
        if self._api_key:  # local servers often need no key
            headers["authorization"] = f"Bearer {self._api_key}"
        body = self.build_body(system, messages, tools)
        started = time.monotonic_ns()
        data, attempts = post_json(
            self._client, self._url, headers=headers, body=body, policy=self._retry
        )
        ended = time.monotonic_ns()
        try:
            choice = data["choices"][0]
            message = parse_message(choice["message"])
            usage = parse_usage(data.get("usage"))
        except (KeyError, IndexError, TypeError, ValueError) as e:
            # The loop stops cleanly on ModelError; anything else would crash the run.
            shape = json.dumps(data)[:500]
            raise ModelError(f"unexpected response ({type(e).__name__}: {e}): {shape}") from e
        return ModelResponse(
            message=message,
            stop_reason=parse_stop_reason(choice.get("finish_reason")),
            usage=usage,
            cost_usd=cost_usd(usage, self._price),
            model=data.get("model", self.name),
            started_ns=started,
            ended_ns=ended,
            attempts=attempts,
        )
