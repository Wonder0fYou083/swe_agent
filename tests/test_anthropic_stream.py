import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from swe_agent.models.anthropic import AnthropicModel, accumulate_stream, to_wire_messages
from swe_agent.models.base import ModelError
from swe_agent.models.http import RetryPolicy
from swe_agent.models.sse import iter_sse
from swe_agent.models.types import ProviderBlock, TextBlock, ToolCall, Usage

THINKING = {"type": "thinking", "thinking": "", "signature": "EqQBCgIYAhIM"}


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


# The streamed form of the same message as TOOL_USE_RESPONSE in test_anthropic_adapter.py.
STREAM = "".join(
    [
        _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 1200,
                        "output_tokens": 1,
                        "cache_creation_input_tokens": 2000,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
        ),
        _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM"},
            },
        ),
        _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Let me run"},
            },
        ),
        _event("ping", {"type": "ping"}),
        _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": " the tests."},
            },
        ),
        _event("content_block_stop", {"type": "content_block_stop", "index": 1}),
        _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "bash",
                    "input": {},
                },
            },
        ),
        _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": '},
            },
        ),
        _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '"pytest -q"}'},
            },
        ),
        _event("content_block_stop", {"type": "content_block_stop", "index": 2}),
        _event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 80},
            },
        ),
        _event("message_stop", {"type": "message_stop"}),
    ]
)


def test_stream_rebuilds_the_same_body_as_a_non_streaming_call() -> None:
    body, first_token_ns = accumulate_stream(iter_sse(STREAM.splitlines()))

    assert body["content"] == [
        THINKING,
        {"type": "text", "text": "Let me run the tests."},
        {"type": "tool_use", "id": "toolu_01", "name": "bash", "input": {"command": "pytest -q"}},
    ]
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["input_tokens"] == 1200
    assert body["usage"]["output_tokens"] == 80  # message_delta's count replaces message_start's
    assert body["usage"]["cache_creation_input_tokens"] == 2000
    assert first_token_ns is not None


ERROR_EVENT = _event(
    "error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
)
CUT_OFF = STREAM.rsplit("event: message_stop", 1)[0]


@pytest.mark.parametrize(
    ("stream", "match"),
    [
        (STREAM.split("event: content_block_start")[0] + ERROR_EVENT, "overloaded"),
        (CUT_OFF, "before message_stop"),
    ],
)
def test_an_error_event_or_a_cut_off_stream_raises_model_error(stream: str, match: str) -> None:
    with pytest.raises(ModelError, match=match):
        accumulate_stream(iter_sse(stream.splitlines()))


def _sse(text: str) -> httpx.Response:
    return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})


def _model(handler: Any) -> AnthropicModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    retry = RetryPolicy(max_attempts=3, base_delay_s=0)
    return AnthropicModel("claude-haiku-5-5", "test-key", client=client, stream=True, retry=retry)


def test_streamed_complete_gives_the_same_response_as_a_non_streaming_call() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _sse(STREAM)

    response = _model(handler).complete("s", [], [])

    assert json.loads(seen[0].content)["stream"] is True
    assert response.message.content == [
        ProviderBlock(provider="anthropic", data=THINKING),
        TextBlock(text="Let me run the tests."),
        ToolCall(id="toolu_01", name="bash", arguments={"command": "pytest -q"}),
    ]
    assert to_wire_messages([response.message])[0]["content"][0] == THINKING  # signature kept
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(input_tokens=1200, output_tokens=80, cache_write_tokens=2000)
    assert response.first_token_ns is not None
    assert response.started_ns <= response.first_token_ns <= response.ended_ns
    assert response.attempts == 1


def _breaks_mid_stream() -> Iterator[bytes]:
    yield STREAM[: len(STREAM) // 2].encode()
    raise httpx.ReadError("connection reset")


@pytest.mark.parametrize(
    "first",
    [httpx.Response(529, text="overloaded"), httpx.Response(200, content=_breaks_mid_stream())],
)
def test_streaming_retries_a_failure_before_or_during_the_stream(first: httpx.Response) -> None:
    queue = [first, _sse(STREAM)]

    response = _model(lambda _: queue.pop(0)).complete("s", [], [])

    assert response.attempts == 2
    assert response.message.tool_calls[0].arguments == {"command": "pytest -q"}


def test_a_bad_request_while_streaming_raises_model_error_without_retrying() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request"}})

    with pytest.raises(ModelError, match=r"HTTP 400: .*invalid_request"):
        _model(handler).complete("s", [], [])
    assert len(calls) == 1


def test_an_error_event_while_streaming_is_not_retried() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _sse(STREAM.split("event: content_block_start")[0] + ERROR_EVENT)

    with pytest.raises(ModelError, match="overloaded"):
        _model(handler).complete("s", [], [])
    assert len(calls) == 1
