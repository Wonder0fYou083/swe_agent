import json
from typing import Any

import httpx
import pytest

from swe_agent.models.anthropic import (
    AnthropicModel,
    from_wire_block,
    parse_usage,
    to_wire_messages,
)
from swe_agent.models.base import ModelError
from swe_agent.models.http import RetryPolicy
from swe_agent.models.types import (
    Endpoint,
    Message,
    ProviderBlock,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from swe_agent.tools.bash import BASH_SPEC

THINKING = {"type": "thinking", "thinking": "", "signature": "EqQBCgIYAhIM"}

# Shaped like a real /v1/messages response. Replace it with one you record yourself.
TOOL_USE_RESPONSE: dict[str, Any] = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        THINKING,
        {"type": "text", "text": "Let me run the tests."},
        {"type": "tool_use", "id": "toolu_01", "name": "bash", "input": {"command": "pytest -q"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 1200,
        "output_tokens": 80,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 0,
    },
}


def test_to_wire_messages_maps_each_block_type() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="fix it")]),
        Message(
            role="assistant",
            content=[
                ProviderBlock(provider="anthropic", data=THINKING),
                ToolCall(id="toolu_01", name="bash", arguments={"command": "ls"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResult(tool_call_id="toolu_01", content="a.py", is_error=False)],
        ),
    ]

    assert to_wire_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "fix it"}]},
        {
            "role": "assistant",
            "content": [
                THINKING,
                {"type": "tool_use", "id": "toolu_01", "name": "bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": "a.py",
                    "is_error": False,
                }
            ],
        },
    ]


def test_unknown_blocks_become_provider_blocks_kept_verbatim() -> None:
    block = from_wire_block(THINKING)

    assert block == ProviderBlock(provider="anthropic", data=THINKING)


def test_parse_usage_treats_missing_cache_counters_as_zero() -> None:
    assert parse_usage({"input_tokens": 5, "output_tokens": 7}) == Usage(
        input_tokens=5, output_tokens=7
    )
    assert parse_usage(
        {"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": None}
    ) == Usage(input_tokens=5, output_tokens=7)


def _model(handler: Any, **kwargs: Any) -> AnthropicModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AnthropicModel("claude-haiku-5-5", "test-key", client=client, **kwargs)


def test_complete_sends_the_expected_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TOOL_USE_RESPONSE)

    _model(handler).complete(
        "be helpful", [Message(role="user", content=[TextBlock(text="hi")])], [BASH_SPEC]
    )

    request = seen[0]
    body = json.loads(request.content)
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-5-5"
    assert body["system"] == "be helpful"
    assert body["tools"][0]["name"] == "bash"
    assert body["tools"][0]["input_schema"] == BASH_SPEC.input_schema
    assert body["max_tokens"] >= 16_000
    assert not {"temperature", "top_p", "top_k"} & body.keys()


def test_complete_parses_the_response() -> None:
    response = _model(lambda _: httpx.Response(200, json=TOOL_USE_RESPONSE)).complete("s", [], [])

    assert response.stop_reason == "tool_use"
    assert response.message.role == "assistant"
    assert response.message.content[0] == ProviderBlock(provider="anthropic", data=THINKING)
    assert response.message.tool_calls == [
        ToolCall(id="toolu_01", name="bash", arguments={"command": "pytest -q"})
    ]
    assert response.usage == Usage(input_tokens=1200, output_tokens=80, cache_write_tokens=2000)
    assert response.cost_usd == pytest.approx((1200 * 0.10 + 80 * 0.50 + 2000 * 0.125) / 1e6)
    assert response.ended_ns >= response.started_ns


def test_a_response_round_trips_back_to_the_same_wire_content() -> None:
    response = _model(lambda _: httpx.Response(200, json=TOOL_USE_RESPONSE)).complete("s", [], [])

    assert to_wire_messages([response.message])[0]["content"] == TOOL_USE_RESPONSE["content"]


def test_a_bad_request_raises_model_error_without_retrying() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error"}}
        )

    with pytest.raises(ModelError):
        _model(handler, retry=RetryPolicy(max_attempts=3, base_delay_s=0)).complete("s", [], [])
    assert len(calls) == 1


@pytest.mark.parametrize("cache", [True, False])
def test_endpoint_records_what_was_asked_for(cache: bool) -> None:
    model = _model(lambda _: httpx.Response(200, json=TOOL_USE_RESPONSE), cache=cache)

    assert model.endpoint == Endpoint(
        provider="anthropic", base_url="https://api.anthropic.com", cache_requested=cache
    )
    assert ("cache_control" in model.build_body("s", [], [])) is cache
