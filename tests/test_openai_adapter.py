import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from swe_agent.core.agent import Agent
from swe_agent.models.base import ModelError
from swe_agent.models.openai import (
    OpenAIModel,
    parse_arguments,
    parse_stop_reason,
    parse_usage,
    to_wire_messages,
)
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

MODEL = "gpt-6-luna"

# One real /chat/completions response (gpt-6-luna, reasoning_effort "none", 2026-11-12):
# the toy-repo task with the bash tool. Its first move is one tool call.
RECORDED: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data" / "openai_tool_call.json").read_text(encoding="utf-8")
)
RECORDED_COMMAND = "ls -la && find . -maxdepth 2 -type f | head -80 && git status --short"


def _response(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _model(handler: Any, **kwargs: Any) -> OpenAIModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIModel(MODEL, "test-key", client=client, **kwargs)


def test_to_wire_messages_maps_each_block_type() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="fix it")]),
        Message(
            role="assistant",
            content=[
                ProviderBlock(provider="anthropic", data={"type": "thinking"}),
                TextBlock(text="Two commands."),
                ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
                ToolCall(id="call_2", name="bash", arguments={"command": "pwd"}),
            ],
        ),
        Message(
            role="user",
            content=[
                ToolResult(tool_call_id="call_1", content="a.py"),
                ToolResult(tool_call_id="call_2", content="/repo"),
            ],
        ),
    ]

    assert to_wire_messages("be helpful", messages) == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "Two commands.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
        {"role": "tool", "tool_call_id": "call_2", "content": "/repo"},
    ]


def test_tool_results_come_before_user_text_in_the_same_message() -> None:
    message = Message(
        role="user",
        content=[TextBlock(text="also check b.py"), ToolResult(tool_call_id="c", content="ok")],
    )

    assert [m["role"] for m in to_wire_messages("s", [message])] == ["system", "tool", "user"]


def test_an_assistant_message_with_only_tool_calls_sends_null_content() -> None:
    message = Message(role="assistant", content=[ToolCall(id="c", name="bash", arguments={})])

    assert to_wire_messages("s", [message])[1]["content"] is None


@pytest.mark.parametrize(
    ("content", "sent"),
    [
        ("timed out; the command was killed", "error: timed out; the command was killed"),
        ("error: unknown tool 'x'", "error: unknown tool 'x'"),  # no second prefix
    ],
)
def test_an_error_result_says_so_in_its_text(content: str, sent: str) -> None:
    message = Message(
        role="user", content=[ToolResult(tool_call_id="c", content=content, is_error=True)]
    )

    assert to_wire_messages("s", [message])[1]["content"] == sent


@pytest.mark.parametrize("raw", ["{not json", "", '["ls"]', '"ls"'])
def test_arguments_that_are_not_a_json_object_are_kept_without_raising(raw: str) -> None:
    assert parse_arguments(raw) == {"_invalid_json": raw}


def test_arguments_sent_as_an_object_are_accepted() -> None:
    assert parse_arguments({"command": "ls"}) == {"command": "ls"}


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "model not found"}},  # some servers answer 200 with an error
        {"choices": []},
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "bash"}}]}}]},  # no id
    ],
)
def test_a_malformed_response_raises_model_error(body: dict[str, Any]) -> None:
    with pytest.raises(ModelError, match="unexpected response"):
        _model(lambda _: httpx.Response(200, json=body)).complete("s", [], [])


def test_parse_usage_subtracts_cached_tokens_from_prompt_tokens() -> None:
    usage = {
        "prompt_tokens": 3200,
        "completion_tokens": 80,
        "prompt_tokens_details": {"cached_tokens": 2000, "cache_write_tokens": 0},
    }

    assert parse_usage(usage) == Usage(input_tokens=1200, output_tokens=80, cache_read_tokens=2000)


def test_parse_usage_treats_missing_fields_as_zero() -> None:
    assert parse_usage(None) == Usage()
    assert parse_usage({"prompt_tokens": 5, "completion_tokens": 7}) == Usage(
        input_tokens=5, output_tokens=7
    )
    assert parse_usage(
        {"prompt_tokens": 5, "completion_tokens": 7, "prompt_tokens_details": None}
    ) == Usage(input_tokens=5, output_tokens=7)


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("function_call", "other"),
        (None, "other"),
    ],
)
def test_parse_stop_reason(finish_reason: str | None, stop_reason: str) -> None:
    assert parse_stop_reason(finish_reason) == stop_reason


def test_complete_sends_the_expected_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=RECORDED)

    _model(handler).complete(
        "be helpful", [Message(role="user", content=[TextBlock(text="hi")])], [BASH_SPEC]
    )

    request = seen[0]
    body = json.loads(request.content)
    assert request.url == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == MODEL
    assert body["messages"][0] == {"role": "system", "content": "be helpful"}
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": BASH_SPEC.description,
                "parameters": BASH_SPEC.input_schema,
            },
        }
    ]
    assert body["max_completion_tokens"] >= 16_000
    assert not {"temperature", "top_p", "max_tokens", "reasoning_effort"} & body.keys()


def test_reasoning_effort_is_sent_only_when_set() -> None:
    model = _model(lambda _: httpx.Response(200, json=RECORDED), reasoning_effort="none")

    assert model.build_body("s", [], [BASH_SPEC])["reasoning_effort"] == "none"


def test_complete_parses_the_recorded_response() -> None:
    response = _model(lambda _: httpx.Response(200, json=RECORDED)).complete("s", [], [])

    assert response.stop_reason == "tool_use"
    assert response.message.role == "assistant"
    assert response.message.text == ""  # content is null next to the tool call
    assert response.message.tool_calls == [
        ToolCall(
            id="call_KkJij3q48YB6HlGj505iMqrU", name="bash", arguments={"command": RECORDED_COMMAND}
        )
    ]
    assert response.usage == Usage(input_tokens=433, output_tokens=39)
    assert response.cost_usd == pytest.approx((433 * 0.10 + 39 * 0.50) / 1e6)
    assert response.model == MODEL
    assert response.ended_ns >= response.started_ns


def test_a_response_round_trips_back_to_the_same_wire_message() -> None:
    response = _model(lambda _: httpx.Response(200, json=RECORDED)).complete("s", [], [])

    sent = to_wire_messages("s", [response.message])[1]
    received = RECORDED["choices"][0]["message"]
    assert sent == {k: received[k] for k in ("role", "content", "tool_calls")}


def test_a_local_server_setup_changes_only_the_url_key_and_max_tokens_field() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_response({"role": "assistant", "content": "hi"}, "stop"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAIModel(
        MODEL,
        "",
        client=client,
        base_url="http://localhost:11434/v1/",
        max_tokens_field="max_tokens",
    )
    model.complete("s", [], [])

    body = json.loads(seen[0].content)
    assert seen[0].url == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body
    assert "tools" not in body  # an empty list is rejected, so it is left out
    assert model.endpoint == Endpoint(
        provider="openai", base_url="http://localhost:11434/v1", cache_requested=False
    )


def test_the_agent_loop_runs_end_to_end_on_this_adapter(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []
    replies = [
        _response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "echo hi"}'},
                    }
                ],
            },
            "tool_calls",
        ),
        _response({"role": "assistant", "content": "Done."}, "stop"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(bodies) - 1])

    result = Agent(_model(handler), cwd=tmp_path, system_prompt="s").run("say hi")

    assert result.stop == "submitted"
    assert result.final_text == "Done."
    tool_message = bodies[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"].startswith("exit code: 0\n<output>\nhi\n")
