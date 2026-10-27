import pytest

from swe_agent.models.base import ModelError
from swe_agent.models.fake import ScriptedModel, bash_call, text_reply
from swe_agent.models.types import Message, ProviderBlock, TextBlock, ToolCall, ToolResult, Usage


def test_message_tool_calls_returns_only_tool_calls() -> None:
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    message = Message(role="assistant", content=[TextBlock(text="let me look"), call])

    assert message.tool_calls == [call]
    assert message.text == "let me look"


def test_every_block_type_round_trips_through_json() -> None:
    message = Message(
        role="assistant",
        content=[
            ProviderBlock(
                provider="anthropic", data={"type": "thinking", "thinking": "", "signature": "abc"}
            ),
            TextBlock(text="hi"),
            ToolCall(id="c1", name="bash", arguments={"command": "ls"}),
            ToolResult(tool_call_id="c1", content="ok", is_error=True),
        ],
    )

    assert Message.model_validate_json(message.model_dump_json()) == message


def test_usage_adds_and_counts_prompt_tokens() -> None:
    total = Usage(
        input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4
    ) + Usage(input_tokens=10, output_tokens=20, cache_read_tokens=30, cache_write_tokens=40)

    assert total == Usage(
        input_tokens=11, output_tokens=22, cache_read_tokens=33, cache_write_tokens=44
    )
    assert total.prompt_tokens == 11 + 33 + 44


def test_scripted_model_returns_responses_in_order() -> None:
    first, second = bash_call("ls"), text_reply("done")
    model = ScriptedModel([first, second])

    assert model.complete("sys", [], []) == first
    assert model.complete("sys", [], []) == second


def test_scripted_model_records_a_copy_of_each_request() -> None:
    model = ScriptedModel([text_reply("done")])
    messages = [Message(role="user", content=[TextBlock(text="task")])]

    model.complete("sys", messages, [])
    messages.append(Message(role="user", content=[TextBlock(text="later")]))

    assert model.requests[0].system == "sys"
    assert len(model.requests[0].messages) == 1


def test_scripted_model_record_ignores_later_edits_to_sent_messages() -> None:
    model = ScriptedModel([text_reply("done")])
    messages = [Message(role="user", content=[TextBlock(text="task")])]

    model.complete("sys", messages, [])
    messages[0].content.append(TextBlock(text="edited"))

    assert model.requests[0].messages[0].text == "task"


def test_scripted_model_raises_when_script_runs_out() -> None:
    model = ScriptedModel([])

    with pytest.raises(ModelError):
        model.complete("sys", [], [])


def test_bash_call_numbers_its_tool_calls() -> None:
    response = bash_call("ls", "pwd", text="two commands")

    assert [c.id for c in response.message.tool_calls] == ["call_1", "call_2"]
    assert [c.arguments["command"] for c in response.message.tool_calls] == ["ls", "pwd"]
    assert response.stop_reason == "tool_use"
