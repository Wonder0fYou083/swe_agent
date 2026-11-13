import httpx

from swe_agent.models.sse import iter_sse

STREAM = (
    "event: message_start\n"
    'data: {"type":"message_start"}\n'
    "\n"
    ": keep-alive\n"
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0}\n'
    "\n"
)


def test_parses_events_from_lines() -> None:
    assert list(iter_sse(STREAM.splitlines())) == [
        ("message_start", '{"type":"message_start"}'),
        ("content_block_delta", '{"type":"content_block_delta","index":0}'),
    ]


def test_works_on_an_httpx_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=STREAM.encode(), headers={"content-type": "text/event-stream"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with client.stream("POST", "https://example.test/v1/messages", json={}) as response:
        events = list(iter_sse(response.iter_lines()))

    assert [name for name, _ in events] == ["message_start", "content_block_delta"]


def test_openai_style_data_only_events() -> None:
    lines = ['data: {"id":"1"}', "", "data: [DONE]", ""]

    assert list(iter_sse(lines)) == [(None, '{"id":"1"}'), (None, "[DONE]")]
