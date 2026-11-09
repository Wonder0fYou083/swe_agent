import httpx
import pytest

from swe_agent.models.base import ModelError
from swe_agent.models.http import RetryPolicy, post_json

POLICY = RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=30.0)


def _post(
    responses: list[httpx.Response | Exception],
) -> tuple[list[float], dict[str, object], int]:
    sleeps: list[float] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.Client(transport=httpx.MockTransport(handler))
    data, attempts = post_json(
        client, "https://example.test/v1", headers={}, body={}, policy=POLICY, sleep=sleeps.append
    )
    return sleeps, data, attempts


def test_returns_the_first_success() -> None:
    sleeps, data, attempts = _post([httpx.Response(200, json={"ok": True})])

    assert data == {"ok": True}
    assert attempts == 1
    assert sleeps == []


def test_retries_overloaded_then_succeeds() -> None:
    sleeps, data, attempts = _post([httpx.Response(529), httpx.Response(200, json={"ok": True})])

    assert data == {"ok": True}
    assert attempts == 2
    assert len(sleeps) == 1


def test_honours_retry_after() -> None:
    sleeps, _, _ = _post(
        [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(200, json={})]
    )

    assert sleeps == [3.0]


def test_retries_connection_errors() -> None:
    _, _, attempts = _post([httpx.ConnectError("refused"), httpx.Response(200, json={})])

    assert attempts == 2


def test_does_not_retry_client_errors() -> None:
    with pytest.raises(ModelError, match="400"):
        _post([httpx.Response(400, text="bad"), httpx.Response(200, json={})])


def test_gives_up_after_max_attempts() -> None:
    with pytest.raises(ModelError, match="3 attempts"):
        _post([httpx.Response(500), httpx.Response(500), httpx.Response(500)])
