"""POST with retry and backoff, shared by every adapter."""

import random
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from swe_agent.models.base import ModelError

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class RetryPolicy(BaseModel):
    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


def backoff_delay(attempt: int, policy: RetryPolicy, retry_after: str | None) -> float:
    def calculate_ceiling() -> float:
        ceiling = min(policy.max_delay_s, policy.base_delay_s * 2 ** (attempt - 1))
        return random.uniform(ceiling / 2, ceiling)

    if retry_after is None:
        return calculate_ceiling()
    try:
        return min(policy.max_delay_s, float(retry_after))
    except ValueError:
        return calculate_ceiling()


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], int]:
    attempts = 0
    error = ""
    while attempts < policy.max_attempts:
        attempts += 1
        retry_after = None
        try:
            response = client.post(url=url, headers=headers, json=body)
            if response.status_code < 400:
                return (response.json(), attempts)
            error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code not in RETRYABLE_STATUS:
                raise ModelError(error)
            retry_after = response.headers.get("retry-after")
        except httpx.TransportError as e:
            error = f"{type(e).__name__}: {e}"
        if attempts < policy.max_attempts:
            sleep(backoff_delay(attempts, policy, retry_after))
    raise ModelError(f"gave up after {attempts} attempts: {error}")
