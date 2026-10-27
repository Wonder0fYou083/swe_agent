"""The interface every model adapter implements."""

from collections.abc import Sequence
from typing import Protocol

from swe_agent.models.types import Message, ModelResponse, ToolSpec


class ModelError(Exception):
    """The model call failed for good (retries used up, or a non-retryable error)."""


class Model(Protocol):
    name: str

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse: ...
