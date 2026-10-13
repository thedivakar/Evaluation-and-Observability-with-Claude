"""Anthropic Messages client abstraction.

The pipeline depends on a narrow Protocol so tests can inject a recorded-response
fake without touching the real API. The production wrapper delegates to the SDK.
"""
from __future__ import annotations

from typing import Any, Protocol

from anthropic import Anthropic
from anthropic.types import Message


class MessageClient(Protocol):
    """Subset of anthropic.Anthropic().messages used by the pipeline."""

    def create(self, **kwargs: Any) -> Message: ...


class AnthropicMessagesClient:
    """Production adapter around anthropic.Anthropic().messages."""

    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Message:
        result: Message = self._client.messages.create(**kwargs)
        return result
