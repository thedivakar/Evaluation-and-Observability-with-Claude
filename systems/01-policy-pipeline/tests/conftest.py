"""Shared pytest fixtures and helpers.

Includes a RecordedClient for deterministic replay of real API responses,
and loaders for the policy-document and recorded-response fixtures.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

DATA_DIR = Path(__file__).parent.parent / "data"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
POLICIES_DIR = DATA_DIR / "policies"
CASSETTES_DIR = FIXTURES_DIR / "cassettes"


def _load_env_file() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key and value:
                os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


@pytest.fixture(scope="session")
def policies_manifest() -> list[dict[str, Any]]:
    return json.loads((POLICIES_DIR / "MANIFEST.json").read_text())


def load_policy_text(policy_id: str) -> str:
    return (POLICIES_DIR / f"{policy_id}.txt").read_text()


def load_cassette(name: str) -> list[dict[str, Any]]:
    """Load a recorded API response sequence."""
    path = CASSETTES_DIR / f"{name}.json"
    return json.loads(path.read_text())


def make_tool_use_message(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    model: str = "claude-haiku-4-5",
    tool_use_id: str = "toolu_test_001",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Message:
    """Construct an SDK Message containing a single tool_use block.

    Used by RecordedClient to synthesize deterministic API responses from
    captured real responses (or from real responses with specific fields mutated
    to drive a failure path). Every field shape mirrors a real API response.
    """
    return Message(
        id="msg_test",
        content=[ToolUseBlock(id=tool_use_id, input=tool_input, name=tool_name, type="tool_use")],
        model=model,
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        stop_details=None,
        type="message",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cache_creation=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier=None,
        ),
    )


def make_text_message(
    text: str,
    *,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Message:
    """Construct an SDK Message containing a single text block."""
    return Message(
        id="msg_test_text",
        content=[TextBlock(citations=None, text=text, type="text")],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        stop_details=None,
        type="message",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cache_creation=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier=None,
        ),
    )


class RecordedClient:
    """Thin recorded-response fake for tests.

    Returns pre-baked Message objects in order. Captures each create() call's
    kwargs so tests can assert on prompt construction. Not a mock — every
    response object is shaped exactly like a real API response.
    """

    def __init__(self, responses: Iterable[Message]) -> None:
        self._responses: list[Message] = list(responses)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if self._index >= len(self._responses):
            raise AssertionError(
                f"RecordedClient exhausted: {self._index + 1} calls but only "
                f"{len(self._responses)} responses queued"
            )
        response = self._responses[self._index]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        return len(self.calls)
