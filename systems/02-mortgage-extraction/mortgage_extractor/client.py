"""Thin Anthropic-SDK wrapper that records and replays API calls.

In production this is a transparent passthrough to ``anthropic.Anthropic`` — the
``call`` method takes the same kwargs as ``client.messages.create`` and returns
the same ``Message`` object. In tests we run with ``mode="replay"`` so calls
hit a JSON-on-disk cache instead of the network, keeping tests offline and
deterministic.

The cache key is a SHA-256 of the request payload (model, messages, tools,
tool_choice, system, max_tokens) so any meaningful change to the request
invalidates the cache. Cache files live under
``fixtures/recorded_responses/<sha8>.json`` and bundle both the request and the
response, which lets tests verify *what was sent* in addition to what came back.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import anthropic
from anthropic.types import Message

from mortgage_extractor.config import RECORDED_RESPONSES_DIR

Mode = Literal["record", "replay", "auto"]


@dataclass
class RecordedCall:
    request: dict[str, Any]
    response: dict[str, Any]


class RecordingClient:
    """Wraps an Anthropic client and persists each call to disk for replay."""

    def __init__(
        self,
        *,
        mode: Mode = "auto",
        cache_dir: Path | None = None,
        real_client: anthropic.Anthropic | None = None,
    ) -> None:
        self.mode: Mode = mode
        self.cache_dir = cache_dir or RECORDED_RESPONSES_DIR
        self._real_client = real_client
        self.last_request: dict[str, Any] | None = None
        self.last_cache_key: str | None = None

    @property
    def real_client(self) -> anthropic.Anthropic:
        if self._real_client is None:
            self._real_client = anthropic.Anthropic()
        return self._real_client

    def call(self, **kwargs: Any) -> Message:
        normalized = self._normalize(kwargs)
        self.last_request = normalized
        key = _cache_key(normalized)
        self.last_cache_key = key
        cache_file = self.cache_dir / f"{key}.json"

        if self.mode in ("replay", "auto") and cache_file.exists():
            return self._load(cache_file)

        if self.mode == "replay":
            raise FileNotFoundError(
                f"No recorded response for key {key} (mode=replay). "
                f"Re-run with mode='record' or 'auto' to populate the cache."
            )

        response = cast(Message, self.real_client.messages.create(**kwargs))
        self._save(cache_file, normalized, response)
        return response

    @staticmethod
    def _normalize(kwargs: dict[str, Any]) -> dict[str, Any]:
        roundtripped = json.loads(json.dumps(kwargs, sort_keys=True, default=_json_default))
        return cast(dict[str, Any], roundtripped)

    def _load(self, path: Path) -> Message:
        payload = json.loads(path.read_text())
        return Message.model_validate(payload["response"])

    def _save(self, path: Path, request: dict[str, Any], response: Message) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "request": request,
            "response": response.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")
