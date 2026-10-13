"""Batch processing via the Message Batches API.

Components:
- BatchClient Protocol — normalised boundary so tests can stub
- AnthropicBatchClient — production adapter around anthropic.Anthropic().messages.batches
- submit / poll / collect — primitives
- process_with_resubmission — top-level flow: Batch 1 → validate → Batch 2 (one round of
  retries for format/consistency failures and per-item batch errors)
- dry_run_sample — validates a small N before authorising bulk batch
- submission_frequency — pure helper mapping SLA to batches-per-day
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from anthropic import Anthropic
from anthropic.types.messages import (
    MessageBatchErroredResult,
    MessageBatchExpiredResult,
    MessageBatchSucceededResult,
)

from policy_extractor.client import MessageClient
from policy_extractor.extractor import EXTRACT_POLICY_TOOL, build_extraction_messages
from policy_extractor.records import (
    ExtractionOutcome,
    PolicyExtraction,
    RetryFutileEscalation,
    ValidationError,
)
from policy_extractor.retry import build_extraction, extract_with_retry
from policy_extractor.summary import summarize_patterns
from policy_extractor.validator import validate_extraction

DEFAULT_EXTRACTOR_MODEL = "claude-haiku-4-5"
BATCH_POLL_INTERVAL_SECONDS = 5.0
BATCH_POLL_TIMEOUT_SECONDS = 86_400.0  # 24h SDK contract


BatchStatus = Literal["succeeded", "errored", "canceled", "expired"]


class SLATooTightError(ValueError):
    """Raised when an SLA is tighter than the batch API's own completion time."""


@dataclass(frozen=True)
class BatchItemResult:
    custom_id: str
    status: BatchStatus
    tool_input: dict[str, Any] | None
    error: str | None


class BatchClient(Protocol):
    """Narrow boundary around the Batches API used by the pipeline."""

    def submit(self, requests: list[dict[str, Any]]) -> str:
        """Submit a batch; return its id."""
        ...

    def collect(self, batch_id: str) -> list[BatchItemResult]:
        """Poll until the batch ends, then return normalised per-item results."""
        ...


class AnthropicBatchClient:
    """Production BatchClient adapter around anthropic.Anthropic().messages.batches."""

    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def submit(self, requests: list[dict[str, Any]]) -> str:
        # The SDK accepts dict-shaped requests at runtime; the TypedDict typing
        # is structurally compatible with our request dicts.
        batch = self._client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
        return batch.id

    def collect(self, batch_id: str) -> list[BatchItemResult]:
        deadline = time.monotonic() + BATCH_POLL_TIMEOUT_SECONDS
        while True:
            batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Batch {batch_id} did not end within 24h.")
            time.sleep(BATCH_POLL_INTERVAL_SECONDS)

        out: list[BatchItemResult] = []
        for item in self._client.messages.batches.results(batch_id):
            result = item.result
            if isinstance(result, MessageBatchSucceededResult):
                tool_input = _extract_tool_input(result.message)
                out.append(
                    BatchItemResult(
                        custom_id=item.custom_id,
                        status="succeeded",
                        tool_input=tool_input,
                        error=None,
                    )
                )
            elif isinstance(result, MessageBatchErroredResult):
                out.append(
                    BatchItemResult(
                        custom_id=item.custom_id,
                        status="errored",
                        tool_input=None,
                        error=str(result.error),
                    )
                )
            elif isinstance(result, MessageBatchExpiredResult):
                out.append(
                    BatchItemResult(
                        custom_id=item.custom_id,
                        status="expired",
                        tool_input=None,
                        error="expired",
                    )
                )
            else:
                out.append(
                    BatchItemResult(
                        custom_id=item.custom_id,
                        status="canceled",
                        tool_input=None,
                        error="canceled",
                    )
                )
        return out


def _extract_tool_input(message: Any) -> dict[str, Any] | None:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


# ---------- Helpers ----------


def submission_frequency(*, sla_hours: float, batch_eta_hours: float) -> int:
    """Return the minimum batches-per-day needed to meet the SLA.

    The Message Batches API guarantees only that a batch completes within 24h —
    there is no latency SLA. We model the per-batch ETA explicitly so callers can
    plug in their measured p50/p90.

    Raises SLATooTightError when sla_hours < batch_eta_hours: the batch's own
    completion time blows the SLA before the result is even available; switch to
    the real-time Messages API.
    """
    if sla_hours <= 0 or batch_eta_hours <= 0:
        raise ValueError("sla_hours and batch_eta_hours must be positive.")
    if sla_hours < batch_eta_hours:
        raise SLATooTightError(
            f"SLA {sla_hours}h is shorter than batch_eta {batch_eta_hours}h. "
            "Batch processing cannot meet this SLA; use the real-time Messages API."
        )
    # The latest a request can be submitted and still meet SLA is (sla_hours - batch_eta_hours)
    # after the previous batch ended. Submission frequency must cover the SLA window so that
    # any arriving request is submitted within (sla_hours - batch_eta_hours) of its arrival.
    head_room = sla_hours - batch_eta_hours
    if head_room <= 0:
        return max(1, math.ceil(24.0 / batch_eta_hours))
    return max(1, math.ceil(24.0 / (head_room + batch_eta_hours)))


def _build_request(
    custom_id: str,
    document_text: str,
    *,
    prior_attempts: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    messages, system = build_extraction_messages(document_text, prior_attempts)
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "tools": [EXTRACT_POLICY_TOOL],
            "tool_choice": {"type": "tool", "name": "extract_policy"},
        },
    }


# ---------- Top-level: process with at most one resubmission round ----------


def process_with_resubmission(
    *,
    batch_client: BatchClient,
    extractor_client: MessageClient,
    policies: list[tuple[str, str]],
    model: str = DEFAULT_EXTRACTOR_MODEL,
) -> dict[str, ExtractionOutcome]:
    """Submit policies as a batch; resubmit format/consistency failures once.

    Returns a dict mapping policy_id → PolicyExtraction or RetryFutileEscalation.

    because the Batch API does not support multi-turn tool conversations
    within a single batch request, format/consistency retries happen on a follow-up
    batch and missing_source escalates immediately.
    """
    del extractor_client  # reserved for future per-item real-time fallback

    # --- Round 1 ---
    round_1_requests = [
        _build_request(pid, doc, model=model) for pid, doc in policies
    ]
    batch_id = batch_client.submit(round_1_requests)
    round_1_results = batch_client.collect(batch_id)

    docs_by_id = dict(policies)
    final: dict[str, ExtractionOutcome] = {}
    retry_queue: list[tuple[str, list[dict[str, Any]]]] = []
    history: dict[str, list[ValidationError]] = {}

    for item in round_1_results:
        pid = item.custom_id
        if item.status == "succeeded" and item.tool_input is not None:
            err = validate_extraction(item.tool_input)
            if err is None:
                final[pid] = build_extraction(
                    policy_id=pid,
                    extraction=item.tool_input,
                    attempt_index=0,
                    history=[],
                )
            elif err.category == "missing_source":
                final[pid] = RetryFutileEscalation(
                    policy_id=pid,
                    field=err.field,
                    category="missing_source",
                    detected_pattern=err.detected_pattern,
                    reason=err.message,
                )
            else:
                history.setdefault(pid, []).append(err)
                retry_queue.append(
                    (
                        pid,
                        [
                            {
                                "extraction": item.tool_input,
                                "error_field": err.field,
                                "error_category": err.category,
                                "error_pattern": err.detected_pattern,
                                "error_message": err.message,
                            }
                        ],
                    )
                )
        else:
            # errored / expired / canceled — resubmit without error feedback context
            history.setdefault(pid, [])
            retry_queue.append((pid, []))

    if not retry_queue:
        return final

    # --- Round 2 (resubmission, single round) ---
    round_2_requests = [
        _build_request(pid, docs_by_id[pid], model=model, prior_attempts=prior or None)
        for pid, prior in retry_queue
    ]
    batch_id_2 = batch_client.submit(round_2_requests)
    round_2_results = batch_client.collect(batch_id_2)

    for item in round_2_results:
        pid = item.custom_id
        prior_history = history.get(pid, [])
        if item.status == "succeeded" and item.tool_input is not None:
            err = validate_extraction(item.tool_input)
            if err is None:
                final[pid] = build_extraction(
                    policy_id=pid,
                    extraction=item.tool_input,
                    attempt_index=1,
                    history=prior_history,
                )
            elif err.category == "missing_source":
                final[pid] = RetryFutileEscalation(
                    policy_id=pid,
                    field=err.field,
                    category="missing_source",
                    detected_pattern=err.detected_pattern,
                    reason=err.message,
                )
            else:
                final[pid] = RetryFutileEscalation(
                    policy_id=pid,
                    field=err.field,
                    category="missing_source",
                    detected_pattern=f"retries_exhausted__{err.detected_pattern}",
                    reason=(
                        "Validator rejected the extraction on both batches. "
                        f"Last failure: {err.message}"
                    ),
                )
        else:
            final[pid] = RetryFutileEscalation(
                policy_id=pid,
                field="<batch_item>",
                category="missing_source",
                detected_pattern=f"batch_item_{item.status}",
                reason=f"Item {item.status} on resubmission: {item.error}",
            )

    return final


# ---------- Dry-run sample ----------


@dataclass
class DryRunSampleResult:
    sampled: int
    succeeded: int
    escalated: int
    first_pass_success_rate: float
    pattern_summary: dict[str, dict[str, Any]] = field(default_factory=dict)


def dry_run_sample(
    *,
    extractor_client: MessageClient,
    policies: list[tuple[str, str]],
    sample_size: int,
    model: str = DEFAULT_EXTRACTOR_MODEL,
) -> DryRunSampleResult:
    """Run sample_size policies through the real-time extractor and report stats.

    Used as a pre-batch guard: before authorising a bulk batch, the operator runs a
    small sample real-time and sees the first-pass success rate plus a pattern
    summary of recurring failures.
    """
    n = min(sample_size, len(policies))
    outcomes: list[ExtractionOutcome] = []
    for pid, doc in policies[:n]:
        outcomes.append(
            extract_with_retry(
                client=extractor_client,
                policy_id=pid,
                document_text=doc,
                model=model,
                max_retries=0,  # first-pass success only — no retries during dry run
            )
        )
    succeeded = sum(1 for o in outcomes if isinstance(o, PolicyExtraction))
    escalated = sum(1 for o in outcomes if isinstance(o, RetryFutileEscalation))
    rate = succeeded / n if n > 0 else 0.0
    summary: dict[str, dict[str, Any]] = {
        pattern: dict(row) for pattern, row in summarize_patterns(outcomes).items()
    }
    return DryRunSampleResult(
        sampled=n,
        succeeded=succeeded,
        escalated=escalated,
        first_pass_success_rate=rate,
        pattern_summary=summary,
    )
