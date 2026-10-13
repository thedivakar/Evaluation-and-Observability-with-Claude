"""Tests for batch processing via the Message Batches API.

Batch submitter takes N documents and submits one Message Batches request
           with each item carrying custom_id = policy_id.
Batch poller returns per-item results keyed by custom_id, distinguishing
           succeeded / errored / expired.
Per-item failures isolated and resubmitted in one follow-up batch.
submission_frequency() returns batches/day; raises SLATooTightError if
           sla_hours < batch_eta_hours.
Items carry the standard retry semantics: format/consistency retries on follow-up
           batch with error feedback; missing_source escalates immediately.
`--dry-run-sample N` validates a sample before authorizing bulk batch.
"""
from __future__ import annotations

import os

import pytest

from policy_extractor.batch import (
    BatchClient,
    BatchItemResult,
    SLATooTightError,
    dry_run_sample,
    process_with_resubmission,
    submission_frequency,
)
from policy_extractor.records import PolicyExtraction, RetryFutileEscalation
from tests.conftest import RecordedClient, load_policy_text, make_tool_use_message

# ---------- Submission frequency (pure helper, simplest first) ----------


def test_ac_02_04_submission_frequency_meets_sla() -> None:
    # SLA 24h, batch ETA 6h: 1 batch is enough (within 24h of submission)
    assert submission_frequency(sla_hours=24.0, batch_eta_hours=6.0) == 1
    # SLA 12h, batch ETA 6h: 1 batch per 6h, so 4 per day (24/6) but only
    # 24/12=2 needed for SLA — return the lower frequency that still meets SLA
    assert submission_frequency(sla_hours=12.0, batch_eta_hours=6.0) == 2


def test_ac_02_04_submission_frequency_raises_when_sla_below_batch_eta() -> None:
    """SLA tighter than the batch's own completion time => batch is the wrong tool."""
    with pytest.raises(SLATooTightError) as excinfo:
        submission_frequency(sla_hours=2.0, batch_eta_hours=6.0)
    assert "real-time" in str(excinfo.value).lower()


def test_ac_02_04_submission_frequency_clamps_to_minimum_one_per_day() -> None:
    """Very loose SLA still returns at least 1/day so a daily cron makes sense."""
    assert submission_frequency(sla_hours=168.0, batch_eta_hours=6.0) == 1


# ---------- Batch submitter constructs custom_id per item ----------


def _fake_batch_client(
    succeed: dict[str, dict[str, object]] | None = None,
    errored: list[str] | None = None,
    expired: list[str] | None = None,
) -> FakeBatchClient:
    return FakeBatchClient(succeed=succeed or {}, errored=errored or [], expired=expired or [])


def test_ac_02_01_submitter_attaches_custom_id_per_policy() -> None:
    """Each batch request carries custom_id = policy_id; only one batch call."""
    succeed = {pid: _ok_extraction_input(pid) for pid in ("POL-2025-001", "POL-2025-002")}
    batch_client = _fake_batch_client(succeed=succeed)

    process_with_resubmission(
        batch_client=batch_client,
        extractor_client=RecordedClient([]),  # not used when batch handles all
        policies=[
            ("POL-2025-001", load_policy_text("POL-2025-001")),
            ("POL-2025-002", load_policy_text("POL-2025-002")),
        ],
    )

    assert len(batch_client.create_calls) == 1
    submitted = batch_client.create_calls[0]
    custom_ids = [r["custom_id"] for r in submitted]
    assert custom_ids == ["POL-2025-001", "POL-2025-002"]
    # Sanity: each request carries the model + tools params
    assert all("params" in r for r in submitted)
    for r in submitted:
        params = r["params"]
        assert params["model"] == "claude-haiku-4-5"
        assert any(t["name"] == "extract_policy" for t in params["tools"])


# ---------- Poller distinguishes succeeded / errored / expired ----------


def test_ac_02_02_poller_returns_results_keyed_by_custom_id() -> None:
    succeed = {"POL-2025-001": _ok_extraction_input("POL-2025-001")}
    batch_client = _fake_batch_client(
        succeed=succeed,
        errored=["POL-2025-002"],
        expired=["POL-2025-003"],
    )
    out = process_with_resubmission(
        batch_client=batch_client,
        extractor_client=RecordedClient(
            # Resubmission round retries POL-2025-002 and POL-2025-003 in a 2nd batch;
            # ensure no extra real-time calls are needed.
            []
        ),
        policies=[
            ("POL-2025-001", load_policy_text("POL-2025-001")),
            ("POL-2025-002", load_policy_text("POL-2025-002")),
            ("POL-2025-003", load_policy_text("POL-2025-003")),
        ],
    )
    # All policies appear in output keyed by id
    assert set(out.keys()) == {"POL-2025-001", "POL-2025-002", "POL-2025-003"}


# ---------- Resubmission of errored/expired items ----------


def test_ac_02_03_errored_items_resubmitted_in_one_followup_batch() -> None:
    """Errored items go into a second batch call. Maximum two batch rounds."""
    # Round 1: POL-001 succeeds, POL-002 errored.
    # Round 2: POL-002 retried, this time succeeds.
    batch_client = SequencedBatchClient(
        rounds=[
            # round 1
            {
                "succeed": {"POL-2025-001": _ok_extraction_input("POL-2025-001")},
                "errored": ["POL-2025-002"],
                "expired": [],
            },
            # round 2 (resubmission)
            {
                "succeed": {"POL-2025-002": _ok_extraction_input("POL-2025-002")},
                "errored": [],
                "expired": [],
            },
        ]
    )
    out = process_with_resubmission(
        batch_client=batch_client,
        extractor_client=RecordedClient([]),
        policies=[
            ("POL-2025-001", load_policy_text("POL-2025-001")),
            ("POL-2025-002", load_policy_text("POL-2025-002")),
        ],
    )
    assert len(batch_client.create_calls) == 2  # exactly one resubmission round
    # Round 2 only contained POL-2025-002
    round_2_ids = [r["custom_id"] for r in batch_client.create_calls[1]]
    assert round_2_ids == ["POL-2025-002"]
    assert isinstance(out["POL-2025-001"], PolicyExtraction)
    assert isinstance(out["POL-2025-002"], PolicyExtraction)


def test_ac_02_03_persistent_errors_after_resubmission_are_returned_as_escalation() -> None:
    """Item still failing after resubmission round returns a RetryFutileEscalation."""
    batch_client = SequencedBatchClient(
        rounds=[
            {"succeed": {}, "errored": ["POL-2025-002"], "expired": []},
            {"succeed": {}, "errored": ["POL-2025-002"], "expired": []},
        ]
    )
    out = process_with_resubmission(
        batch_client=batch_client,
        extractor_client=RecordedClient([]),
        policies=[("POL-2025-002", load_policy_text("POL-2025-002"))],
    )
    result = out["POL-2025-002"]
    assert isinstance(result, RetryFutileEscalation)
    assert result.detected_pattern.startswith("batch_item_")


# ---------- Validation-driven retry between batches ----------


def test_ac_02_05_format_failure_triggers_resubmission_with_error_feedback() -> None:
    """A successful batch response that fails *validation* triggers a 2nd batch
    where the second-round prompt contains the validation error from the first round.
    """
    bad_input = _ok_extraction_input("POL-2025-001")
    bad_input["premium_amount"] = -999.0  # format failure
    good_input = _ok_extraction_input("POL-2025-001")
    batch_client = SequencedBatchClient(
        rounds=[
            {"succeed": {"POL-2025-001": bad_input}, "errored": [], "expired": []},
            {"succeed": {"POL-2025-001": good_input}, "errored": [], "expired": []},
        ]
    )
    out = process_with_resubmission(
        batch_client=batch_client,
        extractor_client=RecordedClient([]),
        policies=[("POL-2025-001", load_policy_text("POL-2025-001"))],
    )
    assert len(batch_client.create_calls) == 2
    # Second-round prompt must reference the prior error
    round_2 = batch_client.create_calls[1][0]
    messages = round_2["params"]["messages"]
    text = " ".join(_text_of(m) for m in messages)
    assert "negative_premium" in text or "-999" in text
    assert isinstance(out["POL-2025-001"], PolicyExtraction)


def test_ac_02_05_missing_source_escalates_immediately_no_resubmission() -> None:
    """Validation-class missing_source on first batch => escalation; no Batch 2."""
    bad_input = _ok_extraction_input("POL-2025-009")
    bad_input["endorsements"] = None  # missing_source
    batch_client = SequencedBatchClient(
        rounds=[
            {"succeed": {"POL-2025-009": bad_input}, "errored": [], "expired": []},
        ]
    )
    out = process_with_resubmission(
        batch_client=batch_client,
        extractor_client=RecordedClient([]),
        policies=[("POL-2025-009", load_policy_text("POL-2025-009"))],
    )
    assert len(batch_client.create_calls) == 1  # no resubmission
    result = out["POL-2025-009"]
    assert isinstance(result, RetryFutileEscalation)
    assert result.detected_pattern == "endorsements_absent"


# ---------- Dry-run sample ----------


def test_ac_02_06_dry_run_sample_returns_success_rate_and_pattern_summary() -> None:
    """The sample uses the real-time extractor path and returns first-pass stats."""
    ok = make_tool_use_message("extract_policy", _ok_extraction_input("POL-2025-001"))
    bad = make_tool_use_message(
        "extract_policy", {**_ok_extraction_input("POL-2025-002"), "endorsements": None}
    )
    extractor_client = RecordedClient([ok, bad])
    sample = dry_run_sample(
        extractor_client=extractor_client,
        policies=[
            ("POL-2025-001", load_policy_text("POL-2025-001")),
            ("POL-2025-002", load_policy_text("POL-2025-002")),
        ],
        sample_size=2,
    )
    assert sample.sampled == 2
    assert sample.succeeded == 1
    assert sample.first_pass_success_rate == 0.5
    assert "endorsements_absent" in sample.pattern_summary


def test_ac_02_06_dry_run_sample_caps_at_input_size() -> None:
    """If sample_size > len(policies), use all policies."""
    ok = make_tool_use_message("extract_policy", _ok_extraction_input("POL-2025-001"))
    sample = dry_run_sample(
        extractor_client=RecordedClient([ok]),
        policies=[("POL-2025-001", load_policy_text("POL-2025-001"))],
        sample_size=10,
    )
    assert sample.sampled == 1


# ---------- Live integration test ----------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.live
def test_live_dry_run_sample_against_real_api() -> None:
    """End-to-end dry-run on two real policies. Verifies the real-time path works."""
    from anthropic import Anthropic

    from policy_extractor.client import AnthropicMessagesClient

    client = AnthropicMessagesClient(Anthropic())
    sample = dry_run_sample(
        extractor_client=client,
        policies=[
            ("POL-2025-001", load_policy_text("POL-2025-001")),
            ("POL-2025-004", load_policy_text("POL-2025-004")),
        ],
        sample_size=2,
    )
    # Should succeed on both well-formed policies
    assert sample.succeeded == 2
    assert sample.first_pass_success_rate == 1.0


# ---------- Helpers and fakes ----------


def _ok_extraction_input(policy_id: str) -> dict[str, object]:
    """A canonical well-formed extraction input dict (model-side tool input shape)."""
    return {
        "policy_type": "auto",
        "premium_amount": 1500.0,
        "deductible": 500.0,
        "coverage_limit": 100000.0,
        "endorsements": [{"name": "Roadside", "limit": None}],
        "exclusions": ["Racing"],
        "confidence": {
            "policy_type": 0.95,
            "premium_amount": 0.92,
            "deductible": 0.95,
            "coverage_limit": 0.95,
            "endorsements": 0.9,
            "exclusions": 0.9,
        },
    }


def _text_of(message: dict[str, object]) -> str:
    """Flatten a single message into a string for substring assertions."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or ""
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


class FakeBatchClient:
    """Minimal BatchClient fake that returns canned results in a single round."""

    def __init__(
        self,
        *,
        succeed: dict[str, dict[str, object]],
        errored: list[str],
        expired: list[str],
    ) -> None:
        self._succeed = succeed
        self._errored = errored
        self._expired = expired
        self.create_calls: list[list[dict[str, object]]] = []

    def submit(self, requests: list[dict[str, object]]) -> str:
        self.create_calls.append(requests)
        return f"batch_{len(self.create_calls)}"

    def collect(self, batch_id: str) -> list[BatchItemResult]:
        # Only honour ids we know about — others remain unanswered.
        results: list[BatchItemResult] = []
        last_requests = self.create_calls[-1]
        ids = [r["custom_id"] for r in last_requests]
        for cid in ids:
            assert isinstance(cid, str)
            if cid in self._succeed:
                results.append(
                    BatchItemResult(
                        custom_id=cid,
                        status="succeeded",
                        tool_input=self._succeed[cid],
                        error=None,
                    )
                )
            elif cid in self._errored:
                results.append(
                    BatchItemResult(
                        custom_id=cid, status="errored", tool_input=None, error="rate_limit"
                    )
                )
            elif cid in self._expired:
                results.append(
                    BatchItemResult(
                        custom_id=cid, status="expired", tool_input=None, error="expired"
                    )
                )
        return results


class SequencedBatchClient:
    """BatchClient fake with one preset response set per round (FIFO)."""

    def __init__(self, rounds: list[dict[str, object]]) -> None:
        self._rounds = rounds
        self._round_index = 0
        self.create_calls: list[list[dict[str, object]]] = []

    def submit(self, requests: list[dict[str, object]]) -> str:
        self.create_calls.append(requests)
        return f"batch_{len(self.create_calls)}"

    def collect(self, batch_id: str) -> list[BatchItemResult]:
        round_data = self._rounds[self._round_index]
        self._round_index += 1
        succeed = round_data["succeed"]
        errored = round_data["errored"]
        expired = round_data["expired"]
        assert isinstance(succeed, dict)
        assert isinstance(errored, list)
        assert isinstance(expired, list)

        results: list[BatchItemResult] = []
        last_requests = self.create_calls[-1]
        ids = [r["custom_id"] for r in last_requests]
        for cid in ids:
            assert isinstance(cid, str)
            if cid in succeed:
                results.append(
                    BatchItemResult(
                        custom_id=cid, status="succeeded", tool_input=succeed[cid], error=None
                    )
                )
            elif cid in errored:
                results.append(
                    BatchItemResult(
                        custom_id=cid, status="errored", tool_input=None, error="rate_limit"
                    )
                )
            elif cid in expired:
                results.append(
                    BatchItemResult(
                        custom_id=cid, status="expired", tool_input=None, error="expired"
                    )
                )
        return results


# Static type check: our fakes satisfy the BatchClient protocol.
_: BatchClient = FakeBatchClient(succeed={}, errored=[], expired=[])
_2: BatchClient = SequencedBatchClient(rounds=[])
