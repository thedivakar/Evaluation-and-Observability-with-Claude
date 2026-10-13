"""Tests for retry-with-error-feedback extraction.

Extractor produces structured policy record via tool_choice-forced tool call.
Validator returns ValidationError with category + detected_pattern.
Format/consistency failures retry up to 3 with error appended to prompt.
Missing_source failures halt immediately, no further API call.
Success record carries retry_count, final_attempt_index, validation_history.
Run-level summarize_patterns aggregator produces frequency table.
"""
from __future__ import annotations

import os

import pytest

from policy_extractor.extractor import EXTRACT_POLICY_TOOL, build_extraction_messages
from policy_extractor.records import PolicyExtraction, RetryFutileEscalation, ValidationError
from policy_extractor.retry import extract_with_retry
from policy_extractor.summary import summarize_patterns
from policy_extractor.validator import validate_extraction
from tests.conftest import RecordedClient, load_policy_text, make_tool_use_message

# ---------- Extractor structure ----------


def test_ac_01_01_extract_tool_has_required_fields() -> None:
    """The extraction tool schema requires every field in the PRD spec."""
    schema = EXTRACT_POLICY_TOOL["input_schema"]
    required = set(schema["required"])
    assert {
        "policy_type",
        "premium_amount",
        "deductible",
        "coverage_limit",
        "endorsements",
        "exclusions",
        "confidence",
    } <= required


def test_ac_01_01_policy_type_uses_resilient_catchall_enum() -> None:
    """policy_type schema includes 'other' so unseen types don't break the pipeline."""
    props = EXTRACT_POLICY_TOOL["input_schema"]["properties"]
    assert "other" in props["policy_type"]["enum"]
    assert set(props["policy_type"]["enum"]) >= {"auto", "home", "umbrella", "other"}


def test_ac_01_01_system_prompt_instructs_null_for_unstated_fields() -> None:
    """The extraction system prompt tells the model to return null when info is absent."""
    _messages, system = build_extraction_messages("dummy doc text")
    assert "null" in system.lower()
    # Few-shot examples for format standardization (e.g., "$1,250.00" -> 1250.0)
    assert system.count("Example") >= 2 or system.lower().count("example") >= 2


def test_ac_01_01_extractor_parses_well_formed_response() -> None:
    """Given a captured tool_use response, the extractor returns a PolicyExtraction."""
    response = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "auto",
            "premium_amount": 1847.62,
            "deductible": 500.0,
            "coverage_limit": 300000.0,
            "endorsements": [
                {"name": "Roadside Assistance", "limit": None},
            ],
            "exclusions": ["Racing"],
            "confidence": {
                "policy_type": 0.99,
                "premium_amount": 0.97,
                "deductible": 0.95,
                "coverage_limit": 0.95,
                "endorsements": 0.92,
                "exclusions": 0.94,
            },
        },
    )
    client = RecordedClient([response])
    result = extract_with_retry(
        client=client,
        policy_id="POL-2025-001",
        document_text=load_policy_text("POL-2025-001"),
    )
    assert isinstance(result, PolicyExtraction)
    assert result.policy_type == "auto"
    assert result.premium_amount == 1847.62


# ---------- Validator ----------


def test_ac_01_02_validator_flags_negative_premium_as_format() -> None:
    err = validate_extraction(
        {
            "policy_type": "auto",
            "premium_amount": -1500.0,
            "deductible": 500.0,
            "coverage_limit": 100000.0,
            "endorsements": [],
            "exclusions": ["x"],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.9,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        }
    )
    assert err is not None
    assert err.category == "format"
    assert err.field == "premium_amount"
    assert err.detected_pattern == "negative_premium"


def test_ac_01_02_validator_flags_null_required_as_missing_source() -> None:
    err = validate_extraction(
        {
            "policy_type": "auto",
            "premium_amount": 1500.0,
            "deductible": 500.0,
            "coverage_limit": 100000.0,
            # endorsements explicitly null = model couldn't find them in source
            "endorsements": None,
            "exclusions": ["x"],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.9,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.5,
                "exclusions": 0.9,
            },
        }
    )
    assert err is not None
    assert err.category == "missing_source"
    assert err.field == "endorsements"
    assert err.detected_pattern == "endorsements_absent"


def test_ac_01_02_validator_flags_consistency_premium_mismatch() -> None:
    err = validate_extraction(
        {
            "policy_type": "home",
            "premium_amount": 2400.0,
            "deductible": 1000.0,
            "coverage_limit": 348000.0,
            "endorsements": [{"name": "Water Backup", "limit": 10000.0}],
            "exclusions": ["Flood"],
            "premium_components": [
                {"name": "Base", "amount": 800.0},
                {"name": "Coverage A", "amount": 1100.0},
                {"name": "Coverage B", "amount": 350.0},
                {"name": "Endorsements", "amount": 200.0},
                {"name": "Discount", "amount": -100.0},
            ],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.9,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        }
    )
    assert err is not None
    assert err.category == "consistency"
    assert err.detected_pattern == "premium_does_not_match_components"
    # The stated 2400.0 vs sum 2350.0 means a $50 discrepancy
    assert "2350" in str(err.observed_value) or "2400" in str(err.observed_value)


def test_ac_01_02_validator_accepts_well_formed() -> None:
    err = validate_extraction(
        {
            "policy_type": "auto",
            "premium_amount": 1847.62,
            "deductible": 500.0,
            "coverage_limit": 300000.0,
            "endorsements": [{"name": "Roadside", "limit": None}],
            "exclusions": ["Racing"],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.9,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        }
    )
    assert err is None


# ---------- Format retry appends error ----------


def test_ac_01_03_format_failure_retries_with_error_appended() -> None:
    bad = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "auto",
            "premium_amount": -1500.0,
            "deductible": 500.0,
            "coverage_limit": 100000.0,
            "endorsements": [{"name": "Roadside", "limit": None}],
            "exclusions": ["Racing"],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.7,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        },
    )
    good = make_tool_use_message(
        "extract_policy",
        {
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
        },
    )
    client = RecordedClient([bad, good])
    result = extract_with_retry(
        client=client,
        policy_id="POL-2025-001",
        document_text=load_policy_text("POL-2025-001"),
        max_retries=3,
    )
    assert isinstance(result, PolicyExtraction)
    assert result.premium_amount == 1500.0

    # Second prompt must contain the validation error message and the prior
    # offending value.
    assert client.call_count == 2
    second_call_messages = client.calls[1]["messages"]
    retry_text = " ".join(_flatten_messages(second_call_messages))
    assert "negative_premium" in retry_text or "-1500" in retry_text
    assert "premium_amount" in retry_text


def test_ac_01_03_consistency_failure_also_retries() -> None:
    """Consistency errors (not just format) also trigger retry."""
    bad = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "home",
            "premium_amount": 2400.0,
            "deductible": 1000.0,
            "coverage_limit": 348000.0,
            "endorsements": [{"name": "Water Backup", "limit": 10000.0}],
            "exclusions": ["Flood"],
            "premium_components": [
                {"name": "Base", "amount": 800.0},
                {"name": "Coverage A", "amount": 1100.0},
                {"name": "Coverage B", "amount": 350.0},
                {"name": "Endorsements", "amount": 200.0},
                {"name": "Discount", "amount": -100.0},
            ],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.7,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        },
    )
    good = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "home",
            "premium_amount": 2350.0,
            "deductible": 1000.0,
            "coverage_limit": 348000.0,
            "endorsements": [{"name": "Water Backup", "limit": 10000.0}],
            "exclusions": ["Flood"],
            "premium_components": [
                {"name": "Base", "amount": 800.0},
                {"name": "Coverage A", "amount": 1100.0},
                {"name": "Coverage B", "amount": 350.0},
                {"name": "Endorsements", "amount": 200.0},
                {"name": "Discount", "amount": -100.0},
            ],
            "confidence": {
                "policy_type": 0.95,
                "premium_amount": 0.95,
                "deductible": 0.95,
                "coverage_limit": 0.95,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        },
    )
    client = RecordedClient([bad, good])
    result = extract_with_retry(
        client=client,
        policy_id="POL-2025-010",
        document_text=load_policy_text("POL-2025-010"),
        max_retries=3,
    )
    assert isinstance(result, PolicyExtraction)
    assert result.premium_amount == 2350.0
    assert client.call_count == 2


# ---------- Missing source halts immediately ----------


def test_ac_01_04_missing_source_halts_immediately() -> None:
    """Null required field => RetryFutileEscalation, exactly one API call."""
    response = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "auto",
            "premium_amount": 1512.88,
            "deductible": 500.0,
            "coverage_limit": 300000.0,
            "endorsements": None,  # model couldn't find Schedule A
            "exclusions": ["Nuclear hazard"],
            "confidence": {
                "policy_type": 0.95,
                "premium_amount": 0.95,
                "deductible": 0.95,
                "coverage_limit": 0.9,
                "endorsements": 0.2,
                "exclusions": 0.9,
            },
        },
    )
    client = RecordedClient([response])
    result = extract_with_retry(
        client=client,
        policy_id="POL-2025-009",
        document_text=load_policy_text("POL-2025-009"),
        max_retries=3,
    )
    assert isinstance(result, RetryFutileEscalation)
    assert result.field == "endorsements"
    assert result.detected_pattern == "endorsements_absent"
    assert client.call_count == 1  # no further API call


# ---------- Success record carries metadata ----------


def test_ac_01_05_success_metadata_after_retry() -> None:
    bad = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "auto",
            "premium_amount": -100.0,
            "deductible": 500.0,
            "coverage_limit": 100000.0,
            "endorsements": [{"name": "x", "limit": None}],
            "exclusions": ["y"],
            "confidence": {
                "policy_type": 0.9,
                "premium_amount": 0.9,
                "deductible": 0.9,
                "coverage_limit": 0.9,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        },
    )
    good = make_tool_use_message(
        "extract_policy",
        {
            "policy_type": "auto",
            "premium_amount": 100.0,
            "deductible": 500.0,
            "coverage_limit": 100000.0,
            "endorsements": [{"name": "x", "limit": None}],
            "exclusions": ["y"],
            "confidence": {
                "policy_type": 0.95,
                "premium_amount": 0.92,
                "deductible": 0.95,
                "coverage_limit": 0.95,
                "endorsements": 0.9,
                "exclusions": 0.9,
            },
        },
    )
    client = RecordedClient([bad, good])
    result = extract_with_retry(
        client=client,
        policy_id="POL-2025-001",
        document_text="dummy",
        max_retries=3,
    )
    assert isinstance(result, PolicyExtraction)
    assert result.retry_count == 1
    assert result.final_attempt_index == 1
    assert len(result.validation_history) == 1
    assert result.validation_history[0].detected_pattern == "negative_premium"


# ---------- Pattern aggregator ----------


def test_ac_01_06_pattern_summary_counts_across_runs() -> None:
    """Aggregator returns a frequency table of detected_patterns and policies affected."""
    ext_1 = PolicyExtraction(
        policy_id="POL-A",
        policy_type="auto",
        premium_amount=1000.0,
        deductible=500.0,
        coverage_limit=100000.0,
        endorsements=[],
        exclusions=[],
        premium_components=None,
        confidence={
            "policy_type": 0.9, "premium_amount": 0.9, "deductible": 0.9,
            "coverage_limit": 0.9, "endorsements": 0.9, "exclusions": 0.9,
        },
        retry_count=1,
        final_attempt_index=1,
        validation_history=[
            ValidationError(
                field="premium_amount",
                observed_value="-100",
                category="format",
                detected_pattern="negative_premium",
                message="negative",
            )
        ],
    )
    ext_2 = PolicyExtraction(
        policy_id="POL-B",
        policy_type="auto",
        premium_amount=1500.0,
        deductible=500.0,
        coverage_limit=100000.0,
        endorsements=[],
        exclusions=[],
        premium_components=None,
        confidence={
            "policy_type": 0.9, "premium_amount": 0.9, "deductible": 0.9,
            "coverage_limit": 0.9, "endorsements": 0.9, "exclusions": 0.9,
        },
        retry_count=1,
        final_attempt_index=1,
        validation_history=[
            ValidationError(
                field="premium_amount",
                observed_value="-200",
                category="format",
                detected_pattern="negative_premium",
                message="negative",
            )
        ],
    )
    esc = RetryFutileEscalation(
        policy_id="POL-C",
        field="endorsements",
        category="missing_source",
        detected_pattern="endorsements_absent",
        reason="Schedule A not enclosed",
    )

    summary = summarize_patterns([ext_1, ext_2, esc])
    # "negative_premium" appears in 2 different policies
    assert summary["negative_premium"]["count"] == 2
    assert set(summary["negative_premium"]["policies"]) == {"POL-A", "POL-B"}
    assert summary["endorsements_absent"]["count"] == 1
    assert summary["endorsements_absent"]["policies"] == ["POL-C"]


# ---------- Live integration test ----------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live API test",
)
@pytest.mark.live
def test_live_extracts_well_formed_policy() -> None:
    """End-to-end against real API on a well-formed clean policy."""
    from anthropic import Anthropic

    from policy_extractor.client import AnthropicMessagesClient

    client = AnthropicMessagesClient(Anthropic())
    result = extract_with_retry(
        client=client,
        policy_id="POL-2025-001",
        document_text=load_policy_text("POL-2025-001"),
        max_retries=3,
    )
    assert isinstance(result, PolicyExtraction)
    assert result.policy_type == "auto"
    assert result.premium_amount is not None
    assert 1800 < result.premium_amount < 1900


# ---------- helpers ----------


def _flatten_messages(messages: list[dict[str, object]]) -> list[str]:
    """Flatten message content blocks into a list of strings for substring search."""
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    if isinstance(text, str):
                        out.append(text)
                    elif isinstance(text, list):
                        for sub in text:
                            if isinstance(sub, dict):
                                t = sub.get("text") or ""
                                if isinstance(t, str):
                                    out.append(t)
    return out
