"""Tests for independent review + within-policy integration pass.

Reviewer is a separate API call with a fresh prompt; no extractor history.
Reviewer returns per-field {agreement, reason?, review_confidence}.
Any disagreement flags review_disagreement (routed to human resolution).
Reviewer prompt contains only source + extracted JSON; no extractor tokens.
Reviewer uses claude-sonnet-4-6; extractor uses claude-haiku-4-5.
Integration pass runs cross-field checks: coverage_limit vs endorsements,
           endorsements vs exclusions non-contradiction, premium vs components.
"""
from __future__ import annotations

import json
import os

import pytest

from policy_extractor.records import Endorsement, PolicyExtraction, PremiumComponent
from policy_extractor.reviewer import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_REVIEWER_MODEL,
    FieldAgreement,
    build_review_messages,
    has_disagreement,
    independent_review,
    integration_pass,
)
from tests.conftest import RecordedClient, load_policy_text, make_tool_use_message

# ---------- Independence + clean prompt ----------


def _extraction_dict() -> dict[str, object]:
    return {
        "policy_id": "POL-2025-001",
        "policy_type": "auto",
        "premium_amount": 1847.62,
        "deductible": 500.0,
        "coverage_limit": 300000.0,
        "endorsements": [{"name": "Roadside Assistance", "limit": None}],
        "exclusions": ["Racing"],
        "confidence": {
            "policy_type": 0.99, "premium_amount": 0.97, "deductible": 0.95,
            "coverage_limit": 0.95, "endorsements": 0.92, "exclusions": 0.94,
        },
    }


def test_ac_03_01_reviewer_prompt_is_a_fresh_message_list() -> None:
    """Reviewer messages contain a single user turn with source + extraction. No assistant turns."""
    messages, system = build_review_messages(
        source_document=load_policy_text("POL-2025-001"),
        extracted_record=_extraction_dict(),
    )
    # Only one message (user); no assistant turns from the extractor.
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    # System prompt is a reviewer system prompt; it should be distinct from extractor's.
    assert "review" in system.lower()


def test_ac_03_04_reviewer_prompt_has_no_extractor_artefacts() -> None:
    """Reviewer prompt must not contain extractor scratchpad, tool-call IDs, or system prompt."""
    messages, system = build_review_messages(
        source_document=load_policy_text("POL-2025-001"),
        extracted_record=_extraction_dict(),
    )
    combined = system + " " + json.dumps(messages)
    # No reasoning/thinking leakage
    assert "<thinking>" not in combined
    # No extractor tool-call IDs
    assert "toolu_" not in combined
    # The reviewer must NOT echo the extractor's system prompt
    forbidden_extractor_phrases = [
        "You extract structured data",  # opening of extractor system prompt
        "Self-rate confidence",         # extractor-only instruction
    ]
    for phrase in forbidden_extractor_phrases:
        assert phrase not in combined, f"reviewer prompt leaks: {phrase!r}"


# ---------- Per-field review output ----------


def _review_tool_input(field_reviews: list[dict[str, object]]) -> dict[str, object]:
    return {"field_reviews": field_reviews}


def test_ac_03_02_reviewer_returns_per_field_agreement_with_review_confidence() -> None:
    response = make_tool_use_message(
        "review_extraction",
        _review_tool_input(
            [
                {"field": "policy_type", "agreement": "agree", "review_confidence": 0.98},
                {"field": "premium_amount", "agreement": "agree", "review_confidence": 0.95},
                {"field": "deductible", "agreement": "agree", "review_confidence": 0.94},
                {"field": "coverage_limit", "agreement": "agree", "review_confidence": 0.93},
                {"field": "endorsements", "agreement": "agree", "review_confidence": 0.92},
                {"field": "exclusions", "agreement": "agree", "review_confidence": 0.91},
            ]
        ),
    )
    client = RecordedClient([response])
    result = independent_review(
        client=client,
        source_document=load_policy_text("POL-2025-001"),
        extracted_record=_extraction_dict(),
    )
    assert isinstance(result.agreements["policy_type"], FieldAgreement)
    assert result.agreements["premium_amount"].agreement == "agree"
    assert result.agreements["premium_amount"].review_confidence == 0.95
    # Reviewer confidence is kept separate from extractor confidence
    assert "review_confidence" in result.agreements["premium_amount"].__dict__


# ---------- Disagreement triggers review_disagreement ----------


def test_ac_03_03_any_disagreement_flags_review_disagreement() -> None:
    response = make_tool_use_message(
        "review_extraction",
        _review_tool_input(
            [
                {"field": "policy_type", "agreement": "agree", "review_confidence": 0.98},
                {
                    "field": "premium_amount",
                    "agreement": "disagree",
                    "reason": "Document states 1847.62 but extracted value differs",
                    "review_confidence": 0.92,
                },
                {"field": "deductible", "agreement": "agree", "review_confidence": 0.94},
                {"field": "coverage_limit", "agreement": "agree", "review_confidence": 0.93},
                {"field": "endorsements", "agreement": "agree", "review_confidence": 0.92},
                {"field": "exclusions", "agreement": "agree", "review_confidence": 0.91},
            ]
        ),
    )
    client = RecordedClient([response])
    result = independent_review(
        client=client,
        source_document=load_policy_text("POL-2025-001"),
        extracted_record=_extraction_dict(),
    )
    assert has_disagreement(result)
    disagreeing = [f for f, a in result.agreements.items() if a.agreement == "disagree"]
    assert disagreeing == ["premium_amount"]


# ---------- Model split ----------


def test_ac_03_05_extractor_and_reviewer_use_different_models() -> None:
    """The default reviewer model is stronger than the default extractor model."""
    assert DEFAULT_EXTRACTOR_MODEL == "claude-haiku-4-5"
    assert DEFAULT_REVIEWER_MODEL == "claude-sonnet-4-6"


def test_ac_03_05_independent_review_uses_reviewer_model_by_default() -> None:
    response = make_tool_use_message(
        "review_extraction",
        _review_tool_input(
            [
                {"field": "policy_type", "agreement": "agree", "review_confidence": 0.99},
                {"field": "premium_amount", "agreement": "agree", "review_confidence": 0.97},
                {"field": "deductible", "agreement": "agree", "review_confidence": 0.95},
                {"field": "coverage_limit", "agreement": "agree", "review_confidence": 0.95},
                {"field": "endorsements", "agreement": "agree", "review_confidence": 0.93},
                {"field": "exclusions", "agreement": "agree", "review_confidence": 0.93},
            ]
        ),
    )
    client = RecordedClient([response])
    independent_review(
        client=client,
        source_document=load_policy_text("POL-2025-001"),
        extracted_record=_extraction_dict(),
    )
    assert client.calls[0]["model"] == DEFAULT_REVIEWER_MODEL


# ---------- Integration pass ----------


def _ok_extraction(**overrides: object) -> PolicyExtraction:
    base = PolicyExtraction(
        policy_id="POL-X",
        policy_type="home",
        premium_amount=2400.0,
        deductible=1000.0,
        coverage_limit=348000.0,
        endorsements=[
            Endorsement(name="Water Backup", limit=10000.0),
            Endorsement(name="Identity Fraud Expense", limit=15000.0),
        ],
        exclusions=["Flood", "Earth movement", "Power failure"],
        premium_components=None,
        confidence={
            "policy_type": 0.95, "premium_amount": 0.95, "deductible": 0.9,
            "coverage_limit": 0.9, "endorsements": 0.9, "exclusions": 0.9,
        },
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_ac_03_06_coverage_limit_must_exceed_endorsement_limits_sum() -> None:
    ext = _ok_extraction(coverage_limit=1000.0)  # less than 10k+15k=25k endorsements
    findings = integration_pass(ext)
    statuses = {f.check_name: f.status for f in findings}
    assert statuses["coverage_limit_exceeds_endorsement_sum"] == "fail"


def test_ac_03_06_coverage_limit_check_passes_when_consistent() -> None:
    ext = _ok_extraction()
    findings = integration_pass(ext)
    statuses = {f.check_name: f.status for f in findings}
    assert statuses["coverage_limit_exceeds_endorsement_sum"] == "pass"


def test_ac_03_06_endorsement_exclusion_contradiction_detected() -> None:
    """An endorsement and an exclusion sharing a content bigram is a contradiction."""
    ext = _ok_extraction(
        endorsements=[Endorsement(name="Water Backup Coverage", limit=10000.0)],
        exclusions=["Water Backup damage from municipal sewer"],
    )
    findings = integration_pass(ext)
    statuses = {f.check_name: f.status for f in findings}
    assert statuses["endorsements_exclusions_non_contradiction"] == "fail"


def test_ac_03_06_endorsement_exclusion_no_contradiction_passes() -> None:
    ext = _ok_extraction(
        endorsements=[Endorsement(name="Roadside Assistance", limit=None)],
        exclusions=["Racing", "Nuclear hazard"],
    )
    findings = integration_pass(ext)
    statuses = {f.check_name: f.status for f in findings}
    assert statuses["endorsements_exclusions_non_contradiction"] == "pass"


def test_ac_03_06_premium_vs_components_consistency() -> None:
    """Cross-field integration check: stated premium matches sum of components."""
    bad = _ok_extraction(
        premium_amount=2400.0,
        premium_components=[
            PremiumComponent(name="Base", amount=800.0),
            PremiumComponent(name="Coverage A", amount=1100.0),
            PremiumComponent(name="Coverage B", amount=350.0),
            PremiumComponent(name="Endorsements", amount=200.0),
            PremiumComponent(name="Discount", amount=-100.0),
            # Sums to 2350.0, stated as 2400.0
        ],
    )
    findings = integration_pass(bad)
    statuses = {f.check_name: f.status for f in findings}
    assert statuses["premium_matches_components_sum"] == "fail"

    good = _ok_extraction(
        premium_amount=2350.0,
        premium_components=[
            PremiumComponent(name="Base", amount=800.0),
            PremiumComponent(name="Coverage A", amount=1100.0),
            PremiumComponent(name="Coverage B", amount=350.0),
            PremiumComponent(name="Endorsements", amount=200.0),
            PremiumComponent(name="Discount", amount=-100.0),
        ],
    )
    findings = integration_pass(good)
    statuses = {f.check_name: f.status for f in findings}
    assert statuses["premium_matches_components_sum"] == "pass"


def test_ac_03_06_premium_components_check_skipped_when_components_absent() -> None:
    ext = _ok_extraction(premium_components=None)
    findings = integration_pass(ext)
    check_names = {f.check_name for f in findings}
    assert "premium_matches_components_sum" not in check_names


# ---------- Live integration test ----------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.live
def test_live_reviewer_returns_per_field_judgement() -> None:
    """End-to-end: real reviewer produces a structured per-field judgement.

    We do NOT assert "all agree" — the reviewer is a quality net, and real-world
    extractions carry simplifications (e.g., coverage_limit collapsing multi-line
    coverage into a single number). The reviewer correctly catching those is the
    feature, not a bug.
    """
    from anthropic import Anthropic

    from policy_extractor.client import AnthropicMessagesClient

    client = AnthropicMessagesClient(Anthropic())
    result = independent_review(
        client=client,
        source_document=load_policy_text("POL-2025-001"),
        extracted_record={
            "policy_id": "POL-2025-001",
            "policy_type": "auto",
            "premium_amount": 1847.62,
            "deductible": 500.0,
            "coverage_limit": 300000.0,
            "endorsements": [
                {"name": "Rental Reimbursement Endorsement (AU-21)", "limit": None},
                {"name": "Roadside Assistance Coverage (AU-44)", "limit": None},
                {"name": "Additional Insured: Anywhere Auto Leasing LLC", "limit": None},
            ],
            "exclusions": [
                "Racing or speed contest activity",
                "Use of vehicle for ride-share / TNC purposes without endorsement",
                "Intentional acts by the insured",
            ],
        },
    )
    # Every field is judged
    assert set(result.agreements.keys()) == {
        "policy_type", "premium_amount", "deductible", "coverage_limit",
        "endorsements", "exclusions",
    }
    # Every judgement carries a review_confidence in range
    for agreement in result.agreements.values():
        assert 0.0 <= agreement.review_confidence <= 1.0
    # The reviewer should agree on policy_type for this clean auto policy
    assert result.agreements["policy_type"].agreement == "agree"
