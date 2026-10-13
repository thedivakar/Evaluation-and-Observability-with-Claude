"""Tests for confidence-based HITL routing with stratified sampling.

Each extracted field carries a confidence in [0.0, 1.0].
Routing function classifies as auto_approve | human_review | spot_check.
            human_review fires on (field conf < 0.90) OR (any reviewer disagreement)
            OR (any integration check failure).
Stratified sampler groups by policy_type, draws configurable % per stratum,
            never silently drops strata with eligible records.
Sliced calibration report: policy_type x field with (samples, mean_pred_conf,
            observed_accuracy, Brier_score) + overall Brier.
Routing decisions written to a single JSON file with per-policy records.
Routing is deterministic on (confidence ∧ reviewer ∧ integration);
            high-confidence + reviewer disagreement => human_review, not auto_approve.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policy_extractor.records import Endorsement, PolicyExtraction
from policy_extractor.reviewer import (
    FieldAgreement,
    IntegrationFinding,
    ReviewResult,
)
from policy_extractor.routing import (
    CalibrationLabel,
    apply_stratified_spot_check,
    calibration_report,
    route_extraction,
    write_routing_decisions,
)

# ---------- Helpers ----------


def _ext(
    *,
    policy_id: str = "POL-X",
    policy_type: str = "auto",
    confidence: dict[str, float] | None = None,
) -> PolicyExtraction:
    conf = confidence or {
        "policy_type": 0.95, "premium_amount": 0.95, "deductible": 0.95,
        "coverage_limit": 0.95, "endorsements": 0.95, "exclusions": 0.95,
    }
    return PolicyExtraction(
        policy_id=policy_id,
        policy_type=policy_type,  # type: ignore[arg-type]
        premium_amount=1000.0,
        deductible=500.0,
        coverage_limit=100000.0,
        endorsements=[Endorsement(name="x", limit=None)],
        exclusions=["y"],
        premium_components=None,
        confidence=conf,
    )


def _agree(field: str) -> FieldAgreement:
    return FieldAgreement(field=field, agreement="agree", reason=None, review_confidence=0.95)


def _disagree(field: str) -> FieldAgreement:
    return FieldAgreement(
        field=field, agreement="disagree", reason="conflict", review_confidence=0.9
    )


def _review(agreements: list[FieldAgreement]) -> ReviewResult:
    return ReviewResult(agreements={a.field: a for a in agreements})


def _integration_pass_all() -> list[IntegrationFinding]:
    return [
        IntegrationFinding(
            check_name="coverage_limit_exceeds_endorsement_sum",
            status="pass",
            details="",
        ),
        IntegrationFinding(
            check_name="endorsements_exclusions_non_contradiction",
            status="pass",
            details="",
        ),
    ]


def _all_agree() -> list[FieldAgreement]:
    return [_agree(f) for f in (
        "policy_type", "premium_amount", "deductible", "coverage_limit",
        "endorsements", "exclusions",
    )]


# ---------- Routing classifier ----------


def test_ac_04_02_all_clear_routes_to_auto_approve() -> None:
    decision = route_extraction(
        extraction=_ext(),
        review=_review(_all_agree()),
        integration_findings=_integration_pass_all(),
    )
    assert decision.decision == "auto_approve"
    assert decision.fields_below_threshold == []
    assert decision.reviewer_disagreements == []
    assert decision.integration_failures == []


def test_ac_04_02_low_confidence_routes_to_human_review() -> None:
    decision = route_extraction(
        extraction=_ext(confidence={
            "policy_type": 0.95, "premium_amount": 0.65, "deductible": 0.95,
            "coverage_limit": 0.95, "endorsements": 0.95, "exclusions": 0.95,
        }),
        review=_review(_all_agree()),
        integration_findings=_integration_pass_all(),
    )
    assert decision.decision == "human_review"
    assert decision.fields_below_threshold == ["premium_amount"]


def test_ac_04_02_integration_failure_routes_to_human_review() -> None:
    failing = [
        IntegrationFinding(
            check_name="premium_matches_components_sum",
            status="fail",
            details="50 dollar discrepancy",
        ),
        IntegrationFinding(
            check_name="endorsements_exclusions_non_contradiction",
            status="pass",
            details="",
        ),
    ]
    decision = route_extraction(
        extraction=_ext(),
        review=_review(_all_agree()),
        integration_findings=failing,
    )
    assert decision.decision == "human_review"
    assert "premium_matches_components_sum" in decision.integration_failures


# ---------- Routing determinism / safety nets ----------


def test_ac_04_06_high_confidence_plus_reviewer_disagreement_still_routes_to_human_review() -> None:
    """All-0.99 extractor confidence cannot override a reviewer disagreement.

    routing is deterministic on (confidence ∧ reviewer ∧ integration).
    The model's self-rated confidence is not the sole gate.
    """
    high_conf = {
        "policy_type": 0.99, "premium_amount": 0.99, "deductible": 0.99,
        "coverage_limit": 0.99, "endorsements": 0.99, "exclusions": 0.99,
    }
    agreements = _all_agree()
    # Replace one with disagree
    agreements[1] = _disagree("premium_amount")
    decision = route_extraction(
        extraction=_ext(confidence=high_conf),
        review=_review(agreements),
        integration_findings=_integration_pass_all(),
    )
    assert decision.decision == "human_review"
    assert decision.reviewer_disagreements == ["premium_amount"]
    # extractor's self-rated confidence is irrelevant here
    assert decision.fields_below_threshold == []


# ---------- Stratified sampler ----------


def test_ac_04_03_stratified_sampler_picks_from_every_stratum_with_eligible_records() -> None:
    """No stratum that has eligible records is silently dropped."""
    decisions = [
        route_extraction(
            extraction=_ext(policy_id=f"AUTO-{i}", policy_type="auto"),
            review=_review(_all_agree()),
            integration_findings=_integration_pass_all(),
        )
        for i in range(10)
    ] + [
        route_extraction(
            extraction=_ext(policy_id=f"HOME-{i}", policy_type="home"),
            review=_review(_all_agree()),
            integration_findings=_integration_pass_all(),
        )
        for i in range(10)
    ] + [
        route_extraction(
            extraction=_ext(policy_id=f"UMB-{i}", policy_type="umbrella"),
            review=_review(_all_agree()),
            integration_findings=_integration_pass_all(),
        )
        for i in range(2)
    ]
    promoted = apply_stratified_spot_check(decisions, sample_pct=0.2, seed=42)
    auto_promoted = [d for d in promoted if d.decision == "spot_check" and "AUTO-" in d.policy_id]
    home_promoted = [d for d in promoted if d.decision == "spot_check" and "HOME-" in d.policy_id]
    umb_promoted = [d for d in promoted if d.decision == "spot_check" and "UMB-" in d.policy_id]
    # 20% of 10 -> 2 each for auto and home
    assert len(auto_promoted) >= 1
    assert len(home_promoted) >= 1
    # 20% of 2 = 0.4 -> ceil to 1, so umbrella stratum is not dropped
    assert len(umb_promoted) >= 1


def test_ac_04_03_only_auto_approve_decisions_are_eligible_for_spot_check() -> None:
    """Human_review decisions are never promoted to spot_check by the sampler."""
    low_conf = {
        "policy_type": 0.95, "premium_amount": 0.5, "deductible": 0.95,
        "coverage_limit": 0.95, "endorsements": 0.95, "exclusions": 0.95,
    }
    decisions = [
        route_extraction(
            extraction=_ext(policy_id=f"AUTO-{i}", policy_type="auto",
                            confidence=low_conf if i < 3 else None),
            review=_review(_all_agree()),
            integration_findings=_integration_pass_all(),
        )
        for i in range(10)
    ]
    promoted = apply_stratified_spot_check(decisions, sample_pct=0.5, seed=42)
    # Of 10 decisions, 3 are human_review (low conf) and 7 are auto_approve.
    # No human_review decision can flip to spot_check.
    for d in promoted:
        if d.policy_id in {"AUTO-0", "AUTO-1", "AUTO-2"}:
            assert d.decision == "human_review"


# ---------- Sliced calibration report ----------


def test_ac_04_04_calibration_report_is_sliced_by_policy_type_and_field() -> None:
    """Report cells are keyed (policy_type, field_name) with Brier + accuracy."""
    labels = [
        # auto + premium_amount: predicted 0.9 conf, was correct
        CalibrationLabel(
            policy_id="POL-A", policy_type="auto", field="premium_amount",
            predicted_confidence=0.9, correct=True,
        ),
        CalibrationLabel(
            policy_id="POL-B", policy_type="auto", field="premium_amount",
            predicted_confidence=0.9, correct=True,
        ),
        # auto + premium_amount: predicted 0.9, was wrong → Brier penalty
        CalibrationLabel(
            policy_id="POL-C", policy_type="auto", field="premium_amount",
            predicted_confidence=0.9, correct=False,
        ),
        # auto + deductible: cleanly correct
        CalibrationLabel(
            policy_id="POL-A", policy_type="auto", field="deductible",
            predicted_confidence=0.95, correct=True,
        ),
        # home + premium_amount: correct
        CalibrationLabel(
            policy_id="POL-D", policy_type="home", field="premium_amount",
            predicted_confidence=0.85, correct=True,
        ),
    ]
    report = calibration_report(labels)
    # Per-cell sliced
    auto_premium = report.cells[("auto", "premium_amount")]
    assert auto_premium.samples == 3
    assert auto_premium.mean_predicted_confidence == 0.9
    # 2/3 correct
    assert abs(auto_premium.observed_accuracy - (2 / 3)) < 1e-9
    # Brier: predicted 0.9, actuals (1,1,0). Per-sample (0.9-1)^2=0.01 + 0.01 + 0.81 = 0.83 / 3
    assert abs(auto_premium.brier_score - (0.01 + 0.01 + 0.81) / 3) < 1e-9
    # The (home, premium_amount) cell exists separately
    assert ("home", "premium_amount") in report.cells
    # Overall Brier is across all 5 samples
    expected_overall = (0.01 + 0.01 + 0.81 + (0.95 - 1) ** 2 + (0.85 - 1) ** 2) / 5
    assert abs(report.overall_brier - expected_overall) < 1e-9


def test_ac_04_04_calibration_report_handles_empty_input() -> None:
    report = calibration_report([])
    assert report.cells == {}
    assert report.overall_brier == 0.0


# ---------- JSON output file ----------


def test_ac_04_05_routing_decisions_written_to_json_file(tmp_path: Path) -> None:
    decisions = [
        route_extraction(
            extraction=_ext(policy_id="POL-A", policy_type="auto"),
            review=_review(_all_agree()),
            integration_findings=_integration_pass_all(),
        ),
        route_extraction(
            extraction=_ext(
                policy_id="POL-B", policy_type="home",
                confidence={
                    "policy_type": 0.95, "premium_amount": 0.5, "deductible": 0.95,
                    "coverage_limit": 0.95, "endorsements": 0.95, "exclusions": 0.95,
                },
            ),
            review=_review(_all_agree()),
            integration_findings=_integration_pass_all(),
        ),
    ]
    path = tmp_path / "routing.json"
    write_routing_decisions(decisions, path)
    parsed: list[dict[str, Any]] = json.loads(path.read_text())
    assert len(parsed) == 2
    record_a = next(r for r in parsed if r["policy_id"] == "POL-A")
    assert record_a["decision"] == "auto_approve"
    assert "confidence_summary" in record_a
    record_b = next(r for r in parsed if r["policy_id"] == "POL-B")
    assert record_b["decision"] == "human_review"
    assert "premium_amount" in record_b["fields_below_threshold"]
    # Required keys
    for required in (
        "policy_id", "decision", "reason", "fields_below_threshold",
        "reviewer_disagreements", "integration_failures", "confidence_summary",
    ):
        assert required in record_b
