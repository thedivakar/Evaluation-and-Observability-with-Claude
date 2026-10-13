"""HITL routing with stratified sampling and confidence calibration.

The routing layer is deliberately deterministic: the decision for each extraction is
a pure function of (extractor confidence, reviewer agreement, integration findings).
The model's self-rated confidence is one of three inputs, not the sole gate.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from policy_extractor.records import PolicyExtraction
from policy_extractor.reviewer import IntegrationFinding, ReviewResult

DEFAULT_CONFIDENCE_THRESHOLD = 0.90

Decision = Literal["auto_approve", "human_review", "spot_check"]


@dataclass
class RoutingDecision:
    policy_id: str
    policy_type: str
    decision: Decision
    reason: str
    fields_below_threshold: list[str]
    reviewer_disagreements: list[str]
    integration_failures: list[str]
    confidence_summary: dict[str, float]


def route_extraction(
    *,
    extraction: PolicyExtraction,
    review: ReviewResult,
    integration_findings: Sequence[IntegrationFinding],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> RoutingDecision:
    """Classify a single extraction as auto_approve or human_review.

    spot_check is assigned later by apply_stratified_spot_check over a batch.
    """
    fields_below = sorted(
        f for f, c in extraction.confidence.items() if c < threshold
    )
    disagreements = sorted(
        f for f, a in review.agreements.items() if a.agreement == "disagree"
    )
    integration_failures = sorted(
        f.check_name for f in integration_findings if f.status == "fail"
    )

    if fields_below or disagreements or integration_failures:
        reasons: list[str] = []
        if fields_below:
            reasons.append(
                f"fields_below_threshold={fields_below}"
            )
        if disagreements:
            reasons.append(f"reviewer_disagreement={disagreements}")
        if integration_failures:
            reasons.append(f"integration_failure={integration_failures}")
        reason = "; ".join(reasons)
        decision: Decision = "human_review"
    else:
        decision = "auto_approve"
        reason = "all confidence at/above threshold, reviewer agrees, integration clean"

    return RoutingDecision(
        policy_id=extraction.policy_id,
        policy_type=extraction.policy_type,
        decision=decision,
        reason=reason,
        fields_below_threshold=fields_below,
        reviewer_disagreements=disagreements,
        integration_failures=integration_failures,
        confidence_summary=dict(extraction.confidence),
    )


def apply_stratified_spot_check(
    decisions: Sequence[RoutingDecision],
    *,
    sample_pct: float,
    seed: int | None = None,
) -> list[RoutingDecision]:
    """Promote a stratified sample of auto_approve decisions to spot_check.

    Groups by policy_type; from each stratum, promotes ceil(sample_pct * stratum_size)
    decisions to spot_check. A stratum with at least one eligible record never gets
    zero spot-checks — this is the drift-detection net.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = defaultdict(list)
    for idx, d in enumerate(decisions):
        if d.decision == "auto_approve":
            by_type[d.policy_type].append(idx)

    promoted_indices: set[int] = set()
    for indices in by_type.values():
        n = len(indices)
        if n == 0:
            continue
        k = max(1, math.ceil(sample_pct * n))
        k = min(k, n)
        promoted_indices.update(rng.sample(indices, k))

    out: list[RoutingDecision] = []
    for idx, decision in enumerate(decisions):
        if idx in promoted_indices:
            out.append(
                RoutingDecision(
                    policy_id=decision.policy_id,
                    policy_type=decision.policy_type,
                    decision="spot_check",
                    reason="stratified drift-detection sample",
                    fields_below_threshold=decision.fields_below_threshold,
                    reviewer_disagreements=decision.reviewer_disagreements,
                    integration_failures=decision.integration_failures,
                    confidence_summary=decision.confidence_summary,
                )
            )
        else:
            out.append(decision)
    return out


def write_routing_decisions(
    decisions: Iterable[RoutingDecision],
    path: Path,
) -> None:
    """Write routing decisions to a single JSON file."""
    payload = [asdict(d) for d in decisions]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------- Calibration ----------


@dataclass(frozen=True)
class CalibrationLabel:
    policy_id: str
    policy_type: str
    field: str
    predicted_confidence: float
    correct: bool


@dataclass
class CalibrationCell:
    policy_type: str
    field: str
    samples: int
    mean_predicted_confidence: float
    observed_accuracy: float
    brier_score: float


@dataclass
class CalibrationReport:
    cells: dict[tuple[str, str], CalibrationCell] = field(default_factory=dict)
    overall_brier: float = 0.0


def calibration_report(labels: Sequence[CalibrationLabel]) -> CalibrationReport:
    """Produce a (policy_type x field) sliced reliability report.

    For each cell, compute mean predicted confidence, observed accuracy, and Brier
    score (mean squared error between predicted_confidence and the binary outcome).
    Also reports the overall Brier across all samples.
    """
    if not labels:
        return CalibrationReport()

    buckets: dict[tuple[str, str], list[CalibrationLabel]] = defaultdict(list)
    for label in labels:
        buckets[(label.policy_type, label.field)].append(label)

    cells: dict[tuple[str, str], CalibrationCell] = {}
    for (ptype, field_name), bucket in buckets.items():
        n = len(bucket)
        mean_conf = sum(b.predicted_confidence for b in bucket) / n
        accuracy = sum(1 for b in bucket if b.correct) / n
        brier = sum(
            (b.predicted_confidence - (1.0 if b.correct else 0.0)) ** 2 for b in bucket
        ) / n
        cells[(ptype, field_name)] = CalibrationCell(
            policy_type=ptype,
            field=field_name,
            samples=n,
            mean_predicted_confidence=mean_conf,
            observed_accuracy=accuracy,
            brier_score=brier,
        )

    overall = sum(
        (label.predicted_confidence - (1.0 if label.correct else 0.0)) ** 2 for label in labels
    ) / len(labels)
    return CalibrationReport(cells=cells, overall_brier=overall)
