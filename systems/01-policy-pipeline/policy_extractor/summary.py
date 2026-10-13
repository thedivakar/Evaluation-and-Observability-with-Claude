"""Run-level aggregator over validation_history + escalation records."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from policy_extractor.records import PolicyExtraction, RetryFutileEscalation


class PatternRow(TypedDict):
    count: int
    policies: list[str]
    categories: list[str]


def summarize_patterns(
    outcomes: Sequence[PolicyExtraction | RetryFutileEscalation],
) -> dict[str, PatternRow]:
    """Aggregate detected_patterns across a run.

    Returns a frequency table:
      {
        "negative_premium": {"count": 5, "policies": ["POL-A", "POL-B", ...],
                             "categories": ["format"]},
        "endorsements_absent": {"count": 1, "policies": ["POL-C"],
                                "categories": ["missing_source"]},
      }

    Counts unique (policy, pattern) pairs — a policy that hit the same pattern
    twice during retries appears once.
    """
    table: dict[str, PatternRow] = {}

    for outcome in outcomes:
        if isinstance(outcome, RetryFutileEscalation):
            _bump(table, outcome.detected_pattern, outcome.policy_id, outcome.category)
        else:
            seen_for_policy: set[str] = set()
            for err in outcome.validation_history:
                if err.detected_pattern in seen_for_policy:
                    continue
                seen_for_policy.add(err.detected_pattern)
                _bump(table, err.detected_pattern, outcome.policy_id, err.category)

    return table


def _bump(table: dict[str, PatternRow], pattern: str, policy_id: str, category: str) -> None:
    row = table.get(pattern)
    if row is None:
        table[pattern] = {"count": 1, "policies": [policy_id], "categories": [category]}
        return
    row["count"] += 1
    if policy_id not in row["policies"]:
        row["policies"].append(policy_id)
    if category not in row["categories"]:
        row["categories"].append(category)
