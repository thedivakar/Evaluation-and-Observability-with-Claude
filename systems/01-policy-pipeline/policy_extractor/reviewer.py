"""Independent review + within-policy integration pass.

Two separate concerns share this module because both are "second-pass" QA steps:
  1. independent_review() — a separate Claude call with no extractor context, returning
     per-field agree/disagree + review_confidence.
  2. integration_pass() — pure cross-field consistency checks within one extraction.

The output of both feeds the routing layer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from anthropic.types import ToolUseBlock

from policy_extractor.client import MessageClient
from policy_extractor.records import Endorsement, PolicyExtraction

DEFAULT_EXTRACTOR_MODEL = "claude-haiku-4-5"
DEFAULT_REVIEWER_MODEL = "claude-sonnet-4-6"

REVIEW_FIELDS = [
    "policy_type",
    "premium_amount",
    "deductible",
    "coverage_limit",
    "endorsements",
    "exclusions",
]


REVIEW_TOOL: dict[str, Any] = {
    "name": "review_extraction",
    "description": (
        "Review an extracted policy record against the source document. For each field, "
        "state whether the extracted value agrees with the document, and self-rate your "
        "review confidence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field_reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": REVIEW_FIELDS},
                        "agreement": {"type": "string", "enum": ["agree", "disagree"]},
                        "reason": {
                            "type": ["string", "null"],
                            "description": "Required when agreement='disagree'.",
                        },
                        "review_confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["field", "agreement", "review_confidence"],
                },
            },
        },
        "required": ["field_reviews"],
    },
}


REVIEWER_SYSTEM_PROMPT = """You review extracted insurance-policy records against the \
original source document. You receive only the source document and the proposed \
extraction — you have no access to the extractor's reasoning, tool-call history, or \
scratchpad. Your judgment must be independent.

For each of the listed fields, state whether the extracted value agrees with the \
source document, and self-rate your review_confidence in [0.0, 1.0]. If you disagree, \
explain briefly *why* — what the document actually says vs what was extracted.

You must call the review_extraction tool exactly once.
"""


@dataclass(frozen=True)
class FieldAgreement:
    field: str
    agreement: Literal["agree", "disagree"]
    reason: str | None
    review_confidence: float


@dataclass
class ReviewResult:
    agreements: dict[str, FieldAgreement]


def build_review_messages(
    *,
    source_document: str,
    extracted_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Construct the reviewer's (messages, system) tuple.

    Critically: nothing from the extractor's prompts, reasoning, or tool-call history
    flows into this prompt. Only the raw source document + the proposed extraction.
    """
    extraction_json = json.dumps(extracted_record, indent=2, sort_keys=True)
    user_content = (
        "<source_document>\n"
        f"{source_document.strip()}\n"
        "</source_document>\n\n"
        "<proposed_extraction>\n"
        f"{extraction_json}\n"
        "</proposed_extraction>"
    )
    return [{"role": "user", "content": user_content}], REVIEWER_SYSTEM_PROMPT


def independent_review(
    *,
    client: MessageClient,
    source_document: str,
    extracted_record: dict[str, Any],
    model: str = DEFAULT_REVIEWER_MODEL,
    max_tokens: int = 2048,
) -> ReviewResult:
    """Run the reviewer and parse its per-field judgement."""
    messages, system = build_review_messages(
        source_document=source_document,
        extracted_record=extracted_record,
    )
    response = client.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=[REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "review_extraction"},
    )
    field_reviews = _parse_field_reviews(response)
    return ReviewResult(
        agreements={
            fr["field"]: FieldAgreement(
                field=fr["field"],
                agreement=fr["agreement"],
                reason=fr.get("reason"),
                review_confidence=float(fr["review_confidence"]),
            )
            for fr in field_reviews
        }
    )


def _parse_field_reviews(message: Any) -> list[dict[str, Any]]:
    for block in message.content:
        if isinstance(block, ToolUseBlock) or getattr(block, "type", None) == "tool_use":
            payload = dict(block.input)
            reviews = payload.get("field_reviews")
            if isinstance(reviews, list):
                return [dict(r) for r in reviews]
    raise ValueError(
        "Reviewer response did not contain a review_extraction tool_use block."
    )


def has_disagreement(result: ReviewResult) -> bool:
    return any(a.agreement == "disagree" for a in result.agreements.values())


# ---------- Integration pass ----------


@dataclass(frozen=True)
class IntegrationFinding:
    check_name: str
    status: Literal["pass", "fail"]
    details: str


def integration_pass(extraction: PolicyExtraction) -> list[IntegrationFinding]:
    """Cross-field consistency checks. Pure function over a single extraction."""
    findings: list[IntegrationFinding] = [
        _check_coverage_limit_vs_endorsements(extraction),
        _check_endorsements_vs_exclusions(extraction),
    ]
    components_check = _check_premium_vs_components(extraction)
    if components_check is not None:
        findings.append(components_check)
    return findings


def _check_coverage_limit_vs_endorsements(extraction: PolicyExtraction) -> IntegrationFinding:
    if extraction.coverage_limit is None or extraction.endorsements is None:
        return IntegrationFinding(
            check_name="coverage_limit_exceeds_endorsement_sum",
            status="pass",
            details="coverage_limit or endorsements absent — check skipped.",
        )
    endorsement_total = sum(
        e.limit for e in extraction.endorsements if e.limit is not None
    )
    if endorsement_total == 0:
        return IntegrationFinding(
            check_name="coverage_limit_exceeds_endorsement_sum",
            status="pass",
            details="No endorsements carry numeric limits.",
        )
    if extraction.coverage_limit >= endorsement_total:
        return IntegrationFinding(
            check_name="coverage_limit_exceeds_endorsement_sum",
            status="pass",
            details=(
                f"coverage_limit={extraction.coverage_limit} >= "
                f"sum(endorsement.limit)={endorsement_total}."
            ),
        )
    return IntegrationFinding(
        check_name="coverage_limit_exceeds_endorsement_sum",
        status="fail",
        details=(
            f"coverage_limit={extraction.coverage_limit} is less than the sum of "
            f"endorsement limits ({endorsement_total}). The primary coverage cannot "
            "be smaller than what the endorsements promise."
        ),
    )


_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "by", "with",
        "from", "at", "is", "are", "was", "were", "be", "been", "being", "as",
        "endorsement", "endorsements", "exclusion", "exclusions", "coverage",
        "policy", "insurance", "any", "all", "this", "that", "these", "those",
    }
)


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-z]+", text)
        if t.lower() not in _STOPWORDS and len(t) > 2
    ]
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _check_endorsements_vs_exclusions(extraction: PolicyExtraction) -> IntegrationFinding:
    if not extraction.endorsements:
        return IntegrationFinding(
            check_name="endorsements_exclusions_non_contradiction",
            status="pass",
            details="No endorsements to compare against exclusions.",
        )
    contradictions: list[tuple[Endorsement, str]] = []
    for endorsement in extraction.endorsements:
        endorsement_bigrams = _bigrams(endorsement.name)
        if not endorsement_bigrams:
            continue
        for exclusion in extraction.exclusions:
            if endorsement_bigrams & _bigrams(exclusion):
                contradictions.append((endorsement, exclusion))
    if not contradictions:
        return IntegrationFinding(
            check_name="endorsements_exclusions_non_contradiction",
            status="pass",
            details="No overlapping noun-phrases between endorsements and exclusions.",
        )
    detail = "; ".join(
        f"endorsement '{e.name}' overlaps with exclusion '{x}'" for e, x in contradictions
    )
    return IntegrationFinding(
        check_name="endorsements_exclusions_non_contradiction",
        status="fail",
        details=f"Potential contradiction(s): {detail}",
    )


def _check_premium_vs_components(extraction: PolicyExtraction) -> IntegrationFinding | None:
    if not extraction.premium_components or extraction.premium_amount is None:
        return None
    component_sum = sum(c.amount for c in extraction.premium_components)
    if abs(component_sum - extraction.premium_amount) <= 0.01:
        return IntegrationFinding(
            check_name="premium_matches_components_sum",
            status="pass",
            details=f"premium={extraction.premium_amount} == sum(components)={component_sum}.",
        )
    return IntegrationFinding(
        check_name="premium_matches_components_sum",
        status="fail",
        details=(
            f"premium={extraction.premium_amount} but sum(components)={component_sum} "
            f"(delta {extraction.premium_amount - component_sum:+.2f})."
        ),
    )
