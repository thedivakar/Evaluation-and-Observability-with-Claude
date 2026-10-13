"""Retry-with-error-feedback orchestration for policy extraction."""
from __future__ import annotations

import logging
from typing import Any

from policy_extractor.client import MessageClient
from policy_extractor.extractor import (
    EXTRACT_POLICY_TOOL,
    build_extraction_messages,
    parse_tool_use,
)
from policy_extractor.records import (
    Endorsement,
    ExtractionOutcome,
    PolicyExtraction,
    PremiumComponent,
    RetryFutileEscalation,
    ValidationError,
)
from policy_extractor.validator import validate_extraction

logger = logging.getLogger(__name__)


DEFAULT_EXTRACTOR_MODEL = "claude-haiku-4-5"


def extract_with_retry(
    *,
    client: MessageClient,
    policy_id: str,
    document_text: str,
    max_retries: int = 3,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 2048,
) -> ExtractionOutcome:
    """Extract one policy with validation-driven retry.

    Returns PolicyExtraction on success, RetryFutileEscalation when the source
    document is missing required information (no retry attempted).
    """
    prior_attempts: list[dict[str, Any]] = []
    history: list[ValidationError] = []

    for attempt_index in range(max_retries + 1):
        messages, system = build_extraction_messages(document_text, prior_attempts)
        response = client.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=[EXTRACT_POLICY_TOOL],
            tool_choice={"type": "tool", "name": "extract_policy"},
        )
        extraction = parse_tool_use(response)

        error = validate_extraction(extraction)
        if error is None:
            return build_extraction(
                policy_id=policy_id,
                extraction=extraction,
                attempt_index=attempt_index,
                history=history,
            )

        history.append(error)
        logger.info(
            "validation_failed",
            extra={
                "policy_id": policy_id,
                "attempt": attempt_index,
                "category": error.category,
                "detected_pattern": error.detected_pattern,
                "field": error.field,
            },
        )

        if error.category == "missing_source":
            return RetryFutileEscalation(
                policy_id=policy_id,
                field=error.field,
                category="missing_source",
                detected_pattern=error.detected_pattern,
                reason=error.message,
            )

        prior_attempts.append(
            {
                "extraction": extraction,
                "error_field": error.field,
                "error_category": error.category,
                "error_pattern": error.detected_pattern,
                "error_message": error.message,
            }
        )

    # All retries exhausted on format/consistency failures — escalate as well.
    last_error = history[-1]
    return RetryFutileEscalation(
        policy_id=policy_id,
        field=last_error.field,
        category="missing_source",
        detected_pattern=f"retries_exhausted__{last_error.detected_pattern}",
        reason=(
            f"Validator rejected the extraction {len(history)} times in a row. Last "
            f"failure: {last_error.message}"
        ),
    )


def build_extraction(
    *,
    policy_id: str,
    extraction: dict[str, Any],
    attempt_index: int,
    history: list[ValidationError],
) -> PolicyExtraction:
    raw_endorsements = extraction.get("endorsements")
    if raw_endorsements is None:
        endorsements: list[Endorsement] | None = None
    else:
        endorsements = [
            Endorsement(name=e["name"], limit=e.get("limit")) for e in raw_endorsements
        ]

    raw_components = extraction.get("premium_components")
    components: list[PremiumComponent] | None = None
    if raw_components is not None:
        components = [
            PremiumComponent(name=c["name"], amount=c["amount"]) for c in raw_components
        ]

    return PolicyExtraction(
        policy_id=policy_id,
        policy_type=extraction["policy_type"],
        premium_amount=extraction["premium_amount"],
        deductible=extraction["deductible"],
        coverage_limit=extraction["coverage_limit"],
        endorsements=endorsements,
        exclusions=list(extraction.get("exclusions") or []),
        premium_components=components,
        confidence=dict(extraction.get("confidence") or {}),
        retry_count=attempt_index,
        final_attempt_index=attempt_index,
        validation_history=list(history),
    )
