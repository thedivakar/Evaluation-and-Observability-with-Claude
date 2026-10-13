"""Validates extracted policy records and categorises failures.

The validator only handles semantic and consistency errors. JSON-syntax errors
are eliminated upstream by tool_choice-forced structured tool calls — the SDK
enforces the schema before this validator runs.
"""
from __future__ import annotations

from typing import Any

from policy_extractor.records import ValidationError


def validate_extraction(extraction: dict[str, Any]) -> ValidationError | None:
    """Return the first ValidationError found, or None if extraction is clean.

    Checks run in this order:
      1. Required fields absent (null) → missing_source
      2. Numeric fields out of legal range (negative premium, etc.) → format
      3. Cross-field consistency (premium vs sum of components) → consistency
    """
    err = _check_required_present(extraction)
    if err is not None:
        return err

    err = _check_numeric_ranges(extraction)
    if err is not None:
        return err

    err = _check_premium_components_consistency(extraction)
    if err is not None:
        return err

    return None


_REQUIRED_FIELDS = ("premium_amount", "deductible", "coverage_limit", "endorsements", "exclusions")


def _check_required_present(extraction: dict[str, Any]) -> ValidationError | None:
    for field_name in _REQUIRED_FIELDS:
        value = extraction.get(field_name)
        if value is None:
            return ValidationError(
                field=field_name,
                observed_value="null",
                category="missing_source",
                detected_pattern=f"{field_name}_absent",
                message=(
                    f"Field {field_name!r} returned null — the source document does not "
                    "contain this information. Retry is futile; escalate to human review."
                ),
            )
    return None


def _check_numeric_ranges(extraction: dict[str, Any]) -> ValidationError | None:
    premium = extraction.get("premium_amount")
    if isinstance(premium, (int, float)) and premium < 0:
        return ValidationError(
            field="premium_amount",
            observed_value=str(premium),
            category="format",
            detected_pattern="negative_premium",
            message=(
                f"premium_amount is {premium}, which is negative. Premium amounts must "
                "be a positive number in USD. Re-read the document and extract the "
                "correct positive value."
            ),
        )

    deductible = extraction.get("deductible")
    if isinstance(deductible, (int, float)) and deductible < 0:
        return ValidationError(
            field="deductible",
            observed_value=str(deductible),
            category="format",
            detected_pattern="negative_deductible",
            message=(
                f"deductible is {deductible}, which is negative. Deductibles must be a "
                "non-negative number in USD."
            ),
        )

    coverage_limit = extraction.get("coverage_limit")
    if isinstance(coverage_limit, (int, float)) and coverage_limit < 0:
        return ValidationError(
            field="coverage_limit",
            observed_value=str(coverage_limit),
            category="format",
            detected_pattern="negative_coverage_limit",
            message=f"coverage_limit is {coverage_limit}, which is negative.",
        )
    return None


def _check_premium_components_consistency(
    extraction: dict[str, Any],
) -> ValidationError | None:
    components = extraction.get("premium_components")
    premium = extraction.get("premium_amount")
    if not components or not isinstance(premium, (int, float)):
        return None
    component_sum = sum(c["amount"] for c in components)
    if abs(component_sum - premium) > 0.01:
        return ValidationError(
            field="premium_amount",
            observed_value=f"stated={premium}, components_sum={component_sum}",
            category="consistency",
            detected_pattern="premium_does_not_match_components",
            message=(
                f"premium_amount is stated as {premium} but the sum of premium_components "
                f"is {component_sum} (delta {premium - component_sum:+.2f}). Re-read "
                "the document and reconcile: either the stated premium is wrong or one "
                "of the components is wrong."
            ),
        )
    return None
