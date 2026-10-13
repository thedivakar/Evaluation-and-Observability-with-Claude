"""Mathematical-consistency validator.

The extractor pulls two redundant values from any line-itemized total: the
``calculated`` sum (computed in Python from the per-component fields) and the
``stated`` total (extracted verbatim from the document). When the document is
internally consistent these agree; when they don't, the discrepancy is
surfaced for human review rather than silently corrected.

The default tolerance is one dollar — small enough to catch real arithmetic
errors, loose enough to absorb cent-level OCR rounding that upstream document
conversion sometimes introduces. Override per-call when a domain demands
zero-tolerance or a percentage-based threshold.
"""
from __future__ import annotations

from mortgage_extractor.config import DEFAULT_TOLERANCE_USD
from mortgage_extractor.models import (
    Discrepancy,
    MortgageExtraction,
    ValidationReport,
)


def validate(
    extraction: MortgageExtraction,
    *,
    tolerance: float = DEFAULT_TOLERANCE_USD,
) -> ValidationReport:
    """Return a consistency report for ``extraction``.

    The validator currently checks one line-itemized total: ``total_monthly_income``,
    comparing ``Income.calculated_monthly_total`` against
    ``Income.stated_monthly_total``. Additional checks would be added the same
    way (compute, compare against stated, emit a :class:`Discrepancy` on mismatch).
    """
    discrepancies: list[Discrepancy] = []

    if extraction.income is not None:
        calculated = extraction.income.calculated_monthly_total
        stated = extraction.income.stated_monthly_total
        if calculated is not None and stated is not None:
            delta = round(calculated - stated, 2)
            if abs(delta) > tolerance:
                discrepancies.append(
                    Discrepancy(
                        field="total_monthly_income",
                        calculated=round(calculated, 2),
                        stated=round(stated, 2),
                        delta=delta,
                    )
                )

    return ValidationReport(
        consistent=not discrepancies,
        discrepancies=discrepancies,
    )
