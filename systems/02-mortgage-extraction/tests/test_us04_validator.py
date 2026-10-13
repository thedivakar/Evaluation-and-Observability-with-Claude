"""Tests for the mathematical-consistency validator."""
from __future__ import annotations

from pathlib import Path

from mortgage_extractor.client import RecordingClient
from mortgage_extractor.models import (
    Borrower,
    Discrepancy,
    Income,
    MortgageExtraction,
    ValidationReport,
)
from mortgage_extractor.pipeline import Pipeline
from mortgage_extractor.validator import validate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "documents"


def _income_only_extraction(**income_kwargs: float | None) -> MortgageExtraction:
    return MortgageExtraction(
        borrower=Borrower(full_name="Test Borrower"),
        income=Income(**income_kwargs),
    )


def test_ac_04_01_extraction_carries_both_calculated_and_stated() -> None:
    extraction = _income_only_extraction(
        base_monthly=4000.0,
        bonus_monthly=500.0,
        commission_monthly=0.0,
        stated_monthly_total=4500.0,
    )
    assert extraction.income is not None
    assert extraction.income.calculated_monthly_total == 4500.0
    assert extraction.income.stated_monthly_total == 4500.0


def test_ac_04_02_consistent_when_within_tolerance() -> None:
    extraction = _income_only_extraction(
        base_monthly=4000.0,
        bonus_monthly=500.50,
        stated_monthly_total=4500.0,
    )
    report = validate(extraction)
    assert report.consistent is True
    assert report.discrepancies == []


def test_ac_04_02_inconsistent_when_beyond_tolerance() -> None:
    extraction = _income_only_extraction(
        base_monthly=4000.0,
        bonus_monthly=500.0,
        stated_monthly_total=5000.0,
    )
    report = validate(extraction)
    assert report.consistent is False
    assert len(report.discrepancies) == 1


def test_ac_04_02_default_tolerance_is_one_dollar() -> None:
    just_within = _income_only_extraction(
        base_monthly=4000.0, stated_monthly_total=4001.0
    )
    assert validate(just_within).consistent is True

    just_outside = _income_only_extraction(
        base_monthly=4000.0, stated_monthly_total=4001.50
    )
    assert validate(just_outside).consistent is False


def test_ac_04_03_discrepancy_carries_field_calc_stated_delta() -> None:
    extraction = _income_only_extraction(
        base_monthly=4000.0,
        bonus_monthly=500.0,
        stated_monthly_total=5000.0,
    )
    report = validate(extraction)
    [disc] = report.discrepancies
    assert isinstance(disc, Discrepancy)
    assert disc.field == "total_monthly_income"
    assert disc.calculated == 4500.0
    assert disc.stated == 5000.0
    assert disc.delta == -500.0


def test_ac_04_04_real_paystub_with_sum_mismatch_is_flagged() -> None:
    document = (FIXTURES / "income_sum_mismatch.txt").read_text()
    pipeline = Pipeline(client=RecordingClient(mode="auto"))
    extraction = pipeline.run(document)

    report = validate(extraction)

    assert report.consistent is False
    assert any(
        d.field == "total_monthly_income" for d in report.discrepancies
    ), f"expected a total_monthly_income discrepancy, got {report.discrepancies!r}"


def test_clean_extraction_yields_consistent_true() -> None:
    extraction = _income_only_extraction(
        base_monthly=4800.0,
        bonus_monthly=400.0,
        commission_monthly=0.0,
        stated_monthly_total=5200.0,
    )
    report = validate(extraction)
    assert report.consistent is True


def test_validation_report_round_trips_through_json() -> None:
    report = ValidationReport(
        consistent=False,
        discrepancies=[
            Discrepancy(
                field="total_monthly_income",
                calculated=4500.0,
                stated=5000.0,
                delta=-500.0,
            )
        ],
    )
    payload = report.model_dump_json()
    restored = ValidationReport.model_validate_json(payload)
    assert restored == report
