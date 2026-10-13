"""Tests for the two-pass tool_choice pipeline."""
from __future__ import annotations

import pytest

from mortgage_extractor.client import RecordingClient
from mortgage_extractor.errors import UnsupportedDocumentTypeError
from mortgage_extractor.models import DocumentType, MortgageExtraction
from mortgage_extractor.pipeline import Pipeline

LOAN_APP_TEXT = """\
UNIFORM RESIDENTIAL LOAN APPLICATION
Borrower: Jordan Chen
Co-Borrower: Priya Chen
Property Address: 4218 Magnolia Street, Sacramento, CA 95816
Property Type: Single Family
Occupancy: Primary Residence
Year Built: 1962
Gross Living Area: 1,840 sq ft
Loan Amount: $485,000
Loan Term: 360 months
Interest Rate: 6.875%
Loan Purpose: Purchase
"""

NOT_A_MORTGAGE_TEXT = """\
PHO BAC RECIPE
For broth:
  - 4 lbs beef bones
  - 1 yellow onion, charred
  - 1 piece ginger
  - Star anise, cloves, cinnamon
Simmer 6-8 hours.
"""


def _pipeline() -> Pipeline:
    return Pipeline(client=RecordingClient(mode="auto"))


def test_ac_02_01_classify_uses_forced_tool_choice() -> None:
    pipeline = _pipeline()
    classification = pipeline.classify_document(LOAN_APP_TEXT)

    assert classification.document_type is DocumentType.LOAN_APPLICATION

    sent = pipeline.client.last_request
    assert sent is not None
    assert sent["tool_choice"] == {"type": "tool", "name": "classify_document"}
    assert [tool["name"] for tool in sent["tools"]] == ["classify_document"]


def test_ac_02_02_extract_uses_any_tool_choice_against_doc_type_tools() -> None:
    pipeline = _pipeline()
    result = pipeline.extract(LOAN_APP_TEXT, DocumentType.LOAN_APPLICATION)

    assert isinstance(result, MortgageExtraction)

    sent = pipeline.client.last_request
    assert sent is not None
    assert sent["tool_choice"] == {"type": "any"}
    tool_names = [tool["name"] for tool in sent["tools"]]
    assert "extract_loan_application" in tool_names
    assert "flag_for_review" in tool_names
    assert len(tool_names) >= 2


def test_ac_02_03_extract_returns_typed_mortgage_extraction() -> None:
    pipeline = _pipeline()
    result = pipeline.extract(LOAN_APP_TEXT, DocumentType.LOAN_APPLICATION)

    assert isinstance(result, MortgageExtraction)
    assert isinstance(result.borrower.full_name, str)
    assert result.borrower.full_name
    assert isinstance(result.loan.amount, float)
    assert result.loan.amount > 0


def test_ac_02_04_other_short_circuits_before_extract() -> None:
    pipeline = _pipeline()

    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        pipeline.run(NOT_A_MORTGAGE_TEXT)

    assert exc_info.value.reason
    classify_request = pipeline.client.last_request
    assert classify_request is not None
    assert classify_request["tool_choice"]["type"] == "tool"


def test_ac_02_04_extract_directly_with_other_raises() -> None:
    pipeline = _pipeline()
    with pytest.raises(UnsupportedDocumentTypeError):
        pipeline.extract(LOAN_APP_TEXT, DocumentType.OTHER)


def test_ac_02_05_default_model_is_haiku_and_overridable() -> None:
    default = Pipeline(client=RecordingClient(mode="replay"))
    assert default.model == "claude-haiku-4-5"

    overridden = Pipeline(
        client=RecordingClient(mode="replay"),
        model="claude-sonnet-4-6",
    )
    assert overridden.model == "claude-sonnet-4-6"
