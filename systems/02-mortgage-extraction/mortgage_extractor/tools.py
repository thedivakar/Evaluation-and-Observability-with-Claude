"""Anthropic tool definitions for the mortgage extraction pipeline.

A "tool" in the Anthropic Messages API is a JSON object with a ``name``,
``description``, and ``input_schema`` (JSON Schema). When the API returns a
``tool_use`` content block, its ``input`` is guaranteed to validate against
this schema, which is how this project enforces structured output.

The canonical extractor tool, :func:`extract_mortgage_data`, is registered for
the second pass of the pipeline (``tool_choice="any"`` against the per-doc-type
extractor set; see :mod:`mortgage_extractor.pipeline`). The classifier tool,
:func:`classify_document`, is registered for the forced first pass.
"""
from __future__ import annotations

from typing import TypedDict

from mortgage_extractor.models import DocumentType
from mortgage_extractor.schema import JsonSchema, mortgage_data_schema


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: JsonSchema


_EXTRACT_DESCRIPTION = (
    "Extract structured mortgage data from a single document. Use null for any "
    "field not explicitly stated in the document — do not infer, default, or "
    "fabricate. When a categorical field's value is not in the listed enum, "
    "emit 'other' and place the actual value in the corresponding *_detail "
    "field."
)


def extract_mortgage_data() -> ToolDefinition:
    """Return the canonical mortgage-data extractor tool definition."""
    return {
        "name": "extract_mortgage_data",
        "description": _EXTRACT_DESCRIPTION,
        "input_schema": mortgage_data_schema(),
    }


_CLASSIFY_DESCRIPTION = (
    "Classify a mortgage document into one of the supported document types, "
    "or 'other' if none apply. Always provide a one-sentence reason describing "
    "the textual cues that drove the classification."
)


def classify_document() -> ToolDefinition:
    """Return the document-classifier tool definition."""
    return {
        "name": "classify_document",
        "description": _CLASSIFY_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": [
                        "loan_application",
                        "appraisal",
                        "income_verification",
                        "other",
                    ],
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One sentence describing the textual cues that drove "
                        "the classification."
                    ),
                },
            },
            "required": ["document_type", "reason"],
        },
    }


def doc_type_extractor(doc_type: DocumentType) -> ToolDefinition:
    """Return a doc-type-tailored extractor tool.

    The canonical schema is shared across document types, but ``required`` is
    narrowed to the fields a given doc type is expected to carry. (A
    loan-application document is expected to have borrower / property / loan
    sections; an appraisal is expected to have property; an income-verification
    document is expected to have borrower / income.) The model still emits the
    full record — these per-type required lists just tell it which sections it
    must populate vs. may leave absent.
    """
    schema = mortgage_data_schema()
    schema["required"] = _required_sections_for(doc_type)
    return {
        "name": f"extract_{doc_type.value}",
        "description": (
            f"Extract structured mortgage data from a {doc_type.value} "
            "document. Use null for any field not explicitly stated. "
            "When a categorical field's value is not in the listed enum, "
            "emit 'other' and write the actual value into the corresponding "
            "*_detail field."
        ),
        "input_schema": schema,
    }


def flag_for_review() -> ToolDefinition:
    """Return the escape-hatch tool the model calls when it cannot extract.

    Registered alongside the doc-type-specific extractor for the
    ``tool_choice="any"`` extraction pass; the model must call one of the two.
    """
    return {
        "name": "flag_for_review",
        "description": (
            "Call this when the document is too unclear, damaged, or off-topic "
            "to extract confidently. Provide a one-sentence reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One sentence describing why extraction is not possible.",
                },
            },
            "required": ["reason"],
        },
    }


def _required_sections_for(doc_type: DocumentType) -> list[str]:
    """Which top-level sections must appear in the extractor's output.

    A loan application document carries borrower, property, and loan details
    together. An appraisal centres on the property. An income-verification
    document identifies the borrower and reports their income; it typically
    does not state the loan amount or describe the property.
    """
    match doc_type:
        case DocumentType.LOAN_APPLICATION:
            return ["borrower", "property", "loan"]
        case DocumentType.APPRAISAL:
            return ["property"]
        case DocumentType.INCOME_VERIFICATION:
            return ["borrower", "income"]
        case DocumentType.OTHER:
            raise ValueError(
                "doc_type_extractor should never be called for DocumentType.OTHER"
            )


__all__ = [
    "ToolDefinition",
    "classify_document",
    "doc_type_extractor",
    "extract_mortgage_data",
    "flag_for_review",
]
