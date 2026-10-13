"""JSON Schema for mortgage document extraction.

The schema is a single canonical structure spanning all three document types
(loan application, appraisal, income verification). Per-document-type extractor
tools share these properties but differ in their `required` lists.

Nullable fields use the union-type idiom (``type: ["<base>", "null"]``) so the
model can return ``null`` for absent fields rather than fabricating values.
Categorical fields that may grow over time use the ``enum + "other" + *_detail``
pattern: when none of the enum members fit, the model emits ``"other"`` and
writes a free-text reason into the sibling ``_detail`` field.
"""
from __future__ import annotations

from typing import Any

JsonSchema = dict[str, Any]


PROPERTY_TYPES = [
    "single_family",
    "condo",
    "townhouse",
    "multi_family_2_4_unit",
    "manufactured",
    "planned_unit_development",
    "other",
]

OCCUPANCY_TYPES = ["primary_residence", "second_home", "investment", "other"]

LOAN_PURPOSES = ["purchase", "refinance_rate_term", "refinance_cash_out", "other"]


def mortgage_data_schema() -> JsonSchema:
    """Return the canonical JSON Schema for mortgage data extraction."""
    return {
        "type": "object",
        "properties": {
            "borrower": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "coborrower_name": {"type": ["string", "null"]},
                    "ssn_last4": {"type": ["string", "null"]},
                    "date_of_birth": {
                        "type": ["string", "null"],
                        "description": "ISO 8601 date (YYYY-MM-DD).",
                    },
                    "email": {"type": ["string", "null"]},
                    "phone": {"type": ["string", "null"]},
                },
                "required": ["full_name"],
            },
            "property": {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "property_type": {
                        "type": "string",
                        "enum": PROPERTY_TYPES,
                    },
                    "property_type_detail": {
                        "type": ["string", "null"],
                        "description": (
                            "Free-text description when property_type is 'other'. "
                            "Null when property_type is a known value."
                        ),
                    },
                    "occupancy_type": {
                        "type": ["string", "null"],
                        "enum": [*OCCUPANCY_TYPES, None],
                    },
                    "occupancy_type_detail": {"type": ["string", "null"]},
                    "year_built": {"type": ["integer", "null"]},
                    "gross_living_area_sqft": {"type": ["integer", "null"]},
                    "hoa_dues_monthly": {"type": ["number", "null"]},
                    "appraised_value": {"type": ["number", "null"]},
                },
                "required": ["address", "property_type"],
            },
            "loan": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "term_months": {"type": ["integer", "null"]},
                    "interest_rate": {
                        "type": ["number", "null"],
                        "description": "Decimal rate, e.g. 0.0625 for 6.25%.",
                    },
                    "loan_purpose": {
                        "type": ["string", "null"],
                        "enum": [*LOAN_PURPOSES, None],
                    },
                    "loan_purpose_detail": {"type": ["string", "null"]},
                    "loan_program": {"type": ["string", "null"]},
                },
                "required": ["amount"],
            },
            "income": {
                "type": "object",
                "properties": {
                    "base_monthly": {"type": ["number", "null"]},
                    "bonus_monthly": {"type": ["number", "null"]},
                    "bonus_ytd": {
                        "type": ["number", "null"],
                        "description": (
                            "Year-to-date bonus. Tracked separately from "
                            "`bonus_monthly` so the YTD column on a paystub "
                            "can be captured even when the current period had "
                            "no bonus."
                        ),
                    },
                    "commission_monthly": {"type": ["number", "null"]},
                    "overtime_monthly": {"type": ["number", "null"]},
                    "other_monthly": {"type": ["number", "null"]},
                    "stated_monthly_total": {
                        "type": ["number", "null"],
                        "description": (
                            "Total monthly income as stated by the document, "
                            "extracted verbatim. The validator cross-checks "
                            "this against the sum of the per-component "
                            "monthly fields above."
                        ),
                    },
                },
            },
        },
        "required": ["borrower", "property", "loan"],
    }


def list_nullable_fields(schema: JsonSchema) -> list[str]:
    """Return dotted paths of every nullable leaf field in the schema.

    A leaf is "nullable" when its declared ``type`` is a list containing
    ``"null"`` (the JSON Schema idiom for union with null). Object-typed
    properties are traversed recursively.
    """
    paths: list[str] = []
    _walk(schema, prefix="", out=paths)
    return paths


def _walk(node: JsonSchema, prefix: str, out: list[str]) -> None:
    properties = node.get("properties")
    if not isinstance(properties, dict):
        return
    for name, value in properties.items():
        path = f"{prefix}.{name}" if prefix else name
        type_field = value.get("type")
        if isinstance(type_field, list) and "null" in type_field:
            out.append(path)
            continue
        if value.get("type") == "object":
            _walk(value, prefix=path, out=out)
