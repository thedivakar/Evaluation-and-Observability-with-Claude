"""Single-shot policy extraction: tool schema, prompt assembly, response parsing."""
from __future__ import annotations

from typing import Any

from anthropic.types import ToolUseBlock

POLICY_TYPES = ["auto", "home", "umbrella", "other"]

EXTRACT_POLICY_TOOL: dict[str, Any] = {
    "name": "extract_policy",
    "description": (
        "Extract structured renewal data from an insurance policy document. Return null "
        "for any field that is not stated in the source document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "policy_type": {
                "type": "string",
                "enum": POLICY_TYPES,
                "description": (
                    "Document type. Use 'auto' for personal auto, 'home' for homeowners, "
                    "'umbrella' for umbrella/excess liability. Use 'other' as a catch-all "
                    "for specialty products (fine art, watercraft, etc.) — never invent "
                    "a new value."
                ),
            },
            "premium_amount": {
                "type": ["number", "null"],
                "description": (
                    "Total annual premium in USD as a positive number, or null if not stated."
                ),
            },
            "deductible": {
                "type": ["number", "null"],
                "description": (
                    "Primary deductible in USD as a non-negative number, or null if not stated."
                ),
            },
            "coverage_limit": {
                "type": ["number", "null"],
                "description": (
                    "Primary aggregate coverage limit in USD. For auto, use the per-accident "
                    "bodily-injury limit. For home, use Coverage A (dwelling). For umbrella, "
                    "use the aggregate limit. Null if not stated."
                ),
            },
            "endorsements": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "limit": {"type": ["number", "null"]},
                    },
                    "required": ["name", "limit"],
                },
                "description": (
                    "List of endorsements with optional per-endorsement limit. Null if the "
                    "document references endorsements but does not enclose them (e.g., "
                    "'see attached Schedule A' with no schedule)."
                ),
            },
            "exclusions": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "List of exclusion clauses. Null if not stated.",
            },
            "premium_components": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["name", "amount"],
                },
                "description": (
                    "If the document itemises the premium (e.g., Base + Coverage A + ... - "
                    "Discount), extract each component with its amount (negative for "
                    "discounts). Omit or set null if the document does not itemise."
                ),
            },
            "confidence": {
                "type": "object",
                "properties": {
                    "policy_type": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "premium_amount": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "deductible": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "coverage_limit": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "endorsements": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "exclusions": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": [
                    "policy_type",
                    "premium_amount",
                    "deductible",
                    "coverage_limit",
                    "endorsements",
                    "exclusions",
                ],
                "description": (
                    "Per-field self-rated confidence in [0.0, 1.0]. Rate 'low' (≤0.8) when "
                    "the document is ambiguous, OCR is messy, or a field required inference."
                ),
            },
        },
        "required": [
            "policy_type",
            "premium_amount",
            "deductible",
            "coverage_limit",
            "endorsements",
            "exclusions",
            "confidence",
        ],
    },
}


SYSTEM_PROMPT = """You extract structured data from insurance policy renewal documents \
into the extract_policy tool. The documents arrive as OCR text and may have inconsistent \
whitespace, column flattening, and mixed casing.

Rules:
1. If a required field is not stated anywhere in the document, return null for that field. \
Do not guess. Do not synthesize plausible values.
2. Normalise numeric fields to plain numbers in USD. Strip currency symbols, commas, and \
spaces. Do not return strings for numeric fields.
3. Self-rate confidence per field. Score below 0.9 when you had to infer, when OCR was \
ambiguous, or when multiple plausible values were present.

Example 1 — currency normalisation:
  Source: "Total Policy Premium .........  $ 1,847.62"
  Extract: premium_amount = 1847.62  (not "$1,847.62", not "1,847.62", not 1847)

Example 2 — missing data (return null, never guess):
  Source: "Endorsements:  see attached Schedule A
           *** (Schedule A referenced but not enclosed) ***"
  Extract: endorsements = null  (the Schedule A is not in this document)

Example 3 — itemised premium:
  Source: "Base Premium $800.00 / Coverage A $1,100.00 / Endorsement charges $200.00 / \
Discount $(100.00) / TOTAL $2,000.00"
  Extract: premium_amount = 2000.0, premium_components = [
    {"name": "Base", "amount": 800.0},
    {"name": "Coverage A", "amount": 1100.0},
    {"name": "Endorsement charges", "amount": 200.0},
    {"name": "Discount", "amount": -100.0}
  ]
"""


def build_extraction_messages(
    document_text: str,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Build (messages, system) for a single extraction attempt.

    On retry, prior_attempts carries each previous extraction + its validation error,
    so the model sees exactly what it produced last time and why it was rejected.
    """
    user_content = f"<document>\n{document_text.strip()}\n</document>"

    if prior_attempts:
        feedback_blocks: list[str] = []
        for i, attempt in enumerate(prior_attempts, start=1):
            feedback_blocks.append(
                f"<prior_attempt index=\"{i}\">\n"
                f"  <extraction>{attempt['extraction']}</extraction>\n"
                f"  <validation_error field=\"{attempt['error_field']}\" "
                f"category=\"{attempt['error_category']}\" "
                f"detected_pattern=\"{attempt['error_pattern']}\">\n"
                f"    {attempt['error_message']}\n"
                f"  </validation_error>\n"
                f"</prior_attempt>"
            )
        user_content += (
            "\n\nYour previous attempts were rejected by the validator. Review the "
            "errors and produce a corrected extraction.\n\n" + "\n".join(feedback_blocks)
        )

    return [{"role": "user", "content": user_content}], SYSTEM_PROMPT


def parse_tool_use(message: Any) -> dict[str, Any]:
    """Extract the first tool_use block's input dict from a Message."""
    for block in message.content:
        if isinstance(block, ToolUseBlock) or getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise ValueError(
        f"No tool_use block found in response (stop_reason={message.stop_reason}, "
        f"blocks={[type(b).__name__ for b in message.content]})"
    )
