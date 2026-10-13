"""System prompts, normalization rules, and few-shot examples.

The prompts in this module teach the extractor *how* to handle the cases the
schema alone cannot enforce: when to return null instead of fabricating, how to
normalize informal numerics, and how to record arithmetic discrepancies
verbatim rather than silently correcting them.

The single source of truth for normalization is :data:`NORMALIZATION_RULES`,
which is interpolated into every extractor system prompt and is the canonical
reference for the rules described in the module docstrings of
:mod:`mortgage_extractor.schema` and :mod:`mortgage_extractor.models`.
"""
from __future__ import annotations

from mortgage_extractor.models import DocumentType

NORMALIZATION_RULES = """\
Normalization rules — apply BEFORE writing values into structured fields:
1. Square footage: emit an INTEGER (e.g., "about 2,400 sq ft" → 2400; "~3,100 SF" → 3100).
2. Currency amounts: strip "$" and commas; emit a NUMBER (e.g., "$485,000" → 485000.0; "$1,234.56" → 1234.56).
3. Percentage fields (interest rates, ratios): emit a DECIMAL (e.g., "6.5%" → 0.065; "6.875%" → 0.06875).
"""

_CATEGORICAL_CRITERIA = """\
Categorical criteria:
1. Return null for any field not explicitly stated in the document. Do not infer, default, or fabricate.
2. Base income is regularly recurring wages paid for ordinary hours. Distinguish it from bonus (irregular, performance-tied), commission (tied to sales volume), and overtime (premium hours). When the document is ambiguous, place the value in the most specific matching field; only use `other_earnings` as a last resort.
3. When a categorical field's value is not in the listed enum, emit "other" and write the actual value into the corresponding `*_detail` field. Do not pick the "closest" enum value.
4. Numeric fields receive numbers (per the normalization rules above), not strings. If the document is ambiguous about a numeric value (e.g., a smudged digit), emit null and rely on the validator / human review path.
"""

_FEW_SHOT_EXAMPLES = """\
<example name="clean">
<input>
PAYSTUB. Employee: Jane Park. Pay frequency: monthly. Period 11/01-11/30/2024.
Base monthly pay: $4,800.00. Monthly bonus: $400.00. Commission this month: $0.00.
Other earnings: none. Stated monthly total: $5,200.00.
</input>
<reasoning>
All four income components are explicitly stated. Commission is explicitly
$0.00 in the document, so we report 0.0 — not null — because the document
*stated* it as zero. The stated_monthly_total of $5,200 equals the sum
($4,800 + $400 + $0 + $0), so the validator will mark this consistent; we
still emit both fields verbatim so the validator can confirm.
</reasoning>
<output>
{"income": {"base_monthly": 4800.00, "bonus_monthly": 400.00, "commission_monthly": 0.00, "other_monthly": null, "stated_monthly_total": 5200.00}}
</output>
</example>

<example name="missing-bonus-returns-null">
<input>
PAYSTUB. Employee: Carlos Mendez. Pay frequency: monthly. Period 03/01-03/31/2025.
Base monthly pay: $3,500.00. The document does not mention bonus, commission, or overtime anywhere.
</input>
<reasoning>
The document does not mention bonus, commission, or overtime. Rule #1 says
to return null when the document does not state a value — fabricating zero
would imply the employee earned zero bonus, not that the document is silent
about bonus. These are different facts. The downstream consumer treats null
as "we don't know" and zero as "we know it's zero"; conflating them hides
the underlying uncertainty.
</reasoning>
<output>
{"income": {"base_monthly": 3500.00, "bonus_monthly": null, "commission_monthly": null, "overtime_monthly": null, "other_monthly": null, "stated_monthly_total": null}}
</output>
</example>

<example name="informal-measurement-normalized">
<input>
APPRAISAL. Subject property at 123 Oak Lane, Portland, OR 97214.
Gross living area: about 2,400 sq ft (above-grade finished).
Property type: single family. Occupancy: primary residence.
</input>
<reasoning>
"about 2,400 sq ft" is an informal expression of a numeric measurement.
Normalization rule #1 (square footage as integer) requires emitting the
integer 2400, not the source phrase and not a rounded variant. The hedge
word "about" does not change the recorded value; if the appraiser were
confident in a different value they would have stated it. The schema enum
includes "single_family" exactly, so property_type uses the enum value and
property_type_detail is null.
</reasoning>
<output>
{"property": {"gross_living_area_sqft": 2400, "property_type": "single_family", "property_type_detail": null, "occupancy_type": "primary_residence"}}
</output>
</example>

<example name="sum-mismatch-recorded-verbatim">
<input>
PAYSTUB. Employee: Aiko Tanaka. Pay frequency: monthly.
Base monthly pay: $4,000.00. Monthly bonus: $500.00. Commission this month: $0.00.
Other earnings: $0.00. Stated monthly total: $5,000.00.
</input>
<reasoning>
The stated_monthly_total ($5,000.00) does not equal the sum of components
($4,500.00). Do NOT silently correct the discrepancy — record both values
verbatim. The mathematical-consistency validator downstream will flag this
so a human can investigate the source document. Lying about either value
would hide the underlying error from the underwriting team. The "right"
answer here is to surface the conflict, not to resolve it.
</reasoning>
<output>
{"income": {"base_monthly": 4000.00, "bonus_monthly": 500.00, "commission_monthly": 0.00, "other_monthly": 0.00, "stated_monthly_total": 5000.00}}
</output>
</example>
"""


def classifier_system_prompt() -> str:
    return (
        "You are a document classifier for a mortgage lender. You will receive "
        "the text of a single mortgage-related document and must call the "
        "`classify_document` tool exactly once with the document's type and a "
        "one-sentence reason describing the textual cues that drove the "
        "classification. Use `other` if the document is not one of the listed "
        "types or is too damaged to classify confidently."
    )


def extractor_system_prompt(doc_type: DocumentType) -> str:
    """Return the system prompt for the given document type's extractor.

    All extractor prompts share the same normalization rules and few-shot
    examples; the income-verification prompt additionally surfaces the full
    explicit categorical criteria block.
    """
    if doc_type is DocumentType.INCOME_VERIFICATION:
        return income_verification_system_prompt()

    intro = (
        "You are an extraction agent for a mortgage lender. The document is a "
        f"{doc_type.value} document. Call `extract_{doc_type.value}` exactly "
        "once with the structured data the document contains, or "
        "`flag_for_review` if the document is unreadable or off-topic.\n\n"
    )
    return intro + NORMALIZATION_RULES + "\n" + _FEW_SHOT_EXAMPLES


def income_verification_system_prompt() -> str:
    """Return the income-verification extractor system prompt.

    This is the canonical extractor prompt: it contains the verbatim null-
    handling criterion as item #1, four numbered categorical criteria, the
    shared :data:`NORMALIZATION_RULES`, and four contrastive few-shot examples
    each with an inline ``<reasoning>`` block.
    """
    intro = (
        "You are an extraction agent for a mortgage lender. The document is an "
        "income-verification document (W-2, paystub, employer letter, etc.). "
        "Call `extract_income_verification` exactly once with the structured "
        "data the document contains, or `flag_for_review` if the document is "
        "unreadable.\n\n"
    )
    return (
        intro
        + _CATEGORICAL_CRITERIA
        + "\n"
        + NORMALIZATION_RULES
        + "\n"
        + _FEW_SHOT_EXAMPLES
    )
