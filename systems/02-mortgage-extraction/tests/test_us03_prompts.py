"""Tests for few-shot prompts, explicit criteria, normalization."""
from __future__ import annotations

import re
from pathlib import Path

from mortgage_extractor import prompts
from mortgage_extractor.client import RecordingClient
from mortgage_extractor.models import DocumentType
from mortgage_extractor.pipeline import Pipeline

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "documents"

VERBATIM_NULL_RULE = (
    "Return null for any field not explicitly stated in the document. "
    "Do not infer, default, or fabricate."
)


def test_ac_03_01_system_prompt_has_numbered_criteria_with_verbatim_first() -> None:
    prompt = prompts.income_verification_system_prompt()
    assert VERBATIM_NULL_RULE in prompt

    numbered = re.findall(r"^\s*\d+\.\s", prompt, flags=re.MULTILINE)
    assert len(numbered) >= 4, f"expected ≥4 numbered criteria, got {len(numbered)}"

    first_criterion = re.search(r"1\.\s+(.+)", prompt)
    assert first_criterion is not None
    assert VERBATIM_NULL_RULE.startswith(first_criterion.group(1).strip()[:40])


def test_ac_03_02_four_contrastive_examples() -> None:
    prompt = prompts.income_verification_system_prompt()
    examples = re.findall(r'<example name="([^"]+)">', prompt)
    assert len(examples) == 4, f"expected exactly 4 examples, got {len(examples)}"

    keywords = {
        "clean": "clean",
        "missing": "missing",
        "informal": "informal",
        "mismatch": "mismatch",
    }
    for kw in keywords.values():
        assert any(kw in name for name in examples), f"missing example for {kw}"


def test_ac_03_02b_each_example_has_inline_reasoning() -> None:
    prompt = prompts.income_verification_system_prompt()
    blocks = re.findall(
        r'<example name="[^"]+">(.*?)</example>',
        prompt,
        flags=re.DOTALL,
    )
    assert len(blocks) == 4
    for i, block in enumerate(blocks):
        assert "<reasoning>" in block, f"example #{i+1} missing <reasoning>"
        assert "</reasoning>" in block, f"example #{i+1} missing </reasoning>"


def test_ac_03_03_missing_bonus_returns_none() -> None:
    document = (FIXTURES / "income_missing_bonus.txt").read_text()
    pipeline = Pipeline(client=RecordingClient(mode="auto"))
    result = pipeline.run(document)

    assert result.income is not None
    assert result.income.bonus_monthly is None, (
        f"extractor fabricated bonus_monthly={result.income.bonus_monthly!r} "
        "instead of returning None"
    )
    assert result.income.bonus_ytd is None, (
        f"extractor fabricated bonus_ytd={result.income.bonus_ytd!r} "
        "instead of returning None"
    )


def test_ac_03_04_informal_sqft_normalized_to_integer() -> None:
    document = (FIXTURES / "appraisal_informal_sqft.txt").read_text()
    pipeline = Pipeline(client=RecordingClient(mode="auto"))
    result = pipeline.run(document)

    assert isinstance(result.property.gross_living_area_sqft, int)
    assert result.property.gross_living_area_sqft == 2400


def test_ac_03_05_normalization_rules_is_single_referenced_constant() -> None:
    assert hasattr(prompts, "NORMALIZATION_RULES")
    rules = prompts.NORMALIZATION_RULES
    assert isinstance(rules, str)

    assert "Square footage" in rules
    assert "Currency" in rules
    assert "Percentage" in rules
    assert "6.5%" in rules and "0.065" in rules
    assert "2,400" in rules and "2400" in rules

    income_prompt = prompts.income_verification_system_prompt()
    appraisal_prompt = prompts.extractor_system_prompt(DocumentType.APPRAISAL)
    assert rules in income_prompt
    assert rules in appraisal_prompt
