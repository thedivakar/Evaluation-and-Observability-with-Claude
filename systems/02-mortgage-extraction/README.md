# Exercise 4. Validate Mathematical Consistency (Solution). Full Project

This folder is the complete project, end of the arc. Every Arc LO is in place: resilient schema, two-pass tool choice, production extractor prompts, and the math-consistency validator. The full reference solution.

## What's complete

The whole package:

- `mortgage_extractor/{schema,tools,pipeline,prompts,validator,models,client,config,errors,__main__}.py`
- `tests/test_us0{1,2,3,4}_*.py`, all 28 acceptance tests
- `fixtures/documents/*.txt`, three real-world-shaped mortgage documents
- `fixtures/recorded_responses/*.json`, cached Anthropic responses for offline test runs
- `scripts/regenerate_fixtures.py`, refresh recorded responses against the live API
- `pyproject.toml`, package config, dev tools

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...   # only needed to regenerate fixtures
```

## Run

```bash
mortgage-extract fixtures/documents/appraisal_informal_sqft.txt        # exit 0
mortgage-extract fixtures/documents/income_sum_mismatch.txt; echo $?    # exit 1 (inconsistent)
```

## Verify (offline, deterministic)

```bash
.venv/bin/pytest tests/ -v
.venv/bin/mypy mortgage_extractor/
.venv/bin/ruff check mortgage_extractor/ tests/
```

Regenerate recorded responses against the live API:

```bash
.venv/bin/python scripts/regenerate_fixtures.py
```

## Layout

| Path | Purpose |
|---|---|
| `mortgage_extractor/schema.py` | JSON Schema for the four extraction tools |
| `mortgage_extractor/tools.py` | Anthropic tool-definition wrappers |
| `mortgage_extractor/models.py` | Typed result objects (`MortgageExtraction`, `Discrepancy`, ...) |
| `mortgage_extractor/prompts.py` | System prompts, few-shot examples, `NORMALIZATION_RULES` |
| `mortgage_extractor/pipeline.py` | Two-pass classify-then-extract orchestration |
| `mortgage_extractor/validator.py` | `calculated_total` vs `stated_total` consistency checks |
| `mortgage_extractor/client.py` | Anthropic SDK wrapper + `RecordingClient` shim |
| `mortgage_extractor/errors.py` | Domain exceptions |
| `mortgage_extractor/config.py` | Model defaults, tolerance |
| `fixtures/documents/` | Mortgage document text fixtures |
| `fixtures/recorded_responses/` | Recorded API responses for deterministic tests |

## Key design moves added in Exercise 4

- **`Income.calculated_monthly_total` returns `None` when no components are set.** An income-less appraisal compares "calculated=None vs stated=None" and the validator returns `consistent=True` because there's nothing to diff. If the property returned `0.0` instead, the validator would silently pass "$0 vs stated $0 = consistent" and mask real bugs.
- **Only `*_monthly` components belong in the sum.** Mixing `bonus_monthly` with `bonus_ytd` produces a number with no meaningful relationship to `stated_monthly_total`. The five-element list in `calculated_monthly_total` enforces the same-unit invariant.
- **$1.00 default tolerance.** Absorbs cent-level OCR rounding that upstream conversion sometimes introduces. Zero-tolerance is a per-call override; the default is sized for the operational reality of receiving OCR'd PDFs.
