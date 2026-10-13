# Exercise 4 — Solution: Deterministic HITL Routing With Stratified Sampling and Calibration

All four user stories are implemented. The pipeline runs end-to-end: extract → review →
integration → route, with deterministic routing on `(confidence ∧ reviewer ∧ integration)`,
a stratified spot-check sampler that never drops a small `policy_type` stratum, and a
sliced calibration report.

This is the final state of the project. It matches the reference solution.

## Verify

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v
```

Target: 45 passed, 3 skipped.

## What changed since `04-hitl-routing/starter/`

- `policy_extractor/routing.py` — `route_extraction`, `apply_stratified_spot_check`, and `calibration_report` now implemented.

## End-to-end demo (needs `ANTHROPIC_API_KEY` set)

```bash
.venv/bin/policy-extractor pipeline data/policies/ --routing-out /tmp/routing.json --spot-check-pct 0.2 --seed 42
cat /tmp/routing.json | python3 -m json.tool | head -40
```
