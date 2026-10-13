# Project 3 — Capstone Working Pack

This pack is organized around the course's capstone instructions and contains the **final reference implementations** for the three systems, plus the evidence-pack layout and reflection/perturbation templates.

## Important: use the final solution folders
The source archive's top-level README and the capstone instructions state that this capstone runs each system from the `solution/` folder of its final exercise. The `starter/` folders are intentionally incomplete fill-in exercises. The earlier 8 failures occurred because tests were run in `04-hitl-routing/starter`; those failures do not mean the capstone reference system is broken.

## Included final systems
- `systems/01-policy-pipeline/`
- `systems/02-mortgage-extraction/`
- `systems/03-supply-chain/`

Each system needs its own `.venv` created and dependencies installed in the course workspace, as described by the exercise README.

## Submission evidence
Fill `capstone-submission/` with outputs from your **actual runs in the course workspace**. Do not submit this pack as a completed evidence pack until the files listed in each `README-evidence-needed.txt` have been captured. The reflection and perturbation templates must quote your own outputs; no test results, routing JSON, or briefing data have been fabricated here.

## Core rubric requirements
1. Passing test output for all three systems.
2. `mypy` and `ruff` output for all three systems.
3. System 1 routing output and calibration report (or the documented offline routing-test fallback if no API key is available).
4. System 2 extraction output, including a consistency discrepancy and a missing-field result.
5. System 3 normal briefing and simulated-timeout briefing.
6. One controlled perturbation per system.
7. Reflection answers that cite exact filenames/values from your artifacts.
8. `environment.txt` and at least one screenshot per system.
