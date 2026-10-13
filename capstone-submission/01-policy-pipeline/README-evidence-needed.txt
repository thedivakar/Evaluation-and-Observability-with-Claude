Required evidence for System 1
- tests.txt: full pytest output from the FINAL solution folder
- static-checks.txt: mypy and ruff output
- pipeline-run.txt: live pipeline output if API access is available; otherwise capture the documented offline routing test run
- routing_decisions.json: generated only by an actual pipeline run; do not invent this file
- calibration-report.txt: output from capstone-submission/calibration_report.py
- screenshots/: at least one screenshot of a real run

Final source folder: ../systems/01-policy-pipeline
Commands (run from that folder):
  .venv/bin/pytest tests/ -v
  .venv/bin/mypy policy_extractor/
  .venv/bin/ruff check policy_extractor/ tests/
  .venv/bin/policy-extractor pipeline data/policies/ --routing-out ../../capstone-submission/01-policy-pipeline/routing_decisions.json --spot-check-pct 0.2 --seed 42
