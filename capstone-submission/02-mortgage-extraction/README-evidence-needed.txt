Required evidence for System 2
- tests.txt: full pytest output from the FINAL solution folder
- static-checks.txt: mypy and ruff output
- extract-run.txt: run on appraisal_informal_sqft.txt
- discrepancy-run.txt: run on income_sum_mismatch.txt and retain the validator's discrepancy output
- screenshots/: at least one screenshot of a real run

Final source folder: ../systems/02-mortgage-extraction
Commands (run from that folder):
  .venv/bin/pytest tests/ -v
  .venv/bin/mypy mortgage_extractor/
  .venv/bin/ruff check mortgage_extractor/ tests/
  .venv/bin/mortgage-extract fixtures/documents/appraisal_informal_sqft.txt
  .venv/bin/mortgage-extract fixtures/documents/income_sum_mismatch.txt
