Required evidence for System 3
- tests.txt: full pytest output from the FINAL solution folder
- static-checks.txt: mypy and ruff output
- investigation-run.txt: normal offline investigation output
- briefing.*: actual generated briefing file
- timeout-run.txt: output from --simulate-timeout showing the failed source as Incomplete
- screenshots/: at least one screenshot of a real run

Final source folder: ../systems/03-supply-chain
Commands (run from that folder):
  .venv/bin/pytest tests/ -q
  .venv/bin/mypy supply_chain_risk/
  .venv/bin/ruff check supply_chain_risk/ tests/
  .venv/bin/supply-chain-investigate meridian --offline
  .venv/bin/supply-chain-investigate meridian --offline --simulate-timeout
