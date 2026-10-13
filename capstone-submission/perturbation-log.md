# Perturbation Log

For each system, make one deliberate change to an input or configuration, predict the outcome, run
it, and record what actually happened. See the starters in the Instructions, or design your own (your
own experiment earns more credit).

---

### System 1 — validated, routed pipeline

- **Change I made (file + what I changed):** In `systems/01-policy-pipeline/tests/test_us01_retry.py`, the required `endorsements` field was set to `None` because Schedule A could not be found.
- **Command I ran:** `.venv\Scripts\python.exe -m pytest tests\test_us01_retry.py::test_ac_01_04_missing_source_halts_immediately -v`
- **What I predicted:** The missing-source validation error would trigger `RetryFutileEscalation` immediately instead of retrying when the required source information was unavailable.
- **What actually happened (paste the key output line):** `test_ac_01_04_missing_source_halts_immediately PASSED`; `1 passed in 0.02s`. The test assertions confirm the field was `endorsements`, the pattern was `endorsements_absent`, and the API call count was `1`.
- **How this differs from the unperturbed run:** With the required source missing, the extraction escalates after one API call rather than making additional retry calls. This avoids wasting calls when the source information is unavailable.

---

### System 2 — schema-enforced two-pass extraction

- **Change I made (file + what I changed):** Tested the existing `fixtures/documents/income_sum_mismatch.txt` fixture containing inconsistent monthly income values. No original source file was modified.
- **Command I ran:** `.\.venv\Scripts\python.exe -m mortgage_extractor fixtures\documents\income_sum_mismatch.txt`
- **What I predicted:** The extraction would return structured JSON, and the validation pass would flag the inconsistent monthly income total.
- **What actually happened (paste the key output line):** `"consistent": false`; calculated total `9642.17`, stated total `10892.17`, and delta `-1250.0`.
- **How this differs from the unperturbed run:** The mismatch fixture caused validation to report a discrepancy rather than mark the extracted income as consistent. This demonstrates why structured extraction needs a separate validation step.

---

### System 3 — multi-source synthesis

- **Change I made (file + what I changed):** Tested the existing simulated timeout scenario using the supply-chain investigation CLI. The logistics source was made unavailable by the scenario; no original source files were modified.
- **Command I ran:** `.\.venv\Scripts\supply-chain-investigate.exe meridian --offline --simulate-timeout logistics`
- **What I predicted:** The logistics timeout would be reported explicitly, affected metrics would be marked incomplete, and the system would still generate a briefing using the remaining sources.
- **What actually happened (paste the key output line):** `Sources unavailable: logistics unavailable (timeout)`. The report marked `late_shipment_count` as missing because of the timeout and still generated the risk briefing. It also escalated `production_capacity_utilization` because no source reported that metric.
- **How this differs from the unperturbed run:** With logistics unavailable, `late_shipment_count` could not be reported. Other available source information remained in the briefing, demonstrating that a source timeout did not stop the whole investigation.
