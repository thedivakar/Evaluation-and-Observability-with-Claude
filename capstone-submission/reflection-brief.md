# Reflection Brief --- Evaluation and Observability Capstone

**Name:** \[Divakar Chadalavada\]\
**Date:** 2026-10-13

> Ground every answer in your own run. Values marked **not captured**
> still need to be verified from the relevant artifact before
> submission.

------------------------------------------------------------------------

## 0. Environment

  -----------------------------------------------------------------------
  Field                               Value
  ----------------------------------- -----------------------------------
  OS & version                        Microsoft Windows NT 10.0.26300.0

  Python version                      Python 3.11.4

  Date run                            2026-10-13

  Ran any system live? (which)        No live API run verified. System
                                      1's live pipeline attempt failed
                                      because API authentication was
                                      unavailable. Tests and
                                      fixture/offline runs were used.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 1. Validated, routed pipeline

  -----------------------------------------------------------------------
  Evidence                            Value
  ----------------------------------- -----------------------------------
  Passing test count                  45 passed, 3 skipped

  Routing output file                 No live `routing_decisions.json`
                                      generated; the live pipeline failed
                                      because API authentication was
                                      unavailable.

  `auto_approve` / `human_review` /   Not available from an actual
  `spot_check` counts                 pipeline run.
  -----------------------------------------------------------------------

**1a. Retry boundary.** From your perturbation run (a required field
removed), paste the escalation record. How many API calls did the system
make, and why is retrying a futile case worse than escalating it?

> In `systems/01-policy-pipeline/tests/test_us01_retry.py`, test
> `test_ac_01_04_missing_source_halts_immediately` sets the required
> `endorsements` field to `None` because Schedule A could not be found.
> The test passed and asserts `result` is a `RetryFutileEscalation`,
> `result.field == "endorsements"`,
> `result.detected_pattern == "endorsements_absent"`, and
> `client.call_count == 1`. This is a test-verified escalation rather
> than a live API pipeline record. Retrying is futile when the source
> information is unavailable; escalation makes the missing information
> explicit and avoids wasting additional calls.

**1b. Reading the router.** Pick one `human_review` record from your
routing output. Which of the three signals (confidence, reviewer,
integration) sent it to a human? If you had trusted the model's
confidence alone, what would have happened?

> I could not select a real `human_review` record because the live
> pipeline failed authentication and did not generate
> `routing_decisions.json`. The routing tests verify that low
> confidence, reviewer disagreement, and integration failure can trigger
> `human_review`. A confidence-only decision could miss reviewer
> disagreement or an integration failure even when model confidence is
> high. Replace this answer with a specific record if a live routing
> output becomes available.

**1c. Where the aggregate lies.** Run the calibration snippet. Quote the
one cell whose accuracy lags its confidence, plus the overall figure.
What does slicing by `policy_type × field` catch that a single number
hides?

> The `umbrella / exclusions` calibration cell had confidence `0.93`,
> accuracy `0.00`, Brier score `0.865`, and `n=2`. The overall Brier
> score was `0.291`. Slicing by policy type and field exposes a poorly
> performing category that a single aggregate score can hide. Because
> this cell has only two samples, it is a warning signal rather than a
> definitive estimate of long-term performance. Evidence:
> `capstone-submission/01-policy-pipeline/calibration-report.txt`.

------------------------------------------------------------------------

## 2. Schema-enforced two-pass extraction

  ----------------------------------------------------------------------------------
  Evidence                            Value
  ----------------------------------- ----------------------------------------------
  Passing test count                  25 passed (`systems/02-mortgage-extraction`
                                      test run)

  Document run                        `fixtures/documents/income_sum_mismatch.txt`

  Classified type                     Not captured in the saved output shown; verify
                                      the document classification output before
                                      submission.
  ----------------------------------------------------------------------------------

**2a. Two guarantees.** Paste your discrepancy-run output. Tool use
already forces valid JSON, yet the validator still catches a bad sum.
Why are these two different guarantees? Name one error each cannot
catch.

> The saved run in
> `capstone-submission/02-mortgage-extraction/perturbation-test.txt`
> returned `"consistent": false` and a discrepancy for
> `total_monthly_income`: calculated `9642.17`, stated `10892.17`, delta
> `-1250.0`. Tool/schema enforcement guarantees the response has the
> required structured shape and valid JSON; it cannot guarantee that
> values are factually correct or arithmetically consistent. The
> deterministic validator checks the income calculation, but it cannot
> detect every plausible-looking value that was extracted incorrectly
> from the source.

**2b. Refusing to fabricate.** Run on a document missing a field. Paste
that field's output. Why null instead of an invented value? Point to the
schema choice that allows it.

> The project includes `fixtures/documents/income_missing_bonus.txt`,
> but the missing-field output and the exact schema declaration have not
> been captured in the evidence reviewed here. Before submitting, run
> the CLI on that fixture and record the relevant field output and the
> schema's nullable/optional declaration. A `null` value should
> represent information absent from the source rather than inventing a
> value; only state the exact schema choice after verifying the
> implementation.
> The `Income` schema in `systems/02-mortgage-extraction/mortgage_extractor/models.py` defines `bonus_monthly` and `bonus_ytd` as `float | None = None`. Running `income_missing_bonus.txt` produced `bonus_monthly: null`, `bonus_ytd: null`, and `stated_monthly_total: null`. Validation returned `consistent: true` with no discrepancies. This demonstrates that missing values are represented explicitly rather than fabricated. However, this run does not demonstrate automatic rejection or escalation of missing bonus information.

**2c. Normalization.** Quote one field where the source text and
extracted value differ in format ("about 2,400 sq ft" → `2400`). Why
normalize at extraction time rather than downstream?

> In the appraisal fixture, the source describes the area as "about
> 2,400 sq ft" and the extracted `gross_living_area_sqft` value is
> `2400`; the saved successful example reports validation as consistent.
> Normalizing at extraction time gives downstream code a predictable
> numeric value for comparisons and calculations, instead of making each
> later step parse different text formats.

------------------------------------------------------------------------

## 3. Multi-source synthesis

  ---------------------------------------------------------------------------------------------
  Evidence                            Value
  ----------------------------------- ---------------------------------------------------------
  Passing test count                  34 passed

  Briefing file                       `capstone-submission/03-supply-chain/investigation.txt`
                                      (normal/offline report); timeout evidence:
                                      `capstone-submission/03-supply-chain/timeout-run.txt`

  Section the conflict landed in      The normal investigation was reported to contain
                                      conflicting on-time delivery rates; verify the exact
                                      section heading in `investigation.txt` before final
                                      submission. The timeout report itself says `## Contested`
                                      → `_none_`.
  ---------------------------------------------------------------------------------------------

**3a. Annotate, don't arbitrate.** Quote one conflicting-metric pair
from your briefing --- both values, sources, dates. Give one way a
reader is better served by the preserved conflict than by a single
reconciled number.

> The normal investigation evidence records on-time delivery rates of
> `95%` from `supplier_audit` (as of `2026-04-10`) and `78%` from
> `logistics` (as of `2026-04-05`). Keeping both values with their
> sources and dates lets a reader see the disagreement and investigate
> freshness or measurement differences rather than being shown a
> potentially misleading reconciled number. Confirm the exact lines and
> section in `investigation.txt` before submitting.

**3b. Source goes dark.** Run with `--simulate-timeout`. Paste the part
of the briefing showing the failed source. How is "unreachable" handled
differently from "nothing to report," and why does the run still finish?

> From `capstone-submission/03-supply-chain/timeout-run.txt`:
> `Sources unavailable: logistics unavailable (timeout)`. The report
> marks `late_shipment_count` as
> `[missing source: timeout reading logistics]`. This distinguishes a
> source that could not be reached from a source that successfully
> returned no finding. The report still completes using information from
> the other available sources; `production_capacity_utilization` is
> separately escalated because no source reported that high-impact
> metric.

**3c. Dates as a guardrail.** Quote two claims about the same supplier
with different dates. How does requiring a date stop a time difference
from reading as a contradiction?

> The normal investigation records on-time delivery as `95%` from
> `supplier_audit` dated `2026-04-10`, versus `78%` from `logistics`
> dated `2026-04-05`. The dates show the observations were made at
> different times, so the values may reflect a change over time or
> differing reporting windows rather than a direct same-time
> contradiction. Keep the source/date annotations visible rather than
> silently choosing one value.

------------------------------------------------------------------------

## 4. Synthesis

**4a. One principle.** Name the single moment in your runs (system +
artifact) where *evaluate the output, don't trust the model's word* most
clearly caught something a trusting design would have shipped.

> System 2,
> `capstone-submission/02-mortgage-extraction/perturbation-test.txt`:
> the extraction produced structured JSON, but the deterministic
> validation reported `consistent: false` and a `-1250.0` difference
> between calculated and stated monthly income. A design that trusted
> valid-looking JSON alone could have passed on an inconsistent total.

**4b. Confidence ≠ correctness.** Pick the system where this mattered
most, and explain why using something you observed.

> System 1's calibration report demonstrates this directly:
> `umbrella / exclusions` had confidence `0.93` but accuracy `0.00`
> (`n=2`, Brier score `0.865`). This shows that a high confidence score
> did not correspond to correctness in that small sample, so confidence
> should be combined with calibration evidence and deterministic routing
> safeguards.

**4c. Apply it.** Describe a real workflow where an LLM pulls structured
results from messy input. Which pattern --- validated retry with
escalation, independent review with deterministic routing, or
provenance-preserving conflict annotation --- would you reach for first,
and what would you instrument to know when it broke?

> For insurance-claim intake, I would start with validated extraction
> plus deterministic routing and escalation: extract claim fields from
> PDFs, validate required fields and cross-field totals, and send
> missing-source or low-confidence cases to a human instead of
> repeatedly retrying a futile extraction. I would instrument
> schema/validation failure rates, missing-field categories,
> retry/API-call counts, escalation rates, human override rates, and
> accuracy by field and document type. These measures would help detect
> when apparently structured output becomes incomplete or unreliable.

------------------------------------------------------------------------

## Before submitting

-   Replace `[Divakar Chadalavada]`.
-   Verify the exact `Classified type` from the mortgage run.
-   Capture the missing-field output and the relevant schema declaration
    for 2b.
-   Open `capstone-submission/03-supply-chain/investigation.txt` and
    confirm the conflicting pair and exact section heading for 3a/3c.
-   Do not claim a live System 1 routing record or routing counts unless
    an actual run produces them.
