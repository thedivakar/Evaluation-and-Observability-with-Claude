"""Print a sliced (policy_type x field) calibration report.

Run from the policy-pipeline project folder, using that project's venv so
`policy_extractor` is importable:

    .venv/bin/python calibration_report.py        # macOS / Linux
    .venv\\Scripts\\python calibration_report.py    # Windows

The labels below are a small fixed set: overall accuracy looks healthy, but one
cell (umbrella / exclusions) is wrong every time despite high confidence. Slicing
surfaces the bad cell that a single aggregate number hides.
"""

from policy_extractor.routing import CalibrationLabel, calibration_report

labels = [
    CalibrationLabel(policy_id="POL-1", policy_type="auto",     field="premium_amount", predicted_confidence=0.95, correct=True),
    CalibrationLabel(policy_id="POL-2", policy_type="auto",     field="premium_amount", predicted_confidence=0.95, correct=True),
    CalibrationLabel(policy_id="POL-3", policy_type="auto",     field="premium_amount", predicted_confidence=0.95, correct=True),
    CalibrationLabel(policy_id="POL-4", policy_type="umbrella",  field="exclusions",     predicted_confidence=0.93, correct=False),
    CalibrationLabel(policy_id="POL-5", policy_type="umbrella",  field="exclusions",     predicted_confidence=0.93, correct=False),
    CalibrationLabel(policy_id="POL-6", policy_type="home",     field="deductible",     predicted_confidence=0.90, correct=True),
]

report = calibration_report(labels)
for (ptype, fname), cell in sorted(report.cells.items()):
    print(
        f"{ptype:9} {fname:15} n={cell.samples} "
        f"conf={cell.mean_predicted_confidence:.2f} "
        f"acc={cell.observed_accuracy:.2f} brier={cell.brier_score:.3f}"
    )
print(f"OVERALL brier={report.overall_brier:.3f}")
