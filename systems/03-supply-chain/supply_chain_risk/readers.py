"""Source readers — adapters over differently-shaped evidence sources that all
converge on the same `Claim` representation.

- `read_audit`     : structured supplier-audit JSON
- `read_logistics` : semi-structured shipment CSV (metrics derived from rows)
- `read_quality`   : internal quality records in SQLite
- `read_news`      : unstructured prose, extracted via a `NewsExtractor`
                     (the Anthropic SDK at runtime, a recorded fixture in tests)

Readers return `ReaderResult` — errors are values, not exceptions — so a single
unreadable source never aborts the investigation.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Protocol

from .models import Claim, FailureContext, ReaderResult

AUDIT = "supplier_audit"
LOGISTICS = "logistics"
QUALITY = "internal_quality"
NEWS = "industry_news"


def read_audit(path: str | Path) -> ReaderResult:
    """Parse the structured supplier audit JSON into Claims."""
    path = Path(path)
    if not path.exists():
        return ReaderResult(
            source=AUDIT,
            ok=False,
            error=FailureContext(
                failure_type="file_not_found",
                attempted=f"open {path}",
                alternatives=["request the audit report from the supplier"],
            ),
        )
    doc = json.loads(path.read_text())
    report_date = date.fromisoformat(doc["report_date"])
    claims = [
        Claim(
            claim=m["statement"],
            evidence=f"{m['label']}: {m['value']} {m['unit']} ({doc['period']})",
            source=AUDIT,
            source_date=report_date,
            confidence=0.9,
            metric_id=m["metric_id"],
            value=float(m["value"]),
            unit=m["unit"],
        )
        for m in doc["metrics"]
    ]
    return ReaderResult(source=AUDIT, ok=True, claims=claims)


def _logistics_claims(rows: list[dict[str, str]]) -> list[Claim]:
    collection = max(date.fromisoformat(r["collection_date"]) for r in rows)
    on_time = sum(1 for r in rows if r["on_time"].strip().lower() == "true")
    on_time_rate = round(100.0 * on_time / len(rows), 1)
    avg_lead = round(sum(int(r["lead_time_days"]) for r in rows) / len(rows), 1)
    late = len(rows) - on_time
    return [
        Claim(
            claim=f"On-time delivery measured {on_time_rate}% across {len(rows)} shipments.",
            evidence=f"{on_time}/{len(rows)} shipments delivered on or before the promised date.",
            source=LOGISTICS,
            source_date=collection,
            confidence=0.85,
            metric_id="on_time_delivery_rate",
            value=on_time_rate,
            unit="percent",
        ),
        Claim(
            claim=f"Average lead time was {avg_lead} days across {len(rows)} shipments.",
            evidence=f"Mean of lead_time_days over {len(rows)} Q1 shipment records.",
            source=LOGISTICS,
            source_date=collection,
            confidence=0.85,
            metric_id="average_lead_time_days",
            value=avg_lead,
            unit="days",
        ),
        Claim(
            claim=f"{late} of {len(rows)} shipments were late this quarter.",
            evidence=f"Count of rows with on_time=false over {len(rows)} shipment records.",
            source=LOGISTICS,
            source_date=collection,
            confidence=0.85,
            metric_id="late_shipment_count",
            value=float(late),
            unit="shipments",
        ),
    ]


# Metrics only this source provides — used to report a coverage gap when it fails.
LOGISTICS_EXCLUSIVE_METRICS = ("late_shipment_count",)


def read_logistics(path: str | Path, *, fail_after: int | None = None) -> ReaderResult:
    """Derive delivery metrics from the semi-structured shipment CSV.

    `fail_after` simulates a source timeout after reading that many rows: the
    reader returns `ok=False` with a `FailureContext` carrying the partial
    results gathered before the failure — it does not raise.
    """
    path = Path(path)
    if not path.exists():
        return ReaderResult(
            source=LOGISTICS,
            ok=False,
            error=FailureContext(
                failure_type="file_not_found",
                attempted=f"open {path}",
                alternatives=["pull the shipment extract from the 3PL portal"],
            ),
        )
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ReaderResult(source=LOGISTICS, ok=True, claims=[])

    if fail_after is not None and fail_after < len(rows):
        partial = _logistics_claims(rows[:fail_after]) if fail_after > 0 else []
        return ReaderResult(
            source=LOGISTICS,
            ok=False,
            error=FailureContext(
                failure_type="timeout",
                attempted=f"read {len(rows)} shipment rows from {path}",
                partial_results=partial,
                alternatives=[
                    "retry the 3PL extract",
                    "proceed with available sources and annotate the gap",
                ],
            ),
        )
    return ReaderResult(source=LOGISTICS, ok=True, claims=_logistics_claims(rows))


def read_quality(path: str | Path) -> ReaderResult:
    """Read internal quality records from SQLite."""
    path = Path(path)
    if not path.exists():
        return ReaderResult(
            source=QUALITY,
            ok=False,
            error=FailureContext(
                failure_type="file_not_found",
                attempted=f"open {path}",
                alternatives=["query the incoming-inspection database directly"],
            ),
        )
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT metric_id, label, value, unit, recorded_date, statement "
            "FROM quality_metrics"
        ).fetchall()
    finally:
        conn.close()
    claims = [
        Claim(
            claim=r["statement"],
            evidence=f"{r['label']}: {r['value']} {r['unit']}",
            source=QUALITY,
            source_date=date.fromisoformat(r["recorded_date"]),
            confidence=0.88,
            metric_id=r["metric_id"],
            value=float(r["value"]),
            unit=r["unit"],
        )
        for r in rows
    ]
    return ReaderResult(source=QUALITY, ok=True, claims=claims)


class NewsExtractor(Protocol):
    """Extracts structured claim dicts from one article's prose.

    Implemented by `AnthropicNewsExtractor` (live SDK call) at runtime and by a
    recorded-fixture extractor in tests. Returns a list of raw claim dicts
    matching the `Claim` field names (plus `needs_identifier`/`candidates`).
    """

    def extract(self, article_text: str) -> list[dict[str, object]]: ...


def read_news(path: str | Path, extractor: NewsExtractor) -> ReaderResult:
    """Extract Claims from an unstructured news article via `extractor`.

    A readable article that yields no supplier-relevant claims returns
    `ok=True, claims=[]` (a valid empty result), distinct from a missing file
    (`ok=False`). When the extractor surfaces an ambiguous entity match, the
    resulting Claim carries `needs_identifier=True` and the candidate list —
    the reader does not pick one heuristically.
    """
    path = Path(path)
    if not path.exists():
        return ReaderResult(
            source=NEWS,
            ok=False,
            error=FailureContext(
                failure_type="file_not_found",
                attempted=f"open {path}",
                alternatives=["re-fetch the article from the news source"],
            ),
        )
    raw = extractor.extract(path.read_text())
    claims = [
        Claim(
            claim=str(d["claim"]),
            evidence=str(d["evidence"]),
            source=NEWS,
            source_date=date.fromisoformat(str(d["source_date"])),
            confidence=float(d["confidence"]),  # type: ignore[arg-type]
            metric_id=str(d["metric_id"]),
            needs_identifier=bool(d.get("needs_identifier", False)),
            candidates=tuple(d.get("candidates", []) or ()),  # type: ignore[arg-type]
        )
        for d in raw
    ]
    return ReaderResult(source=NEWS, ok=True, claims=claims)
