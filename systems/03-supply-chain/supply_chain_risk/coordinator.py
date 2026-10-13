"""Coordinator — deterministic control-flow over the source readers.

Runs each scoped reader, collects successful claims into shared memory, and hands
them to synthesis. Error propagation is local-recovery: a reader that fails
returns a `ReaderResult` with structured `FailureContext`; the coordinator
proceeds on the remaining sources and annotates the gap in the briefing rather
than aborting. Only successful claims are written to shared memory — failures are
recorded as coverage gaps, not vectorized.

This is intentionally a plain control-flow function, not an LLM-driven
orchestrator (multi-agent orchestration is a separate module).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .memory import SharedMemory
from .models import ReaderResult
from .readers import (
    LOGISTICS_EXCLUSIVE_METRICS,
    NewsExtractor,
    read_audit,
    read_logistics,
    read_news,
    read_quality,
)
from .synthesis import Briefing, build_briefing

# Metrics each source is the sole expected provider of (for gap attribution).
EXCLUSIVE_METRICS = {"logistics": LOGISTICS_EXCLUSIVE_METRICS}


@dataclass
class InvestigationResult:
    briefing: Briefing
    reader_results: list[ReaderResult]


def investigate(
    supplier: str,
    data_dir: str | Path,
    extractor: NewsExtractor,
    *,
    simulate_logistics_timeout: bool = False,
    logistics_fail_after: int = 5,
    memory: SharedMemory | None = None,
) -> InvestigationResult:
    data_dir = Path(data_dir)
    results: list[ReaderResult] = [
        read_audit(data_dir / "audit.json"),
        read_logistics(
            data_dir / "logistics.csv",
            fail_after=logistics_fail_after if simulate_logistics_timeout else None,
        ),
        read_quality(data_dir / "quality.sqlite"),
    ]
    for article in sorted((data_dir / "news").glob("*.txt")):
        results.append(read_news(article, extractor))

    ok_claims = [c for r in results if r.ok for c in r.claims]
    present = {c.metric_id for c in ok_claims}

    unavailable: dict[str, str] = {}
    unavailable_sources: list[str] = []
    for r in results:
        if r.ok or r.error is None:
            continue
        reason = f"{r.source} unavailable ({r.error.failure_type})"
        unavailable_sources.append(reason)
        for metric_id in EXCLUSIVE_METRICS.get(r.source, ()):
            if metric_id not in present:
                unavailable[metric_id] = f"{r.error.failure_type} reading {r.source}"

    mem = memory if memory is not None else SharedMemory()
    mem.add_claims(ok_claims)  # only successful claims are vectorized
    briefing = build_briefing(
        supplier,
        ok_claims,
        mem,
        unavailable=unavailable,
        unavailable_sources=unavailable_sources,
    )
    return InvestigationResult(briefing=briefing, reader_results=results)
