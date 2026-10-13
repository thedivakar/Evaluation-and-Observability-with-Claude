"""Structured error propagation with partial results, via the coordinator."""
from __future__ import annotations

from pathlib import Path

from supply_chain_risk.coordinator import investigate
from supply_chain_risk.readers import read_logistics
from supply_chain_risk.synthesis import INCOMPLETE


# a simulated timeout returns ok=False with FailureContext
# AND partial results, without raising.
def test_logistics_timeout_returns_partial_and_context(data_dir: Path) -> None:
    result = read_logistics(data_dir / "logistics.csv", fail_after=10)
    assert result.ok is False
    assert result.error is not None
    err = result.error
    assert err.failure_type == "timeout"
    assert err.attempted and err.alternatives
    assert err.partial_results, "partial results gathered before the failure must be kept"
    # partial results are claims computed from the rows read before timeout
    assert all(c.source == "logistics" for c in err.partial_results)


# coordinator proceeds; failed source's exclusive metric is
# Incomplete with failure context; briefing names the unavailable source.
def test_coordinator_proceeds_and_annotates_gap(data_dir, news_extractor) -> None:  # type: ignore[no-untyped-def]
    result = investigate(
        "Meridian Components", data_dir, news_extractor, simulate_logistics_timeout=True
    )
    briefing = result.briefing
    out = briefing.render()
    # the run completed and still produced corroborated findings from other sources
    assert briefing.section("Well-Established")
    # logistics' exclusive metric is now an Incomplete coverage gap, citing the failure
    incomplete_ids = {f.metric_id for f in briefing.section(INCOMPLETE)}
    assert "late_shipment_count" in incomplete_ids
    gap = next(f for f in briefing.section(INCOMPLETE) if f.metric_id == "late_shipment_count")
    assert "timeout" in gap.coverage
    # the briefing explicitly states the source was unavailable
    assert "Sources unavailable" in out and "logistics" in out


# access failure (ok=False, error) vs valid empty result (ok=True, []).
def test_access_failure_distinct_from_empty_result(data_dir, news_extractor, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # access failure: file missing
    missing = read_logistics(tmp_path / "nope.csv")
    assert missing.ok is False and missing.error is not None

    # valid empty result: the unrelated article yields zero claims, ok=True
    from supply_chain_risk.readers import read_news

    empty = read_news(data_dir / "news" / "unrelated_macro.txt", news_extractor)
    assert empty.ok is True and empty.claims == [] and empty.error is None


# a single source failure never aborts the whole investigation.
def test_single_failure_does_not_abort(data_dir, news_extractor) -> None:  # type: ignore[no-untyped-def]
    result = investigate(
        "Meridian Components", data_dir, news_extractor, simulate_logistics_timeout=True
    )
    # all four readers ran (audit, logistics, quality, + news articles)
    sources_run = {r.source for r in result.reader_results}
    assert {"supplier_audit", "logistics", "internal_quality", "industry_news"} <= sources_run
    # the failed reader is present and marked not-ok; others succeeded
    logistics = next(r for r in result.reader_results if r.source == "logistics")
    assert logistics.ok is False
    assert any(r.ok for r in result.reader_results)


# only successful claims are vectorized; the failure's partial results
# (and the contested logistics value) are not silently added to shared memory.
def test_only_successful_claims_vectorized(data_dir, news_extractor) -> None:  # type: ignore[no-untyped-def]
    from supply_chain_risk.memory import SharedMemory

    mem = SharedMemory()
    investigate(
        "Meridian Components",
        data_dir,
        news_extractor,
        simulate_logistics_timeout=True,
        memory=mem,
    )
    # nothing from the failed logistics read made it into the store
    stored_sources = {c.source for c in mem.search("delivery shipments late", k=10)}
    assert "logistics" not in stored_sources


# Healthy run (no timeout) keeps the on-time conflict Contested across both sources.
def test_healthy_run_is_contested(data_dir, news_extractor) -> None:  # type: ignore[no-untyped-def]
    result = investigate("Meridian Components", data_dir, news_extractor)
    contested = {f.metric_id for f in result.briefing.section("Contested")}
    assert "on_time_delivery_rate" in contested
