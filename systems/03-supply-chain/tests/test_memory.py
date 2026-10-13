"""Shared-memory vector store with cross-source semantic retrieval."""
from __future__ import annotations

from datetime import date

from supply_chain_risk.memory import SharedMemory
from supply_chain_risk.models import Claim


# SharedMemory wraps Chroma with a local embedding function (offline).
def test_shared_memory_constructs_offline() -> None:
    mem = SharedMemory()
    assert mem.count() == 0


# add_claims stores provenance metadata that round-trips intact.
def test_provenance_round_trips(seeded_memory) -> None:  # type: ignore[no-untyped-def]
    hits = seeded_memory.search("on-time delivery performance", k=5)
    assert hits
    for c in hits:
        assert isinstance(c, Claim)
        assert c.source and isinstance(c.source_date, date)
    audit_ot = [c for c in hits if c.metric_id == "on_time_delivery_rate"]
    assert any(c.source == "supplier_audit" and c.source_date == date(2026, 4, 10)
               for c in audit_ot)


# semantic search surfaces cross-source related claims ahead of noise.
def test_search_retrieves_cross_source_corroborating_claims(seeded_memory) -> None:  # type: ignore[no-untyped-def]
    hits = seeded_memory.search("on-time delivery rate for the supplier", k=4)
    sources = {c.source for c in hits if c.metric_id == "on_time_delivery_rate"}
    # both the audit (95%) and logistics (78%) on-time claims are retrieved
    assert {"supplier_audit", "logistics"} <= sources
    # and they rank ahead of an unrelated finding like financial distress
    top2_metrics = [c.metric_id for c in hits[:2]]
    assert "supplier_financial_distress" not in top2_metrics


# related_to finds similar claims across sources, excluding itself.
def test_related_to_excludes_self_and_finds_cross_source(seeded_memory) -> None:  # type: ignore[no-untyped-def]
    audit_ot = Claim(
        claim="During Q1 2026, Meridian Components achieved an on-time delivery rate "
        "of 95% across all customer purchase orders, consistent with our internal "
        "service-level commitments.",
        evidence="On-time delivery: 95 percent (Q1 2026)",
        source="supplier_audit",
        source_date=date(2026, 4, 10),
        confidence=0.9,
        metric_id="on_time_delivery_rate",
        value=95.0,
        unit="percent",
    )
    related = seeded_memory.related_to(audit_ot, k=3)
    assert all(not (c.source == "supplier_audit" and c.metric_id == "on_time_delivery_rate"
                    and c.value == 95.0) for c in related)
    # the logistics on-time claim (different source, same metric) is surfaced
    assert any(c.source == "logistics" and c.metric_id == "on_time_delivery_rate"
               for c in related)


# deterministic: same query yields the same ordering across instances.
def test_deterministic_results() -> None:
    from tests.conftest import load_all_claims

    claims = load_all_claims()
    a, b = SharedMemory(), SharedMemory()
    a.add_claims(claims)
    b.add_claims(claims)
    q = "defect rate quality"
    ra = [(c.source, c.metric_id) for c in a.search(q, k=4)]
    rb = [(c.source, c.metric_id) for c in b.search(q, k=4)]
    assert ra == rb
