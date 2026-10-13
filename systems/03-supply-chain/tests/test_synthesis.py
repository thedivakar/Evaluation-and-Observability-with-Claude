"""Conflict annotation and coverage-tagged synthesis briefing."""
from __future__ import annotations

from datetime import date

from supply_chain_risk.memory import SharedMemory
from supply_chain_risk.models import Claim
from supply_chain_risk.synthesis import (
    CONTESTED,
    INCOMPLETE,
    SECTIONS,
    WELL_ESTABLISHED,
    build_briefing,
    group_by_metric,
)


def _claim(metric_id: str, text: str, source: str, value: float | None = None,
           *, conf: float = 0.8, dt: date = date(2026, 4, 1),
           needs_identifier: bool = False, candidates: tuple[str, ...] = ()) -> Claim:
    return Claim(
        claim=text, evidence=text, source=source, source_date=dt, confidence=conf,
        metric_id=metric_id, value=value, unit="percent" if value is not None else None,
        needs_identifier=needs_identifier, candidates=candidates,
    )


# group_by_metric merges near-duplicate metric ids via shared memory.
def test_group_merges_near_duplicate_metrics() -> None:
    c1 = _claim("defect_rate_ppm", "Defect rate measured 190 ppm in incoming QC.",
                "internal_quality", 190.0)
    c2 = _claim("defect_rate", "Defect rate reported at 180 ppm by the supplier audit.",
                "supplier_audit", 180.0)
    mem = SharedMemory()
    mem.add_claims([c1, c2])
    merged = group_by_metric([c1, c2], mem)
    assert len(merged) == 1, "near-duplicate defect metrics should merge into one group"
    # Without memory they stay separate (exact metric_id only).
    assert len(group_by_metric([c1, c2], None)) == 2


def test_group_keeps_distinct_metrics_separate(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    groups = {g.metric_id for g in group_by_metric(all_claims, seeded_memory)}
    assert {"on_time_delivery_rate", "defect_rate_ppm", "average_lead_time_days"} <= groups
    # unrelated metrics are not collapsed together
    assert "port_disruption" in groups and "supplier_financial_distress" in groups


# classification: agree -> Well-Established, disagree -> Contested,
# tracked-but-absent -> Incomplete.
def test_classification(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    b = build_briefing("Meridian Components", all_claims, seeded_memory)
    cls = {f.metric_id: f.classification for f in b.findings}
    assert cls["on_time_delivery_rate"] == CONTESTED          # 95 vs 78
    assert cls["defect_rate_ppm"] == WELL_ESTABLISHED         # 180 vs 190 (agree)
    assert cls["average_lead_time_days"] == WELL_ESTABLISHED  # 12 vs 12.0
    assert cls["production_capacity_utilization"] == INCOMPLETE  # tracked, no source


# Contested preserves every value with source+date; no arbitration.
def test_contested_preserves_both_values(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    b = build_briefing("Meridian Components", all_claims, seeded_memory)
    out = b.render()
    contested = b.section(CONTESTED)
    assert [f.metric_id for f in contested] == ["on_time_delivery_rate"]
    # both conflicting values appear, each attributed and dated
    assert "95.0" in out and "78.0" in out
    assert "supplier_audit" in out and "logistics" in out
    assert "2026-04-10" in out and "2026-04-05" in out
    # the system does NOT arbitrate: no averaged/resolved single value
    assert "86.5" not in out


# content-appropriate rendering: table / prose / structured list.
def test_content_appropriate_rendering(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    out = build_briefing("Meridian Components", all_claims, seeded_memory).render()
    assert "| Metric | Value | As of | Source |" in out          # logistics as table
    assert "> " in out                                            # news as prose
    assert "- Reported values by source:" in out                 # comparison as list


# escalation by explicit criteria, independent of confidence/sentiment.
def test_escalation_criteria(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    b = build_briefing("Meridian Components", all_claims, seeded_memory)
    escalated = {f.metric_id for f in b.escalations}
    assert escalated == {
        "on_time_delivery_rate",          # high-impact + Contested
        "production_capacity_utilization",  # high-impact + Incomplete
        "supplier_financial_distress",    # needs_identifier
    }
    # defect_rate is corroborated (high confidence) but NOT escalated — escalation
    # is not driven by confidence; the ambiguous finding (confidence 0.4) IS.
    by_id = {f.metric_id: f for f in b.findings}
    assert by_id["defect_rate_ppm"].escalate is False
    assert by_id["supplier_financial_distress"].escalate is True


def test_escalation_ignores_confidence_directly() -> None:
    # A low-confidence, single-source, non-high-impact finding must not escalate.
    low_conf = _claim("field_return_rate", "Field returns at 1.4%.", "internal_quality",
                      1.4, conf=0.05)
    b = build_briefing("X", [low_conf])
    assert b.section(WELL_ESTABLISHED)[0].escalate is False


# exactly three sections, each populated or explicit none, with attribution.
def test_three_sections_with_attribution(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    b = build_briefing("Meridian Components", all_claims, seeded_memory)
    out = b.render()
    for name in SECTIONS:
        assert f"## {name}" in out
    level2 = [ln for ln in out.splitlines() if ln.startswith("## ")]
    assert level2 == [f"## {name}" for name in SECTIONS]
    # every finding traces to at least one source (or is an explicit gap)
    for f in b.findings:
        assert f.classification == INCOMPLETE or f.sources


def test_empty_section_renders_none() -> None:
    # Only a well-established finding -> Contested and Incomplete say "none".
    c = _claim("revenue", "ok", "supplier_audit", 100.0)
    out = build_briefing("X", [c], tracked_metrics=()).render()
    assert "## Contested\n_none_" in out
    assert "## Incomplete\n_none_" in out


# per-finding coverage badges describe support level.
def test_coverage_badges(all_claims, seeded_memory) -> None:  # type: ignore[no-untyped-def]
    out = build_briefing("Meridian Components", all_claims, seeded_memory).render()
    assert "corroborated across 2 sources" in out   # defect_rate / lead_time
    assert "single source only" in out              # e.g. field_return_rate
    assert "missing source:" in out                 # production_capacity_utilization
