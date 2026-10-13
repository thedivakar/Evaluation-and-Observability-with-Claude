"""Synthesis — turn provenance-carrying claims into an executive briefing.

The synthesis is deterministic Python (not an LLM call), so provenance
preservation and conflict annotation are *guaranteed*, not merely prompted:

- claims are grouped by the metric they describe (with shared-memory retrieval
  catching near-duplicate metric ids);
- each group is classified Well-Established / Contested / Incomplete;
- conflicts are **annotated, not arbitrated** — every conflicting value is kept
  with its source and date, and no single/averaged value is emitted;
- each finding carries a coverage annotation and an escalation flag driven by
  explicit criteria (never by model confidence or sentiment).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import SharedMemory, _claim_id
from .models import Claim

WELL_ESTABLISHED = "Well-Established"
CONTESTED = "Contested"
INCOMPLETE = "Incomplete"
SECTIONS = (WELL_ESTABLISHED, CONTESTED, INCOMPLETE)

REL_TOL = 0.10  # values within 10% are treated as agreeing

# Metrics the investigation always tries to assess; a tracked metric with no
# usable source is reported as Incomplete (a coverage gap, not a fabricated one).
TRACKED_METRICS = (
    "on_time_delivery_rate",
    "defect_rate_ppm",
    "average_lead_time_days",
    "production_capacity_utilization",
)
# Metrics whose contested/incomplete state warrants human escalation.
HIGH_IMPACT = frozenset(
    {"on_time_delivery_rate", "defect_rate_ppm", "production_capacity_utilization"}
)
LOGISTICS = "logistics"
NEWS = "industry_news"


def _tokens(metric_id: str) -> set[str]:
    return set(metric_id.split("_"))


@dataclass
class MetricGroup:
    metric_id: str
    claims: list[Claim]


def group_by_metric(claims: list[Claim], memory: SharedMemory | None = None) -> list[MetricGroup]:
    """Cluster claims describing the same metric.

    Claims with an identical metric_id always group together. When `memory` is
    given, near-duplicate metric ids (different strings, same concept) are merged
    too: two claims unify if shared-memory retrieval ranks them as neighbours and
    their metric ids share at least two tokens (e.g. defect_rate / defect_rate_ppm).
    """
    parent = list(range(len(claims)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_metric: dict[str, list[int]] = {}
    for i, c in enumerate(claims):
        by_metric.setdefault(c.metric_id, []).append(i)
    for idxs in by_metric.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    if memory is not None:
        index_by_id = {_claim_id(c): i for i, c in enumerate(claims)}
        for i, c in enumerate(claims):
            for r in memory.related_to(c, k=3):
                if r.metric_id == c.metric_id:
                    continue
                if len(_tokens(r.metric_id) & _tokens(c.metric_id)) >= 2:
                    jdx = index_by_id.get(_claim_id(r))
                    if jdx is not None:
                        union(i, jdx)

    clusters: dict[int, list[Claim]] = {}
    for i, c in enumerate(claims):
        clusters.setdefault(find(i), []).append(c)

    groups: list[MetricGroup] = []
    for members in clusters.values():
        counts: dict[str, int] = {}
        for c in members:
            counts[c.metric_id] = counts.get(c.metric_id, 0) + 1
        canonical = sorted(counts, key=lambda m: (-counts[m], m))[0]
        groups.append(MetricGroup(metric_id=canonical, claims=members))
    groups.sort(key=lambda g: g.metric_id)
    return groups


@dataclass
class Finding:
    metric_id: str
    classification: str
    claims: list[Claim]
    coverage: str
    escalate: bool
    escalation_reason: str = ""

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.claims:
            if c.source not in seen:
                seen.append(c.source)
        return seen


def _source_values(claims: list[Claim]) -> dict[str, float]:
    values: dict[str, float] = {}
    for c in claims:
        if c.value is not None and c.source not in values:
            values[c.source] = c.value
    return values


def _disagree(values: list[float]) -> bool:
    lo, hi = min(values), max(values)
    denom = max(abs(lo), abs(hi), 1e-9)
    return (hi - lo) / denom > REL_TOL


def _classify(group: MetricGroup) -> tuple[str, str]:
    """Return (classification, coverage_badge) for a present metric group."""
    n_sources = len({c.source for c in group.claims})
    values = _source_values(group.claims)
    if len(values) >= 2 and _disagree(list(values.values())):
        return CONTESTED, f"{n_sources} sources, conflicting"
    if n_sources >= 2:
        return WELL_ESTABLISHED, f"corroborated across {n_sources} sources"
    return WELL_ESTABLISHED, "single source only"


def _escalation(metric_id: str, classification: str, claims: list[Claim]) -> tuple[bool, str]:
    """Explicit escalation criteria — never confidence/sentiment.

    Few-shot intent:
      on_time_delivery_rate Contested (high-impact) -> escalate
      defect_rate_ppm Well-Established               -> do not escalate
      production_capacity_utilization Incomplete     -> escalate (high-impact gap)
      supplier_financial_distress needs_identifier   -> escalate for clarification
    """
    if any(c.needs_identifier for c in claims):
        return True, "ambiguous supplier identity — request an identifier"
    if metric_id in HIGH_IMPACT and classification == CONTESTED:
        return True, "high-impact metric is contested across sources"
    if metric_id in HIGH_IMPACT and classification == INCOMPLETE:
        return True, "high-impact metric has no usable source"
    return False, ""


@dataclass
class Briefing:
    supplier: str
    findings: list[Finding] = field(default_factory=list)
    unavailable_sources: list[str] = field(default_factory=list)

    def section(self, name: str) -> list[Finding]:
        return [f for f in self.findings if f.classification == name]

    @property
    def escalations(self) -> list[Finding]:
        return [f for f in self.findings if f.escalate]

    def render(self) -> str:
        lines = [f"# Supply Chain Risk Briefing — {self.supplier}", ""]
        if self.unavailable_sources:
            lines.append(f"> Sources unavailable: {', '.join(self.unavailable_sources)}")
            lines.append("")
        for name in SECTIONS:
            lines.append(f"## {name}")
            findings = self.section(name)
            if not findings:
                lines.append("_none_")
                lines.append("")
                continue
            for f in findings:
                lines.extend(_render_finding(f))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _render_finding(f: Finding) -> list[str]:
    flag = "  ⚠️ ESCALATE" if f.escalate else ""
    out = [f"### {f.metric_id}  _[{f.coverage}]_{flag}"]
    if f.escalate:
        out.append(f"- escalation: {f.escalation_reason}")

    if f.classification == INCOMPLETE:
        out.append(f"- {f.coverage}")
        out.append("")
        return out

    # Cross-source comparison as a structured list (annotate, don't arbitrate).
    numeric = [c for c in f.claims if c.value is not None]
    if numeric:
        out.append("- Reported values by source:")
        for c in numeric:
            unit = f" {c.unit}" if c.unit else ""
            out.append(f"    - {c.value}{unit} — {c.source} (as of {c.source_date})")

    # Logistics data rendered as a Markdown table.
    logistics = [c for c in f.claims if c.source == LOGISTICS and c.value is not None]
    if logistics:
        out.append("")
        out.append("| Metric | Value | As of | Source |")
        out.append("| --- | --- | --- | --- |")
        for c in logistics:
            unit = f" {c.unit}" if c.unit else ""
            out.append(f"| {c.metric_id} | {c.value}{unit} | {c.source_date} | {c.source} |")

    # News findings rendered as prose.
    news = [c for c in f.claims if c.source == NEWS]
    for c in news:
        out.append(f"> {c.claim} ({c.source}, {c.source_date})")

    out.append("")
    return out


def build_briefing(
    supplier: str,
    claims: list[Claim],
    memory: SharedMemory | None = None,
    *,
    tracked_metrics: tuple[str, ...] = TRACKED_METRICS,
    unavailable: dict[str, str] | None = None,
    unavailable_sources: list[str] | None = None,
) -> Briefing:
    """Assemble the coverage-tagged briefing.

    `unavailable` maps a metric_id to a failure reason (from a source that could
    not be read) so its metric is reported under Incomplete with that context.
    """
    unavailable = unavailable or {}
    groups = group_by_metric(claims, memory)
    findings: list[Finding] = []
    present: set[str] = set()

    for g in groups:
        present.add(g.metric_id)
        classification, coverage = _classify(g)
        escalate, reason = _escalation(g.metric_id, classification, g.claims)
        findings.append(
            Finding(
                metric_id=g.metric_id,
                classification=classification,
                claims=g.claims,
                coverage=coverage,
                escalate=escalate,
                escalation_reason=reason,
            )
        )

    # Incomplete: tracked metrics with no usable source, and metrics whose source
    # failed (access failure -> coverage gap, distinct from a valid empty result).
    incomplete_ids = {m for m in tracked_metrics if m not in present}
    incomplete_ids |= {m for m in unavailable if m not in present}
    for metric_id in sorted(incomplete_ids):
        reason = unavailable.get(metric_id, "no source reported this metric")
        escalate, esc_reason = _escalation(metric_id, INCOMPLETE, [])
        findings.append(
            Finding(
                metric_id=metric_id,
                classification=INCOMPLETE,
                claims=[],
                coverage=f"missing source: {reason}",
                escalate=escalate,
                escalation_reason=esc_reason,
            )
        )

    findings.sort(key=lambda f: (SECTIONS.index(f.classification), f.metric_id))
    return Briefing(
        supplier=supplier,
        findings=findings,
        unavailable_sources=unavailable_sources or [],
    )
