"""Provenance-preserving claim model and mixed-source readers."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from supply_chain_risk.models import Claim, ReaderResult
from supply_chain_risk.readers import (
    read_audit,
    read_logistics,
    read_news,
    read_quality,
)


# Claim model carries the required provenance/uncertainty fields.
def test_claim_has_required_fields() -> None:
    c = Claim(
        claim="x",
        evidence="y",
        source="supplier_audit",
        source_date=date(2026, 4, 10),
        confidence=0.9,
        metric_id="on_time_delivery_rate",
        value=95.0,
        unit="percent",
    )
    assert c.source == "supplier_audit"
    assert c.source_date == date(2026, 4, 10)
    assert c.metric_id == "on_time_delivery_rate"
    assert 0.0 <= c.confidence <= 1.0


# source_date and confidence are enforced, not optional/free-form.
def test_claim_requires_source_date() -> None:
    with pytest.raises(ValidationError):
        Claim(  # type: ignore[call-arg]
            claim="x", evidence="y", source="s", confidence=0.5, metric_id="m"
        )


def test_claim_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        Claim(
            claim="x",
            evidence="y",
            source="s",
            source_date=date(2026, 1, 1),
            confidence=1.4,
            metric_id="m",
        )


def test_claim_rejects_unparseable_date() -> None:
    with pytest.raises(ValidationError):
        Claim(
            claim="x",
            evidence="y",
            source="s",
            source_date="not-a-date",  # type: ignore[arg-type]
            confidence=0.5,
            metric_id="m",
        )


# audit reader: structured JSON -> Claims, carrying report_date.
def test_read_audit(data_dir: Path) -> None:
    result = read_audit(data_dir / "audit.json")
    assert result.ok and result.source == "supplier_audit"
    by_metric = {c.metric_id: c for c in result.claims}
    assert by_metric["on_time_delivery_rate"].value == 95.0
    assert by_metric["on_time_delivery_rate"].source_date == date(2026, 4, 10)
    assert all(c.source == "supplier_audit" for c in result.claims)


# logistics reader: semi-structured CSV -> Claims, derived metrics.
def test_read_logistics(data_dir: Path) -> None:
    result = read_logistics(data_dir / "logistics.csv")
    assert result.ok and result.source == "logistics"
    by_metric = {c.metric_id: c for c in result.claims}
    assert by_metric["on_time_delivery_rate"].value == 78.0  # 39/50
    assert by_metric["on_time_delivery_rate"].source_date == date(2026, 4, 5)
    assert by_metric["average_lead_time_days"].value == pytest.approx(12.0, abs=0.5)


# news reader: prose -> Claims via the extractor, dated, sourced.
def test_read_news_extracts_dated_claims(news_dir: Path, news_extractor) -> None:  # type: ignore[no-untyped-def]
    result = read_news(news_dir / "port_strike.txt", news_extractor)
    assert result.ok and result.source == "industry_news"
    assert result.claims, "expected at least one extracted claim"
    c = result.claims[0]
    assert c.source == "industry_news"
    assert c.metric_id == "port_disruption"
    assert c.source_date == date(2026, 3, 17)


# the live extractor uses the SDK with a structured-output contract.
def test_anthropic_extractor_uses_structured_output() -> None:
    from supply_chain_risk.news_extraction import AnthropicNewsExtractor

    calls: dict[str, object] = {}

    class _RecordingMessages:
        def create(self, **kwargs: object) -> object:
            calls.update(kwargs)

            class _Block:
                type = "text"
                text = '{"claims": []}'

            class _Resp:
                content = (_Block(),)

            return _Resp()

    class _Client:
        messages = _RecordingMessages()

    extractor = AnthropicNewsExtractor("Meridian Components", client=_Client())
    assert extractor.extract("some article") == []
    # structured output is requested via output_config.format (not a prefill)
    fmt = calls["output_config"]["format"]  # type: ignore[index]
    assert fmt["type"] == "json_schema"
    assert "claims" in fmt["schema"]["properties"]
    # the prompt states the goal/contract and names the supplier
    assert "Meridian Components" in calls["system"]  # type: ignore[operator]


# empty-but-readable source yields ok=True, claims==[] (not an error).
def test_empty_result_is_ok_not_error(news_dir: Path, news_extractor) -> None:  # type: ignore[no-untyped-def]
    result = read_news(news_dir / "unrelated_macro.txt", news_extractor)
    assert result.ok is True
    assert result.claims == []
    assert result.error is None


# a missing file is an access failure (ok=False), distinctly.
def test_missing_file_is_access_failure(tmp_path: Path) -> None:
    result = read_audit(tmp_path / "nope.json")
    assert result.ok is False
    assert result.error is not None
    assert result.error.failure_type == "file_not_found"


# ambiguous supplier mention is flagged, not heuristically resolved.
def test_ambiguous_match_sets_needs_identifier(news_dir: Path, news_extractor) -> None:  # type: ignore[no-untyped-def]
    result = read_news(news_dir / "ambiguous_meridian.txt", news_extractor)
    assert result.ok is True
    ambiguous = [c for c in result.claims if c.needs_identifier]
    assert len(ambiguous) == 1, "the ambiguous entity claim must be flagged"
    c = ambiguous[0]
    assert len(c.candidates) == 2
    assert "Meridian Components" in c.candidates


# (quality source, supports multi-source corroboration downstream)
def test_read_quality(data_dir: Path) -> None:
    result = read_quality(data_dir / "quality.sqlite")
    assert result.ok and result.source == "internal_quality"
    by_metric = {c.metric_id: c for c in result.claims}
    assert by_metric["defect_rate_ppm"].value == 190.0
    assert by_metric["defect_rate_ppm"].source_date == date(2026, 4, 8)


# Provenance round-trip foundation: every reader claim carries source + date.
def test_all_readers_emit_provenance(data_dir: Path, news_dir: Path, news_extractor) -> None:  # type: ignore[no-untyped-def]
    results: list[ReaderResult] = [
        read_audit(data_dir / "audit.json"),
        read_logistics(data_dir / "logistics.csv"),
        read_quality(data_dir / "quality.sqlite"),
        read_news(news_dir / "port_strike.txt", news_extractor),
    ]
    for r in results:
        for c in r.claims:
            assert c.source and isinstance(c.source_date, date)
