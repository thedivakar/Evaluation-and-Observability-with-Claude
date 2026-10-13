"""Core data models for the supply-chain risk investigation system.

The `Claim` is the single structured intermediate representation that every
source reader emits and every downstream step consumes. It carries provenance
(`source`, `source_date`) and uncertainty (`confidence`) so attribution and
temporal context survive synthesis. `ReaderResult` / `FailureContext` make
errors values rather than exceptions at the source boundary.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    """A single finding about a supplier, with its provenance.

    `metric_id` groups claims that describe the same quantity (e.g. several
    sources reporting on-time delivery). `value`/`unit` are populated for
    numeric metrics so conflicts can be detected; qualitative findings leave
    `value` as None. `needs_identifier`/`candidates` flag an ambiguous entity
    match the reader refused to resolve heuristically.
    """

    model_config = ConfigDict(frozen=True)

    claim: str
    evidence: str
    source: str
    source_date: date
    confidence: float = Field(ge=0.0, le=1.0)
    metric_id: str
    value: float | None = None
    unit: str | None = None
    needs_identifier: bool = False
    candidates: tuple[str, ...] = ()


class FailureContext(BaseModel):
    """Structured error context propagated when a source cannot be read.

    Enables the coordinator to make an informed recovery decision instead of
    aborting: it knows the failure type, what was attempted, any partial
    results gathered before the failure, and alternative approaches.
    """

    failure_type: str
    attempted: str
    partial_results: list[Claim] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)


class ReaderResult(BaseModel):
    """The outcome of reading one source.

    A successful read sets `ok=True` and carries `claims` (possibly empty — a
    valid empty result). An access failure sets `ok=False` and populates
    `error`; the two are deliberately distinguishable.
    """

    source: str
    ok: bool
    claims: list[Claim] = Field(default_factory=list)
    error: FailureContext | None = None
