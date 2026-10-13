"""Domain records for the extraction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Category = Literal["format", "missing_source", "consistency"]


@dataclass(frozen=True)
class Endorsement:
    name: str
    limit: float | None


@dataclass(frozen=True)
class PremiumComponent:
    name: str
    amount: float


@dataclass(frozen=True)
class ValidationError:
    """A single validation failure on an extraction attempt."""

    field: str
    observed_value: str
    category: Category
    detected_pattern: str
    message: str


@dataclass(frozen=True)
class RetryFutileEscalation:
    """Emitted when retry is futile because the source document is missing data."""

    policy_id: str
    field: str
    category: Literal["missing_source"]
    detected_pattern: str
    reason: str


@dataclass
class PolicyExtraction:
    """A successful extraction record."""

    policy_id: str
    policy_type: Literal["auto", "home", "umbrella", "other"]
    premium_amount: float | None
    deductible: float | None
    coverage_limit: float | None
    endorsements: list[Endorsement] | None
    exclusions: list[str]
    premium_components: list[PremiumComponent] | None
    confidence: dict[str, float]
    retry_count: int = 0
    final_attempt_index: int = 0
    validation_history: list[ValidationError] = field(default_factory=list)


ExtractionOutcome = PolicyExtraction | RetryFutileEscalation
