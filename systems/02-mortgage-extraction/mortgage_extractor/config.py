"""Configuration constants for the mortgage extraction pipeline."""
from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-6"

DEFAULT_MAX_TOKENS = 1500
DEFAULT_TOLERANCE_USD = 1.00

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
RECORDED_RESPONSES_DIR = FIXTURES_DIR / "recorded_responses"
