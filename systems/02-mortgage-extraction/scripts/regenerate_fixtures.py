#!/usr/bin/env python
"""Regenerate the recorded-API-response cache by hitting the live Anthropic API.

Deletes ``fixtures/recorded_responses/*.json`` and replays every fixture
document under ``fixtures/documents/`` through the pipeline in ``record`` mode.
Requires ``ANTHROPIC_API_KEY`` in the environment.

Run after any schema change that would invalidate cached request hashes, or to
refresh fixtures against a newer model.
"""
from __future__ import annotations

import logging
import sys

from mortgage_extractor.client import RecordingClient
from mortgage_extractor.config import FIXTURES_DIR, RECORDED_RESPONSES_DIR
from mortgage_extractor.errors import (
    FlaggedForReviewError,
    UnsupportedDocumentTypeError,
)
from mortgage_extractor.pipeline import Pipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    log = logging.getLogger("regenerate_fixtures")

    documents_dir = FIXTURES_DIR / "documents"
    if not documents_dir.exists():
        log.error("no documents dir at %s", documents_dir)
        return 2

    deleted = 0
    for stale in RECORDED_RESPONSES_DIR.glob("*.json"):
        stale.unlink()
        deleted += 1
    log.info("removed %d stale recorded response(s)", deleted)

    pipeline = Pipeline(client=RecordingClient(mode="record"))
    documents = sorted(documents_dir.glob("*.txt"))
    if not documents:
        log.warning("no documents to record under %s", documents_dir)
        return 0

    for path in documents:
        log.info("recording: %s", path.name)
        text = path.read_text()
        try:
            pipeline.run(text)
        except (UnsupportedDocumentTypeError, FlaggedForReviewError) as exc:
            log.info("  → %s: %s", type(exc).__name__, exc)

    log.info("done; cache now has %d entr(y/ies)", len(list(RECORDED_RESPONSES_DIR.glob("*.json"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
