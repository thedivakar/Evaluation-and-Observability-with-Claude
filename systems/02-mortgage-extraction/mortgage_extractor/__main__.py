"""Command-line entry point: extract a mortgage document and validate the result."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mortgage_extractor.client import RecordingClient
from mortgage_extractor.config import DEFAULT_MODEL
from mortgage_extractor.errors import (
    FlaggedForReviewError,
    UnsupportedDocumentTypeError,
)
from mortgage_extractor.pipeline import Pipeline
from mortgage_extractor.validator import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mortgage-extract",
        description=(
            "Extract structured data from a single mortgage document and "
            "report mathematical-consistency findings."
        ),
    )
    parser.add_argument("document", type=Path, help="Path to a plain-text mortgage document.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "record", "replay"],
        default="auto",
        help=(
            "Recording-client mode: 'auto' uses cache when present, "
            "'record' always hits API, 'replay' fails if uncached."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Emit INFO-level logs to stderr."
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    try:
        document_text = args.document.read_text()
    except OSError as exc:
        print(f"error: cannot read {args.document}: {exc}", file=sys.stderr)
        return 2

    pipeline = Pipeline(model=args.model, client=RecordingClient(mode=args.mode))

    try:
        extraction = pipeline.run(document_text)
    except UnsupportedDocumentTypeError as exc:
        print(f"unsupported document: {exc.reason}", file=sys.stderr)
        return 3
    except FlaggedForReviewError as exc:
        print(f"flagged for review: {exc.reason}", file=sys.stderr)
        return 4

    report = validate(extraction)
    payload = {
        "extraction": extraction.model_dump(mode="json"),
        "validation": report.model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
