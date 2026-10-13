"""CLI entry point for the policy-extractor pipeline.

Subcommands: extract, batch,
review, route, pipeline (end-to-end).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic

from policy_extractor.batch import (
    AnthropicBatchClient,
    DryRunSampleResult,
    dry_run_sample,
    process_with_resubmission,
)
from policy_extractor.client import AnthropicMessagesClient
from policy_extractor.records import (
    ExtractionOutcome,
    PolicyExtraction,
    RetryFutileEscalation,
)
from policy_extractor.retry import extract_with_retry
from policy_extractor.reviewer import (
    has_disagreement,
    independent_review,
    integration_pass,
)
from policy_extractor.routing import (
    apply_stratified_spot_check,
    route_extraction,
    write_routing_decisions,
)
from policy_extractor.summary import summarize_patterns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="policy-extractor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract a single policy document (US-01).")
    p_extract.add_argument("document", type=Path)
    p_extract.add_argument("--policy-id", required=True)
    p_extract.add_argument("--max-retries", type=int, default=3)

    p_batch = sub.add_parser("batch", help="Submit a batch of policies (US-02).")
    p_batch.add_argument("policies_dir", type=Path)
    p_batch.add_argument(
        "--dry-run-sample", type=int, default=3,
        help="Run N policies real-time first as a sample. Use 0 to skip the sample.",
    )
    p_batch.add_argument(
        "--sample-threshold", type=float, default=0.7,
        help="Minimum sample first-pass success rate to authorise the full batch.",
    )
    p_batch.add_argument("--force", action="store_true", help="Skip the sample-rate gate.")

    p_pipeline = sub.add_parser(
        "pipeline", help="End-to-end: extract → review → integration → route (US-01..US-04).",
    )
    p_pipeline.add_argument("policies_dir", type=Path)
    p_pipeline.add_argument("--routing-out", type=Path, default=Path("routing_decisions.json"))
    p_pipeline.add_argument("--spot-check-pct", type=float, default=0.1)
    p_pipeline.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr,
    )

    if args.command == "extract":
        return _cmd_extract(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "pipeline":
        return _cmd_pipeline(args)
    return 2


def _cmd_extract(args: argparse.Namespace) -> int:
    text = args.document.read_text()
    client = AnthropicMessagesClient(Anthropic())
    outcome = extract_with_retry(
        client=client,
        policy_id=args.policy_id,
        document_text=text,
        max_retries=args.max_retries,
    )
    print(json.dumps(_serialize_outcome(outcome), indent=2, sort_keys=True))
    return 0 if isinstance(outcome, PolicyExtraction) else 1


def _load_policies(policies_dir: Path) -> list[tuple[str, str]]:
    policies = []
    for path in sorted(policies_dir.glob("POL-*.txt")):
        policies.append((path.stem, path.read_text()))
    if not policies:
        raise FileNotFoundError(f"No POL-*.txt files in {policies_dir}.")
    return policies


def _cmd_batch(args: argparse.Namespace) -> int:
    policies = _load_policies(args.policies_dir)
    extractor_client = AnthropicMessagesClient(Anthropic())

    if args.dry_run_sample > 0:
        sample = dry_run_sample(
            extractor_client=extractor_client,
            policies=policies,
            sample_size=args.dry_run_sample,
        )
        _print_sample(sample)
        if not args.force and sample.first_pass_success_rate < args.sample_threshold:
            print(
                f"Sample first-pass success rate {sample.first_pass_success_rate:.0%} "
                f"below threshold {args.sample_threshold:.0%}; aborting. Use --force to override.",
                file=sys.stderr,
            )
            return 3

    batch_client = AnthropicBatchClient(Anthropic())
    results = process_with_resubmission(
        batch_client=batch_client,
        extractor_client=extractor_client,
        policies=policies,
    )
    print(json.dumps({pid: _serialize_outcome(o) for pid, o in results.items()}, indent=2))
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    policies = _load_policies(args.policies_dir)
    extractor_client = AnthropicMessagesClient(Anthropic())

    decisions = []
    outcomes: list[ExtractionOutcome] = []
    for pid, doc in policies:
        outcome = extract_with_retry(
            client=extractor_client, policy_id=pid, document_text=doc, max_retries=3,
        )
        outcomes.append(outcome)
        if isinstance(outcome, RetryFutileEscalation):
            continue
        review = independent_review(
            client=extractor_client,
            source_document=doc,
            extracted_record={
                "policy_id": outcome.policy_id,
                "policy_type": outcome.policy_type,
                "premium_amount": outcome.premium_amount,
                "deductible": outcome.deductible,
                "coverage_limit": outcome.coverage_limit,
                "endorsements": [
                    {"name": e.name, "limit": e.limit}
                    for e in (outcome.endorsements or [])
                ],
                "exclusions": outcome.exclusions,
            },
        )
        integration = integration_pass(outcome)
        decision = route_extraction(
            extraction=outcome,
            review=review,
            integration_findings=integration,
        )
        if has_disagreement(review):
            disagreeing_fields = [
                f for f, a in review.agreements.items() if a.agreement == "disagree"
            ]
            logging.info(
                "review_disagreement policy_id=%s fields=%s",
                outcome.policy_id,
                disagreeing_fields,
            )
        decisions.append(decision)

    decisions = apply_stratified_spot_check(
        decisions, sample_pct=args.spot_check_pct, seed=args.seed,
    )
    write_routing_decisions(decisions, args.routing_out)
    print(json.dumps({
        "decisions_written": len(decisions),
        "auto_approve": sum(1 for d in decisions if d.decision == "auto_approve"),
        "human_review": sum(1 for d in decisions if d.decision == "human_review"),
        "spot_check": sum(1 for d in decisions if d.decision == "spot_check"),
        "escalations": sum(1 for o in outcomes if isinstance(o, RetryFutileEscalation)),
        "pattern_summary": summarize_patterns(outcomes),
        "output_path": str(args.routing_out),
    }, indent=2, default=dict))
    return 0


def _serialize_outcome(outcome: ExtractionOutcome) -> dict[str, object]:
    if isinstance(outcome, PolicyExtraction):
        return {"kind": "extraction", **asdict(outcome)}
    return {"kind": "escalation", **asdict(outcome)}


def _print_sample(sample: DryRunSampleResult) -> None:
    print(json.dumps(asdict(sample), indent=2), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
