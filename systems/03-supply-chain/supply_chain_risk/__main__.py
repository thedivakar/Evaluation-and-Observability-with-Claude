"""CLI: run a supply-chain risk investigation for one supplier and print the briefing.

    supply-chain-investigate meridian
    supply-chain-investigate meridian --offline        # replay recorded extraction
    supply-chain-investigate meridian --simulate-timeout

By default the news reader calls the Anthropic SDK (needs ANTHROPIC_API_KEY).
`--offline` replays the recorded extraction fixtures so the demo runs without a key.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .coordinator import investigate
from .news_extraction import AnthropicNewsExtractor, RecordedNewsExtractor
from .readers import NewsExtractor

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supply-chain-investigate")
    parser.add_argument("supplier", help="supplier slug, e.g. 'meridian'")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument(
        "--offline", action="store_true", help="replay recorded news extraction (no API key)"
    )
    parser.add_argument(
        "--simulate-timeout", action="store_true", help="simulate the logistics source timing out"
    )
    parser.add_argument("--supplier-name", default=None, help="display name for the briefing")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_root) / args.supplier
    if not data_dir.exists():
        parser.error(f"no data directory for supplier {args.supplier!r} at {data_dir}")

    supplier_name = args.supplier_name or args.supplier.title()
    extractor: NewsExtractor
    if args.offline:
        extractor = RecordedNewsExtractor(data_dir / "news", ROOT / "fixtures" / "news_extraction")
    else:
        extractor = AnthropicNewsExtractor(supplier_name)

    result = investigate(
        supplier_name,
        data_dir,
        extractor,
        simulate_logistics_timeout=args.simulate_timeout,
    )
    print(result.briefing.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
