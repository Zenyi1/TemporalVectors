#!/usr/bin/env python
"""Fetch natural counterfactual pairs from Wikipedia revision history.

Usage:
    python scripts/fetch_wikipedia_pairs.py --config configs/default.yaml
    python scripts/fetch_wikipedia_pairs.py --config configs/debug.yaml --max-pairs 20
"""

import argparse
import itertools
import logging
import sys
from pathlib import Path

import yaml

from temporal_vectors.config import DATA_PROC, SEED, WIKI_TARGET_YEARS
from temporal_vectors.data.pair_builder import build_pairs, save_pairs, validate_dataset
from temporal_vectors.data.wikipedia_fetcher import (
    DOMAIN_ARTICLES,
    create_site,
    fetch_article_pairs,
)
from temporal_vectors.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia natural pairs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Stop after collecting this many valid pairs (for debugging)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/processed/natural_pairs.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = {}
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    seed_everything(config.get("seed", SEED))

    years = config.get("wiki_target_years", WIKI_TARGET_YEARS)
    domains = config.get("wiki_domains", list(DOMAIN_ARTICLES.keys()))
    rate_limit = config.get("wiki_rate_limit_seconds", 1.0)
    max_pairs = args.max_pairs or config.get("max_pairs")
    output_path = args.output or DATA_PROC / "natural_pairs.jsonl"

    logger.info("Target years: %s", years)
    logger.info("Domains: %s", domains)
    logger.info("Max pairs: %s", max_pairs or "unlimited")

    # Connect to Wikipedia
    site = create_site()
    logger.info("Connected to Wikipedia")

    # Generate all year-pair combinations
    year_combos = list(itertools.combinations(sorted(years), 2))
    logger.info("Year combinations: %s", year_combos)

    # Fetch pairs for each domain, article, and year combination
    raw_pairs: list[dict] = []
    total_articles = 0

    for domain in domains:
        articles = DOMAIN_ARTICLES.get(domain, [])
        logger.info("Domain '%s': %d articles", domain, len(articles))

        for article in articles:
            for year_old, year_new in year_combos:
                pairs = fetch_article_pairs(
                    site, article, domain, year_old, year_new, rate_limit
                )
                raw_pairs.extend(pairs)
                total_articles += 1

                if pairs:
                    logger.info(
                        "  %s (%d->%d): %d pairs", article, year_old, year_new, len(pairs)
                    )

                # Early exit if we have enough
                if max_pairs and len(raw_pairs) >= max_pairs * 3:
                    break
            if max_pairs and len(raw_pairs) >= max_pairs * 3:
                break
        if max_pairs and len(raw_pairs) >= max_pairs * 3:
            break

    logger.info("Fetched %d raw pairs from %d article-year queries", len(raw_pairs), total_articles)

    # Filter and build valid pairs
    valid_pairs = build_pairs(raw_pairs, pair_type="natural")

    if max_pairs:
        valid_pairs = valid_pairs[:max_pairs]

    # Validate
    report = validate_dataset(valid_pairs)
    logger.info("Validation: %d total, %d valid, %d invalid", report["total"], report["valid"], report["invalid"])
    logger.info("Domain distribution: %s", report["domain_percentages"])

    if report["invalid"] > 0:
        logger.warning("Invalid pairs: %s", report["errors"][:5])

    # Save
    save_pairs(valid_pairs, output_path)
    logger.info("Done. Saved %d pairs to %s", len(valid_pairs), output_path)

    # Check against minimum target
    min_target = config.get("min_natural_pairs", 500)
    if len(valid_pairs) < min_target:
        logger.warning(
            "Below target: %d pairs (need %d). Consider adding more articles.",
            len(valid_pairs),
            min_target,
        )


if __name__ == "__main__":
    main()
