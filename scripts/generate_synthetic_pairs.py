#!/usr/bin/env python
"""Generate synthetic temporal pairs and non-temporal control pairs.

Usage:
    python scripts/generate_synthetic_pairs.py --config configs/default.yaml
    python scripts/generate_synthetic_pairs.py --config configs/debug.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml

from temporal_vectors.config import DATA_PROC, SEED, WIKI_TARGET_YEARS
from temporal_vectors.data.pair_builder import build_pairs, save_pairs, validate_dataset
from temporal_vectors.data.synthetic_pairs import generate_control_pairs, generate_synthetic_pairs
from temporal_vectors.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic and control pairs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--synthetic-output",
        type=Path,
        default=None,
        help="Output path for synthetic pairs",
    )
    parser.add_argument(
        "--control-output",
        type=Path,
        default=None,
        help="Output path for control pairs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {}
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    seed = config.get("seed", SEED)
    seed_everything(seed)

    years = config.get("wiki_target_years", WIKI_TARGET_YEARS)
    syn_output = args.synthetic_output or DATA_PROC / "synthetic_pairs.jsonl"
    ctl_output = args.control_output or DATA_PROC / "control_pairs.jsonl"

    # ── Synthetic pairs ──────────────────────────────────────────
    logger.info("Generating synthetic pairs...")
    raw_synthetic = generate_synthetic_pairs(years=years, seed=seed)
    synthetic = build_pairs(raw_synthetic, pair_type="synthetic")

    syn_report = validate_dataset(synthetic)
    logger.info(
        "Synthetic: %d total, %d valid, %d invalid",
        syn_report["total"], syn_report["valid"], syn_report["invalid"],
    )
    logger.info("Synthetic domains: %s", syn_report["domain_percentages"])

    save_pairs(synthetic, syn_output)

    min_syn = config.get("min_synthetic_pairs", 300)
    if len(synthetic) < min_syn:
        logger.warning("Below target: %d synthetic pairs (need %d)", len(synthetic), min_syn)

    # ── Control pairs ────────────────────────────────────────────
    logger.info("Generating control pairs...")
    raw_control = generate_control_pairs(seed=seed)
    control = build_pairs(raw_control, pair_type="control")

    ctl_report = validate_dataset(control)
    logger.info(
        "Control: %d total, %d valid, %d invalid",
        ctl_report["total"], ctl_report["valid"], ctl_report["invalid"],
    )

    save_pairs(control, ctl_output)

    min_ctl = config.get("min_control_pairs", 200)
    if len(control) < min_ctl:
        logger.warning("Below target: %d control pairs (need %d)", len(control), min_ctl)

    # ── Summary ──────────────────────────────────────────────────
    logger.info("Done. Synthetic: %d pairs, Control: %d pairs", len(synthetic), len(control))


if __name__ == "__main__":
    main()
