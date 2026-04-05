#!/usr/bin/env python
"""Compute temporal vectors and run linearity tests.

Usage:
    python scripts/compute_temporal_vectors.py --layers 7 14 21 27
    python scripts/compute_temporal_vectors.py --layers 14 --pair-type natural --method pca
"""

import argparse
import logging
from pathlib import Path

from temporal_vectors.config import HIDDEN_DIR, VECTOR_DIR, DATA_PROC, TARGET_LAYERS, SEED
from temporal_vectors.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute temporal vectors")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=TARGET_LAYERS,
        help="Layer indices (default: 7 14 21 27)",
    )
    parser.add_argument(
        "--pair-type", type=str, default="natural",
        choices=["natural", "synthetic", "control"],
        help="Which pair dataset to use (default: natural)",
    )
    parser.add_argument(
        "--method", type=str, default="mean", choices=["mean", "pca"],
        help="Direction estimation method (default: mean)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: outputs/vectors/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    hidden_dir = HIDDEN_DIR / args.pair_type
    pairs_path = DATA_PROC / f"{args.pair_type}_pairs.jsonl"
    output_dir = args.output_dir or VECTOR_DIR / args.pair_type

    logger.info("Hidden states: %s", hidden_dir)
    logger.info("Pairs: %s", pairs_path)
    logger.info("Method: %s", args.method)
    logger.info("Output: %s", output_dir)

    from temporal_vectors.analysis.temporal_vectors import extract_and_save

    all_results = extract_and_save(
        hidden_dir=hidden_dir,
        pairs_path=pairs_path,
        output_dir=output_dir,
        layers=args.layers,
        method=args.method,
    )

    #print summary table
    print("\n" + "=" * 80)
    print(f"{'Layer':>6} {'Parallelism':>13} {'Additivity':>12} {'Scaling':>9} {'LOO':>7}")
    print("-" * 80)
    for layer, res in sorted(all_results.items()):
        print(
            f"{layer:>6d} {res['parallelism_mean']:>10.3f} ± {res['parallelism_std']:.3f}"
            f" {res['additivity_residual']:>10.3f} {res['scaling_correlation']:>9.3f}"
            f" {res['loo_stability']:>7.3f}"
        )
    print("=" * 80)

    #print cross-domain breakdown for each layer
    for layer, res in sorted(all_results.items()):
        domains = res["cross_domain"]
        if domains:
            print(f"\nLayer {layer} cross-domain parallelism:")
            for domain, score in sorted(domains.items()):
                print(f"  {domain:>12s}: {score:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
