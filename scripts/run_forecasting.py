#!/usr/bin/env python
"""Run temporal forecasting experiments.

Usage:
    python scripts/run_forecasting.py --layers 7 14 21 27
    python scripts/run_forecasting.py --layers 14 --pair-type synthetic
"""

import argparse
import logging
from pathlib import Path

import torch

from temporal_vectors.config import (
    HIDDEN_DIR, VECTOR_DIR, DATA_PROC, TARGET_LAYERS,
    FORECAST_ALPHA_RANGE, SEED, RESULTS_DIR,
)
from temporal_vectors.utils.reproducibility import seed_everything
from temporal_vectors.utils.io import read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal forecasting experiments")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=TARGET_LAYERS,
    )
    parser.add_argument(
        "--pair-types", type=str, nargs="+",
        default=["natural", "synthetic", "control"],
    )
    parser.add_argument(
        "--direction-from", type=str, default="synthetic",
        help="Which pair type's temporal direction to use (default: synthetic)",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=FORECAST_ALPHA_RANGE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    from temporal_vectors.analysis.forecasting import run_forecasting

    all_results = {}

    for pair_type in args.pair_types:
        logger.info("Forecasting: %s pairs", pair_type)
        pairs = read_jsonl(DATA_PROC / f"{pair_type}_pairs.jsonl")
        pair_results = {}

        for layer in args.layers:
            direction = torch.load(
                VECTOR_DIR / args.direction_from / f"temporal_direction_layer_{layer}.pt",
                map_location="cpu", weights_only=False,
            )
            result = run_forecasting(
                hidden_dir=HIDDEN_DIR / pair_type,
                direction=direction,
                alphas=args.alphas,
                pairs=pairs,
                layer=layer,
            )
            pair_results[layer] = result

        all_results[pair_type] = pair_results

    #save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, RESULTS_DIR / "forecasting_results.pt")

    #print summary
    for pair_type in args.pair_types:
        print(f"\n{'=' * 85}")
        print(f"  {pair_type.upper()} (direction from {args.direction_from})")
        print(f"{'=' * 85}")
        print(f"{'Layer':>6} {'Identity':>10} {'Random':>10}", end="")
        for alpha in args.alphas:
            print(f" {'a='+str(alpha):>10}", end="")
        print(f" {'Best a':>8}")
        print("-" * 85)

        for layer in sorted(args.layers):
            r = all_results[pair_type][layer]
            print(f"{layer:>6d} {r['identity']['cosine_mean']:>10.4f} {r['random']['cosine_mean']:>10.4f}", end="")
            for alpha in args.alphas:
                print(f" {r['temporal'][alpha]['cosine_mean']:>10.4f}", end="")
            print(f" {r['best_alpha']:>8.1f}")

        #domain breakdown at best layer
        best_layer = max(
            args.layers,
            key=lambda l: all_results[pair_type][l]["best_temporal"]["cosine_mean"],
        )
        domains = all_results[pair_type][best_layer].get("domain_cosine", {})
        if domains:
            print(f"\n  Domain breakdown (layer {best_layer}, a={all_results[pair_type][best_layer]['best_alpha']}):")
            for d, v in sorted(domains.items(), key=lambda x: -x[1]):
                print(f"    {d:>12s}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
