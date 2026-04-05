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

    from temporal_vectors.analysis.forecasting import run_forecasting, run_rigorous_forecasting

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
            #original h_new cosine evaluation
            result = run_forecasting(
                hidden_dir=HIDDEN_DIR / pair_type,
                direction=direction,
                alphas=args.alphas,
                pairs=pairs,
                layer=layer,
            )
            #delta-focused with bootstrap CIs and permutation tests
            rigorous = run_rigorous_forecasting(
                hidden_dir=HIDDEN_DIR / pair_type,
                direction=direction,
                pairs=pairs,
                layer=layer,
            )
            result["rigorous"] = rigorous
            pair_results[layer] = result

        all_results[pair_type] = pair_results

    #save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, RESULTS_DIR / "forecasting_results.pt")

    #print rigorous summary
    for pair_type in args.pair_types:
        print(f"\n{'=' * 90}")
        print(f"  {pair_type.upper()} -- Delta-focused evaluation (a=0.5, direction from {args.direction_from})")
        print(f"{'=' * 90}")
        print(f"{'Layer':>6} {'Delta cos':>10} {'95% CI':>18} {'R^2':>8} {'Random':>8} {'p-value':>10} {'Sig?':>6}")
        print("-" * 90)

        for layer in sorted(args.layers):
            rig = all_results[pair_type][layer]["rigorous"]
            ci = rig["delta_cosine_ci"]
            pt = rig["permutation_test"]
            de = rig["delta_eval"]
            sig = "*" if pt["significant"] else ""
            print(
                f"{layer:>6d} {ci['point']:>10.4f}"
                f" [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                f" {de['r_squared_mean']:>8.4f}"
                f" {rig['random_baseline_mean']:>8.4f}"
                f" {pt['p_value']:>10.4f} {sig:>6}"
            )

        #domain breakdown
        best_layer = sorted(args.layers)[-1]
        domains = all_results[pair_type][best_layer]["rigorous"].get("domain_delta", {})
        if domains:
            print(f"\n  Domain breakdown (layer {best_layer}):")
            for d, v in sorted(domains.items(), key=lambda x: -x[1]["point"]):
                print(f"    {d:>12s}: {v['point']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}] (n={v['n']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
