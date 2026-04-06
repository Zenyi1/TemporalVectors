#!/usr/bin/env python
"""Phase 6: statistical tests, ablations, and variance decomposition.

Usage:
    python scripts/run_analysis.py --full
    python scripts/run_analysis.py --layers 14 --pair-type synthetic
"""

import argparse
import logging

import torch

from temporal_vectors.config import (
    HIDDEN_DIR, VECTOR_DIR, DATA_PROC, TARGET_LAYERS, RESULTS_DIR, SEED,
)
from temporal_vectors.utils.reproducibility import seed_everything
from temporal_vectors.utils.io import read_jsonl
from temporal_vectors.analysis.ablations import (
    linearity_with_cis,
    natural_by_change_type,
    dataset_size_ablation,
    direction_method_comparison,
    variance_decomposition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6 analysis and ablations")
    parser.add_argument("--layers", type=int, nargs="+", default=TARGET_LAYERS)
    parser.add_argument("--full", action="store_true", help="Run all analyses")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    all_results = {}

    #load all pair datasets
    pair_types = ["synthetic", "natural", "control"]
    pairs_data = {}
    for pt in pair_types:
        path = DATA_PROC / f"{pt}_pairs.jsonl"
        if path.exists():
            pairs_data[pt] = read_jsonl(path)

    # --- 1. linearity metrics with CIs ---
    print("\n" + "=" * 90)
    print("  LINEARITY METRICS WITH BOOTSTRAP CIs")
    print("=" * 90)

    for pt in pair_types:
        if pt not in pairs_data:
            continue
        print(f"\n  {pt.upper()}")
        print(f"  {'Layer':>6} {'Parallelism':>12} {'95% CI':>22} {'Additivity res':>15} {'95% CI':>22}")
        print("  " + "-" * 80)

        ci_results = {}
        for layer in args.layers:
            res = linearity_with_cis(HIDDEN_DIR / pt, layer, pairs_data[pt])
            ci_results[layer] = res
            p = res["parallelism"]
            a = res["additivity_residual"]
            print(
                f"  {layer:>6d} {p['point']:>12.4f}"
                f" [{p['ci_lower']:.4f}, {p['ci_upper']:.4f}]"
                f" {a['point']:>15.4f}"
                f" [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]"
            )
        all_results[f"{pt}_linearity_cis"] = ci_results

    # --- 2. natural pair breakdown by change type ---
    if "natural" in pairs_data:
        print("\n" + "=" * 90)
        print("  NATURAL PAIRS BY CHANGE TYPE")
        print("=" * 90)

        for layer in args.layers:
            ct_results = natural_by_change_type(
                HIDDEN_DIR / "natural", layer, pairs_data["natural"]
            )
            print(f"\n  Layer {layer}:")
            print(f"  {'Type':>15} {'Parallelism':>12} {'95% CI':>22} {'n':>6}")
            print("  " + "-" * 60)
            for ct, v in sorted(ct_results.items(), key=lambda x: -x[1]["point"]):
                print(
                    f"  {ct:>15} {v['point']:>12.4f}"
                    f" [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
                    f" {v['n']:>6d}"
                )
            all_results[f"natural_change_type_layer_{layer}"] = ct_results

    # --- 3. dataset size ablation ---
    if "synthetic" in pairs_data:
        print("\n" + "=" * 90)
        print("  DATASET SIZE ABLATION (synthetic)")
        print("=" * 90)

        for layer in args.layers:
            abl = dataset_size_ablation(
                HIDDEN_DIR / "synthetic", layer, pairs_data["synthetic"]
            )
            print(f"\n  Layer {layer}:")
            print(f"  {'Fraction':>10} {'n_pairs':>8} {'Dir cosine':>12} {'Parallelism':>12}")
            print("  " + "-" * 50)
            for frac in sorted(abl.keys()):
                v = abl[frac]
                print(
                    f"  {frac:>10.1f} {v['n_pairs']:>8d}"
                    f" {v['direction_cosine_mean']:>12.4f}"
                    f" {v['parallelism_mean']:>12.4f}"
                )
            all_results[f"size_ablation_layer_{layer}"] = abl

    # --- 4. direction method comparison ---
    if "synthetic" in pairs_data:
        print("\n" + "=" * 90)
        print("  DIRECTION METHOD: MEAN vs PCA")
        print("=" * 90)

        for layer in args.layers:
            comp = direction_method_comparison(
                HIDDEN_DIR / "synthetic", layer, pairs_data["synthetic"]
            )
            print(f"\n  Layer {layer}: mean-PCA cosine = {comp['mean_pca_cosine']:.4f}")
            for method in ["mean", "pca"]:
                p = comp[method]["parallelism"]
                a = comp[method]["additivity_residual"]
                print(
                    f"    {method:>5}: parallelism={p['point']:.4f}"
                    f" [{p['ci_lower']:.4f}, {p['ci_upper']:.4f}],"
                    f" additivity_res={a:.4f}"
                )
            all_results[f"method_comparison_layer_{layer}"] = comp

    # --- 5. variance decomposition ---
    print("\n" + "=" * 90)
    print("  VARIANCE DECOMPOSITION")
    print("=" * 90)

    for pt in pair_types:
        if pt not in pairs_data:
            continue
        print(f"\n  {pt.upper()}")
        print(f"  {'Layer':>6} {'Temporal':>10} {'Domain':>10} {'Residual':>10}")
        print("  " + "-" * 40)

        for layer in args.layers:
            vd = variance_decomposition(
                HIDDEN_DIR / pt, layer, pairs_data[pt]
            )
            print(
                f"  {layer:>6d}"
                f" {vd['temporal_fraction']:>10.4f}"
                f" {vd['domain_fraction']:>10.4f}"
                f" {vd['residual_fraction']:>10.4f}"
            )
            all_results[f"{pt}_variance_layer_{layer}"] = vd

    #save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, RESULTS_DIR / "phase6_analysis_results.pt")
    print(f"\nSaved to {RESULTS_DIR / 'phase6_analysis_results.pt'}")
    print("Done.")


if __name__ == "__main__":
    main()
