#!/usr/bin/env python
"""Compute Causal Inner Product analysis.

Usage:
    python scripts/compute_cip.py --layers 7 14 21 27
"""

import argparse
import logging
from pathlib import Path

import torch

from temporal_vectors.config import (
    HIDDEN_DIR, VECTOR_DIR, DATA_PROC, TARGET_LAYERS, SEED, RESULTS_DIR,
)
from temporal_vectors.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIP analysis")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=TARGET_LAYERS,
    )
    parser.add_argument(
        "--reg", type=float, default=1e-4,
        help="Regularisation for Sigma_V inverse (default: 1e-4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    from temporal_vectors.models.loader import load_model_and_tokenizer
    from temporal_vectors.analysis.cip import (
        CausalInnerProduct, extract_unembedding,
        run_cip_analysis, run_orthogonality_test,
    )

    logger.info("Loading model for unembedding extraction...")
    model, _ = load_model_and_tokenizer()
    unembedding = extract_unembedding(model)
    logger.info("Unembedding shape: %s", unembedding.shape)

    #free model memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    cip = CausalInnerProduct(unembedding, reg=args.reg)

    #run CIP analysis for each pair type using synthetic temporal direction
    temporal_dir_path = VECTOR_DIR / "synthetic"
    control_dir_path = VECTOR_DIR / "control"

    all_results = {}
    for pair_type in ["natural", "synthetic", "control"]:
        logger.info("CIP analysis: %s pairs", pair_type)
        results = run_cip_analysis(
            cip_obj=cip,
            hidden_dir=HIDDEN_DIR / pair_type,
            pairs_path=DATA_PROC / f"{pair_type}_pairs.jsonl",
            temporal_dir_path=temporal_dir_path,
            layers=args.layers,
        )
        all_results[pair_type] = results

    #orthogonality test: temporal vs control directions
    logger.info("Orthogonality test: temporal vs control")
    ortho = run_orthogonality_test(cip, temporal_dir_path, control_dir_path, args.layers)
    all_results["orthogonality"] = ortho

    #save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, RESULTS_DIR / "cip_results.pt")

    #print summary
    print("\n" + "=" * 90)
    print(f"{'':>6} {'Synthetic':^25} {'Natural':^25} {'Control':^25}")
    print(f"{'Layer':>6} {'EUC':>10} {'CIP':>10} {'EUC':>10} {'CIP':>10} {'EUC':>10} {'CIP':>10}")
    print("-" * 90)
    for layer in sorted(args.layers):
        syn = all_results["synthetic"][layer]
        nat = all_results["natural"][layer]
        ctrl = all_results["control"][layer]
        print(
            f"{layer:>6d}"
            f" {syn['euc_parallelism_mean']:>10.3f} {syn['cip_parallelism_mean']:>10.3f}"
            f" {nat['euc_parallelism_mean']:>10.3f} {nat['cip_parallelism_mean']:>10.3f}"
            f" {ctrl['euc_parallelism_mean']:>10.3f} {ctrl['cip_parallelism_mean']:>10.3f}"
        )
    print("=" * 90)

    print("\nOrthogonality (temporal vs control direction):")
    print(f"{'Layer':>6} {'EUC cos':>10} {'CIP cos':>10} {'Pass (<0.2)':>12}")
    print("-" * 42)
    for layer in sorted(args.layers):
        o = ortho[layer]
        status = "yes" if o["orthogonal"] else "NO"
        print(f"{layer:>6d} {o['euc_cosine']:>10.3f} {o['cip_cosine']:>10.3f} {status:>12}")

    #domain breakdown for synthetic
    print("\nSynthetic CIP parallelism by domain:")
    for layer in sorted(args.layers):
        domains = all_results["synthetic"][layer]["domain_cip"]
        print(f"  Layer {layer}:", end="")
        for d, v in sorted(domains.items()):
            print(f"  {d}={v:.3f}", end="")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
