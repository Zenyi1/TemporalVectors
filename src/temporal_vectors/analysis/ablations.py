"""Phase 6: statistical tests, ablations, and variance decomposition."""

import logging
from pathlib import Path

import torch
import numpy as np

from temporal_vectors.config import HIDDEN_DIR, VECTOR_DIR, TARGET_LAYERS
from temporal_vectors.analysis.temporal_vectors import (
    load_hidden_states, compute_deltas, compute_temporal_direction,
    parallelism_score, additivity_residual,
)
from temporal_vectors.utils.stats import bootstrap_ci, permutation_test
from temporal_vectors.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def linearity_with_cis(
    hidden_dir: Path,
    layer: int,
    pairs: list[dict],
    n_bootstrap: int = 10_000,
) -> dict:
    """Parallelism and additivity with bootstrap CIs."""
    deltas = compute_deltas(hidden_dir, layer)
    direction = compute_temporal_direction(deltas, method="mean")
    cos_sims = parallelism_score(deltas, direction).numpy()

    #parallelism CI
    par_point, par_lo, par_hi = bootstrap_ci(cos_sims, n_bootstrap=n_bootstrap)

    #additivity residual CI (bootstrap over pairs)
    rng = np.random.RandomState(42)
    n = len(cos_sims)
    add_boots = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sub_deltas = deltas[idx]
        sub_dir = compute_temporal_direction(sub_deltas, method="mean")
        add_boots.append(additivity_residual(sub_deltas, sub_dir))
    add_point = additivity_residual(deltas, direction)
    add_lo = float(np.percentile(add_boots, 2.5))
    add_hi = float(np.percentile(add_boots, 97.5))

    return {
        "layer": layer,
        "n_pairs": n,
        "parallelism": {"point": par_point, "ci_lower": par_lo, "ci_upper": par_hi},
        "additivity_residual": {"point": add_point, "ci_lower": add_lo, "ci_upper": add_hi},
    }


def natural_by_change_type(
    hidden_dir: Path,
    layer: int,
    pairs: list[dict],
    n_bootstrap: int = 10_000,
) -> dict:
    """Break down natural pair parallelism by change type."""
    deltas = compute_deltas(hidden_dir, layer)
    direction = compute_temporal_direction(deltas, method="mean")
    cos_sims = parallelism_score(deltas, direction).numpy()

    change_types = [p.get("metadata", {}).get("change_type", "unknown") for p in pairs]
    unique = sorted(set(change_types))
    results = {}
    for ct in unique:
        mask = np.array([c == ct for c in change_types])
        if mask.sum() < 3:
            continue
        vals = cos_sims[mask]
        pt, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
        results[ct] = {"point": pt, "ci_lower": lo, "ci_upper": hi, "n": int(mask.sum())}
    return results


def dataset_size_ablation(
    hidden_dir: Path,
    layer: int,
    pairs: list[dict],
    fractions: list[float] | None = None,
    n_repeats: int = 10,
) -> dict:
    """Ablation: how does direction quality change with dataset size?

    Subsamples pairs at different fractions, recomputes direction,
    measures parallelism against the full-data direction.
    """
    fractions = fractions or [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    deltas = compute_deltas(hidden_dir, layer)
    full_direction = compute_temporal_direction(deltas, method="mean")
    n = deltas.size(0)

    results = {}
    for frac in fractions:
        k = max(2, int(n * frac))
        cosines = []
        parallelisms = []
        for seed in range(n_repeats):
            rng = torch.Generator().manual_seed(seed)
            idx = torch.randperm(n, generator=rng)[:k]
            sub_dir = compute_temporal_direction(deltas[idx], method="mean")
            #cosine with full direction (stability)
            cosines.append(float(full_direction @ sub_dir))
            #mean parallelism of subsample direction on full data
            par = float(parallelism_score(deltas, sub_dir).mean())
            parallelisms.append(par)

        results[frac] = {
            "n_pairs": k,
            "direction_cosine_mean": float(np.mean(cosines)),
            "direction_cosine_std": float(np.std(cosines)),
            "parallelism_mean": float(np.mean(parallelisms)),
            "parallelism_std": float(np.std(parallelisms)),
        }
    return results


def direction_method_comparison(
    hidden_dir: Path,
    layer: int,
    pairs: list[dict],
    n_bootstrap: int = 10_000,
) -> dict:
    """Compare mean vs PCA direction methods."""
    deltas = compute_deltas(hidden_dir, layer)

    results = {}
    for method in ["mean", "pca"]:
        direction = compute_temporal_direction(deltas, method=method)
        cos_sims = parallelism_score(deltas, direction).numpy()
        pt, lo, hi = bootstrap_ci(cos_sims, n_bootstrap=n_bootstrap)
        add_res = additivity_residual(deltas, direction)
        results[method] = {
            "parallelism": {"point": pt, "ci_lower": lo, "ci_upper": hi},
            "additivity_residual": add_res,
        }

    #cosine between the two directions
    mean_dir = compute_temporal_direction(deltas, method="mean")
    pca_dir = compute_temporal_direction(deltas, method="pca")
    results["mean_pca_cosine"] = float(mean_dir @ pca_dir)

    return results


def variance_decomposition(
    hidden_dir: Path,
    layer: int,
    pairs: list[dict],
) -> dict:
    """Decompose delta variance into temporal, domain, and residual components.

    temporal = variance explained by temporal direction
    domain = additional variance explained by per-domain means (after removing temporal)
    residual = remaining variance
    """
    deltas = compute_deltas(hidden_dir, layer)
    direction = compute_temporal_direction(deltas, method="mean")
    n = deltas.size(0)

    total_var = deltas.norm(dim=1).pow(2).mean().item()

    #temporal component
    projections = (deltas @ direction).unsqueeze(1) * direction.unsqueeze(0)
    temporal_var = projections.norm(dim=1).pow(2).mean().item()

    #domain component (per-domain mean of residuals)
    residuals_after_temporal = deltas - projections
    domains = [p.get("domain", "unknown") for p in pairs]
    unique = sorted(set(domains))
    domain_projections = torch.zeros_like(deltas)
    for d in unique:
        mask = torch.tensor([dom == d for dom in domains])
        if mask.sum() > 0:
            domain_mean = residuals_after_temporal[mask].mean(dim=0)
            domain_projections[mask] = domain_mean.unsqueeze(0)
    domain_var = domain_projections.norm(dim=1).pow(2).mean().item()

    residual_var = total_var - temporal_var - domain_var

    return {
        "layer": layer,
        "total_variance": total_var,
        "temporal_fraction": temporal_var / total_var,
        "domain_fraction": domain_var / total_var,
        "residual_fraction": residual_var / total_var,
        "temporal_variance": temporal_var,
        "domain_variance": domain_var,
        "residual_variance": residual_var,
        "n_domains": len(unique),
        "domains": unique,
    }
