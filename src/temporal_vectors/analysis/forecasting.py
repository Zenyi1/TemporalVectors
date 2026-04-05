"""Temporal forecasting via vector arithmetic."""

import logging
from pathlib import Path

import torch
import numpy as np

from temporal_vectors.config import (
    FORECAST_ALPHA_RANGE, HIDDEN_DIR, VECTOR_DIR, DATA_PROC, TARGET_LAYERS,
)
from temporal_vectors.analysis.temporal_vectors import load_hidden_states
from temporal_vectors.utils.io import read_jsonl
from temporal_vectors.utils.stats import bootstrap_ci, permutation_test

logger = logging.getLogger(__name__)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity between two (N, D) tensors."""
    a_norm = a / a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_norm = b / b.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (a_norm * b_norm).sum(dim=1)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise MSE between two (N, D) tensors."""
    return (a - b).pow(2).mean(dim=1)


def forecast_temporal(
    h_old: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """h_forecast = h_old + alpha * direction."""
    return h_old + alpha * direction.unsqueeze(0)


def baseline_identity(h_old: torch.Tensor) -> torch.Tensor:
    """Baseline: predict h_new = h_old (no change)."""
    return h_old.clone()


def baseline_random(h_old: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Baseline: h_old + random unit direction."""
    rng = torch.Generator().manual_seed(seed)
    rand_dir = torch.randn(h_old.size(1), generator=rng)
    rand_dir = rand_dir / rand_dir.norm()
    #scale by mean delta norm for fair comparison
    return h_old + rand_dir.unsqueeze(0)


def evaluate_forecast(
    h_forecast: torch.Tensor,
    h_new: torch.Tensor,
) -> dict[str, float]:
    """Evaluate forecast quality."""
    cos = cosine_similarity(h_forecast, h_new)
    err = mse(h_forecast, h_new)
    #relative improvement over identity (h_old)
    return {
        "cosine_mean": float(cos.mean()),
        "cosine_std": float(cos.std()),
        "mse_mean": float(err.mean()),
        "mse_std": float(err.std()),
    }


def run_forecasting(
    hidden_dir: Path,
    direction: torch.Tensor,
    alphas: list[float] | None = None,
    pairs: list[dict] | None = None,
    layer: int = 14,
) -> dict:
    """Run forecasting experiments for a single layer.

    Tests temporal vector arithmetic against baselines.
    """
    alphas = alphas or FORECAST_ALPHA_RANGE
    states = load_hidden_states(hidden_dir, layer)
    h_old = states[0::2]
    h_new = states[1::2]

    #compute mean delta norm for scaling random baseline
    delta_norm = (h_new - h_old).norm(dim=1).mean().item()

    results = {"layer": layer, "n_pairs": h_old.size(0)}

    #baselines
    results["identity"] = evaluate_forecast(baseline_identity(h_old), h_new)
    #random baseline scaled to match mean delta magnitude
    rng = torch.Generator().manual_seed(42)
    rand_dir = torch.randn(h_old.size(1), generator=rng)
    rand_dir = rand_dir / rand_dir.norm() * delta_norm
    h_random = h_old + rand_dir.unsqueeze(0)
    results["random"] = evaluate_forecast(h_random, h_new)

    #temporal vector at different alphas
    results["temporal"] = {}
    for alpha in alphas:
        scaled_dir = direction * delta_norm * alpha
        h_forecast = forecast_temporal(h_old, scaled_dir, alpha=1.0)
        metrics = evaluate_forecast(h_forecast, h_new)
        metrics["alpha"] = alpha
        results["temporal"][alpha] = metrics

    #find best alpha
    best_alpha = max(alphas, key=lambda a: results["temporal"][a]["cosine_mean"])
    results["best_alpha"] = best_alpha
    results["best_temporal"] = results["temporal"][best_alpha]

    #per-domain breakdown at best alpha if pairs provided
    if pairs:
        scaled_dir = direction * delta_norm * best_alpha
        h_forecast = forecast_temporal(h_old, scaled_dir, alpha=1.0)
        cos = cosine_similarity(h_forecast, h_new)
        domains = [p.get("domain", "unknown") for p in pairs]
        unique = sorted(set(domains))
        results["domain_cosine"] = {}
        for d in unique:
            mask = torch.tensor([dom == d for dom in domains])
            results["domain_cosine"][d] = float(cos[mask].mean())

    logger.info(
        "Layer %d: identity=%.4f, random=%.4f, best_temporal(a=%.1f)=%.4f",
        layer,
        results["identity"]["cosine_mean"],
        results["random"]["cosine_mean"],
        best_alpha,
        results["best_temporal"]["cosine_mean"],
    )
    return results


def evaluate_delta(
    direction: torch.Tensor,
    h_old: torch.Tensor,
    h_new: torch.Tensor,
    alpha: float,
    delta_norm: float,
) -> dict:
    """Evaluate how well predicted delta matches actual delta.

    Predicted delta: alpha * gamma (scaled)
    Actual delta: h_new - h_old
    """
    actual_delta = h_new - h_old
    predicted_delta = (direction * delta_norm * alpha).unsqueeze(0).expand_as(actual_delta)

    #cosine between predicted and actual delta
    delta_cos = cosine_similarity(predicted_delta, actual_delta)
    #residual: what fraction of actual delta is NOT explained
    projections = (actual_delta @ direction).unsqueeze(1) * direction.unsqueeze(0)
    residuals = actual_delta - projections
    r_squared = 1.0 - (residuals.norm(dim=1).pow(2) / actual_delta.norm(dim=1).pow(2).clamp(min=1e-8))
    #L2 distance between predicted and actual delta
    delta_l2 = (predicted_delta - actual_delta).norm(dim=1)
    actual_l2 = actual_delta.norm(dim=1)
    relative_error = delta_l2 / actual_l2.clamp(min=1e-8)

    return {
        "delta_cosine_mean": float(delta_cos.mean()),
        "delta_cosine_std": float(delta_cos.std()),
        "r_squared_mean": float(r_squared.mean()),
        "r_squared_std": float(r_squared.std()),
        "relative_error_mean": float(relative_error.mean()),
        "relative_error_std": float(relative_error.std()),
        "delta_cosines": delta_cos,  #keep raw for stats
    }


def run_rigorous_forecasting(
    hidden_dir: Path,
    direction: torch.Tensor,
    pairs: list[dict] | None = None,
    layer: int = 14,
    n_bootstrap: int = 10000,
    n_random_baselines: int = 100,
) -> dict:
    """Forecasting with delta-focused metrics, CIs, and significance tests."""
    states = load_hidden_states(hidden_dir, layer)
    h_old = states[0::2]
    h_new = states[1::2]
    actual_delta = h_new - h_old
    delta_norm = actual_delta.norm(dim=1).mean().item()

    results = {"layer": layer, "n_pairs": h_old.size(0)}

    #delta-focused evaluation at best alpha (0.5 from prior results)
    alpha = 0.5
    delta_eval = evaluate_delta(direction, h_old, h_new, alpha, delta_norm)
    results["delta_eval"] = {k: v for k, v in delta_eval.items() if k != "delta_cosines"}

    #bootstrap CI on delta cosine
    cos_vals = delta_eval["delta_cosines"].numpy()
    point, ci_lo, ci_hi = bootstrap_ci(cos_vals, n_bootstrap=n_bootstrap)
    results["delta_cosine_ci"] = {"point": point, "ci_lower": ci_lo, "ci_upper": ci_hi}

    #random baseline distribution: repeat with N random directions
    random_cosines = []
    for seed in range(n_random_baselines):
        rng = torch.Generator().manual_seed(seed)
        rand_dir = torch.randn(h_old.size(1), generator=rng)
        rand_dir = rand_dir / rand_dir.norm()
        rand_delta = (rand_dir * delta_norm * alpha).unsqueeze(0).expand_as(actual_delta)
        rand_cos = cosine_similarity(rand_delta, actual_delta)
        random_cosines.append(float(rand_cos.mean()))
    results["random_baseline_mean"] = float(np.mean(random_cosines))
    results["random_baseline_std"] = float(np.std(random_cosines))

    #permutation test: temporal direction cosines vs random direction cosines
    #use cosines from best random seed for fair comparison
    best_random_seed = int(np.argmax(random_cosines))
    rng = torch.Generator().manual_seed(best_random_seed)
    rand_dir = torch.randn(h_old.size(1), generator=rng)
    rand_dir = rand_dir / rand_dir.norm()
    rand_delta = (rand_dir * delta_norm * alpha).unsqueeze(0).expand_as(actual_delta)
    rand_cos_per_pair = cosine_similarity(rand_delta, actual_delta).numpy()

    diff, p_value = permutation_test(cos_vals, rand_cos_per_pair, n_permutations=10000)
    results["permutation_test"] = {
        "observed_diff": diff,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

    #per-domain with CIs if pairs provided
    if pairs:
        domains = [p.get("domain", "unknown") for p in pairs]
        unique = sorted(set(domains))
        results["domain_delta"] = {}
        for d in unique:
            mask = np.array([dom == d for dom in domains])
            domain_cos = cos_vals[mask]
            pt, lo, hi = bootstrap_ci(domain_cos, n_bootstrap=n_bootstrap)
            results["domain_delta"][d] = {"point": pt, "ci_lower": lo, "ci_upper": hi, "n": int(mask.sum())}

    logger.info(
        "Layer %d: delta_cos=%.3f [%.3f, %.3f], random=%.3f, p=%.4f",
        layer,
        results["delta_cosine_ci"]["point"],
        results["delta_cosine_ci"]["ci_lower"],
        results["delta_cosine_ci"]["ci_upper"],
        results["random_baseline_mean"],
        results["permutation_test"]["p_value"],
    )
    return results
