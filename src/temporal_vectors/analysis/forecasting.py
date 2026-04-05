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
        "Layer %d: identity=%.4f, random=%.4f, best_temporal(α=%.1f)=%.4f",
        layer,
        results["identity"]["cosine_mean"],
        results["random"]["cosine_mean"],
        best_alpha,
        results["best_temporal"]["cosine_mean"],
    )
    return results
