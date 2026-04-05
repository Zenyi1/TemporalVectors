"""Temporal vector extraction and linearity tests."""

import logging
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA

from temporal_vectors.config import HIDDEN_SIZE, TARGET_LAYERS, VECTOR_DIR
from temporal_vectors.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def load_hidden_states(hidden_dir: Path, layer: int) -> torch.Tensor:
    """Load hidden state tensor for a given layer."""
    path = hidden_dir / f"layer_{layer}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["tensor"]


def compute_deltas(hidden_dir: Path, layer: int) -> torch.Tensor:
    """Compute delta vectors (h_new - h_old) for all pairs at a layer.

    Assumes hidden states are stored in alternating old/new order.

    Returns:
        Tensor of shape (n_pairs, hidden_size).
    """
    states = load_hidden_states(hidden_dir, layer)
    h_old = states[0::2]  #even indices
    h_new = states[1::2]  #odd indices
    return h_new - h_old


def compute_temporal_direction(deltas: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Estimate a single temporal direction from delta vectors.

    Args:
        deltas: (n_pairs, hidden_size) delta vectors.
        method: "mean" for average direction, "pca" for first principal component.

    Returns:
        Unit vector of shape (hidden_size,).
    """
    if method == "mean":
        direction = deltas.mean(dim=0)
    elif method == "pca":
        pca = PCA(n_components=1)
        pca.fit(deltas.numpy())
        direction = torch.from_numpy(pca.components_[0]).float()
    else:
        raise ValueError(f"Unknown method: {method}")
    return direction / direction.norm()


def parallelism_score(deltas: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of each delta with the temporal direction.

    Returns:
        Tensor of shape (n_pairs,) with cosine similarities.
    """
    deltas_norm = deltas / deltas.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return deltas_norm @ direction


def additivity_residual(deltas: torch.Tensor, direction: torch.Tensor) -> float:
    """Fraction of variance not explained by the temporal direction.

    Lower is better. Returns value in [0, 1].
    """
    projections = (deltas @ direction).unsqueeze(1) * direction.unsqueeze(0)
    residuals = deltas - projections
    residual_var = residuals.norm(dim=1).pow(2).mean()
    total_var = deltas.norm(dim=1).pow(2).mean()
    return (residual_var / total_var).item()


def scaling_consistency(
    deltas: torch.Tensor, direction: torch.Tensor, pairs: list[dict]
) -> float:
    """Correlation between time gap and projection magnitude.

    Tests if larger time gaps produce proportionally larger projections.
    """
    projections = (deltas @ direction).numpy()
    time_gaps = np.array([p["year_new"] - p["year_old"] for p in pairs])
    if time_gaps.std() == 0:
        return 0.0
    correlation = np.corrcoef(time_gaps, np.abs(projections))[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def cross_domain_consistency(
    deltas: torch.Tensor, direction: torch.Tensor, pairs: list[dict]
) -> dict[str, float]:
    """Per-domain mean parallelism with the global temporal direction."""
    cos_sims = parallelism_score(deltas, direction).numpy()
    domains = [p.get("domain", "unknown") for p in pairs]
    unique = sorted(set(domains))
    result = {}
    for d in unique:
        mask = np.array([dom == d for dom in domains])
        result[d] = float(np.mean(cos_sims[mask]))
    return result


def leave_one_out_stability(deltas: torch.Tensor, n_folds: int = 50) -> float:
    """Stability of temporal direction under leave-one-out resampling.

    Computes direction on n-1 samples, measures cosine sim with full direction.
    Uses random subset of folds for efficiency.
    """
    full_direction = compute_temporal_direction(deltas, method="mean")
    n = deltas.size(0)
    indices = torch.randperm(n)[:n_folds]
    similarities = []
    for idx in indices:
        mask = torch.ones(n, dtype=torch.bool)
        mask[idx] = False
        loo_direction = compute_temporal_direction(deltas[mask], method="mean")
        sim = (full_direction @ loo_direction).item()
        similarities.append(sim)
    return float(np.mean(similarities))


def run_linearity_tests(
    hidden_dir: Path,
    layer: int,
    pairs: list[dict],
    method: str = "mean",
) -> dict:
    """Run all four linearity tests for a given layer.

    Returns dict with all metrics and the temporal direction.
    """
    deltas = compute_deltas(hidden_dir, layer)
    direction = compute_temporal_direction(deltas, method=method)
    cos_sims = parallelism_score(deltas, direction)

    results = {
        "layer": layer,
        "method": method,
        "n_pairs": deltas.size(0),
        "parallelism_mean": float(cos_sims.mean()),
        "parallelism_std": float(cos_sims.std()),
        "additivity_residual": additivity_residual(deltas, direction),
        "scaling_correlation": scaling_consistency(deltas, direction, pairs),
        "cross_domain": cross_domain_consistency(deltas, direction, pairs),
        "loo_stability": leave_one_out_stability(deltas),
    }

    logger.info(
        "Layer %d: parallelism=%.3f, additivity_res=%.3f, scaling=%.3f, loo=%.3f",
        layer,
        results["parallelism_mean"],
        results["additivity_residual"],
        results["scaling_correlation"],
        results["loo_stability"],
    )
    return results, direction


def extract_and_save(
    hidden_dir: Path,
    pairs_path: Path,
    output_dir: Path | None = None,
    layers: list[int] | None = None,
    method: str = "mean",
) -> dict[int, dict]:
    """Run full temporal vector extraction pipeline for all layers.

    Saves temporal direction vectors and test results.
    """
    layers = layers or TARGET_LAYERS
    output_dir = output_dir or VECTOR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(pairs_path)

    all_results = {}
    for layer in layers:
        results, direction = run_linearity_tests(hidden_dir, layer, pairs, method)
        all_results[layer] = results

        torch.save(direction, output_dir / f"temporal_direction_layer_{layer}.pt")

    #save summary
    torch.save(all_results, output_dir / "linearity_results.pt")
    return all_results
