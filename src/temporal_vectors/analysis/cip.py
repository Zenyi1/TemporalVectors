"""Causal Inner Product (CIP) following Park et al. (2024).

The CIP reweights hidden-state directions by how much they affect the
model's output distribution, using the unembedding matrix W_U.

Sigma_V = W_U^T @ W_U  (vocabulary covariance)
CIP(u, v) = u^T @ Sigma_V @ v
CIP_cos(u, v) = CIP(u, v) / (CIP_norm(u) * CIP_norm(v))
"""

import logging
from pathlib import Path

import torch
import numpy as np

from temporal_vectors.config import VECTOR_DIR, HIDDEN_DIR, DATA_PROC, TARGET_LAYERS
from temporal_vectors.utils.io import read_jsonl
from temporal_vectors.analysis.temporal_vectors import compute_deltas, load_hidden_states

logger = logging.getLogger(__name__)


class CausalInnerProduct:
    """Causal inner product using the unembedding matrix."""

    def __init__(self, unembedding: torch.Tensor, reg: float = 1e-4):
        """
        Args:
            unembedding: W_U of shape (vocab_size, hidden_size).
            reg: regularisation for numerical stability.
        """
        logger.info("Computing vocabulary covariance (Sigma_V)...")
        #sigma_v = W_U^T @ W_U, shape (hidden_size, hidden_size)
        self.sigma_v = (unembedding.T @ unembedding).float()
        #add regularisation
        self.sigma_v += reg * torch.eye(self.sigma_v.size(0))
        logger.info("Sigma_V shape: %s", self.sigma_v.shape)

    def cip(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """CIP(u, v) = u^T @ Sigma_V @ v."""
        return u @ self.sigma_v @ v

    def cip_norm(self, u: torch.Tensor) -> torch.Tensor:
        """CIP norm = sqrt(CIP(u, u))."""
        return torch.sqrt(self.cip(u, u).clamp(min=1e-12))

    def cip_cosine(self, u: torch.Tensor, v: torch.Tensor) -> float:
        """CIP cosine similarity."""
        return (self.cip(u, v) / (self.cip_norm(u) * self.cip_norm(v))).item()

    def cip_projection(self, u: torch.Tensor, v: torch.Tensor) -> float:
        """Scalar projection of u onto v in CIP space."""
        return (self.cip(u, v) / self.cip_norm(v)).item()

    def batch_cip_cosine(self, vectors: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """CIP cosine similarity of each row vector with a direction.

        Args:
            vectors: (N, hidden_size) tensor.
            direction: (hidden_size,) tensor.

        Returns:
            (N,) tensor of CIP cosine similarities.
        """
        #(N, H) @ (H, H) -> (N, H), then dot with direction -> (N,)
        sv_vectors = vectors @ self.sigma_v
        numerators = sv_vectors @ direction
        dir_norm = self.cip_norm(direction)
        vec_norms = torch.sqrt((sv_vectors * vectors).sum(dim=1).clamp(min=1e-12))
        return numerators / (vec_norms * dir_norm)


def extract_unembedding(model) -> torch.Tensor:
    """Extract unembedding matrix from a causal LM.

    For LLaMA this is the lm_head weight, shape (vocab_size, hidden_size).
    """
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().cpu().float()
    raise ValueError("Cannot find unembedding matrix (lm_head) in model")


def run_cip_analysis(
    cip_obj: CausalInnerProduct,
    hidden_dir: Path,
    pairs_path: Path,
    temporal_dir_path: Path,
    layers: list[int],
) -> dict[int, dict]:
    """Run CIP analysis comparing temporal vs control directions.

    For each layer:
    - CIP parallelism: CIP cosine of each delta with temporal direction
    - Compare with Euclidean parallelism
    """
    pairs = read_jsonl(pairs_path)
    all_results = {}

    for layer in layers:
        direction = torch.load(
            temporal_dir_path / f"temporal_direction_layer_{layer}.pt",
            map_location="cpu", weights_only=False,
        )
        deltas = compute_deltas(hidden_dir, layer)

        #euclidean parallelism
        euc_cos = (deltas / deltas.norm(dim=1, keepdim=True).clamp(min=1e-8)) @ direction
        #cip parallelism
        cip_cos = cip_obj.batch_cip_cosine(deltas, direction)

        #per-domain breakdown
        domains = [p.get("domain", "unknown") for p in pairs]
        unique_domains = sorted(set(domains))
        domain_euc = {}
        domain_cip = {}
        for d in unique_domains:
            mask = torch.tensor([dom == d for dom in domains])
            domain_euc[d] = float(euc_cos[mask].mean())
            domain_cip[d] = float(cip_cos[mask].mean())

        results = {
            "layer": layer,
            "n_pairs": deltas.size(0),
            "euc_parallelism_mean": float(euc_cos.mean()),
            "euc_parallelism_std": float(euc_cos.std()),
            "cip_parallelism_mean": float(cip_cos.mean()),
            "cip_parallelism_std": float(cip_cos.std()),
            "domain_euc": domain_euc,
            "domain_cip": domain_cip,
        }
        all_results[layer] = results

        logger.info(
            "Layer %d: euc=%.3f, cip=%.3f",
            layer, results["euc_parallelism_mean"], results["cip_parallelism_mean"],
        )

    return all_results


def run_orthogonality_test(
    cip_obj: CausalInnerProduct,
    temporal_dir_path: Path,
    control_dir_path: Path,
    layers: list[int],
) -> dict[int, dict]:
    """Test that temporal direction is orthogonal to control direction in CIP space.

    Target: |CIP_cos| < 0.2
    """
    results = {}
    for layer in layers:
        temporal = torch.load(
            temporal_dir_path / f"temporal_direction_layer_{layer}.pt",
            map_location="cpu", weights_only=False,
        )
        control = torch.load(
            control_dir_path / f"temporal_direction_layer_{layer}.pt",
            map_location="cpu", weights_only=False,
        )
        euc_cos = float((temporal @ control).item())
        cip_cos = cip_obj.cip_cosine(temporal, control)

        results[layer] = {
            "euc_cosine": euc_cos,
            "cip_cosine": cip_cos,
            "orthogonal": abs(cip_cos) < 0.2,
        }
        logger.info(
            "Layer %d orthogonality: euc=%.3f, cip=%.3f, pass=%s",
            layer, euc_cos, cip_cos, results[layer]["orthogonal"],
        )
    return results
