"""Reproducibility utilities: seeding and config hashing."""

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds for all RNGs to ensure reproducible results.

    Args:
        seed: The random seed to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_hash(config: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest of a config dict.

    Useful for tagging result files with the exact configuration
    that produced them.

    Args:
        config: Dictionary of configuration values (must be JSON-serialisable).

    Returns:
        First 12 characters of the SHA-256 hex digest.
    """
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
