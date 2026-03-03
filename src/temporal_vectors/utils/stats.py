"""Statistical utilities: bootstrap confidence intervals and permutation tests."""

import numpy as np
from numpy.typing import ArrayLike


def bootstrap_ci(
    values: ArrayLike,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        values: 1-D array of observed values.
        statistic: Which statistic to bootstrap ("mean" or "median").
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(values))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, ci_lower, ci_upper


def permutation_test(
    group_a: ArrayLike,
    group_b: ArrayLike,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Two-sample permutation test for difference in means.

    Args:
        group_a: Observed values for group A.
        group_b: Observed values for group B.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (observed_difference, p_value).
        observed_difference = mean(A) - mean(B).
        p_value = fraction of permutations with |diff| >= |observed|.
    """
    rng = np.random.RandomState(seed)
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    combined = np.concatenate([a, b])
    n_a = len(a)

    observed_diff = float(np.mean(a) - np.mean(b))
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return observed_diff, p_value
