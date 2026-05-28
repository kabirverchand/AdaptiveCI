"""
Synthetic data generators for missing-not-at-random examples.
"""

from __future__ import annotations

import numpy as np


def _validate_probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")
    
def generate_truncated_samples_bounded(
    n: int,
    eps: float,
    q: float,
    level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate bounded samples with MCAR missingness and threshold corruption.

    Parameters
    ----------
    n:
        Number of samples to generate.
    eps:
        Probability that an observation is subject to threshold corruption.
    q:
        Probability that an observation survives the MCAR missingness step.
    level:
        Threshold used by the corruption step.
    rng:
        Optional NumPy random number generator for reproducible simulations.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n!r}")
    _validate_probability(eps, "eps")
    _validate_probability(q, "q")

    random = np.random.default_rng() if rng is None else rng

    mcar_observed = random.binomial(1, q, n).astype(bool)
    samples = random.uniform(-1, 1, n)
    mcar_samples = samples * mcar_observed

    corruption_ind = random.binomial(1, eps, n).astype(bool)
    truncated_ind = samples >= level

    corrupted_samples = truncated_ind * samples

    observed = mcar_samples * (1 - corruption_ind) + corrupted_samples * corruption_ind
    observed[observed == 0] = np.nan

    return observed


def generate_truncated_samples_gaussian(
    n: int,
    eps: float,
    q: float,
    level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate bounded samples with MCAR missingness and threshold corruption.

    Parameters
    ----------
    n:
        Number of samples to generate.
    eps:
        Probability that an observation is subject to threshold corruption.
    q:
        Probability that an observation survives the MCAR missingness step.
    level:
        Threshold used by the corruption step.
    rng:
        Optional NumPy random number generator for reproducible simulations.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n!r}")
    _validate_probability(eps, "eps")
    _validate_probability(q, "q")

    random = np.random.default_rng() if rng is None else rng

    mcar_observed = random.binomial(1, q, n).astype(bool)
    samples = random.normal(0, 1, n)
    mcar_samples = samples * mcar_observed

    corruption_ind = random.binomial(1, eps, n).astype(bool)
    truncated_ind = samples >= level

    corrupted_samples = truncated_ind * samples

    observed = mcar_samples * (1 - corruption_ind) + corrupted_samples * corruption_ind
    observed[observed == 0] = np.nan

    return observed


__all__ = ["generate_truncated_samples_bounded"]
