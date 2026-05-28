"""
Empirical quantiles for data with missing observations.

This module implements the lower and upper empirical quantile functions.

The missing symbol ``*`` is represented by ``None`` or ``NaN``. Missing
observations are counted in ``n`` but contribute zero to the indicators.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def is_missing(value: object) -> bool:
    """
    Return whether ``value`` represents the missing symbol ``*``.
    """
    try:
        return bool(np.isnan(np.asarray(value, dtype=float)))
    except (TypeError, ValueError):
        return False


def _as_observation_array(observations: Iterable[object]) -> np.ndarray:
    """
    Return observations as a one-dimensional NumPy float array.
    """
    try:
        values = np.asarray(observations, dtype=float)
    except (TypeError, ValueError):
        try:
            values = np.fromiter(observations, dtype=float)
        except TypeError as exc:
            raise TypeError(
                "observations must be an iterable of numbers or missing values"
            ) from exc

    if values.ndim != 1:
        raise ValueError(f"observations must be one-dimensional; got shape {values.shape}")
    if values.size == 0:
        raise ValueError("observations must contain at least one entry")
    return values


def _observed_sorted(observations: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    values = _as_observation_array(observations)
    observed = np.sort(values[~np.isnan(values)])
    return values, observed


def observed_values(observations: Iterable[object]) -> np.ndarray:
    """
    Return sorted, non-missing observations as a NumPy array.
    """
    _, observed = _observed_sorted(observations)
    return observed


def phat_n(observations: Iterable[object]) -> float:
    """
    Compute phat_n = n^{-1} sum_i 1{Z_i != *}.
    """
    values = _as_observation_array(observations)
    observed_indicator = np.logical_not(np.isnan(values))
    return float(np.mean(observed_indicator))


def z_min(observations: Iterable[object]) -> float:
    """
    Minimum observed value, with ``-inf`` when all entries are missing.
    """
    observed = observed_values(observations)
    return float(observed[0]) if observed.size else float("-inf")


def z_max(observations: Iterable[object]) -> float:
    """
    Maximum observed value, with ``inf`` when all entries are missing.
    """
    observed = observed_values(observations)
    return float(observed[-1]) if observed.size else float("inf")


def _validate_p(p: float) -> None:
    if not np.logical_and(0 < p, p < 1):
        raise ValueError(f"p must be in (0, 1); got {p!r}")


def lower_empirical_quantile(observations: Iterable[object], p: float) -> float:
    """
    Compute the lower empirical quantile function ``Q_n^-(p)``.

    Parameters
    ----------
    observations:
        Iterable containing numeric observations plus missing entries encoded
        as ``None`` or ``NaN``.
    p:
        Quantile level in ``(0, 1)``.
    """
    _validate_p(p)
    values, observed = _observed_sorted(observations)
    n = values.size
    m = observed.size

    if m == 0:
        return float("inf")

    if np.greater(p, np.divide(m, n)):
        return float(observed[-1])

    rank = int(np.clip(np.ceil(np.multiply(n, p)), 1, m))
    return float(observed[rank - 1])


def upper_empirical_quantile(observations: Iterable[object], p: float) -> float:
    """Compute the upper empirical quantile function ``Q_n^+(p)``.

    Parameters
    ----------
    observations:
        Iterable containing numeric observations plus missing entries encoded
        as ``None`` or ``NaN``.
    p:
        Quantile level in ``(0, 1)``.
    """
    _validate_p(p)
    values, observed = _observed_sorted(observations)
    n = values.size
    m = observed.size

    if m == 0:
        return float("-inf")

    if np.greater_equal(p, np.divide(m, n)):
        return float(observed[0])

    rank = int(np.clip(np.ceil(np.subtract(m, np.multiply(n, p))), 1, m))
    return float(observed[rank - 1])


__all__ = [
    "is_missing",
    "observed_values",
    "phat_n",
    "lower_empirical_quantile",
    "upper_empirical_quantile",
    "z_max",
    "z_min",
]
