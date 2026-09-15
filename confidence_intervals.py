"""
Adaptive confidence intervals for observations with missing values.

Missing observations are represented by ``None`` or ``NaN``. They count toward
the total sample size ``n`` but are ignored by sample summaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import scipy.optimize
import scipy.stats


Interval = dict[str, float]


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


def _observed_values(observations: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    values = _as_observation_array(observations)
    observed = values[~np.isnan(values)]
    return values, observed


def _observed_sorted(observations: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    values, observed = _observed_values(observations)
    return values, np.sort(observed)


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")


def _interval(left: float, right: float) -> Interval:
    return {"left": float(left), "right": float(right)}


def _observed_fraction(values: np.ndarray) -> float:
    return float(np.mean(~np.isnan(values)))


def _lower_empirical_quantile_saturated(
    observations: Iterable[object],
    p: float,
) -> float:
    values, observed = _observed_sorted(observations)
    n = values.size
    m = observed.size

    if m == 0:
        return float("inf")
    if p > m / n:
        return float(observed[-1])

    rank = min(max(math.ceil(n * p), 1), m)
    return float(observed[rank - 1])


def _upper_empirical_quantile_saturated(
    observations: Iterable[object],
    p: float,
) -> float:
    values, observed = _observed_sorted(observations)
    n = values.size
    m = observed.size

    if m == 0:
        return float("-inf")
    if p >= m / n:
        return float(observed[0])

    rank = min(max(math.ceil(m - n * p), 1), m)
    return float(observed[rank - 1])


def max_min_avg(observations: Iterable[object]) -> float:
    """
    Return the average of the maximum and minimum observed values.
    """
    _, observed = _observed_values(observations)
    if observed.size == 0:
        raise ValueError("observations must include at least one non-missing value")
    return float((observed.max() + observed.min()) / 2)


def compute_sample_mean(observations: Iterable[object]) -> float:
    """
    Return the sample mean of the observed values.
    """
    _, observed = _observed_values(observations)
    return float(np.mean(observed)) if observed.size else float("nan")


def phat_minus(observations: Iterable[object], alpha: float) -> float:
    """
    Lower confidence adjustment for the observed-data fraction.
    """
    _validate_alpha(alpha)
    values = _as_observation_array(observations)
    n = values.size
    phatn = _observed_fraction(values)
    log_term = math.log(6 / alpha)
    adjustment = math.sqrt(2 * phatn * (1 - phatn) * log_term / n)
    adjustment += 3 * log_term / n
    return float(max(phatn - adjustment, 0))


def adaptive_CI_finite_var(
    observations: Iterable[object],
    alpha: float,
    sigma_max: float,
) -> Interval:
    """
    Compute an adaptive confidence interval under a finite-variance assumption.
    """
    _validate_alpha(alpha)
    values = _as_observation_array(observations)
    phatn = _observed_fraction(values)
    phatminus = phat_minus(values, alpha)
    center = compute_sample_mean(values)

    if phatn == 0 or phatminus == 0:
        radius = float("inf")
    else:
        finite_variance_term = math.sqrt((1 - phatminus) / phatminus)
        sampling_term = math.sqrt(2 / (values.size * phatn * phatminus * alpha))
        radius = sigma_max * (finite_variance_term + sampling_term)

    return _interval(center - radius, center + radius)


def adaptive_CI_subG(
    observations: Iterable[object],
    alpha: float,
    sigma_max: float,
) -> Interval:
    """
    Compute an adaptive confidence interval under a sub-Gaussian assumption.
    """
    _validate_alpha(alpha)
    values = _as_observation_array(observations)
    phatn = _observed_fraction(values)
    phatminus = phat_minus(values, alpha)
    center = compute_sample_mean(values)

    if phatminus == 0:
        radius = float("inf")
    elif phatminus >= 1:
        radius = 0.0
    else:
        r_sg_1 = math.sqrt(math.log(4) * math.log(1 / phatminus))
        r_sg_2 = 2.2 * (1 - phatminus) / phatminus
        r_sg_2 *= math.sqrt(math.log(2 / (1 - phatminus)))
        sampling_term = 4 * math.sqrt(math.log(4) * math.log(2/phatminus) * math.log(4 / alpha) / (values.size * phatn))
        radius = sigma_max * min(r_sg_1, r_sg_2) + sigma_max * sampling_term

    return _interval(center - radius, center + radius)

def adaptive_CI_bounded(
    observations: Iterable[object],
    alpha: float,
    a: float,
    b: float,
) -> Interval:
    """
    Compute an adaptive confidence interval under a sub-Gaussian assumption.
    """
    _validate_alpha(alpha)
    values = _as_observation_array(observations)
    phatn = _observed_fraction(values)
    phatminus = phat_minus(values, alpha)
    sample_mean = compute_sample_mean(values)

    if phatminus == 0:
        radius = float("inf")
    elif phatminus >= 1:
        radius = 0.0
    else:
        sampling_term = (b-a) * math.sqrt(math.log(4/alpha) / (2 * values.size * phatn))
        Delta_L = (1-phatminus) * a - sampling_term
        Delta_R = (1-phatminus) * b + sampling_term
        center = phatminus * sample_mean
    return _interval(center - Delta_L, center + Delta_R)


def adaptive_CI_sym(observations: Iterable[object], alpha: float) -> Interval:
    """
    Construct the adaptive confidence interval for symmetric distributions.
    """
    _validate_alpha(alpha)
    values = _as_observation_array(observations)
    quantile_level = 0.5 + math.sqrt(math.log(4 / alpha) / (2 * values.size))
    return _interval(
        _upper_empirical_quantile_saturated(values, quantile_level),
        _lower_empirical_quantile_saturated(values, quantile_level),
    )


def compute_CI_Gaussian(
    observations: Iterable[object],
    t: float,
    lambda_1: float,
    lambda_2: float,
    beta: float,
    sigma_bar: float = 100,
) -> Interval:
    """
    Compute the Gaussian Scheffe--Tukey interval.
    """
    _validate_alpha(beta)
    values = _as_observation_array(observations)
    quantile_level = lambda_1 * (1 - scipy.stats.norm.cdf(t))
    quantile_level += math.sqrt(lambda_2 * math.log(4 / beta) / (2 * values.size))
    left_endpoint = _upper_empirical_quantile_saturated(values, quantile_level) - sigma_bar * t
    right_endpoint = _lower_empirical_quantile_saturated(values, quantile_level) + sigma_bar * t
    return _interval(left_endpoint, right_endpoint)


def adaptive_CI_Gaussian(
    observations: Iterable[object],
    lambda_1: float,
    lambda_2: float,
    beta: float,
    sigma_bar: float = 100,
) -> Interval:
    """
    Choose ``t`` by minimizing the Gaussian Scheffe--Tukey interval length.
    """

    def compute_length(t: float) -> float:
        candidate = compute_CI_Gaussian(observations, t, lambda_1, lambda_2, beta, sigma_bar)
        return candidate["right"] - candidate["left"]

    result = scipy.optimize.minimize_scalar(compute_length, bounds=(0, 100), method="bounded",)
    return compute_CI_Gaussian(observations, result.x, lambda_1, lambda_2, beta, sigma_bar)


__all__ = [
    "Interval",
    "adaptive_CI_Gaussian",
    "adaptive_CI_finite_var",
    "adaptive_CI_subG",
    "adaptive_CI_Sym",
    "compute_CI_Gaussian",
    "compute_sample_mean",
    "max_min_avg",
    "phat_minus",
]
