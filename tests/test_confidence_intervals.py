import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.stats

from confidence_intervals import (
    adaptive_CI_Gaussian,
    adaptive_CI_finite_var,
    adaptive_CI_subG,
    adaptive_TS,
    compute_CI_Gaussian,
    compute_sample_mean,
    max_min_avg,
    phat_minus,
)


class ConfidenceIntervalTests(unittest.TestCase):
    def test_max_min_avg_ignores_missing_values(self):
        observations = np.array([1, np.nan, 5, -2])

        self.assertEqual(max_min_avg(observations), 1.5)

    def test_max_min_avg_raises_when_all_values_are_missing(self):
        observations = np.array([np.nan, None], dtype=float)

        with self.assertRaises(ValueError):
            max_min_avg(observations)

    def test_compute_sample_mean_ignores_missing_values(self):
        observations = np.array([1, np.nan, 2, 3])

        self.assertEqual(compute_sample_mean(observations), 2)

    def test_compute_sample_mean_returns_nan_when_all_values_are_missing(self):
        observations = np.array([np.nan, None], dtype=float)

        self.assertTrue(np.isnan(compute_sample_mean(observations)))

    def test_phat_minus_matches_formula(self):
        observations = np.array([1, 2, np.nan, 4, 5])
        alpha = 0.1
        n = observations.size
        phatn = np.mean(~np.isnan(observations))
        log_term = np.log(6 / alpha)
        expected = phatn - np.sqrt(2 * phatn * (1 - phatn) * log_term / n)
        expected -= 3 * log_term / n
        expected = np.maximum(expected, 0)

        self.assertEqual(phat_minus(observations, alpha), expected)

    def test_adaptive_ci_finite_var_matches_formula(self):
        observations = np.linspace(-2, 2, 1000)
        alpha = 0.05
        sigma_max = 2
        phatn = np.mean(~np.isnan(observations))
        phatminus = phat_minus(observations, alpha)
        center = np.mean(observations)
        diff = np.sqrt((1 - phatminus) / phatminus)
        diff += np.sqrt(2 / (observations.size * phatn * alpha))
        expected = {
            "left": center - sigma_max * diff,
            "right": center + sigma_max * diff,
        }

        np.testing.assert_allclose(
            list(adaptive_CI_finite_var(observations, alpha, sigma_max).values()),
            list(expected.values()),
        )

    def test_adaptive_ci_finite_var_returns_infinite_interval_when_phatminus_is_zero(self):
        observations = np.array([1, np.nan, np.nan])

        self.assertEqual(
            adaptive_CI_finite_var(observations, 0.05, 1),
            {"left": float("-inf"), "right": float("inf")},
        )

    def test_adaptive_ci_subg_matches_formula(self):
        observations = np.linspace(-1, 1, 1000)
        alpha = 0.05
        sigma_max = 3
        phatminus = phat_minus(observations, alpha)
        center = np.mean(observations)
        r_sg_1 = np.sqrt(np.log(4) * np.log(1 / phatminus))
        r_sg_2 = 1.5 * (1 - phatminus) / phatminus
        r_sg_2 *= np.sqrt(np.log(2 / (1 - phatminus)))
        radius = sigma_max * np.minimum(r_sg_1, r_sg_2)
        expected = {"left": center - radius, "right": center + radius}

        np.testing.assert_allclose(
            list(adaptive_CI_subG(observations, alpha, sigma_max).values()),
            list(expected.values()),
        )

    def test_adaptive_ts_uses_saturated_empirical_quantiles(self):
        observations = np.append(np.arange(1, 81), np.full(20, np.nan))

        self.assertEqual(adaptive_TS(observations, 0.05), {"left": 16.0, "right": 65.0})

    def test_compute_ci_gaussian_matches_fixed_t_formula(self):
        observations = np.append(np.arange(1, 81), np.full(20, np.nan))
        alpha = 0.05
        t = 1.0
        scale = 10
        quantile_level = 1 - scipy.stats.norm.cdf(t)
        quantile_level += np.sqrt(np.log(4 / alpha) / (2 * observations.size))
        lower_rank = int(np.clip(np.ceil(observations.size * quantile_level), 1, 80))
        upper_rank = int(np.clip(np.ceil(80 - observations.size * quantile_level), 1, 80))
        lower = lower_rank
        upper = upper_rank
        expected = {"left": upper - scale * t, "right": lower + scale * t}

        self.assertEqual(compute_CI_Gaussian(observations, t, alpha, scale), expected)

    def test_adaptive_ci_gaussian_uses_optimizer_t(self):
        observations = np.append(np.arange(1, 81), np.full(20, np.nan))

        with patch(
            "confidence_intervals.scipy.optimize.minimize_scalar",
            return_value=SimpleNamespace(x=1.0),
        ):
            self.assertEqual(
                adaptive_CI_Gaussian(observations, 0.05, 10),
                compute_CI_Gaussian(observations, 1.0, 0.05, 10),
            )

    def test_invalid_alpha_raises_value_error(self):
        observations = np.array([1, 2, 3])

        for alpha in (0, 1, -0.1, 1.1):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    phat_minus(observations, alpha)
                with self.assertRaises(ValueError):
                    adaptive_TS(observations, alpha)

    def test_empty_observations_raise_value_error(self):
        with self.assertRaises(ValueError):
            compute_sample_mean([])
        with self.assertRaises(ValueError):
            adaptive_CI_finite_var([], 0.05, 1)


if __name__ == "__main__":
    unittest.main()
