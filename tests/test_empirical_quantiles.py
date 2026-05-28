import unittest

import numpy as np

from empirical_quantiles import (
    lower_empirical_quantile,
    observed_values,
    phat_n,
    upper_empirical_quantile,
    z_max,
    z_min,
)


class EmpiricalQuantileTests(unittest.TestCase):
    def test_phat_counts_none_and_nan_as_missing(self):
        observations = np.array([1, np.nan, 2, np.nan, 3])

        self.assertEqual(phat_n(observations), 3 / 5)

    def test_observed_values_returns_sorted_numpy_array(self):
        observations = np.array([None, 4, np.nan, -2, 7], dtype=float)

        np.testing.assert_array_equal(observed_values(observations), np.array([-2, 4, 7]))

    def test_z_min_and_z_max_ignore_missing_values(self):
        observations = np.array([None, 4, np.nan, -2, 7], dtype=float)

        self.assertEqual(z_min(observations), -2)
        self.assertEqual(z_max(observations), 7)

    def test_z_min_and_z_max_for_all_missing_values(self):
        observations = np.array([None, np.nan, None], dtype=float)

        self.assertEqual(z_min(observations), float("-inf"))
        self.assertEqual(z_max(observations), float("inf"))

    def test_lower_empirical_quantile_without_missing_values(self):
        observations = np.array([3, 1, 4, 2])

        self.assertEqual(lower_empirical_quantile(observations, 0.25), 1)
        self.assertEqual(lower_empirical_quantile(observations, 0.50), 2)
        self.assertEqual(lower_empirical_quantile(observations, 0.75), 3)
        self.assertEqual(lower_empirical_quantile(observations, 0.99), 4)

    def test_lower_empirical_quantile_counts_missing_values_in_n(self):
        observations = np.array([10, np.nan, 30, 20, np.nan])

        self.assertEqual(lower_empirical_quantile(observations, 0.20), 10)
        self.assertEqual(lower_empirical_quantile(observations, 0.40), 20)
        self.assertEqual(lower_empirical_quantile(observations, 0.60), 30)

    def test_lower_empirical_quantile_returns_z_max_when_p_exceeds_phat(self):
        observations = np.array([10, np.nan, 30, 20, np.nan])

        self.assertEqual(lower_empirical_quantile(observations, 0.61), 30)
        self.assertEqual(
            lower_empirical_quantile(np.array([None, np.nan], dtype=float), 0.5),
            float("inf"),
        )

    def test_upper_empirical_quantile_counts_missing_values_in_n(self):
        observations = np.array([10, np.nan, 30, 20, np.nan])

        self.assertEqual(upper_empirical_quantile(observations, 0.20), 20)
        self.assertEqual(upper_empirical_quantile(observations, 0.40), 10)

    def test_upper_empirical_quantile_returns_z_min_at_and_above_phat(self):
        observations = np.array([10, np.nan, 30, 20, np.nan])

        self.assertEqual(upper_empirical_quantile(observations, 0.60), 10)
        self.assertEqual(upper_empirical_quantile(observations, 0.90), 10)
        self.assertEqual(
            upper_empirical_quantile(np.array([None, np.nan], dtype=float), 0.5),
            float("-inf"),
        )

    def test_quantile_functions_handle_ties(self):
        observations = np.array([1, 1, 1, 5, 9, np.nan])

        self.assertEqual(lower_empirical_quantile(observations, 0.50), 1)
        self.assertEqual(lower_empirical_quantile(observations, 2 / 3), 5)
        self.assertEqual(upper_empirical_quantile(observations, 1 / 6), 5)
        self.assertEqual(upper_empirical_quantile(observations, 2 / 6), 1)

    def test_invalid_p_raises_value_error(self):
        observations = np.array([1, 2, 3])

        for p in (0, 1, -0.1, 1.1):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    lower_empirical_quantile(observations, p)
                with self.assertRaises(ValueError):
                    upper_empirical_quantile(observations, p)

    def test_empty_observations_raise_value_error(self):
        with self.assertRaises(ValueError):
            lower_empirical_quantile([], 0.5)
        with self.assertRaises(ValueError):
            upper_empirical_quantile([], 0.5)


if __name__ == "__main__":
    unittest.main()
