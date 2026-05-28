import unittest

import numpy as np

from synthetic_data import generate_truncated_samples_bounded


class FakeRNG:
    def __init__(self, mcar_observed, samples, corrupted):
        self.binomial_results = [
            np.asarray(mcar_observed, dtype=int),
            np.asarray(corrupted, dtype=int),
        ]
        self.samples = np.asarray(samples, dtype=float)

    def binomial(self, n, p, size):
        return self.binomial_results.pop(0)

    def uniform(self, low, high, size):
        return self.samples


class SyntheticDataTests(unittest.TestCase):
    def test_generate_truncated_samples_bounded_applies_missingness_and_corruption(self):
        rng = FakeRNG(
            mcar_observed=[1, 0, 1, 1, 1],
            samples=[-0.8, -0.2, 0.0, 0.4, 0.9],
            corrupted=[0, 0, 1, 1, 1],
        )

        samples = generate_truncated_samples_bounded(
            n=5,
            eps=0.5,
            q=0.8,
            level=0.25,
            rng=rng,
        )

        expected = np.array([-0.8, np.nan, np.nan, 0.4, 0.9])
        np.testing.assert_allclose(samples, expected, equal_nan=True)

    def test_generate_truncated_samples_bounded_keeps_observed_uncorrupted_zero(self):
        rng = FakeRNG(
            mcar_observed=[1],
            samples=[0.0],
            corrupted=[0],
        )

        samples = generate_truncated_samples_bounded(
            n=1,
            eps=0,
            q=1,
            level=0.25,
            rng=rng,
        )

        np.testing.assert_array_equal(samples, np.array([0.0]))

    def test_generate_truncated_samples_bounded_all_mcar_missing_when_q_is_zero(self):
        samples = generate_truncated_samples_bounded(
            n=10,
            eps=0,
            q=0,
            level=0,
            rng=np.random.default_rng(123),
        )

        self.assertTrue(np.all(np.isnan(samples)))

    def test_generate_truncated_samples_bounded_without_corruption_has_bounded_values(self):
        samples = generate_truncated_samples_bounded(
            n=100,
            eps=0,
            q=1,
            level=0.5,
            rng=np.random.default_rng(123),
        )

        self.assertFalse(np.any(np.isnan(samples)))
        self.assertTrue(np.all(samples >= -1))
        self.assertTrue(np.all(samples <= 1))

    def test_generate_truncated_samples_bounded_validates_inputs(self):
        with self.assertRaises(ValueError):
            generate_truncated_samples_bounded(0, eps=0.1, q=0.9, level=0)
        with self.assertRaises(ValueError):
            generate_truncated_samples_bounded(10, eps=-0.1, q=0.9, level=0)
        with self.assertRaises(ValueError):
            generate_truncated_samples_bounded(10, eps=0.1, q=1.1, level=0)


if __name__ == "__main__":
    unittest.main()
