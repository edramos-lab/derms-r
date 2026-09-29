"""Canonical DERMS-R metric properties."""
import math
import unittest
import torch
from derms.metric import derms_r
from derms.config import ALPHA, BETA, GAMMA, DELTA, P0, SIGMA0, RATIOS


EXAMPLE_FOLDS = [0.982, 0.991, 0.987, 0.979, 0.986]


class MetricTests(unittest.TestCase):
    def test_rejects_percentage_scale_accuracy(self):
        with self.assertRaisesRegex(ValueError, '0-1'):
            derms_r([98.7, 99.1], 1_000_000, 0.25)

    def test_requires_at_least_two_folds(self):
        with self.assertRaisesRegex(ValueError, 'two CV folds'):
            derms_r([0.9], 1_000_000, 0.25)

    def test_sample_standard_deviation(self):
        scored = derms_r(EXAMPLE_FOLDS, 1_000_000, 0.25)
        expected = torch.as_tensor(EXAMPLE_FOLDS, dtype=torch.float64).std(unbiased=True).item()
        self.assertAlmostEqual(scored['sd_accuracy'], expected)
        self.assertNotAlmostEqual(scored['sd_accuracy'], torch.as_tensor(EXAMPLE_FOLDS, dtype=torch.float64).std(unbiased=False).item())

    def test_perfect_stability_is_one(self):
        scored = derms_r([0.9, 0.9, 0.9, 0.9, 0.9], 1_000_000, 1.0)
        self.assertEqual(scored['sd_accuracy'], 0.0)
        self.assertEqual(scored['stability'], 1.0)

    def test_higher_variability_lowers_score(self):
        stable = derms_r([0.90, 0.90, 0.90, 0.90, 0.90], 1_000_000, 0.25)
        variable = derms_r([0.80, 0.85, 0.90, 0.95, 1.00], 1_000_000, 0.25)
        self.assertAlmostEqual(stable['mean_accuracy'], variable['mean_accuracy'])
        self.assertGreater(variable['sd_accuracy'], stable['sd_accuracy'])
        self.assertGreater(stable['derms_r'], variable['derms_r'])

    def test_higher_parameter_count_lowers_score(self):
        small = derms_r(EXAMPLE_FOLDS, 1_000_000, 0.25)
        large = derms_r(EXAMPLE_FOLDS, 4_000_000, 0.25)
        self.assertGreater(small['derms_r'], large['derms_r'])

    def test_higher_data_ratio_lowers_score(self):
        scores = [derms_r(EXAMPLE_FOLDS, 1_000_000, ratio)['derms_r'] for ratio in RATIOS]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_higher_mean_accuracy_raises_score(self):
        low = derms_r([0.80, 0.80, 0.80, 0.80, 0.80], 1_000_000, 0.25)
        high = derms_r([0.90, 0.90, 0.90, 0.90, 0.90], 1_000_000, 0.25)
        self.assertGreater(high['derms_r'], low['derms_r'])

    def test_reference_complexity_is_one(self):
        scored = derms_r(EXAMPLE_FOLDS, int(P0), 1.0)
        self.assertAlmostEqual(scored['complexity_efficiency'], 1.0)

    def test_full_dataset_data_efficiency_is_one(self):
        scored = derms_r(EXAMPLE_FOLDS, 1_000_000, 1.0)
        self.assertAlmostEqual(scored['data_efficiency'], 1.0)

    def test_canonical_known_value(self):
        low, high = 0.8 - 0.003 / math.sqrt(2), 0.8 + 0.003 / math.sqrt(2)
        scored = derms_r([low, high], 4_000_000, 0.25)
        self.assertAlmostEqual(scored['mean_accuracy'], 0.8)
        self.assertAlmostEqual(scored['sd_accuracy'], 0.003)
        self.assertAlmostEqual(scored['derms_r'], 0.32)
        self.assertAlmostEqual(scored['stability'], 0.5)

    def test_defaults_match_canonical_baseline(self):
        self.assertEqual((ALPHA, BETA, GAMMA, DELTA, P0, SIGMA0), (2.0, 0.5, 0.5, 0.5, 1_000_000.0, 0.001))

    def test_parameters_are_configurable(self):
        baseline = derms_r(EXAMPLE_FOLDS, 2_000_000, 0.25)
        heavy = derms_r(EXAMPLE_FOLDS, 2_000_000, 0.25, beta=1.0)
        self.assertNotAlmostEqual(baseline['derms_r'], heavy['derms_r'])
        self.assertAlmostEqual(heavy['complexity_efficiency'], 0.5)


if __name__ == '__main__':
    unittest.main()
