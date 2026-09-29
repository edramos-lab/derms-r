"""Score every architecture × ratio before selecting r*."""
import unittest
from derms.analysis import ABLATIONS, SENSITIVITY, attach_architecture_optima, select_operating_point
from derms.metric import derms_r
from derms.config import RATIOS
import pandas as pd


class OperatingPointTests(unittest.TestCase):
    def test_does_not_select_highest_accuracy_ratio(self):
        scores = {}
        for ratio, acc in [(0.10, 0.90), (0.25, 0.94), (0.50, 0.97), (0.75, 0.98), (1.00, 0.99)]:
            scores[ratio] = derms_r([acc] * 5, 1_000_000, ratio)['derms_r']
        self.assertGreater(scores[1.00], 0)
        r_star, score_star = select_operating_point(scores)
        best_accuracy_ratio = 1.00
        self.assertNotEqual(r_star, best_accuracy_ratio)
        self.assertEqual(r_star, 0.10)
        self.assertEqual(score_star, scores[0.10])
        self.assertEqual(score_star, max(scores.values()))

    def test_architecture_optima_are_distinct_from_operating_points(self):
        rows = []
        for model, accuracies in {
            'small': {0.10: 0.90, 0.25: 0.91, 0.50: 0.92, 0.75: 0.925, 1.00: 0.93},
            'large': {0.10: 0.80, 0.25: 0.88, 0.50: 0.95, 0.75: 0.97, 1.00: 0.99},
        }.items():
            for ratio, acc in accuracies.items():
                scored = derms_r([acc] * 5, 1_000_000 if model == 'small' else 8_000_000, ratio)
                rows.append({
                    'dataset': 'demo',
                    'architecture': model,
                    'data_ratio': ratio,
                    'derms_r_raw': scored['derms_r'],
                })
        df = attach_architecture_optima(pd.DataFrame(rows))
        small = df[df.architecture == 'small']
        self.assertTrue((small.optimal_ratio == 0.10).all())
        self.assertEqual(small.optimal_derms_r.nunique(), 1)
        self.assertGreater(small.derms_r_raw.max(), small.derms_r_raw.min())

    def test_canonical_ratio_set(self):
        self.assertEqual(tuple(RATIOS), (0.10, 0.25, 0.50, 0.75, 1.00))
        self.assertIn(0.75, RATIOS)

    def test_ablations_are_not_canonical(self):
        self.assertNotIn('full', ABLATIONS)
        self.assertIn('performance_complexity', ABLATIONS)
        self.assertEqual(SENSITIVITY['canonical']['alpha'], 2.0)

    def test_tie_prefers_smaller_ratio(self):
        r_star, score = select_operating_point({0.10: 1.5, 1.00: 1.5})
        self.assertEqual(r_star, 0.10)
        self.assertEqual(score, 1.5)


if __name__ == '__main__':
    unittest.main()
