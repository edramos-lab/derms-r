"""Independent DERMS-R component identities."""
import math
import unittest
from derms.components import complexity_efficiency, data_efficiency, performance, stability
from derms.config import RATIOS


class ComponentTests(unittest.TestCase):
    def test_performance_is_accuracy_squared_by_default(self):
        self.assertAlmostEqual(performance(0.8).item(), 0.64)

    def test_canonical_data_efficiency_table(self):
        expected = {0.10: math.sqrt(10), 0.25: 2.0, 0.50: math.sqrt(2),
                    0.75: 1.0 / math.sqrt(0.75), 1.00: 1.0}
        for ratio in RATIOS:
            self.assertAlmostEqual(data_efficiency(ratio).item(), expected[ratio], places=4)

    def test_complexity_at_p0_is_one(self):
        self.assertAlmostEqual(complexity_efficiency(1_000_000).item(), 1.0)

    def test_complexity_four_million_params(self):
        self.assertAlmostEqual(complexity_efficiency(4_000_000).item(), 0.5)

    def test_zero_sd_stability_is_one(self):
        self.assertEqual(stability(0.0).item(), 1.0)

    def test_canonical_stability_example(self):
        self.assertAlmostEqual(stability(0.003).item(), 0.5)

    def test_canonical_ratios_include_r075(self):
        self.assertEqual(tuple(RATIOS), (0.10, 0.25, 0.50, 0.75, 1.00))


if __name__ == '__main__':
    unittest.main()
