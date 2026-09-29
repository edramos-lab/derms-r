"""Normalized DERMS-R is a separate quantity from the raw score."""
import unittest
import pandas as pd
from derms.normalization import add_normalized_column, normalize_within_ratio


class NormalizationTests(unittest.TestCase):
    def test_within_ratio_peak_is_one(self):
        normalized = normalize_within_ratio({'a': 2.0, 'b': 1.0, 'c': 0.5})
        self.assertEqual(normalized['a'], 1.0)
        self.assertEqual(normalized['b'], 0.5)
        self.assertEqual(normalized['c'], 0.25)

    def test_normalization_does_not_overwrite_raw(self):
        df = pd.DataFrame({
            'dataset': ['x', 'x', 'x', 'x'],
            'architecture': ['a', 'b', 'a', 'b'],
            'data_ratio': [0.10, 0.10, 1.00, 1.00],
            'derms_r_raw': [4.0, 2.0, 3.0, 1.5],
        })
        raw = list(df['derms_r_raw'])
        out = add_normalized_column(df)
        self.assertEqual(list(out['derms_r_raw']), raw)
        self.assertEqual(out.loc[0, 'derms_r_normalized'], 1.0)
        self.assertEqual(out.loc[1, 'derms_r_normalized'], 0.5)
        self.assertEqual(out.loc[2, 'derms_r_normalized'], 1.0)
        self.assertEqual(out.loc[3, 'derms_r_normalized'], 0.5)

    def test_normalization_is_per_dataset(self):
        df = pd.DataFrame({
            'dataset': ['x', 'x', 'y'],
            'architecture': ['a', 'b', 'a'],
            'data_ratio': [0.25, 0.25, 0.25],
            'derms_r_raw': [2.0, 1.0, 8.0],
        })
        out = add_normalized_column(df)
        self.assertEqual(out.loc[0, 'derms_r_normalized'], 1.0)
        self.assertEqual(out.loc[2, 'derms_r_normalized'], 1.0)


if __name__ == '__main__':
    unittest.main()
