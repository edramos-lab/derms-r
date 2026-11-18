"""Near-duplicate clustering: perceptual-hash units for CV and the development/test check."""
import unittest

import numpy as np

from derms.datasets import make_units, near_duplicate_overlap, folds_for


def phash_hex(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes().hex()


def record(i, label, bits, sha=None):
    return {'id': f'c{label}/img{i}.png', 'label': label, 'sha256': sha or f'sha{i}', 'phash': phash_hex(bits)}


def flipped(bits, n):
    out = np.array(bits, dtype=np.uint8)
    out[:n] ^= 1
    return out


class NearDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.base = self.rng.integers(0, 2, 256, dtype=np.uint8)

    def test_units_merge_images_within_threshold_only(self):
        records = [record(0, 0, self.base), record(1, 0, flipped(self.base, 10)),
                   record(2, 0, flipped(self.base, 40)), record(3, 1, self.rng.integers(0, 2, 256, dtype=np.uint8))]
        sizes = lambda t: sorted(len(u) for u in make_units(records, None, t)[0])
        self.assertEqual(sizes(0), [1, 1, 1, 1])
        self.assertEqual(sizes(16), [1, 1, 2])

    def test_chain_through_a_middle_image_joins_one_unit(self):
        a = self.base
        b = flipped(a, 12)
        c = np.array(b); c[200:212] ^= 1  # 12 bits from b, 24 from a
        units, _ = make_units([record(0, 0, a), record(1, 0, b), record(2, 0, c)], None, 16)
        self.assertEqual([len(u) for u in units], [3])

    def test_conflicting_labels_in_a_near_duplicate_unit_are_refused(self):
        records = [record(0, 0, self.base), record(1, 1, flipped(self.base, 4))]
        with self.assertRaises(ValueError):
            make_units(records, None, 16)

    def test_overlap_counts_test_images_close_to_development(self):
        dev = [record(0, 0, self.base), record(1, 1, self.rng.integers(0, 2, 256, dtype=np.uint8))]
        test = [record(2, 0, flipped(self.base, 12)), record(3, 1, self.rng.integers(0, 2, 256, dtype=np.uint8))]
        self.assertEqual(near_duplicate_overlap(test, dev, 16), 1)
        self.assertEqual(near_duplicate_overlap(test, dev, 8), 0)

    def test_overlap_without_hashes_is_an_error(self):
        dev = [{'id': 'a', 'label': 0, 'sha256': 'x'}]
        with self.assertRaises(RuntimeError):
            near_duplicate_overlap(dev, dev, 16)

    def test_near_duplicates_never_cross_cv_folds(self):
        records = []
        for cls in range(2):
            for k in range(10):
                seed = self.rng.integers(0, 2, 256, dtype=np.uint8)
                records += [record(len(records), cls, seed), record(len(records) + 1, cls, flipped(seed, 8))]
        plan = folds_for(records, None, 42, 'StratifiedGroupKFold', near_dup_threshold=16)
        index = {r['id']: i for i, r in enumerate(records)}
        for fold in plan['folds']:
            train, val = set(fold['train']), set(fold['val'])
            self.assertFalse(train & val)
            for i in val:  # the twin of every validation image is also in validation
                twin = i + 1 if i % 2 == 0 else i - 1
                self.assertIn(twin, val)
        self.assertEqual(plan['near_dup_threshold'], 16)
        self.assertEqual(plan['independent_units'], 20)


if __name__ == '__main__':
    unittest.main()
