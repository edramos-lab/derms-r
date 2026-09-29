"""E2E filename grouping and augmentation defaults."""
import unittest
from pathlib import Path
from prepare_dataset import base_id, UnionFind
from derms.datasets import inferred_groups, all_split_dirs
from derms.augmentation import training_augmentations, LabABShift
import albumentations as A


class PrepareDatasetTests(unittest.TestCase):
    def test_base_id_strips_crop_and_copy_suffixes(self):
        self.assertEqual(base_id('Sap_001 (3).jpg'), 'Sap_001')
        self.assertEqual(base_id('Snap_018 - Copy.jpg'), 'Snap_018')
        self.assertEqual(base_id('Snap_018 - copy (2).png'), 'Snap_018')
        self.assertEqual(base_id('Snap_018.jpg'), 'Snap_018')

    def test_union_find_merges_identity(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        self.assertEqual(uf.find(0), uf.find(1))
        self.assertNotEqual(uf.find(0), uf.find(2))

    def test_inferred_groups_prefix_class(self):
        records = [
            {'id': 'benign/Snap_018 (2).jpg'},
            {'id': 'benign/Snap_018 - Copy.jpg'},
            {'id': 'pre_b/Snap_018.jpg'},
        ]
        groups = inferred_groups(records)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], 'benign/Snap_018')
        self.assertEqual(groups[2], 'pre_b/Snap_018')

    def test_split_dir_detection(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'train' / 'a').mkdir(parents=True)
            (root / 'test' / 'a').mkdir(parents=True)
            self.assertEqual(all_split_dirs(root), ('train', 'test'))


class AugmentationTests(unittest.TestCase):
    def test_e2e_pipeline_includes_stain_and_noise(self):
        cfg = dict(flip_probability=0.5, rotation=20.0, rotation_probability=1.0,
                   hsv_probability=0.8, lab_probability=0.8,
                   brightness_contrast_probability=0.8,
                   gauss_noise_probability=0.15, blur_probability=0.15)
        ops = training_augmentations(cfg)
        kinds = [type(op) for op in ops]
        self.assertIn(A.HorizontalFlip, kinds)
        self.assertIn(A.Rotate, kinds)
        self.assertIn(A.HueSaturationValue, kinds)
        self.assertIn(LabABShift, kinds)
        self.assertIn(A.RandomBrightnessContrast, kinds)
        self.assertTrue(A.GaussNoise in kinds or any('GaussNoise' in type(op).__name__ for op in ops))
        self.assertIn(A.Blur, kinds)

    def test_lab_shift_keeps_uint8_rgb(self):
        import numpy as np
        image = np.full((16, 16, 3), 120, dtype=np.uint8)
        out = LabABShift(p=1.0).apply(image, a_shift=3.0, b_shift=-4.0)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (16, 16, 3))


if __name__ == '__main__':
    unittest.main()
