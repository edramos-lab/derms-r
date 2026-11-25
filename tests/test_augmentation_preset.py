"""MRI augmentation preset: geometry and intensity only, defaults unchanged."""
import unittest
import numpy as np
import albumentations as A
from derms.augmentation import training_augmentations, AUGMENTATION_PRESETS

COLOUR = ('HueSaturationValue', 'LabABShift')


class AugmentationPresetTests(unittest.TestCase):
    def names(self, cfg):
        return [type(op).__name__ for op in training_augmentations(cfg)]

    def test_default_preset_is_unchanged(self):
        self.assertEqual(self.names({}), self.names({'augmentation_preset': 'default'}))
        self.assertEqual(self.names({}), ['HorizontalFlip', 'Rotate', 'HueSaturationValue', 'LabABShift',
                                          'RandomBrightnessContrast', 'GaussNoise', 'Blur'])

    def test_mri_preset_has_no_colour_shifts(self):
        names = self.names({'augmentation_preset': 'mri'})
        self.assertTrue(set(COLOUR).isdisjoint(names))
        self.assertEqual(names, ['HorizontalFlip', 'Affine', 'RandomBrightnessContrast', 'RandomGamma',
                                 'GaussNoise', 'Blur'])

    def test_mri_preset_runs_and_keeps_shape_and_dtype(self):
        pipeline = A.Compose([A.Resize(224, 224)] + training_augmentations({'augmentation_preset': 'mri'}), seed=1)
        image = np.random.default_rng(0).integers(0, 255, (180, 150, 3), dtype=np.uint8)
        out = pipeline(image=image)['image']
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            training_augmentations({'augmentation_preset': 'stain'})

    def test_presets_are_declared(self):
        self.assertEqual(AUGMENTATION_PRESETS, ('default', 'mri'))


if __name__ == '__main__':
    unittest.main()
