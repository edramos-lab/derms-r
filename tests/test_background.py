import unittest

import numpy as np

from derms import background as bg
from derms.datasets import Images


def smear(cell_colour=(120, 70, 190)):
    """White glass with one stained disc, and a dark caption in the bottom strip."""
    arr = np.full((bg.SIDE, bg.SIDE, 3), 248, dtype=np.uint8)
    yy, xx = np.mgrid[0:bg.SIDE, 0:bg.SIDE]
    arr[(yy - 90) ** 2 + (xx - 100) ** 2 < 35 ** 2] = cell_colour
    arr[-12:, -60:] = 10
    return arr


class MaskTests(unittest.TestCase):
    def test_mask_covers_the_cell_and_not_the_glass_or_the_strip(self):
        mask = bg.cell_mask(smear())
        self.assertGreater(mask[90, 100], 0.9)
        self.assertLess(mask[10, 10], 0.1)
        self.assertEqual(float(mask[int(bg.SIDE * (1 - bg.STRIP)) + 2:].max()), 0.0)

    def test_enclosed_glass_pocket_is_not_filled(self):
        arr = smear()
        yy, xx = np.mgrid[0:bg.SIDE, 0:bg.SIDE]
        arr[(yy - 90) ** 2 + (xx - 100) ** 2 < 12 ** 2] = 248  # 450 px of glass inside the cell
        self.assertLess(bg.cell_mask(arr)[90, 100], 0.3)


class RandomizeTests(unittest.TestCase):
    def test_same_seed_repeats_and_different_seeds_differ(self):
        arr = smear()
        a = bg.randomize(arr, np.random.default_rng(1))
        self.assertTrue(np.array_equal(a, bg.randomize(arr, np.random.default_rng(1))))
        self.assertFalse(np.array_equal(a, bg.randomize(arr, np.random.default_rng(2))))

    def test_glass_changes_but_the_cell_stays_in_place_and_the_strip_is_blank(self):
        arr = smear()
        out = bg.randomize(arr, np.random.default_rng(3), protocol='giemsa_acidic')
        self.assertGreater(np.abs(out[10:30, 10:30].astype(int) - arr[10:30, 10:30].astype(int)).mean(), 2)
        self.assertLess(out[90, 100].astype(int).sum(), 560)  # still a dark stained cell, not glass
        self.assertTrue(np.array_equal(out[-8:, -8:].reshape(-1, 3)[0], bg.FILL))

    def test_background_ignores_the_image_and_so_the_class(self):
        a = bg.make_background(64, 64, np.random.default_rng(5), 'wright')[0]
        b = bg.make_background(64, 64, np.random.default_rng(5), 'wright')[0]
        self.assertTrue(np.array_equal(a, b))

    def test_every_protocol_produces_a_valid_field(self):
        for name in bg.PROTOCOLS:
            field, used = bg.make_background(64, 64, np.random.default_rng(0), name)
            self.assertEqual(used, name)
            self.assertTrue(0 <= field.min() and field.max() <= 255)

    def test_ring_sits_outside_the_cell(self):
        mask = bg.cell_mask(smear())
        ring = bg.ring(mask)
        self.assertTrue(ring.any())
        self.assertFalse((ring & (mask > 0.5)).any())

    def test_native_swap_moves_cells_onto_another_images_glass(self):
        cells, host = smear((120, 70, 190)), smear((200, 120, 120))
        host[:] = np.minimum(host, 235)  # host glass is greyer
        out = bg.native_swap(cells, bg.cell_mask(cells), host, bg.cell_mask(host))
        self.assertTrue(abs(int(out[10, 10, 0]) - 235) < 6)
        self.assertLess(out[90, 100].astype(int).sum(), 560)


class ImagesIntegrationTests(unittest.TestCase):
    class Base:
        def __init__(self, arr):
            from PIL import Image
            self.image = Image.fromarray(arr)
        def __getitem__(self, i):
            return self.image, 1

    def test_training_randomizes_and_evaluation_only_blanks_the_strip(self):
        import torch
        arr = np.full((300, 300, 3), 240, dtype=np.uint8)
        arr[100:160, 100:160] = (120, 70, 190)
        arr[-20:, -80:] = 0
        cfg = {'background_randomization': True}
        train = Images(self.Base(arr), [0], cfg, True, 0)
        torch.manual_seed(0); np.random.seed(0); first = train[0][0]
        np.random.seed(1); second = train[0][0]
        self.assertFalse(torch.equal(first, second))
        evaluation = Images(self.Base(arr), [0], cfg, False, 0)
        a, b = evaluation[0][0], evaluation[0][0]
        self.assertTrue(torch.equal(a, b))
        raw = Images(self.Base(arr), [0], {}, False, 0)[0][0]
        self.assertFalse(torch.equal(a, raw))  # the option blanks the strip even in evaluation


if __name__ == '__main__':
    unittest.main()
