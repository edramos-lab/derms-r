import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from derms import engine

VAL_LOSS = [0.9, 0.4, 0.7, 0.8]  # best validation loss at epoch 2; last epoch is 4


class EpochSelectionTests(unittest.TestCase):
    def run_fit(self, selection):
        # the extra value is the final re-evaluation of whichever epoch was kept
        losses = iter(VAL_LOSS + [VAL_LOSS[-1] if selection == 'last' else min(VAL_LOSS)])
        cfg = {'device': 'cpu', 'pretrained': False, 'lr': 1e-3, 'weight_decay': 0.0, 'epoch_selection': selection}
        batches = [(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(engine, 'model_for', lambda *a, **k: nn.Linear(3, 2)), \
                mock.patch.object(engine, 'loader', lambda *a, **k: batches), \
                mock.patch.object(engine, 'evaluate',
                                  lambda *a, **k: ({'loss': next(losses)}, [0], [0], [[1.0, 0.0]])), \
                mock.patch.object(engine, 'save_classification_reports', lambda *a, **k: None):
            _, metrics, epoch, _ = engine.fit(None, [0], [1], cfg, 'tiny', 2, Path(tmp), 'id', 0, len(VAL_LOSS))
        return metrics['loss'], epoch

    def test_best_keeps_the_lowest_validation_loss_epoch(self):
        self.assertEqual(self.run_fit('best'), (0.4, 2))

    def test_last_keeps_the_final_epoch_whatever_validation_says(self):
        self.assertEqual(self.run_fit('last'), (0.8, 4))

    def test_missing_option_behaves_as_legacy_best(self):
        self.assertEqual(self.run_fit(None), (0.4, 2))


if __name__ == '__main__':
    unittest.main()
