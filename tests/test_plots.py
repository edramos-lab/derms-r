"""Confusion matrices and ROC curves are conventional reports, not DERMS-R."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from derms.plots import classification_scores, save_classification_reports
from export_models import state_dict_from_checkpoint
import torch


class PlotTests(unittest.TestCase):
    def test_confusion_and_roc_are_written(self):
        y = [0, 0, 1, 1, 0, 1]
        pred = [0, 1, 1, 1, 0, 0]
        probs = np.array([
            [0.9, 0.1], [0.4, 0.6], [0.2, 0.8],
            [0.3, 0.7], [0.8, 0.2], [0.6, 0.4],
        ])
        scores = classification_scores(y, pred, probs, 2)
        self.assertEqual(scores['confusion_matrix'], [[2, 1], [1, 2]])
        self.assertIsInstance(scores['roc_auc_macro_ovr'], float)
        with tempfile.TemporaryDirectory() as temp:
            payload = save_classification_reports(temp, y, pred, probs, ['a', 'b'], stem='test')
            folder = Path(temp)
            self.assertTrue((folder / 'test_confusion_matrix.png').exists())
            self.assertTrue((folder / 'test_roc_curve.png').exists())
            saved = json.loads((folder / 'test_classification_report.json').read_text())
            self.assertEqual(saved['classes'], ['a', 'b'])
            self.assertEqual(saved['confusion_matrix'], payload['confusion_matrix'])

    def test_missing_class_does_not_crash_auc(self):
        y = [0, 0, 0]
        pred = [0, 0, 0]
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        scores = classification_scores(y, pred, probs, 2)
        self.assertIsNone(scores['roc_auc_macro_ovr'])
        self.assertIsNone(scores['roc_auc_per_class']['1'])


class ExportHelperTests(unittest.TestCase):
    def test_checkpoint_keys(self):
        weights = {'layer.weight': torch.ones(1)}
        self.assertEqual(state_dict_from_checkpoint({'model': weights}), weights)
        self.assertEqual(state_dict_from_checkpoint({'model_state_dict': weights}), weights)
        self.assertEqual(state_dict_from_checkpoint(weights), weights)


if __name__ == '__main__':
    unittest.main()
