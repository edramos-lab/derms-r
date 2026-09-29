#!/usr/bin/env python3
"""Offline regression tests. Uses synthetic images/tiny networks; no pretrained downloads."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from derms import engine as b
from derms.metric import derms_r
import analyze
from derms.tracking import Tracker
from derms import datasets as dataset_module

spec = importlib.util.spec_from_file_location('locked_evaluator', Path(__file__).with_name('test.py'))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def tiny(*args, **kwargs):
    n = args[1] if len(args) > 1 else 2
    return nn.Sequential(nn.Conv2d(3, 4, 3, stride=16), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(4, n))


def make_data(root, folders=True):
    rng = np.random.default_rng(100)
    for split, count in ([('development', 25), ('test', 5)] if folders else [('', 30)]):
        for c in ('a', 'b'):
            folder = root / 'all' / split / c
            folder.mkdir(parents=True)
            for i in range(count):
                Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(folder / f'{i}.png')


def call(argv, fn=b.main):
    with patch.object(sys, 'argv', ['script'] + argv), contextlib.redirect_stdout(io.StringIO()):
        fn()


class ProtocolTests(unittest.TestCase):
    def test_tracking_mirror_failure(self):
        from unittest.mock import MagicMock
        tracker = Tracker('id', 'name', 'dataset', {})
        tracker.run = MagicMock()
        tracker.run.log.side_effect = RuntimeError('network unavailable')
        with self.assertWarns(UserWarning):
            tracker.epoch({'epoch': 1, 'train_loss': .5, 'lr': .01, 'val_accuracy': .8})
        payload = tracker.run.log.call_args.args[0]
        self.assertEqual(payload['val/accuracy'], .8)
        self.assertFalse(any('derms' in key.lower() for key in payload))

    def test_derms_known_value(self):
        import math
        low, high = 0.8 - 0.003 / math.sqrt(2), 0.8 + 0.003 / math.sqrt(2)
        result = derms_r([low, high], 4_000_000, 0.25)
        self.assertAlmostEqual(result['derms_r'], .32)
        self.assertAlmostEqual(result['stability'], .5)
        self.assertIn(0.75, b.RATIOS)
        self.assertEqual(tuple(b.RATIOS), (0.10, 0.25, 0.50, 0.75, 1.00))

    def test_nested_group_splits(self):
        records = [dict(id=str(i), label=i // 30, sha256=f'h{i//2}') for i in range(60)]
        groups = [f'p{i//3}' for i in range(60)]
        plan = b.folds_for(records, groups, 12)
        self.assertEqual(plan['cv'], 'StratifiedKFold')
        seen = []
        for fold in plan['folds']:
            tr, va = set(fold['train']), set(fold['val'])
            self.assertFalse(tr & va)
            self.assertFalse({records[i]['sha256'] for i in tr} & {records[i]['sha256'] for i in va})
            self.assertFalse({groups[i] for i in tr} & {groups[i] for i in va})
            previous = set()
            for r in b.RATIOS:
                subset = set(fold['subsets'][f'{r:.2f}'])
                self.assertTrue(previous <= subset <= tr)
                self.assertEqual({records[i]['label'] for i in subset}, {0, 1})
                previous = subset
            self.assertEqual(previous, tr)
            seen.extend(va)
        self.assertEqual(sorted(seen), list(range(60)))
        self.assertEqual(plan, b.folds_for(records, groups, 12))
        # transitive duplicate/patient unions are kept intact
        self.assertEqual(plan['independent_units'], 10)

    def test_official_partition_dispatch(self):
        calls = []
        def fake(*args, **kwargs):
            calls.append(kwargs)
            return list(range(10))
        with patch.object(b.tv, 'CIFAR10', side_effect=fake), patch.object(b.tv, 'CIFAR100', side_effect=fake), \
             patch.object(b.tv, 'Flowers102', side_effect=fake), patch.object(b.tv, 'Food101', side_effect=fake):
            for name in b.DATASETS[:-1]:
                b.official(name, '/unused')
        self.assertEqual(calls, [{'train': True, 'download': False}, {'train': True, 'download': False},
            {'split': 'train', 'download': False}, {'split': 'val', 'download': False},
            {'split': 'train', 'download': False}])

    def test_full_workflow_and_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, out = root / 'data', root / 'out'
            make_data(data)
            args = ['--data', str(data), '--output', str(out), '--datasets', 'all', '--models',
                    'mobilenetv4', '--epochs', '1', '--batch-size', '8', '--threads', '1',
                    '--no-pretrained', '--no-augmentation', '--all-independent-images',
                    '--warmup', '1', '--latency-repeats', '2', '--device', 'cpu']
            original_inventory = b.inventory
            def dev_only(ds):
                if isinstance(ds, b.tv.ImageFolder) and Path(ds.root).name == 'test':
                    raise AssertionError('Test data was opened during development')
                return original_inventory(ds)
            with patch.object(b, 'model_for', side_effect=tiny), patch.object(dataset_module, 'inventory', side_effect=dev_only):
                call(args)
                call(args + ['--resume'])
                call(args + ['--resume', '--lock', 'all'])
            call(['--output', str(out)], analyze.main)
            self.assertTrue((out / 'analysis/derms_summary.csv').exists())
            summary = b.read_json(out / 'all' / 'summary.json')
            for row in summary:
                self.assertEqual(row['data_ratio'], row['ratio'])
                self.assertIn('fold_1_accuracy', row)
                self.assertIn('derms_r_raw', row)
                self.assertIn('optimal_ratio', row)
                self.assertIn(row['data_ratio'], (0.10, 0.25, 0.50, 0.75, 1.00))
            predictions = list(out.glob('all/*/r*/fold*/predictions.npz'))
            self.assertEqual(len(predictions), 25)
            with np.load(predictions[0]) as saved:
                np.testing.assert_allclose(saved['probabilities'].sum(1), 1, atol=1e-6)
            self.assertEqual(len(summary), 5)
            self.assertEqual(max(r['derms_r_normalized'] for r in summary), 1.)
            self.assertTrue((out / 'all' / 'fold_results.csv').exists())
            self.assertTrue((out / 'all' / 'architecture_optima.csv').exists())
            self.assertTrue(list(out.glob('all/*/r*/fold*/val_confusion_matrix.png')))
            self.assertTrue(list(out.glob('all/*/r*/fold*/val_roc_curve.png')))
            with patch.object(b, 'model_for', side_effect=tiny):
                call(['--output', str(out), '--dataset', 'all'], t.main)
            result = b.read_json(out / 'all' / 'locked' / 'test_result.json')
            self.assertEqual(result['n_test'], 10)
            self.assertTrue((out / 'all' / 'locked' / 'test_confusion_matrix.png').exists())
            self.assertTrue((out / 'all' / 'locked' / 'test_roc_curve.png').exists())
            self.assertIn('roc_auc_macro_ovr', result['metrics'])
            self.assertIn('confusion_matrix', result['metrics'])
            with patch.object(b, 'inventory', side_effect=AssertionError('repeat accessed data')):
                call(['--output', str(out), '--dataset', 'all'], t.main)
            # Config changes cannot be resumed.
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                call(args + ['--resume', '--lr', '0.001'])
            # Stale split manifest cannot be reused.
            splitfile = out / 'all' / 'splits.json'
            saved = splitfile.read_text()
            plan = json.loads(saved)
            plan['folds'][0]['val'][0] = plan['folds'][0]['train'][0]
            b.atomic_json(splitfile, plan)
            with self.assertRaisesRegex(RuntimeError, 'stale/modified'):
                call(args + ['--resume', '--prepare-only'])
            splitfile.write_text(saved)
            # Changed image content invalidates the fingerprint.
            Image.fromarray(np.zeros((24,24,3), dtype=np.uint8)).save(data / 'all/development/a/0.png')
            with self.assertRaisesRegex(RuntimeError, 'stale/modified'):
                call(args + ['--resume', '--prepare-only'])

    def test_test_overlap_rejected_and_holdout_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_data(root)
            cfg = dict(data=str(root), all_groups=None, all_independent_images=True, seed=1, holdout=.2)
            ds, records, classes, groups, partition = b.development('all', cfg)
            plan = dict(records=records, classes=classes, groups=groups, partition=partition)
            (root / 'all/test/a/0.png').write_bytes((root / 'all/development/a/0.png').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'Data leakage'):
                t.untouched_test('all', cfg, plan, False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_data(root, False)
            cfg['data'] = str(root)
            _, records, classes, groups, partition = b.development('all', cfg)
            self.assertFalse(set(partition['train_indices']) & set(partition['test_indices']))
            self.assertEqual(len(partition['test_indices']), 12)
            self.assertEqual(partition, b.development('all', cfg)[4])
            ds, _ = t.untouched_test('all', cfg, dict(records=records, classes=classes,
                                                     groups=groups, partition=partition), False)
            self.assertEqual(len(ds), 12)

    def test_epoch_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_data(root)
            ds = b.tv.ImageFolder(root / 'all/development')
            cfg = dict(device='cpu', pretrained=False, lr=.001, weight_decay=.01, batch_size=4,
                       workers=0, augmentation=True, flip_probability=.5, rotation=10., rotation_probability=.3)
            torch.set_num_threads(1)
            real_save = b.atomic_torch
            def interrupt(path, value):
                real_save(path, value)
                if Path(path).name == 'last.pt' and value['epoch'] == 1:
                    raise InterruptedError('simulated crash after epoch commit')
            with patch.object(b, 'model_for', side_effect=tiny), contextlib.redirect_stdout(io.StringIO()):
                b.fit(ds, [0,1,25,26], [2,27], cfg, 'mobilenetv4', 2, root / 'full', 'same', 42, 2)
                with patch.object(b, 'atomic_torch', side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        b.fit(ds, [0,1,25,26], [2,27], cfg, 'mobilenetv4', 2, root / 'resumed', 'same', 42, 2)
                b.fit(ds, [0,1,25,26], [2,27], cfg, 'mobilenetv4', 2, root / 'resumed', 'same', 42, 2)
            full, resumed = [b.load_checkpoint(root / name / 'last.pt') for name in ('full', 'resumed')]
            self.assertEqual(full['history'], resumed['history'])
            for key in full['model']:
                self.assertTrue(torch.equal(full['model'][key], resumed['model'][key]), key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
