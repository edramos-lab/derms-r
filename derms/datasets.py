#!/usr/bin/env python3
"""DERMS-R development-only benchmark. See README.md for protocol and limitations."""
from __future__ import annotations
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets as tv

from .config import *
from .utils import *
from .augmentation import training_augmentations


def labels(ds):
    if isinstance(ds, Subset):
        return labels(ds.dataset)[ds.indices]
    if isinstance(ds, ConcatDataset):
        return np.concatenate([labels(d) for d in ds.datasets])
    for key in ('targets', '_labels'):
        if hasattr(ds, key):
            return np.asarray(getattr(ds, key), dtype=np.int64)
    raise TypeError(f'No metadata-only label accessor for {type(ds)}')


def sample_id(ds, i):
    if isinstance(ds, Subset):
        return sample_id(ds.dataset, ds.indices[i])
    if isinstance(ds, ConcatDataset):
        k = int(np.searchsorted(ds.cumulative_sizes, i, side='right'))
        offset = ds.cumulative_sizes[k-1] if k else 0
        return f'{k}/' + sample_id(ds.datasets[k], i-offset)
    if isinstance(ds, tv.ImageFolder):
        return Path(ds.samples[i][0]).relative_to(ds.root).as_posix()
    if hasattr(ds, '_image_files'):
        return Path(ds._image_files[i]).as_posix()
    return str(i)


def image_array(ds, i):
    image, _ = ds[i]
    return np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image, dtype=np.uint8)


def inventory(ds):
    """Hash decoded RGB content so differently encoded exact duplicates also match."""
    y = labels(ds)
    records = []
    for i in range(len(ds)):
        arr = image_array(ds, i)
        h = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
        records.append({'id': sample_id(ds, i), 'label': int(y[i]), 'sha256': h})
    return records


def inferred_groups(records):
    """Patient/sample groups from filenames, matching E2E-ImageClassification."""
    from prepare_dataset import base_id
    groups = []
    for record in records:
        relative = Path(record['id'])
        class_name = relative.parts[0] if len(relative.parts) > 1 else ''
        groups.append(f'{class_name}/{base_id(relative.name)}')
    return groups


def resolve_groups(cfg, records, prefix=''):
    if cfg.get('all_groups'):
        return all_groups(cfg['all_groups'], records, prefix)
    if cfg.get('all_independent_images'):
        return None
    return inferred_groups(records)


def all_split_dirs(root):
    root = Path(root)
    for train_name, test_name in (('development', 'test'), ('train', 'test'), ('Training', 'Testing')):
        if (root / train_name).is_dir() and (root / test_name).is_dir():
            return train_name, test_name
    return None, None


def all_groups(path, records, prefix=''):
    if not path:
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'path', 'group'} <= rows[0].keys():
        raise ValueError('ALL groups CSV requires path,group columns (paths relative to ALL root).')
    mapping = {}
    for row in rows:
        key = row['path'].replace('\\', '/')
        if key in mapping or not row['group'].strip():
            raise ValueError('Duplicate CSV path or empty patient/group ID.')
        mapping[key] = row['group']
    return [mapping[prefix + r['id']] for r in records]


def make_units(records, groups=None):
    """Union exact duplicates and optional patient groups; prohibit label conflicts."""
    parents = list(range(len(records)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, r in enumerate(records):
        keys = [('content', r['sha256'])]
        if groups is not None:
            keys.append(('patient', groups[i]))
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    buckets = {}
    for i in range(len(records)):
        buckets.setdefault(root(i), []).append(i)
    units = list(buckets.values())
    uy = []
    for members in units:
        values = {records[i]['label'] for i in members}
        if len(values) != 1:
            raise ValueError('Exact duplicates/patient group has conflicting class labels; curate data first.')
        uy.append(values.pop())
    return units, np.asarray(uy)


def expand(units, selected):
    return sorted(i for u in selected for i in units[int(u)])


def nested(units, uy, eligible, seed):
    rng = np.random.default_rng(seed)
    orders = [rng.permutation([u for u in eligible if uy[u] == c]).tolist() for c in np.unique(uy)]
    if any(not x for x in orders):
        raise ValueError('Every training subset must contain every class.')
    return {f'{r:.2f}': expand(units, [u for order in orders
            for u in order[:max(1, int(math.floor(len(order)*r)))]] ) for r in RATIOS}


def folds_for(records, groups, seed):
    units, uy = make_units(records, groups)
    counts = np.bincount(uy)
    if min(counts) < FOLDS:
        raise ValueError(f'{FOLDS}-fold stratification needs at least {FOLDS} independent units in every class.')
    folds = []
    for k, (tr, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(units, uy)):
        subsets = nested(units, uy, tr, seed + 100 + k)
        folds.append({'train': expand(units, tr), 'val': expand(units, va), 'subsets': subsets})
    final_subsets = nested(units, uy, np.arange(len(units)), seed + 900)
    return {'folds': folds, 'final_subsets': final_subsets, 'independent_units': len(units),
            'cv': 'StratifiedKFold',
            'stratification_unit': 'image' if len(units) == len(records) else 'duplicate/patient group'}


def official(name, root, test=False, download=False):
    # Training never instantiates official test dataset objects.
    if name in ('cifar10', 'cifar100'):
        return (tv.CIFAR10 if name == 'cifar10' else tv.CIFAR100)(
            root, train=not test, download=download)
    if name == 'flowers102':
        if test:
            return tv.Flowers102(root, split='test', download=download)
        return ConcatDataset([tv.Flowers102(root, split=s, download=download) for s in ('train', 'val')])
    if name == 'food101':
        return tv.Food101(root, split='test' if test else 'train', download=download)
    raise ValueError(name)


def development(name, cfg, download=False):
    root = Path(cfg['data']) / name
    partition = {'kind': 'official'}
    groups = None
    if name != 'all':
        ds = official(name, root, download=download)
        records = inventory(ds)
        classes = list(getattr(ds, 'classes', [str(i) for i in sorted(set(labels(ds)))]))
    else:
        train_name, test_name = all_split_dirs(root)
        if train_name is None and any((root / name).is_dir() for name in
                                     ('development', 'train', 'Training', 'test', 'Testing')):
            raise ValueError('ALL requires a matched pair development/test, train/test, or Training/Testing, or neither for flat holdout.')
        if train_name:
            ds = tv.ImageFolder(root / train_name)
            records = inventory(ds)
            groups = resolve_groups(cfg, records, prefix=f'{train_name}/')
            partition = {'kind': 'all_folders', 'train_name': train_name, 'test_name': test_name}
        else:
            full = tv.ImageFolder(root)
            all_records = inventory(full)
            full_groups = resolve_groups(cfg, all_records)
            units, uy = make_units(all_records, full_groups)
            tr, te = train_test_split(np.arange(len(units)), test_size=cfg['holdout'],
                                     random_state=cfg['seed'], stratify=uy)
            train_ids, test_ids = expand(units, tr), expand(units, te)
            ds = Subset(full, train_ids)
            records = [all_records[i] for i in train_ids]
            groups = [full_groups[i] for i in train_ids] if full_groups else None
            partition = {'kind': 'all_holdout', 'train_indices': train_ids, 'test_indices': test_ids,
                         'inventory_hash': digest(all_records),
                         'test_ids': [all_records[i]['id'] for i in test_ids]}
        classes = list(ds.dataset.classes if isinstance(ds, Subset) else ds.classes)
    return ds, records, classes, groups, partition


def prepare(name, cfg, output, download):
    print(f'{name}: fingerprinting development images and checking splits', flush=True)
    ds, records, classes, groups, partition = development(name, cfg, download)
    plan = {'schema': SCHEMA, 'dataset': name, 'config_hash': digest(cfg),
            'records': records, 'classes': classes, 'groups': groups, 'partition': partition,
            **folds_for(records, groups, cfg['seed'])}
    plan['hash'] = digest(plan)
    path = output / name / 'splits.json'
    if path.exists():
        if read_json(path) != plan:
            raise RuntimeError(f'{path}: stale/modified split manifest or changed dataset. Use a new output directory.')
    else:
        atomic_json(path, plan)
    return ds, plan


def transform(cfg, augment, seed):
    ops = [A.Resize(224, 224)]
    if augment:
        ops += training_augmentations(cfg)
    ops += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(ops, seed=seed)


class Images(Dataset):
    def __init__(self, base, indices, cfg, augment, seed):
        self.base, self.indices = base, indices
        self.tf = transform(cfg, augment, seed)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        image, target = self.base[j]
        arr = np.asarray(image.convert('RGB'))
        return self.tf(image=arr)['image'], int(target)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    # Albumentations owns an independent RNG: reseed each worker explicitly.
    torch.utils.data.get_worker_info().dataset.tf.set_random_seed(seed)


def loader(ds, indices, cfg, training, seed):
    return DataLoader(Images(ds, indices, cfg, training and cfg['augmentation'], seed),
                      batch_size=cfg['batch_size'], shuffle=training, drop_last=False,
                      num_workers=cfg['workers'], pin_memory=cfg['device'].startswith('cuda'),
                      worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed))

