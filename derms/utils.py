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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_torch(path, value):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # Only load artifacts produced by this trusted local run (pickle format).
    return torch.load(path, map_location='cpu', weights_only=False)


@contextlib.contextmanager
def exclusive(output):
    """OS lock released automatically on crash; lock file itself is harmless."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with open(output / '.writer.lock', 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Another process is writing this benchmark output.') from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment():
    packages = {k: importlib.metadata.version(k) for k in
                ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scikit-learn',
                 'albumentations', 'opencv-python-headless', 'Pillow', 'scipy', 'fvcore')}
    return {'packages': packages, 'python': sys.version, 'platform': platform.platform(),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'cpu': platform.processor(), 'threads': torch.get_num_threads()}

