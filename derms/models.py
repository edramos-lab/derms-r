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


def verify_registry(names):
    available = set(timm.list_models(pretrained=True))
    report = {}
    for name in names:
        identifier = REGISTRY[name]
        if identifier not in available or not timm.is_model(identifier):
            raise RuntimeError(f'{name}: {identifier} unavailable in timm {timm.__version__}; '
                               'no substitution permitted. Install the pinned timm version.')
        cfg = timm.get_pretrained_cfg(identifier)
        report[name] = {'identifier': identifier, 'pretrained_config': cfg.to_dict()}
    return report


def model_for(name, nclasses, pretrained):
    # No fallback after download/model construction failure.
    model = timm.create_model(REGISTRY[name], pretrained=pretrained, num_classes=nclasses)
    return model

