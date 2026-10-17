"""Bundled sample dataset, in the spirit of ``sklearn.datasets``.

Provides the verified CIFAR-10 DERMS-R benchmark results from the paper
(12 architectures x 5 dataset-utilization ratios, 5-fold cross-validation
each), so that the reproduction scripts and tests in this package are
fully self-contained and do not require access to the original MLflow
tracking server or GPU training infrastructure.
"""
from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["load_cifar10_benchmark"]


def load_cifar10_benchmark() -> pd.DataFrame:
    """Load the verified CIFAR-10 DERMS-R benchmark results.

    Twelve pretrained ``timm`` architectures, each evaluated at five
    canonical dataset-utilization ratios (0.10, 0.25, 0.50, 0.75, 1.00)
    with 5-fold cross-validation, from the
    ``derms-r-benchmark - final - CIFAR10`` experiment.

    Returns
    -------
    DataFrame of shape (60, 18)
        One row per (architecture, dataset_ratio) pair. Columns:
        ``architecture``, ``dataset_ratio``, ``mean_val_accuracy``,
        ``std_val_accuracy``, ``n_parameters``, ``f1_macro``, ``mcc``,
        ``latency_ms``, the four DERMS-R components, ``derms_r_raw``,
        and ``fold_1_accuracy`` through ``fold_5_accuracy``.

    Examples
    --------
    >>> from dermsr.datasets import load_cifar10_benchmark
    >>> df = load_cifar10_benchmark()
    >>> df.shape
    (60, 18)
    >>> sorted(df["architecture"].unique())  # doctest: +NORMALIZE_WHITESPACE
    ['convnextv2_tiny', 'deit3_base', 'efficientnet_b0', 'efficientnetv2_s',
     'efficientvit_b0', 'maxvit_tiny', 'mobilenetv4', 'nextvit_base',
     'resnet50', 'swin_tiny', 'vit_base', 'xception41']
    """
    with resources.as_file(resources.files("dermsr.data").joinpath("cifar10_benchmark.csv")) as path:
        return pd.read_csv(path)
