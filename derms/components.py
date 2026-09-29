"""Four multiplicative DERMS-R components. Canonical assembly lives in metric.py."""
from __future__ import annotations
import math
import torch
from .config import ALPHA, BETA, GAMMA, DELTA, P0, SIGMA0


def _f64(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _finite_nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return value


def performance(mean_accuracy, alpha=ALPHA):
    """Q = Abar^alpha."""
    _finite_nonnegative('alpha', alpha)
    mean_accuracy = _f64(mean_accuracy)
    if not ((mean_accuracy >= 0.0) & (mean_accuracy <= 1.0)).item():
        raise ValueError('Accuracy must be represented on the 0-1 scale.')
    return mean_accuracy.pow(alpha)


def data_efficiency(data_ratio, gamma=GAMMA):
    """D(r) = r^(-gamma)."""
    _finite_nonnegative('gamma', gamma)
    if not 0.0 < data_ratio <= 1.0:
        raise ValueError('data_ratio must satisfy 0 < r <= 1.')
    return _f64(data_ratio).pow(-gamma)


def complexity_efficiency(num_params, beta=BETA, p0=P0):
    """C(P) = (P / P0)^(-beta)."""
    _finite_nonnegative('beta', beta)
    if num_params <= 0:
        raise ValueError('num_params must be greater than zero.')
    if p0 <= 0:
        raise ValueError('p0 must be greater than zero.')
    return _f64(num_params / p0).pow(-beta)


def stability(sd_accuracy, delta=DELTA, sigma0=SIGMA0):
    """R(s) = (1 + s / sigma0)^(-delta)."""
    _finite_nonnegative('delta', delta)
    sd_accuracy = _f64(sd_accuracy)
    if float(sd_accuracy) < 0:
        raise ValueError('sd_accuracy must be nonnegative.')
    if sigma0 <= 0:
        raise ValueError('sigma0 must be greater than zero.')
    return (1.0 + sd_accuracy / sigma0).pow(-delta)
