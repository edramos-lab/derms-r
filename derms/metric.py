"""Canonical DERMS-R calculation. Training code must not redefine this equation."""
from __future__ import annotations
from typing import Dict, Sequence
import torch
from .config import ALPHA, BETA, GAMMA, DELTA, P0, SIGMA0
from .components import (
    complexity_efficiency,
    data_efficiency,
    performance,
    stability,
)


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def metric_kwargs(cfg) -> Dict[str, float]:
    return {
        'alpha': cfg['alpha'],
        'beta': cfg['beta'],
        'gamma': cfg['gamma'],
        'delta': cfg['delta'],
        'p0': cfg['p0'],
        'sigma0': cfg['sigma0'],
    }


def derms_r(
    fold_accuracies: Sequence[float],
    num_params: int,
    data_ratio: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    p0: float = P0,
    sigma0: float = SIGMA0,
) -> Dict[str, float]:
    acc = torch.as_tensor(fold_accuracies, dtype=torch.float64)
    if acc.numel() < 2:
        raise ValueError('DERMS-R requires at least two CV folds.')
    if torch.any((acc < 0.0) | (acc > 1.0)):
        raise ValueError('Accuracy must be represented on the 0-1 scale.')
    if num_params <= 0:
        raise ValueError('num_params must be greater than zero.')
    if not 0.0 < data_ratio <= 1.0:
        raise ValueError('data_ratio must satisfy 0 < r <= 1.')
    if p0 <= 0:
        raise ValueError('p0 must be greater than zero.')
    if sigma0 <= 0:
        raise ValueError('sigma0 must be greater than zero.')

    mean_acc = acc.mean()
    sd_acc = acc.std(unbiased=True)
    performance_component = performance(mean_acc, alpha=alpha)
    data_efficiency_component = data_efficiency(data_ratio, gamma=gamma)
    complexity_efficiency_component = complexity_efficiency(num_params, beta=beta, p0=p0)
    stability_component = stability(sd_acc, delta=delta, sigma0=sigma0)
    score = (
        performance_component
        * data_efficiency_component
        * complexity_efficiency_component
        * stability_component
    )
    return {
        'mean_accuracy': mean_acc.item(),
        'sd_accuracy': sd_acc.item(),
        'performance': performance_component.item(),
        'data_efficiency': data_efficiency_component.item(),
        'complexity_efficiency': complexity_efficiency_component.item(),
        'stability': stability_component.item(),
        'derms_r': score.item(),
    }
