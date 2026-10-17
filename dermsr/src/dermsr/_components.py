"""The four multiplicative DERMS-R components.

Canonical assembly lives in :mod:`dermsr.metric`. These are pure functions
of scalars or numpy arrays with no side effects, mirroring the style of
``sklearn.metrics`` building blocks.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray]


def _check_finite_positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}.")
    return value


def performance(mean_accuracy: ArrayLike, alpha: float = 2.0) -> ArrayLike:
    """Performance component :math:`Q = \\bar{A}^\\alpha`.

    Parameters
    ----------
    mean_accuracy : float or ndarray
        Mean validation accuracy on the 0-1 scale.
    alpha : float, default=2.0
        Performance exponent. Must be finite and strictly positive.

    Returns
    -------
    float or ndarray
        The performance component.
    """
    _check_finite_positive("alpha", alpha)
    mean_accuracy = np.asarray(mean_accuracy, dtype=np.float64)
    if np.any((mean_accuracy < 0.0) | (mean_accuracy > 1.0)):
        raise ValueError("mean_accuracy must be represented on the 0-1 scale.")
    result = mean_accuracy ** alpha
    return float(result) if np.ndim(result) == 0 else result


def data_efficiency(dataset_ratio: ArrayLike, gamma: float = 0.5) -> ArrayLike:
    """Data-efficiency component :math:`D(r) = r^{-\\gamma}`.

    Parameters
    ----------
    dataset_ratio : float or ndarray
        Fraction of the available training data used, in ``(0, 1]``.
    gamma : float, default=0.5
        Data-efficiency exponent. Must be finite and strictly positive.

    Returns
    -------
    float or ndarray
        The data-efficiency component.
    """
    _check_finite_positive("gamma", gamma)
    dataset_ratio = np.asarray(dataset_ratio, dtype=np.float64)
    if np.any((dataset_ratio <= 0.0) | (dataset_ratio > 1.0)):
        raise ValueError("dataset_ratio must satisfy 0 < r <= 1.")
    result = dataset_ratio ** (-gamma)
    return float(result) if np.ndim(result) == 0 else result


def complexity_efficiency(
    n_parameters: ArrayLike, beta: float = 0.5, p0: float = 1_000_000.0
) -> ArrayLike:
    """Complexity-efficiency component :math:`C(P) = (P / P_0)^{-\\beta}`.

    Parameters
    ----------
    n_parameters : float or ndarray
        Raw (unscaled) trainable parameter count.
    beta : float, default=0.5
        Complexity exponent. Must be finite and strictly positive.
    p0 : float, default=1_000_000.0
        Reference parameter scale. Fixed by the canonical DERMS-R
        specification; only change it for explicit sensitivity analysis.

    Returns
    -------
    float or ndarray
        The complexity-efficiency component.
    """
    _check_finite_positive("beta", beta)
    _check_finite_positive("p0", p0)
    n_parameters = np.asarray(n_parameters, dtype=np.float64)
    if np.any(n_parameters <= 0):
        raise ValueError("n_parameters must be greater than zero.")
    result = (n_parameters / p0) ** (-beta)
    return float(result) if np.ndim(result) == 0 else result


def stability(
    std_accuracy: ArrayLike, delta: float = 0.5, epsilon: float = 0.001
) -> ArrayLike:
    """Stability component :math:`R(s) = (1 + s/\\epsilon)^{-\\delta}`.

    Parameters
    ----------
    std_accuracy : float or ndarray
        Sample standard deviation of validation accuracy across
        cross-validation folds. Must be nonnegative.
    delta : float, default=0.5
        Stability exponent. Must be finite and strictly positive.
    epsilon : float, default=0.001
        Reference scale for the stability penalty. Fixed by the canonical
        DERMS-R specification; only change it for explicit sensitivity
        analysis.

    Returns
    -------
    float or ndarray
        The stability component.
    """
    _check_finite_positive("delta", delta)
    _check_finite_positive("epsilon", epsilon)
    std_accuracy = np.asarray(std_accuracy, dtype=np.float64)
    if np.any(std_accuracy < 0):
        raise ValueError("std_accuracy must be nonnegative.")
    result = (1.0 + std_accuracy / epsilon) ** (-delta)
    return float(result) if np.ndim(result) == 0 else result
