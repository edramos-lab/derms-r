"""The canonical DERMS-R metric.

Matches the public interface specified in the paper's "Reproducibility and
Reference Implementation" section (Section 16): a single ``derms_score``
function with the documented signature and defaults. Do not redefine this
equation elsewhere; import it from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from ._components import complexity_efficiency, data_efficiency, performance, stability

ArrayLike = Union[float, int, np.ndarray]

__all__ = ["DERMSRResult", "derms_score", "normalize"]


@dataclass(frozen=True)
class DERMSRResult:
    """Decomposed DERMS-R result.

    Attributes
    ----------
    score : float or ndarray
        The raw DERMS-R score, equal to the product of the four
        components below.
    performance : float or ndarray
        :math:`Q = \\bar{A}^\\alpha`.
    data_efficiency : float or ndarray
        :math:`D(r) = r^{-\\gamma}`.
    complexity_efficiency : float or ndarray
        :math:`C(P) = (P/P_0)^{-\\beta}`.
    stability : float or ndarray
        :math:`R(s) = (1 + s/\\epsilon)^{-\\delta}`.
    """

    score: ArrayLike
    performance: ArrayLike
    data_efficiency: ArrayLike
    complexity_efficiency: ArrayLike
    stability: ArrayLike


def derms_score(
    mean_val_accuracy: ArrayLike,
    dataset_ratio: ArrayLike,
    n_parameters: ArrayLike,
    std_val_accuracy: ArrayLike,
    alpha: float = 2.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.5,
    epsilon: float = 0.001,
    p0: float = 1_000_000.0,
    return_components: bool = False,
) -> Union[ArrayLike, DERMSRResult]:
    """Compute the DERMS-R score.

    .. math::

        \\mathrm{DERMS\\text{-}R} =
        \\bar{A}_{\\mathrm{val}}^{\\alpha}
        \\cdot r^{-\\gamma}
        \\cdot (P/P_0)^{-\\beta}
        \\cdot \\left(1 + \\frac{\\sigma_{\\mathrm{val}}}{\\epsilon}\\right)^{-\\delta}

    This is the canonical, unnormalized score for a single candidate
    architecture evaluated at a single operating point (dataset ratio).
    To compare several candidates on a relative [0, 1] scale, divide each
    raw score by the maximum raw score in the candidate set (see
    :func:`normalize`).

    Parameters
    ----------
    mean_val_accuracy : float or ndarray of shape (n_candidates,)
        Mean validation accuracy across cross-validation folds, on the
        0-1 scale.
    dataset_ratio : float or ndarray of shape (n_candidates,)
        Fraction of the available training data used, in ``(0, 1]``.
    n_parameters : float or ndarray of shape (n_candidates,)
        Raw (unscaled) trainable parameter count.
    std_val_accuracy : float or ndarray of shape (n_candidates,)
        Sample standard deviation of validation accuracy across
        cross-validation folds (``ddof=1``), on the 0-1 scale.
    alpha : float, default=2.0
        Performance exponent.
    beta : float, default=0.5
        Complexity-efficiency exponent.
    gamma : float, default=0.5
        Data-efficiency exponent.
    delta : float, default=0.5
        Stability exponent.
    epsilon : float, default=0.001
        Stability reference scale (:math:`\\sigma_0` in the paper).
    p0 : float, default=1_000_000.0
        Complexity reference scale. Fixed by the canonical specification;
        only change it for explicit sensitivity analysis.
    return_components : bool, default=False
        If True, return a :class:`DERMSRResult` with the score and each
        of the four components, instead of just the raw score.

    Returns
    -------
    float, ndarray, or DERMSRResult
        The raw DERMS-R score (or scores, for array inputs), or a
        :class:`DERMSRResult` if ``return_components=True``.

    Examples
    --------
    >>> from dermsr import derms_score
    >>> round(derms_score(
    ...     mean_val_accuracy=0.89924,
    ...     dataset_ratio=0.10,
    ...     n_parameters=2_141_818,
    ...     std_val_accuracy=0.0034544174617437214,
    ... ), 4)
    0.8279
    """
    q = performance(mean_val_accuracy, alpha=alpha)
    d = data_efficiency(dataset_ratio, gamma=gamma)
    c = complexity_efficiency(n_parameters, beta=beta, p0=p0)
    r = stability(std_val_accuracy, delta=delta, epsilon=epsilon)
    score = q * d * c * r

    if not return_components:
        return score
    return DERMSRResult(
        score=score, performance=q, data_efficiency=d, complexity_efficiency=c, stability=r
    )


def normalize(scores: ArrayLike) -> ArrayLike:
    """Max-normalize raw DERMS-R scores within a candidate set.

    .. math::

        \\mathrm{DERMS\\text{-}R}^{*}_i = \\frac{\\mathrm{DERMS\\text{-}R}_i}{\\max_j \\mathrm{DERMS\\text{-}R}_j}

    Parameters
    ----------
    scores : array-like of shape (n_candidates,)
        Raw DERMS-R scores for a defined candidate set.

    Returns
    -------
    ndarray of shape (n_candidates,)
        Normalized scores in ``(0, 1]``, where 1 marks the highest-scoring
        candidate in the set. Not an absolute or dataset-independent
        efficiency measure -- it is only meaningful relative to the
        candidate set it was computed from.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-D array-like of raw DERMS-R scores.")
    if np.any(scores <= 0):
        raise ValueError("scores must be strictly positive.")
    return scores / scores.max()
