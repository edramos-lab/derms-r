"""A scikit-learn-flavored scorer around :func:`dermsr.metric.derms_score`.

``DERMSRScorer`` plays the same role as the callables returned by
``sklearn.metrics.make_scorer``: a configured, reusable object that turns
raw cross-validation fold scores into a single number. It cannot be a
drop-in ``sklearn`` scorer in the strict ``scorer(estimator, X, y)`` sense,
because DERMS-R also needs the candidate's parameter count and the dataset
ratio it was trained on -- neither of which scikit-learn's scoring API
exposes. Those are therefore bound at construction time, and the scorer is
then called with the per-fold accuracies produced by, e.g.,
``sklearn.model_selection.cross_val_score``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .metric import DERMSRResult, derms_score

__all__ = ["DERMSRScorer"]


class DERMSRScorer:
    """Reusable DERMS-R scorer bound to one candidate's fixed properties.

    Parameters
    ----------
    n_parameters : float
        Raw (unscaled) trainable parameter count of the candidate model.
    dataset_ratio : float
        Fraction of the available training data this candidate was
        trained on, in ``(0, 1]``.
    alpha, beta, gamma, delta, epsilon, p0 : float
        DERMS-R hyperparameters, passed through to
        :func:`dermsr.metric.derms_score`.

    Examples
    --------
    >>> from dermsr import DERMSRScorer
    >>> scorer = DERMSRScorer(n_parameters=2_141_818, dataset_ratio=0.10)
    >>> fold_accuracies = [0.9002, 0.8940, 0.9020, 0.8977, 0.9023]  # e.g. from
    ...                                                              # sklearn.model_selection.cross_val_score
    >>> round(scorer(fold_accuracies), 4)
    0.8279
    """

    def __init__(
        self,
        n_parameters: float,
        dataset_ratio: float,
        alpha: float = 2.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.5,
        epsilon: float = 0.001,
        p0: float = 1_000_000.0,
    ) -> None:
        self.n_parameters = n_parameters
        self.dataset_ratio = dataset_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.p0 = p0

    def __call__(self, fold_accuracies: Sequence[float], return_components: bool = False):
        """Score a set of cross-validation fold accuracies.

        Parameters
        ----------
        fold_accuracies : sequence of float, length >= 2
            Per-fold validation accuracy, on the 0-1 scale.
        return_components : bool, default=False
            If True, return a :class:`dermsr.metric.DERMSRResult` instead
            of just the raw score.

        Returns
        -------
        float or DERMSRResult
        """
        fold_accuracies = np.asarray(fold_accuracies, dtype=np.float64)
        if fold_accuracies.ndim != 1 or fold_accuracies.size < 2:
            raise ValueError("fold_accuracies must be a 1-D sequence with at least 2 folds.")
        mean_accuracy = float(fold_accuracies.mean())
        std_accuracy = float(fold_accuracies.std(ddof=1))
        return derms_score(
            mean_val_accuracy=mean_accuracy,
            dataset_ratio=self.dataset_ratio,
            n_parameters=self.n_parameters,
            std_val_accuracy=std_accuracy,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            epsilon=self.epsilon,
            p0=self.p0,
            return_components=return_components,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DERMSRScorer(n_parameters={self.n_parameters!r}, "
            f"dataset_ratio={self.dataset_ratio!r}, alpha={self.alpha!r}, "
            f"beta={self.beta!r}, gamma={self.gamma!r}, delta={self.delta!r}, "
            f"epsilon={self.epsilon!r}, p0={self.p0!r})"
        )
