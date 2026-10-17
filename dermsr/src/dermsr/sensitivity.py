"""Sensitivity analysis across alternative DERMS-R hyperparameterizations.

Reproduces the scenario table and per-architecture rank comparison from
the paper's Sensitivity Analysis section.
"""
from __future__ import annotations

from typing import Dict, Mapping

import pandas as pd

from .metric import derms_score

__all__ = ["DEFAULT_SCENARIOS", "sensitivity_analysis"]

#: The five canonical sensitivity scenarios from the paper, each varying
#: (alpha, beta, gamma, delta) while holding epsilon and p0 fixed.
DEFAULT_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Baseline": dict(alpha=2.0, beta=0.5, gamma=0.5, delta=0.5),
    "Performance-heavy": dict(alpha=4.0, beta=0.25, gamma=0.25, delta=0.25),
    "Complexity-heavy": dict(alpha=2.0, beta=1.0, gamma=0.5, delta=0.5),
    "Data-heavy": dict(alpha=2.0, beta=0.5, gamma=1.0, delta=0.5),
    "Stability-heavy": dict(alpha=2.0, beta=0.5, gamma=0.5, delta=1.0),
}


def sensitivity_analysis(
    results: pd.DataFrame,
    scenarios: Mapping[str, Mapping[str, float]] = DEFAULT_SCENARIOS,
    epsilon: float = 0.001,
    p0: float = 1_000_000.0,
    ascending: bool = False,
) -> pd.DataFrame:
    """Recompute DERMS-R ranks under alternative hyperparameterizations.

    For each architecture, its operating point (accuracy, standard
    deviation, parameter count, dataset ratio) is held fixed while
    ``(alpha, beta, gamma, delta)`` are varied per *scenario*. This
    isolates the effect of the weighting hyperparameters from the effect
    of choosing a different operating point.

    Parameters
    ----------
    results : DataFrame
        Must contain the columns ``architecture``, ``mean_val_accuracy``,
        ``std_val_accuracy``, ``n_parameters``, and ``dataset_ratio`` --
        one row per architecture at a fixed, chosen operating point
        (e.g. each architecture's optimal ratio).
    scenarios : mapping of name -> {alpha, beta, gamma, delta}, optional
        Scenario definitions. Defaults to :data:`DEFAULT_SCENARIOS`.
    epsilon, p0 : float
        Held fixed across all scenarios, per the canonical specification.
    ascending : bool, default=False
        Passed to ``Series.rank``; DERMS-R scores are ranked with the
        highest score receiving rank 1 (``ascending=False``).

    Returns
    -------
    DataFrame
        Indexed by ``architecture``, one integer-valued rank column per
        scenario, sorted by the ``"Baseline"`` scenario's rank if present.
    """
    required = {"architecture", "mean_val_accuracy", "std_val_accuracy", "n_parameters", "dataset_ratio"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    ranks = {}
    for name, params in scenarios.items():
        scores = results.apply(
            lambda row: derms_score(
                mean_val_accuracy=row["mean_val_accuracy"],
                dataset_ratio=row["dataset_ratio"],
                n_parameters=row["n_parameters"],
                std_val_accuracy=row["std_val_accuracy"],
                epsilon=epsilon,
                p0=p0,
                **params,
            ),
            axis=1,
        )
        ranks[name] = scores.rank(ascending=ascending, method="min").astype(int)

    # Build with the default positional index first, then assign the
    # architecture labels directly (NOT via the `index=` constructor
    # kwarg): passing a Series with its own 0..n index there triggers
    # pandas' reindex-based alignment against each column's identical
    # 0..n index, which has no overlap with the architecture-name labels
    # and silently produces an all-NaN frame.
    out = pd.DataFrame(ranks)
    out.index = pd.Index(results["architecture"].to_numpy(), name="architecture")
    if "Baseline" in out.columns:
        out = out.sort_values("Baseline")
    return out
