"""Functions reproducing the manuscript's DERMS-R tables from raw results.

Each function takes the long-format benchmark DataFrame (as returned by
:func:`dermsr.datasets.load_cifar10_benchmark`, or any DataFrame with the
same columns) and returns a DataFrame matching one table in the paper.
"""
from __future__ import annotations

import pandas as pd

from .metric import derms_score, normalize

__all__ = [
    "table_predictive_performance",
    "table_derms_results",
    "table_rank_matrix",
    "table_ablation",
]


def table_predictive_performance(results: pd.DataFrame, ratio: float = 1.00) -> pd.DataFrame:
    """Reproduce the manuscript's predictive-performance table (Table 2).

    Parameters
    ----------
    results : DataFrame
        Must contain ``architecture``, ``dataset_ratio``,
        ``mean_val_accuracy``, ``f1_macro``, ``mcc``, ``n_parameters``,
        and optionally ``latency_ms``.
    ratio : float, default=1.00
        The dataset ratio (operating point) to report, e.g. the full-data
        point used in the paper for a fair, data-efficiency-independent
        comparison.

    Returns
    -------
    DataFrame
        One row per architecture at the requested ratio, sorted by mean
        validation accuracy descending.
    """
    subset = results[results["dataset_ratio"] == ratio].copy()
    if subset.empty:
        raise ValueError(f"No rows found at dataset_ratio={ratio!r}.")
    cols = ["architecture", "mean_val_accuracy", "f1_macro", "mcc", "n_parameters"]
    if "latency_ms" in subset.columns:
        cols.append("latency_ms")
    return subset[cols].sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)


def table_derms_results(
    results: pd.DataFrame,
    alpha: float = 2.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.5,
    epsilon: float = 0.001,
    p0: float = 1_000_000.0,
) -> pd.DataFrame:
    """Reproduce the manuscript's DERMS-R decomposition table (Table 3).

    For each architecture, selects its optimal ratio
    :math:`r^{*} = \\arg\\max_r \\mathrm{DERMS\\text{-}R}_{i,r}` over all
    ratios present in ``results``, and reports the score decomposition
    and max-normalized score at that point.

    Parameters
    ----------
    results : DataFrame
        Must contain ``architecture``, ``dataset_ratio``,
        ``mean_val_accuracy``, ``std_val_accuracy``, ``n_parameters``.
    alpha, beta, gamma, delta, epsilon, p0 : float
        DERMS-R hyperparameters.

    Returns
    -------
    DataFrame
        One row per architecture at its optimal ratio, sorted by raw
        DERMS-R score descending, with columns ``architecture``,
        ``optimal_ratio``, ``data_efficiency``, ``complexity_efficiency``,
        ``stability``, ``derms_r_raw``, ``derms_r_normalized``.
    """
    rows = []
    for arch, group in results.groupby("architecture"):
        best = None
        for _, row in group.iterrows():
            decomposed = derms_score(
                mean_val_accuracy=row["mean_val_accuracy"],
                dataset_ratio=row["dataset_ratio"],
                n_parameters=row["n_parameters"],
                std_val_accuracy=row["std_val_accuracy"],
                alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, p0=p0,
                return_components=True,
            )
            if best is None or decomposed.score > best[1].score:
                best = (row["dataset_ratio"], decomposed)
        ratio_star, result = best
        rows.append({
            "architecture": arch,
            "optimal_ratio": ratio_star,
            "data_efficiency": result.data_efficiency,
            "complexity_efficiency": result.complexity_efficiency,
            "stability": result.stability,
            "derms_r_raw": result.score,
        })

    out = pd.DataFrame(rows).sort_values("derms_r_raw", ascending=False).reset_index(drop=True)
    out["derms_r_normalized"] = normalize(out["derms_r_raw"].to_numpy())
    return out


def table_rank_matrix(
    results: pd.DataFrame,
    alpha: float = 2.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.5,
    epsilon: float = 0.001,
    p0: float = 1_000_000.0,
) -> pd.DataFrame:
    """Reproduce the manuscript's rank matrix (Table 4): architecture x ratio.

    Each cell is the architecture's DERMS-R rank within that ratio block
    (1 = best), consistent with the block definition used for the
    Friedman test in :mod:`dermsr.statistics` (blocks = dataset ratios).

    Returns
    -------
    DataFrame
        Index is ``architecture``, columns are the ratios present in
        ``results``, plus an ``average_rank`` column, sorted ascending
        by average rank.
    """
    scored = results.copy()
    scored["derms_r_raw"] = scored.apply(
        lambda row: derms_score(
            mean_val_accuracy=row["mean_val_accuracy"],
            dataset_ratio=row["dataset_ratio"],
            n_parameters=row["n_parameters"],
            std_val_accuracy=row["std_val_accuracy"],
            alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, p0=p0,
        ),
        axis=1,
    )
    pivot = scored.pivot(index="architecture", columns="dataset_ratio", values="derms_r_raw")
    ranked = pivot.rank(ascending=False, axis=0)
    ranked["average_rank"] = ranked.mean(axis=1)
    return ranked.sort_values("average_rank")


def table_ablation(
    results: pd.DataFrame,
    alpha: float = 2.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.5,
    epsilon: float = 0.001,
    p0: float = 1_000_000.0,
) -> pd.DataFrame:
    """Reproduce the manuscript's component-ablation table (Table 6).

    Evaluated at each architecture's optimal ratio (as in
    :func:`table_derms_results`), reports four nested formulations:
    performance+complexity only (``D_AP``), adding data efficiency
    (``D_APr``), adding stability instead (``D_APsigma``), and the full
    DERMS-R score, which by construction equals ``D_AP * data_efficiency
    * stability``.

    Returns
    -------
    DataFrame
        One row per architecture, sorted by ``derms_r_raw`` descending.
    """
    full = table_derms_results(
        results, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, p0=p0
    )
    # Recover the performance component implied by the full score and the
    # other three components, to avoid recomputing it from scratch.
    full["performance"] = full["derms_r_raw"] / (
        full["data_efficiency"] * full["complexity_efficiency"] * full["stability"]
    )
    full["D_AP"] = full["performance"] * full["complexity_efficiency"]
    full["D_APr"] = full["D_AP"] * full["data_efficiency"]
    full["D_APsigma"] = full["D_AP"] * full["stability"]
    cols = ["architecture", "optimal_ratio", "D_AP", "D_APr", "D_APsigma", "derms_r_raw"]
    return full[cols].rename(columns={"derms_r_raw": "Full_DERMS_R"})
