"""Non-parametric statistical comparison of DERMS-R rankings.

Implements the Friedman test, the Iman-Davenport correction, and the
Nemenyi post-hoc critical difference, following Demsar (2006) and
Garcia & Herrera (2008), as used in the paper's Statistical Analysis
section. These are general rank-comparison tools and are not specific
to DERMS-R: they operate on any (blocks x treatments) rank matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "FriedmanResult",
    "rank_matrix",
    "friedman_test",
    "nemenyi_critical_difference",
    "critical_pairs",
]

# Nemenyi two-tailed critical values q_alpha(0.05) for k = 2..20 treatments,
# from the standard table in Demsar (2006), Table 5(b).
_NEMENYI_Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


@dataclass(frozen=True)
class FriedmanResult:
    """Result of :func:`friedman_test`.

    Attributes
    ----------
    chi2_f : float
        The Friedman chi-squared statistic.
    df : int
        Degrees of freedom, ``k - 1``.
    k : int
        Number of treatments (e.g. architectures) compared.
    n : int
        Number of independent blocks (e.g. dataset ratios).
    f_f : float
        The Iman-Davenport F statistic, generally a less conservative
        approximation than ``chi2_f``.
    f_df1, f_df2 : int
        Degrees of freedom for ``f_f``: ``k - 1`` and ``(k - 1)(n - 1)``.
    p_value : float or None
        Exact chi-squared p-value for ``chi2_f``, if ``scipy`` is
        installed; otherwise None (compare ``chi2_f`` against a critical
        value table instead, e.g. ``scipy.stats.chi2.ppf(0.95, df)`` or a
        standard reference table).
    average_ranks : Series
        Mean rank per treatment across blocks (ascending = better, i.e.
        lower mean rank is a stronger treatment), sorted ascending.
    """

    chi2_f: float
    df: int
    k: int
    n: int
    f_f: float
    f_df1: int
    f_df2: int
    p_value: Optional[float]
    average_ranks: pd.Series


def rank_matrix(
    data: pd.DataFrame, block_col: str, treatment_col: str, value_col: str, ascending: bool = False
) -> pd.DataFrame:
    """Build a (treatments x blocks) rank matrix from long-format data.

    Parameters
    ----------
    data : DataFrame
        Long-format results with one row per (block, treatment) pair.
    block_col : str
        Column identifying the independent experimental block (e.g. a
        dataset ratio or dataset name). See the paper's "Important
        Statistical Unit" discussion: cross-validation folds must not be
        used as blocks here, since they are already absorbed into each
        treatment's mean/standard-deviation accuracy upstream.
    treatment_col : str
        Column identifying the treatment being compared (e.g. the
        architecture name).
    value_col : str
        Column to rank within each block (e.g. the raw DERMS-R score).
    ascending : bool, default=False
        If False (default), the highest value in each block receives
        rank 1 (appropriate for a score where higher is better, such as
        DERMS-R).

    Returns
    -------
    DataFrame
        Rows are treatments, columns are blocks, values are ranks.
    """
    pivot = data.pivot(index=treatment_col, columns=block_col, values=value_col)
    return pivot.rank(ascending=ascending, axis=0)


def friedman_test(ranks: pd.DataFrame) -> FriedmanResult:
    """Run the Friedman test and Iman-Davenport correction on a rank matrix.

    Parameters
    ----------
    ranks : DataFrame of shape (k treatments, n blocks)
        A rank matrix as produced by :func:`rank_matrix`.

    Returns
    -------
    FriedmanResult
    """
    k, n = ranks.shape[0], ranks.shape[1]
    if k < 2 or n < 2:
        raise ValueError("Friedman test requires at least 2 treatments and 2 blocks.")

    avg_rank = ranks.mean(axis=1).sort_values()
    chi2_f = (12 * n) / (k * (k + 1)) * ((avg_rank ** 2).sum() - k * (k + 1) ** 2 / 4)
    df = k - 1
    denom = n * (k - 1) - chi2_f
    f_f = ((n - 1) * chi2_f) / denom if denom != 0 else float("inf")
    f_df1, f_df2 = k - 1, (k - 1) * (n - 1)

    p_value = None
    try:
        from scipy import stats as _stats

        p_value = float(1 - _stats.chi2.cdf(chi2_f, df))
    except ImportError:
        pass

    return FriedmanResult(
        chi2_f=float(chi2_f),
        df=df,
        k=k,
        n=n,
        f_f=float(f_f),
        f_df1=f_df1,
        f_df2=f_df2,
        p_value=p_value,
        average_ranks=avg_rank,
    )


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi post-hoc critical difference for all-pairs comparison.

    .. math::

        CD = q_\\alpha \\sqrt{\\frac{k(k+1)}{6n}}

    Parameters
    ----------
    k : int
        Number of treatments compared.
    n : int
        Number of independent blocks.
    alpha : float, default=0.05
        Significance level. Only ``0.05`` is currently tabulated.

    Returns
    -------
    float
        The critical difference. Two treatments' average ranks are
        considered significantly different if they differ by more than
        this value.
    """
    if alpha != 0.05:
        raise NotImplementedError("Only alpha=0.05 is tabulated; pass a custom q_alpha manually if needed.")
    if k not in _NEMENYI_Q_ALPHA_05:
        raise ValueError(
            f"No tabulated q_alpha(0.05) for k={k} treatments; "
            f"supported range is {min(_NEMENYI_Q_ALPHA_05)}-{max(_NEMENYI_Q_ALPHA_05)}."
        )
    q_alpha = _NEMENYI_Q_ALPHA_05[k]
    return q_alpha * float(np.sqrt(k * (k + 1) / (6 * n)))


def critical_pairs(average_ranks: pd.Series, critical_difference: float) -> pd.DataFrame:
    """List all treatment pairs whose average ranks differ by more than CD.

    Parameters
    ----------
    average_ranks : Series
        Mean rank per treatment, as returned in
        :attr:`FriedmanResult.average_ranks`.
    critical_difference : float
        E.g. from :func:`nemenyi_critical_difference`.

    Returns
    -------
    DataFrame
        Columns ``treatment_a``, ``treatment_b``, ``rank_difference``,
        sorted by ``rank_difference`` descending. Only pairs exceeding
        ``critical_difference`` are included.
    """
    names = list(average_ranks.index)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = abs(average_ranks[names[i]] - average_ranks[names[j]])
            if diff > critical_difference:
                rows.append((names[i], names[j], diff))
    out = pd.DataFrame(rows, columns=["treatment_a", "treatment_b", "rank_difference"])
    return out.sort_values("rank_difference", ascending=False).reset_index(drop=True)
