"""Operating-point selection, ablations, and sensitivity analysis.

Only the complete four-component equation is canonical DERMS-R.
Ablation and sensitivity variants are diagnostic and must not be labeled canonical.
"""
from __future__ import annotations
from .config import ALPHA, BETA, GAMMA, DELTA, P0, SIGMA0, RATIOS
from .metric import derms_r

# Named ablations: exponents set to 0 remove that multiplicative term (x^0 = 1).
ABLATIONS = {
    'performance_complexity': {'gamma': 0.0, 'delta': 0.0},
    'performance_complexity_data': {'delta': 0.0},
    'performance_complexity_stability': {'gamma': 0.0},
}

SENSITIVITY = {
    'canonical': dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),
    'performance_heavy': dict(alpha=4.0, beta=0.25, gamma=0.25, delta=0.25),
    'complexity_heavy': dict(alpha=2.0, beta=1.0, gamma=0.5, delta=0.5),
    'data_efficiency_heavy': dict(alpha=2.0, beta=0.5, gamma=1.0, delta=0.5),
    'stability_heavy': dict(alpha=2.0, beta=0.5, gamma=0.5, delta=1.0),
}


def fold_accuracies_from_row(row):
    n = int(row.number_of_folds)
    return [float(getattr(row, f'fold_{k}_accuracy')) for k in range(1, n + 1)]


def select_operating_point(scores_by_ratio):
    """r* = argmax_r DERMS-R(i,r) and DERMS-R*(i) = max_r DERMS-R(i,r).

    Only experimentally evaluated ratios may be supplied; missing ratios are not interpolated.
    Ties keep the smaller ratio (more data-efficient operating point).
    """
    if not scores_by_ratio:
        raise ValueError('No operating points to select.')
    for ratio in scores_by_ratio:
        if not 0.0 < float(ratio) <= 1.0:
            raise ValueError('data_ratio must satisfy 0 < r <= 1.')
    best_ratio = max(scores_by_ratio, key=lambda ratio: (scores_by_ratio[ratio], -float(ratio)))
    return float(best_ratio), float(scores_by_ratio[best_ratio])


def attach_architecture_optima(df, architecture_col='architecture', ratio_col='data_ratio',
                               score_col='derms_r_raw', dataset_col='dataset'):
    frame = df.copy()
    group_cols = [column for column in (dataset_col, architecture_col) if column in frame.columns]
    frame['optimal_ratio'] = float('nan')
    frame['optimal_derms_r'] = float('nan')
    for _, group in frame.groupby(group_cols, sort=False):
        scores = {float(ratio): float(score) for ratio, score in zip(group[ratio_col], group[score_col])}
        ratio_star, score_star = select_operating_point(scores)
        frame.loc[group.index, 'optimal_ratio'] = ratio_star
        frame.loc[group.index, 'optimal_derms_r'] = score_star
    return frame


def score_with_overrides(row, overrides, base=None):
    params = dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA, p0=P0, sigma0=SIGMA0)
    if base:
        params.update(base)
    params.update(overrides)
    return derms_r(
        fold_accuracies_from_row(row),
        int(row.num_parameters),
        float(row.data_ratio),
        **params,
    )


def canonical_ratios_only(ratios=RATIOS):
    if tuple(float(r) for r in ratios) != tuple(float(r) for r in RATIOS):
        raise ValueError('Canonical DERMS-R operating points are {0.10, 0.25, 0.50, 0.75, 1.00}.')
    return tuple(RATIOS)
