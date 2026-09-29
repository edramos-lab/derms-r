"""Within-ratio DERMS-R normalization. Raw scores are never overwritten."""
from __future__ import annotations
import pandas as pd


def normalize_within_ratio(scores_by_architecture):
    """Map architecture -> DERMS-R^Norm at one fixed dataset ratio.

    DERMS-R^Norm_{i,r} = DERMS-R_{i,r} / max_j DERMS-R_{j,r}.
    A value of 1.0 is highest in the comparison set, not 100% accuracy.
    """
    if not scores_by_architecture:
        raise ValueError('No scores to normalize.')
    peak = max(scores_by_architecture.values())
    if peak <= 0:
        return {key: 0.0 for key in scores_by_architecture}
    return {key: value / peak for key, value in scores_by_architecture.items()}


def add_normalized_column(df, score_col='derms_r_raw', ratio_col='data_ratio',
                          dataset_col='dataset', out_col='derms_r_normalized'):
    """Add a within-(dataset, ratio) normalized column; leave raw scores intact."""
    if out_col == score_col:
        raise ValueError('Normalized scores must not overwrite raw DERMS-R.')
    frame = df.copy()
    keys = [column for column in (dataset_col, ratio_col) if column in frame.columns]
    if not keys:
        raise ValueError('data_ratio (and dataset, when present) are required for normalization.')
    frame[out_col] = frame.groupby(keys, sort=False)[score_col].transform(
        lambda values: values / values.max() if len(values) and values.max() > 0 else 0.0)
    return frame
