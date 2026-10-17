import pandas as pd
import pytest

from dermsr import derms_score
from dermsr.datasets import load_cifar10_benchmark
from dermsr.statistics import (
    critical_pairs,
    friedman_test,
    nemenyi_critical_difference,
    rank_matrix,
)
from dermsr.tables import table_rank_matrix


@pytest.fixture
def ranks():
    benchmark = load_cifar10_benchmark()
    # table_rank_matrix includes an "average_rank" column; drop it for the
    # raw (treatments x blocks) matrix expected by friedman_test.
    full = table_rank_matrix(benchmark)
    return full.drop(columns="average_rank")


def test_rank_matrix_shape(ranks):
    assert ranks.shape == (12, 5)


def test_rank_matrix_from_long_format_matches_table_rank_matrix(ranks):
    benchmark = load_cifar10_benchmark()
    benchmark = benchmark.assign(
        derms_r_raw=benchmark.apply(
            lambda row: derms_score(
                row["mean_val_accuracy"], row["dataset_ratio"], row["n_parameters"], row["std_val_accuracy"]
            ),
            axis=1,
        )
    )
    direct = rank_matrix(
        benchmark, block_col="dataset_ratio", treatment_col="architecture", value_col="derms_r_raw"
    )
    # `ranks` (from table_rank_matrix) is sorted by average rank, while
    # `direct` (from rank_matrix) keeps the pivot's alphabetical order;
    # sort both axes on both frames so only the values are compared.
    # check_column_type=False: `ranks`'s column Index was coerced from
    # float64 to object when the (now-dropped) string-named
    # "average_rank" column was added to it, which is a cosmetic pandas
    # quirk unrelated to the actual rank values being compared here.
    pd.testing.assert_frame_equal(
        direct.sort_index(axis=0).sort_index(axis=1),
        ranks.sort_index(axis=0).sort_index(axis=1),
        check_names=False,
        check_column_type=False,
    )


def test_friedman_matches_published_statistic(ranks):
    result = friedman_test(ranks)
    assert result.k == 12
    assert result.n == 5
    assert result.df == 11
    # Published: chi2_F = 47.95, F_F = 27.22 (manuscript Section 13.4).
    assert result.chi2_f == pytest.approx(47.95, abs=0.01)
    assert result.f_f == pytest.approx(27.22, abs=0.01)
    assert result.f_df1 == 11
    assert result.f_df2 == 44


def test_friedman_requires_min_shape():
    with pytest.raises(ValueError):
        friedman_test(pd.DataFrame({"block_1": [1]}))  # 1 treatment, 1 block


def test_nemenyi_critical_difference_matches_published_value():
    cd = nemenyi_critical_difference(k=12, n=5)
    assert cd == pytest.approx(7.4522, abs=0.001)


def test_nemenyi_unsupported_k_raises():
    with pytest.raises(ValueError):
        nemenyi_critical_difference(k=1000, n=5)


def test_critical_pairs_matches_published_count(ranks):
    result = friedman_test(ranks)
    cd = nemenyi_critical_difference(k=result.k, n=result.n)
    pairs = critical_pairs(result.average_ranks, cd)
    # Published: 8 of 66 pairs exceed the critical difference.
    assert len(pairs) == 8
