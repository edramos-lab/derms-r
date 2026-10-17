import pytest

from dermsr.datasets import load_cifar10_benchmark
from dermsr.tables import (
    table_ablation,
    table_derms_results,
    table_predictive_performance,
    table_rank_matrix,
)


@pytest.fixture
def benchmark():
    return load_cifar10_benchmark()


def test_table_predictive_performance_top_is_deit3(benchmark):
    table = table_predictive_performance(benchmark, ratio=1.00)
    assert len(table) == 12
    assert table.iloc[0]["architecture"] == "deit3_base"
    assert table.iloc[0]["mean_val_accuracy"] == pytest.approx(0.9856, abs=1e-3)


def test_table_predictive_performance_invalid_ratio_raises(benchmark):
    with pytest.raises(ValueError):
        table_predictive_performance(benchmark, ratio=0.42)


def test_table_derms_results_matches_published_ranking(benchmark):
    table = table_derms_results(benchmark)
    assert len(table) == 12
    assert table.iloc[0]["architecture"] == "efficientvit_b0"
    assert table.iloc[0]["derms_r_raw"] == pytest.approx(0.8279, abs=1e-3)
    assert table.iloc[0]["derms_r_normalized"] == pytest.approx(1.0)
    assert table.iloc[-1]["architecture"] == "deit3_base"


def test_table_rank_matrix_shape_and_best(benchmark):
    table = table_rank_matrix(benchmark)
    assert table.shape == (12, 6)  # 5 ratios + average_rank
    assert table.index[0] == "efficientvit_b0"  # lowest (best) average rank


def test_table_ablation_full_matches_derms_results(benchmark):
    derms_table = table_derms_results(benchmark)
    ablation_table = table_ablation(benchmark)
    merged = derms_table.merge(ablation_table, on="architecture", suffixes=("", "_ablation"))
    assert (merged["derms_r_raw"] - merged["Full_DERMS_R"]).abs().max() < 1e-9
