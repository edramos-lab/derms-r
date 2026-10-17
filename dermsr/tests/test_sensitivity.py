import pandas as pd
import pytest

from dermsr.datasets import load_cifar10_benchmark
from dermsr.sensitivity import DEFAULT_SCENARIOS, sensitivity_analysis
from dermsr.tables import table_derms_results


@pytest.fixture
def optimal_operating_points():
    benchmark = load_cifar10_benchmark()
    derms_table = table_derms_results(benchmark)
    # join back accuracy/std/params at the optimal ratio for each architecture
    merged = derms_table.merge(
        benchmark,
        left_on=["architecture", "optimal_ratio"],
        right_on=["architecture", "dataset_ratio"],
        suffixes=("", "_bench"),
    )
    return merged[["architecture", "mean_val_accuracy", "std_val_accuracy", "n_parameters", "dataset_ratio"]]


def test_default_scenarios_has_five_entries():
    assert len(DEFAULT_SCENARIOS) == 5
    assert set(DEFAULT_SCENARIOS) == {
        "Baseline", "Performance-heavy", "Complexity-heavy", "Data-heavy", "Stability-heavy",
    }


def test_sensitivity_analysis_matches_published_ranks(optimal_operating_points):
    ranks = sensitivity_analysis(optimal_operating_points)
    assert ranks.loc["efficientvit_b0", "Baseline"] == 1
    assert ranks.loc["deit3_base", "Baseline"] == 12
    # Rank 1 under every scenario (published finding: EfficientViT-B0 never
    # drops from rank 1 across all five reweightings).
    assert (ranks.loc["efficientvit_b0"] == 1).all()


def test_sensitivity_analysis_requires_columns():
    with pytest.raises(ValueError):
        sensitivity_analysis(pd.DataFrame({"architecture": ["a"]}))
