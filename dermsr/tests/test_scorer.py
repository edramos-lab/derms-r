import pytest

from dermsr import DERMSRScorer
from dermsr.metric import DERMSRResult


def test_scorer_matches_manual_computation():
    scorer = DERMSRScorer(n_parameters=2_141_818, dataset_ratio=0.10)
    fold_accuracies = [0.9002, 0.8940, 0.9020, 0.8977, 0.9023]
    score = scorer(fold_accuracies)
    assert score == pytest.approx(0.827874, rel=1e-4)


def test_scorer_return_components():
    scorer = DERMSRScorer(n_parameters=2_141_818, dataset_ratio=0.10)
    fold_accuracies = [0.9002, 0.8940, 0.9020, 0.8977, 0.9023]
    result = scorer(fold_accuracies, return_components=True)
    assert isinstance(result, DERMSRResult)


def test_scorer_requires_at_least_two_folds():
    scorer = DERMSRScorer(n_parameters=2_141_818, dataset_ratio=0.10)
    with pytest.raises(ValueError):
        scorer([0.9])


def test_scorer_repr_contains_key_params():
    scorer = DERMSRScorer(n_parameters=2_141_818, dataset_ratio=0.10)
    text = repr(scorer)
    assert "2141818" in text
    assert "0.1" in text
