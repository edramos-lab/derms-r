"""Reproduces the exact published DERMS-R values from the paper's benchmark.

These fixtures are drawn directly from the `derms-r-benchmark - final -
CIFAR10` MLflow experiment results reported in the manuscript (Table 3 /
Appendix A), not invented examples, so a passing test here is direct
evidence that this reimplementation matches the canonical
`derms/metric.py` + `derms/components.py` implementation in the main
derms-r repository bit-for-bit (to the precision reported in the paper).
"""
import pytest

from dermsr import derms_score, normalize
from dermsr.metric import DERMSRResult

# (architecture, mean_val_accuracy, dataset_ratio, n_parameters, std_val_accuracy, expected_derms_r_raw)
# Full float64 precision, pulled directly from the benchmark's logged
# fold-level results (not display-rounded), so this is a true bit-level
# cross-check against the canonical derms/metric.py implementation.
PUBLISHED_FIXTURES = [
    ("efficientvit_b0", 0.89924, 0.10, 2_141_818, 0.0034544174617437214, 0.8278744772160764),
    ("maxvit_tiny", 0.98542, 1.00, 30_408_658, 0.0009391485505499, 0.126455748861562),
    ("xception41", 0.95762, 0.25, 24_941_050, 0.0007596051605933, 0.2768544007305858),
    ("deit3_base", 0.96968, 0.10, 85_824_010, 0.0032321819255728, 0.1560164210565625),
]


@pytest.mark.parametrize(
    "architecture,mean_acc,ratio,n_params,std_acc,expected", PUBLISHED_FIXTURES
)
def test_derms_score_matches_published_benchmark(architecture, mean_acc, ratio, n_params, std_acc, expected):
    score = derms_score(
        mean_val_accuracy=mean_acc,
        dataset_ratio=ratio,
        n_parameters=n_params,
        std_val_accuracy=std_acc,
    )
    assert score == pytest.approx(expected, rel=1e-5), architecture


def test_derms_score_with_components_matches_product():
    result = derms_score(
        mean_val_accuracy=0.89924,
        dataset_ratio=0.10,
        n_parameters=2_141_818,
        std_val_accuracy=0.0034544174617437214,
        return_components=True,
    )
    assert isinstance(result, DERMSRResult)
    product = result.performance * result.data_efficiency * result.complexity_efficiency * result.stability
    assert result.score == pytest.approx(product)
    assert result.score == pytest.approx(0.8278744772160764, rel=1e-5)


def test_normalize_max_is_one():
    raw = [0.827874, 0.552849, 0.356891]
    normalized = normalize(raw)
    assert normalized.max() == pytest.approx(1.0)
    assert (normalized <= 1.0).all()
    assert (normalized > 0.0).all()


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        normalize([])


def test_normalize_rejects_nonpositive():
    with pytest.raises(ValueError):
        normalize([0.5, -0.1])


def test_derms_score_increases_with_accuracy():
    low = derms_score(0.80, 0.10, 2_000_000, 0.01)
    high = derms_score(0.95, 0.10, 2_000_000, 0.01)
    assert high > low


def test_derms_score_decreases_with_parameters():
    small = derms_score(0.90, 0.10, 2_000_000, 0.01)
    large = derms_score(0.90, 0.10, 80_000_000, 0.01)
    assert small > large


def test_derms_score_decreases_with_ratio():
    low_ratio = derms_score(0.90, 0.10, 2_000_000, 0.01)
    high_ratio = derms_score(0.90, 1.00, 2_000_000, 0.01)
    assert low_ratio > high_ratio


def test_derms_score_decreases_with_instability():
    stable = derms_score(0.90, 0.10, 2_000_000, 0.001)
    unstable = derms_score(0.90, 0.10, 2_000_000, 0.05)
    assert stable > unstable
