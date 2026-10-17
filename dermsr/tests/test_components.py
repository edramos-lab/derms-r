import math

import pytest

from dermsr._components import complexity_efficiency, data_efficiency, performance, stability


def test_performance_basic():
    assert performance(1.0, alpha=2.0) == 1.0
    assert performance(0.5, alpha=2.0) == pytest.approx(0.25)


def test_performance_rejects_out_of_range():
    with pytest.raises(ValueError):
        performance(1.1)
    with pytest.raises(ValueError):
        performance(-0.1)


def test_data_efficiency_basic():
    assert data_efficiency(1.0, gamma=0.5) == pytest.approx(1.0)
    assert data_efficiency(0.25, gamma=0.5) == pytest.approx(2.0)


def test_data_efficiency_rejects_out_of_range():
    with pytest.raises(ValueError):
        data_efficiency(0.0)
    with pytest.raises(ValueError):
        data_efficiency(1.5)


def test_complexity_efficiency_basic():
    assert complexity_efficiency(1_000_000.0, beta=0.5, p0=1_000_000.0) == pytest.approx(1.0)
    assert complexity_efficiency(4_000_000.0, beta=0.5, p0=1_000_000.0) == pytest.approx(0.5)


def test_complexity_efficiency_rejects_nonpositive():
    with pytest.raises(ValueError):
        complexity_efficiency(0.0)
    with pytest.raises(ValueError):
        complexity_efficiency(-1.0)


def test_stability_limiting_behavior():
    # sigma -> 0 means no stability penalty.
    assert stability(0.0, delta=0.5, epsilon=0.001) == pytest.approx(1.0)
    # Larger dispersion always penalizes more.
    assert stability(0.01, delta=0.5, epsilon=0.001) < stability(0.001, delta=0.5, epsilon=0.001)


def test_stability_rejects_negative():
    with pytest.raises(ValueError):
        stability(-0.001)


def test_exponents_must_be_finite_positive():
    with pytest.raises(ValueError):
        performance(0.5, alpha=0.0)
    with pytest.raises(ValueError):
        data_efficiency(0.5, gamma=-1.0)
    with pytest.raises(ValueError):
        complexity_efficiency(1_000_000.0, beta=float("nan"))
    with pytest.raises(ValueError):
        stability(0.5, delta=math.inf)
