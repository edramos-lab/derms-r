from dermsr.datasets import load_cifar10_benchmark


def test_load_cifar10_benchmark_shape():
    df = load_cifar10_benchmark()
    assert df.shape == (60, 18)


def test_load_cifar10_benchmark_architectures():
    df = load_cifar10_benchmark()
    assert df["architecture"].nunique() == 12
    assert set(df.groupby("architecture").size().unique()) == {5}  # 5 ratios each


def test_load_cifar10_benchmark_ratios():
    df = load_cifar10_benchmark()
    assert sorted(df["dataset_ratio"].unique()) == [0.10, 0.25, 0.50, 0.75, 1.00]
