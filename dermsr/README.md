# dermsr

A reference implementation of **DERMS-R** (Data-Efficient Robust Model
Score with Ratio-based data efficiency), a multi-objective score for
neural architecture selection that jointly accounts for predictive
performance, training-data efficiency, parameter complexity, and
cross-validation stability.

This package implements the "Reproducibility and Reference
Implementation" section of the companion paper, *DERMS-R: A
Data-Efficient Robust Model Score for Multi-Objective Evaluation of Deep
Learning Architectures* (Ramos, 2026), and is built in the style of
`scikit-learn`'s `sklearn.metrics` module: small, well-documented,
input-validated functions rather than a stateful framework.

## Install

```bash
pip install -e .            # core package
pip install -e ".[stats]"   # + scipy, for exact Friedman p-values
pip install -e ".[dev]"     # + pytest, for running the test suite
```

## Quick start

```python
from dermsr import derms_score, DERMSRScorer

# Compute a single DERMS-R score directly:
score = derms_score(
    mean_val_accuracy=0.89924,
    dataset_ratio=0.10,
    n_parameters=2_141_818,
    std_val_accuracy=0.0034544174617437214,
)
print(round(score, 4))  # 0.8279

# Or use the scorer, which computes mean/std from raw CV fold accuracies
# (e.g. from sklearn.model_selection.cross_val_score):
scorer = DERMSRScorer(n_parameters=2_141_818, dataset_ratio=0.10)
score = scorer([0.9002, 0.8940, 0.9020, 0.8977, 0.9023])
```

To compare several candidate architectures on a common [0, 1] scale,
max-normalize their raw scores:

```python
from dermsr import normalize

raw_scores = [0.8279, 0.5528, 0.3569]
normalized = normalize(raw_scores)  # [1.0, 0.668, 0.431]
```

## The formula

```
DERMS-R = mean_val_accuracy^alpha
          * dataset_ratio^(-gamma)
          * (n_parameters / p0)^(-beta)
          * (1 + std_val_accuracy / epsilon)^(-delta)
```

with canonical defaults `alpha=2.0`, `beta=gamma=delta=0.5`,
`p0=1_000_000.0`, `epsilon=0.001`. See `dermsr.metric.derms_score` for
the full docstring, or Section 3 of the paper for the derivation.

## Bundled dataset

`dermsr.datasets.load_cifar10_benchmark()` returns the verified
12-architecture x 5-ratio CIFAR-10 benchmark results underlying the
paper's tables and figures (60 rows, 5-fold cross-validation each), so
the reproduction scripts and tests below are fully self-contained and
need no access to the original MLflow tracking server.

## Reproducing the manuscript

```bash
python scripts/reproduce_tables.py       # Tables 2, 3, 4, 6
python scripts/sensitivity_analysis.py   # Table 5
python scripts/statistical_analysis.py   # Friedman / Iman-Davenport / Nemenyi (Section 13.4)
```

Each script's output matches the corresponding table in the paper
exactly; the test suite (`tests/test_metric.py`,
`tests/test_statistics.py`, `tests/test_sensitivity.py`,
`tests/test_tables.py`) asserts this with the published numbers as
fixtures.

NetScore-style and ML-ModelExplorer-style (Theissler et al., 2020)
comparisons from the paper are not reproduced here, since those are
external metrics used only for comparison in the paper, not part of the
DERMS-R implementation itself.

## Package layout

```
src/dermsr/
    _components.py   performance, data_efficiency, complexity_efficiency, stability
    metric.py        derms_score(), normalize(), DERMSRResult
    scorer.py        DERMSRScorer, a reusable sklearn-flavored scorer
    sensitivity.py   the 5 canonical hyperparameter scenarios + rank comparison
    statistics.py    Friedman test, Iman-Davenport, Nemenyi critical difference
    tables.py        functions reproducing manuscript Tables 2-6
    datasets.py      load_cifar10_benchmark(), a bundled sample dataset
tests/               pytest suite; reproduces published numbers as fixtures
scripts/             runnable scripts reproducing the manuscript's tables/analyses
```

## Testing

```bash
pytest  # runs the unit tests AND the doctests in src/dermsr (see pyproject.toml)
```

## Citing

See `CITATION.cff`. The paper does not yet have a persistent DOI; update
`CITATION.cff` (and this README) once an archived release is made.

## License

MIT. See `LICENSE`.
