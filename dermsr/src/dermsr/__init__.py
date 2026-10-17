"""dermsr: a reference implementation of the DERMS-R model-selection score.

DERMS-R ("Data-Efficient Robust Model Score with Ratio-based data
efficiency") jointly scores a model candidate's predictive performance,
training-data efficiency, parameter complexity, and cross-validation
stability. See the companion paper, "DERMS-R: A Data-Efficient Robust
Model Score for Multi-Objective Evaluation of Deep Learning
Architectures" (Ramos, 2026), Section 16 ("Reproducibility and Reference
Implementation"), which this package implements.

Quick start
-----------
>>> from dermsr import derms_score
>>> derms_score(
...     mean_val_accuracy=0.89924,
...     dataset_ratio=0.10,
...     n_parameters=2_141_818,
...     std_val_accuracy=0.0034544174617437214,
... )  # doctest: +ELLIPSIS
0.82787...
"""
from ._components import complexity_efficiency, data_efficiency, performance, stability
from .metric import DERMSRResult, derms_score, normalize
from .scorer import DERMSRScorer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "derms_score",
    "normalize",
    "DERMSRResult",
    "DERMSRScorer",
    "performance",
    "data_efficiency",
    "complexity_efficiency",
    "stability",
]
