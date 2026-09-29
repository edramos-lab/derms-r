"""DERMS-R: Data-Efficient Robust Model Score."""
from .config import ALPHA, BETA, DELTA, FOLDS, GAMMA, P0, RATIOS, SCHEMA, SIGMA0
from .metric import derms_r, trainable_parameter_count
from .analysis import select_operating_point

__version__ = "3.2.0"
__all__ = [
    'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'P0', 'SIGMA0', 'RATIOS', 'FOLDS', 'SCHEMA',
    'derms_r', 'trainable_parameter_count', 'select_operating_point',
]
