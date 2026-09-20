"""DERMS-R: fractional accuracy/SD, millions of trainable parameters."""
import math

def components(a, p_m, r, sigma, cfg):
    if not all(math.isfinite(v) for v in (a, p_m, r, sigma)):
        raise ValueError('DERMS-R inputs must be finite.')
    if not all(math.isfinite(cfg[k]) and cfg[k] >= 0 for k in ('alpha', 'beta', 'gamma', 'delta')) or not math.isfinite(cfg['epsilon']) or cfg['epsilon'] <= 0:
        raise ValueError('Exponents must be finite/nonnegative and epsilon positive.')
    if not (0 <= a <= 1 and p_m > 0 and 0 < r <= 1 and sigma >= 0):
        raise ValueError('Invalid DERMS-R inputs.')
    accuracy = a ** cfg['alpha']
    complexity = p_m ** -cfg['beta']
    data = r ** -cfg['gamma']
    stability = (1 + sigma / cfg['epsilon']) ** -cfg['delta']
    return {'accuracy_component': accuracy, 'complexity_component': complexity,
            'data_component': data, 'stability_component': stability,
            'derms_raw': accuracy * complexity * data * stability}

