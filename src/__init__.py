"""Physics-constrained hybrid model package for PV performance ratio prediction."""

from .data import load_and_preprocess_data, get_split, HGB_FEATS, COLS
from .models import HybridPV, MLPBaseline, Norm
from .training import train, predict, violation_rate, score, read_params
from .visualization import (
    plot_age_extrapolation,
    plot_monotonicity_violations,
    plot_residual_bounded_correction,
    plot_anomaly_detection_curves,
    plot_learned_degradation_rates
)

__all__ = ['load_and_preprocess_data',
            'get_split',
            'HGB_FEATS',
            'COLS',
            'HybridPV',
            'MLPBaseline',
            'Norm',
            'train',
            'predict',
            'violation_rate',
            'score',
            'read_params',
            'plot_age_extrapolation',
            'plot_monotonicity_violations',
            'plot_residual_bounded_correction',
            'plot_anomaly_detection_curves',
            'plot_learned_degradation_rates']