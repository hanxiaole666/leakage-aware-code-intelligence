"""评估层：检测指标、可信校准、统计显著性。"""
from .metrics import compute_metrics, metrics_table
from .calibration import (
    ece, brier, auroc, aurs, risk_coverage, risk_coverage_full, temperature_scale,
    conformal_accept, ece_adaptive, isotonic_calibrate,
)
from .stats import paired_bootstrap, mcnemar, holm_bonferroni, mean_std, summarize_seeds

__all__ = [
    "compute_metrics", "metrics_table", "ece", "brier", "auroc", "aurs", "risk_coverage",
    "temperature_scale", "conformal_accept", "risk_coverage_full", "ece_adaptive",
    "isotonic_calibrate", "paired_bootstrap", "mcnemar",
    "holm_bonferroni", "mean_std", "summarize_seeds",
]
