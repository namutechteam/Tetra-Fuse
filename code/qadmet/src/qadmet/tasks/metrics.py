"""task → metric function mapping.

TDC ``group.evaluate_many()`` directly accepts metric strings, but for OOF stacking/local
sanity checks the same metric must be computable independently. This module only provides
a simple sklearn-based implementation mapping.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Lazy import — pay the sklearn metrics module import cost only once.
_metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}


def _lazy_init() -> None:
    if _metric_fns:
        return
    from sklearn.metrics import (
        average_precision_score,
        mean_absolute_error,
        roc_auc_score,
    )
    from scipy.stats import spearmanr, pearsonr

    def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho)

    def _pearson(y_true, y_pred):
        r, _ = pearsonr(y_true, y_pred)
        return float(r)

    def _rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Relative Absolute Error: OpenADMET PXR challenge primary metric.
        # mean|y-pred| / mean|y - mean(y)|. Trivial mean-predictor scores 1.0.
        denom = float(np.mean(np.abs(y_true - y_true.mean()))) + 1e-12
        return float(np.mean(np.abs(y_pred - y_true)) / denom)

    _metric_fns.update({
        "mae": lambda y, p: float(mean_absolute_error(y, p)),
        "auroc": lambda y, p: float(roc_auc_score(y, p)),
        "auprc": lambda y, p: float(average_precision_score(y, p)),
        "spearman": _spearman,
        "rae": _rae,
        "pearson": _pearson,
    })


def metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """metric name → callable. Unsupported key raises KeyError."""
    _lazy_init()
    if name not in _metric_fns:
        raise KeyError(f"Unsupported metric: {name!r}. Supported: {sorted(_metric_fns)}")
    return _metric_fns[name]


def compute_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute metric after NaN masking. Returns NaN if there are 0 valid samples."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    return metric_fn(name)(y_true[mask], y_pred[mask])
