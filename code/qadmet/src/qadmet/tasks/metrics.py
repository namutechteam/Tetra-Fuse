"""task → metric 함수 매핑.

TDC ``group.evaluate_many()`` 가 metric 문자열을 직접 받지만, OOF 스택킹·로컬
sanity check 시 동일 메트릭을 자체 계산할 수 있어야 한다. 본 모듈은 단순한
sklearn 기반 구현체 매핑만 제공한다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# 지연 import — sklearn metrics 모듈 자체 import 비용을 한 번만.
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
    """metric 이름 → callable. 미지원 키는 KeyError."""
    _lazy_init()
    if name not in _metric_fns:
        raise KeyError(f"미지원 metric: {name!r}. 지원: {sorted(_metric_fns)}")
    return _metric_fns[name]


def compute_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NaN 마스킹 후 메트릭 계산. 유효 샘플 0 개면 NaN 반환."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    return metric_fn(name)(y_true[mask], y_pred[mask])
