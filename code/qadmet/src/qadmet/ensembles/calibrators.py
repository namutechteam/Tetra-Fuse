"""``Calibrator`` ABC + Platt / Isotonic implementation.

This module provides tools for calibrating prob predictions of classification tasks. The old ``run_stacking_v6_hybrid``'s
v6 hybrid blend stage required aligning base output distributions, and the new ``HybridBlendStrategy``
and future stacking pipelines (per-base prob calibration) use it.

Status (Step 4.6):
    - ``Calibrator`` ABC + ``CALIBRATORS`` registry.
    - ``PlattCalibrator`` (sklearn ``LogisticRegression`` 1D) — sigmoid calibration.
    - ``IsotonicCalibrator`` (sklearn ``IsotonicRegression``) — monotonic piecewise constant.
    - sklearn is lazy-imported (not registered in sys.modules on ``import qadmet.ensembles``).

Usage pattern:
    >>> cal = CALIBRATORS.build_with_kwargs("platt")
    >>> cal.fit(probs_val, y_val)
    >>> probs_test_calibrated = cal.transform(probs_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from qadmet.ensembles.registry import CALIBRATORS


class Calibrator(ABC):
    """Common interface for classification prob calibrators.

    Why:
        v6 hybrid blend and stacking pipelines need to align the fact that prob distributions
        differ per base (sigmoid output / soft-label / learned prob). Regression tasks
        do not require calibration, so this ABC assumes classification-only usage.

    Implementation responsibilities:
        - ``fit(p, y)``: train on 1D prob array ``p`` (values ∈ [0, 1]) + 0/1 labels ``y``.
        - ``transform(p)``: convert input prob to calibrated prob (output ∈ [0, 1]).

    Common validation:
        - ``p``, ``y`` must be 1D + matching length.
        - ``y`` must be 0/1 integer (or castable to integer).
        - ``p`` must be in [0, 1] range (NaN allowed — finite mask at fit, NaN passthrough at transform).
    """

    @abstractmethod
    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        ...

    @abstractmethod
    def transform(self, p: np.ndarray) -> np.ndarray:
        ...


def _validate_fit_inputs(
    p: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``(p, y)`` to 1D ndarray + validate shape/values.

    Returns:
        ``(p_arr, y_arr)`` before finite mask is applied. NaN handling is the caller's responsibility.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    y_arr = np.asarray(y).ravel()
    if p_arr.shape != y_arr.shape:
        raise ValueError(
            f"p, y length mismatch: p={p_arr.shape} y={y_arr.shape}"
        )
    if p_arr.size == 0:
        raise ValueError("Empty input (p/y length 0).")
    # Validate y is 0/1.
    y_int = y_arr.astype(np.int64)
    if not np.array_equal(y_int, y_arr):
        raise ValueError(
            "Calibrator: y is not 0/1 integer (classification label required)."
        )
    uniq = set(np.unique(y_int).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Calibrator: y allows only {{0, 1}}. actual={sorted(uniq)}"
        )
    # p range — NaN is passed through and handled via finite mask at fit.
    finite = np.isfinite(p_arr)
    if finite.any():
        if (p_arr[finite] < 0.0).any() or (p_arr[finite] > 1.0).any():
            raise ValueError(
                "Calibrator: p must be in [0, 1] range (NaN allowed)."
            )
    return p_arr, y_int.astype(np.float64)


@CALIBRATORS.register("platt")
class PlattCalibrator(Calibrator):
    """Platt scaling — train ``sigmoid(a * logit(p) + b)``.

    Why:
        Equivalent to sklearn ``CalibratedClassifierCV(method='sigmoid')`` — train
        LogisticRegression 1D using the logit of base prob as a single feature. When base is
        over/under-confident on the training set (when `a` is not 1), the calibration effect is large.

    Args:
        eps: margin for 0/1 clipping during ``logit`` conversion. Default ``1e-6``.
        C: inverse regularization strength of sklearn LogisticRegression. Default ``1e10`` — nearly unregularized
            (Platt's original intent). Setting it smaller applies weak regularization, flattening the calibration curve.

    Training artifacts:
        ``coef_``, ``intercept_`` — copies of sklearn model parameters (for debugging).

    Note:
        - If the training data has only one class, LogisticRegression fails → fit
          raises an explicit ValueError.
        - At transform, input NaN passes through as NaN unchanged.
    """

    def __init__(self, *, eps: float = 1e-6, C: float = 1e10) -> None:
        if not (0.0 < float(eps) < 0.5):
            raise ValueError(f"eps must be in (0, 0.5) range: {eps!r}")
        if float(C) <= 0.0:
            raise ValueError(f"C must be positive: {C!r}")
        self.eps = float(eps)
        self.C = float(C)
        self._model: Any | None = None
        self._coef: float | None = None
        self._intercept: float | None = None

    def _logit(self, p: np.ndarray) -> np.ndarray:
        """``logit(p)`` — convert after clipping ``p`` to ``[eps, 1-eps]``."""
        clipped = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p_arr, y_arr = _validate_fit_inputs(p, y)
        finite = np.isfinite(p_arr)
        if finite.sum() < 2:
            raise ValueError(
                "PlattCalibrator.fit: finite (p, y) rows are fewer than 2."
            )
        p_fin = p_arr[finite]
        y_fin = y_arr[finite].astype(np.int64)
        if len(set(y_fin.tolist())) < 2:
            raise ValueError(
                "PlattCalibrator.fit: cannot train because y has only one class."
            )
        z = self._logit(p_fin).reshape(-1, 1)
        # solver='lbfgs' default, max_iter conservative.
        model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
        model.fit(z, y_fin)
        self._model = model
        self._coef = float(model.coef_.ravel()[0])
        self._intercept = float(model.intercept_.ravel()[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "PlattCalibrator: call fit first."
            )
        p_arr = np.asarray(p, dtype=np.float64).ravel()
        out = np.full_like(p_arr, np.nan)
        finite = np.isfinite(p_arr)
        if finite.any():
            z = self._logit(p_arr[finite]).reshape(-1, 1)
            # predict_proba is (n, 2) → P(y=1) is the second column.
            out[finite] = self._model.predict_proba(z)[:, 1]
        return out

    @property
    def coef_(self) -> float | None:
        return self._coef

    @property
    def intercept_(self) -> float | None:
        return self._intercept


@CALIBRATORS.register("isotonic")
class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibration — monotonically increasing piecewise constant mapping.

    Why:
        Equivalent to sklearn ``CalibratedClassifierCV(method='isotonic')``. More
        flexible than Platt but has over-fit risk — generally recommended when data ≥ 1000.

    Args:
        out_of_bounds: handling of inputs outside the training range.
            - ``"clip"`` (default): clip to the training prob's [min, max] then map.
            - ``"nan"``: outputs NaN for out-of-range values.
        y_min: output lower bound. Default ``0.0``.
        y_max: output upper bound. Default ``1.0``.

    Training artifacts:
        ``X_thresholds_``, ``y_thresholds_`` — sklearn IsotonicRegression's trained
        mapping point copies (for debugging).

    Note:
        - At transform, input NaN passes through as NaN unchanged.
        - If training data has only one class, IsotonicRegression itself trains, but
          the mapping is meaningless, so it is rejected with an explicit ValueError.
    """

    def __init__(
        self,
        *,
        out_of_bounds: str = "clip",
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> None:
        if out_of_bounds not in {"clip", "nan"}:
            raise ValueError(
                f"out_of_bounds must be one of 'clip' | 'nan': {out_of_bounds!r}"
            )
        if not (0.0 <= float(y_min) < float(y_max) <= 1.0):
            raise ValueError(
                f"y_min < y_max and within [0, 1]: y_min={y_min!r} y_max={y_max!r}"
            )
        self.out_of_bounds = out_of_bounds
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self._model: Any | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p_arr, y_arr = _validate_fit_inputs(p, y)
        finite = np.isfinite(p_arr)
        if finite.sum() < 2:
            raise ValueError(
                "IsotonicCalibrator.fit: finite (p, y) rows are fewer than 2."
            )
        p_fin = p_arr[finite]
        y_fin = y_arr[finite]
        if len(set(y_fin.astype(np.int64).tolist())) < 2:
            raise ValueError(
                "IsotonicCalibrator.fit: cannot train because y has only one class."
            )
        model = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=True,
            out_of_bounds=self.out_of_bounds,
        )
        model.fit(p_fin, y_fin)
        self._model = model
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "IsotonicCalibrator: call fit first."
            )
        p_arr = np.asarray(p, dtype=np.float64).ravel()
        out = np.full_like(p_arr, np.nan)
        finite = np.isfinite(p_arr)
        if finite.any():
            out[finite] = self._model.predict(p_arr[finite])
        return out

    @property
    def X_thresholds_(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return np.asarray(self._model.X_thresholds_, dtype=np.float64).copy()

    @property
    def y_thresholds_(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return np.asarray(self._model.y_thresholds_, dtype=np.float64).copy()


__all__ = ["Calibrator", "PlattCalibrator", "IsotonicCalibrator"]
