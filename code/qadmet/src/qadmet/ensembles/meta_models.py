"""``META_MODELS`` registry — regression/classification estimator wrapper for meta-learner.

Legacy ``run_stacking_v4_oof_xgb.stack_one_task`` directly instantiated ``xgb.XGBRegressor`` /
``xgb.XGBClassifier`` / ``Ridge`` / ``LogisticRegression`` via task-type branching —
this part is generalized via this module's decorator registration + ``EnsembleSpec.meta_model``
string lookup.

Currently registered (Step 4.3 skeleton):
    - ``"xgboost_regressor"`` — ``XGBoostRegressor``
    - ``"xgboost_classifier"`` — ``XGBoostClassifier``
    - ``"ridge"`` — ``RidgeMeta``
    - ``"logistic_regression"`` — ``LogRegMeta``

Heavy dependencies (``xgboost``, ``sklearn``) are all lazily imported inside ``__init__`` or ``fit``
— they are lazily imported. The invariant that they do not enter sys.modules merely from ``import qadmet.ensembles``
is enforced by tests.

Usage pattern (planned — called from Step 4.4 OOFXGBoost):

>>> meta = META_MODELS.build_with_kwargs("xgboost_regressor", n_estimators=200)
>>> meta.fit(X_val, y_val)
>>> y_test_pred = meta.predict(X_test)

Classification follows the contract that ``predict`` returns the P(y=1) probability (equivalent to ``sklearn``'s
``predict_proba(X)[:, 1]``). The meta-learner does not need to reapply a
sigmoid/softmax transform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from qadmet.ensembles.registry import META_MODELS


class MetaModel(ABC):
    """Estimator contract for meta-learner.

    sklearn-like ``fit`` / ``predict`` interface. Classifiers are required to have ``predict``
    return a probability (P(y=1)) — on the same scale as the base's prob.

    Note:
        At Step 4.3, only the skeleton is registered, implemented so that ``fit`` / ``predict``
        are also functional. Unit tests only cover lazy import verification + regression/classification dispatch
        + default argument sanity checks. Actual ensemble integration is at Step 4.4
        ``OOFXGBoostStrategy``.
    """

    is_classifier: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------- xgboost


class _XGBoostBase(MetaModel):
    """Common xgboost argument container — default values shared by regression/classification.

    Matches the legacy v4 hyperparameters (``n_estimators=100, max_depth=4,
    learning_rate=0.1, tree_method='hist'``). Overriding via recipe ``meta_kwargs`` is
    possible.

    Additional arguments are accepted as ``**extra`` and spread directly into the xgb constructor — sklearn-style API
    compatibility is maintained.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        tree_method: str = "hist",
        **extra: Any,
    ) -> None:
        if int(n_estimators) <= 0:
            raise ValueError(f"n_estimators must be positive: {n_estimators}")
        if int(max_depth) <= 0:
            raise ValueError(f"max_depth must be positive: {max_depth}")
        if float(learning_rate) <= 0:
            raise ValueError(f"learning_rate must be positive: {learning_rate}")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.tree_method = str(tree_method)
        self.extra = dict(extra)
        self._impl: Any = None

    def _build_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": self.tree_method,
            **self.extra,
        }


@META_MODELS.register("xgboost_regressor")
class XGBoostRegressor(_XGBoostBase):
    """XGBoost regression meta-learner (equivalent to legacy v4 ``xgb.XGBRegressor``).

    Why:
        Boosted trees are effective at capturing nonlinear interactions between base predictions (e.g., in regions where base A is strong,
        the residual of base B). Adopted as the meta-learner for regression tasks
        in v4 OOF.
    """

    is_classifier = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        import xgboost as xgb  # lazy

        self._impl = xgb.XGBRegressor(**self._build_kwargs())
        self._impl.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("XGBoostRegressor: call fit first.")
        return np.asarray(
            self._impl.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )


@META_MODELS.register("xgboost_classifier")
class XGBoostClassifier(_XGBoostBase):
    """XGBoost classification meta-learner — ``predict`` returns ``P(y=1)``.

    Adopts the legacy v4 ``XGBClassifier(use_label_encoder=False, eval_metric='logloss')`` arguments
    as defaults. Behaves as ``predict_proba(X)[:, 1]``.
    """

    is_classifier = True

    def __init__(
        self,
        *,
        eval_metric: str = "logloss",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eval_metric = str(eval_metric)

    def _build_kwargs(self) -> dict[str, Any]:
        kw = super()._build_kwargs()
        kw["eval_metric"] = self.eval_metric
        return kw

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        import xgboost as xgb  # lazy

        y_arr = np.asarray(y).astype(int)
        if not set(np.unique(y_arr)).issubset({0, 1}):
            raise ValueError(
                f"XGBoostClassifier: y must be binary 0/1: {np.unique(y_arr)}"
            )
        self._impl = xgb.XGBClassifier(**self._build_kwargs())
        self._impl.fit(np.asarray(X, dtype=np.float64), y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("XGBoostClassifier: call fit first.")
        proba = self._impl.predict_proba(np.asarray(X, dtype=np.float64))
        # binary classifier → (N, 2). Positive class prob.
        return np.asarray(proba[:, 1], dtype=np.float64)


# ---------------------------------------------------------------------------- linear


@META_MODELS.register("ridge")
class RidgeMeta(MetaModel):
    """Linear regression (L2 reg) meta-learner — equivalent to legacy v4 ``Ridge(alpha=1.0)``.

    Additional arguments are accepted as ``**extra`` and passed directly to ``sklearn.linear_model.Ridge``.
    """

    is_classifier = False

    def __init__(self, *, alpha: float = 1.0, **extra: Any) -> None:
        if float(alpha) < 0:
            raise ValueError(f"alpha must be non-negative: {alpha}")
        self.alpha = float(alpha)
        self.extra = dict(extra)
        self._impl: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeMeta":
        from sklearn.linear_model import Ridge  # lazy

        self._impl = Ridge(alpha=self.alpha, **self.extra)
        self._impl.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("RidgeMeta: call fit first.")
        return np.asarray(
            self._impl.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )


@META_MODELS.register("logistic_regression")
class LogRegMeta(MetaModel):
    """Logistic regression classification meta-learner — ``predict`` is ``P(y=1)``.

    Equivalent to legacy v4 ``LogisticRegression(max_iter=1000)``. For lightweight cases where a linear meta-learner
    is needed (XGBoost overfits), used as a fallback.
    """

    is_classifier = True

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 1000,
        **extra: Any,
    ) -> None:
        if float(C) <= 0:
            raise ValueError(f"C must be positive: {C}")
        if int(max_iter) <= 0:
            raise ValueError(f"max_iter must be positive: {max_iter}")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.extra = dict(extra)
        self._impl: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegMeta":
        from sklearn.linear_model import LogisticRegression  # lazy

        y_arr = np.asarray(y).astype(int)
        if not set(np.unique(y_arr)).issubset({0, 1}):
            raise ValueError(
                f"LogRegMeta: y must be binary 0/1: {np.unique(y_arr)}"
            )
        self._impl = LogisticRegression(
            C=self.C, max_iter=self.max_iter, **self.extra
        )
        self._impl.fit(np.asarray(X, dtype=np.float64), y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("LogRegMeta: call fit first.")
        proba = self._impl.predict_proba(np.asarray(X, dtype=np.float64))
        return np.asarray(proba[:, 1], dtype=np.float64)


__all__ = [
    "MetaModel",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "RidgeMeta",
    "LogRegMeta",
]
