"""``META_MODELS`` registry — meta-learner 의 회귀/분류 추정기 래퍼.

옛 ``run_stacking_v4_oof_xgb.stack_one_task`` 가 ``xgb.XGBRegressor`` /
``xgb.XGBClassifier`` / ``Ridge`` / ``LogisticRegression`` 을 task type 분기로
직접 인스턴스화하던 부분을, 본 모듈의 데코레이터 등록 + ``EnsembleSpec.meta_model``
문자열로 lookup 가능한 형태로 일반화한다.

현재 등록 (Step 4.3 골격):
    - ``"xgboost_regressor"`` — ``XGBoostRegressor``
    - ``"xgboost_classifier"`` — ``XGBoostClassifier``
    - ``"ridge"`` — ``RidgeMeta``
    - ``"logistic_regression"`` — ``LogRegMeta``

heavy 의존 (``xgboost``, ``sklearn``) 은 모두 ``__init__`` 또는 ``fit`` 안에서
지연 import 한다. ``import qadmet.ensembles`` 만으로는 sys.modules 에 들어오지 않는
것이 invariants — 테스트로 강제한다.

사용 패턴 (예정 — Step 4.4 OOFXGBoost 에서 호출):

>>> meta = META_MODELS.build_with_kwargs("xgboost_regressor", n_estimators=200)
>>> meta.fit(X_val, y_val)
>>> y_test_pred = meta.predict(X_test)

분류는 ``predict`` 가 P(y=1) 확률을 반환하는 계약 (``sklearn`` 의
``predict_proba(X)[:, 1]`` 과 동등). meta-learner 가 sigmoid/softmax 변환을
다시 적용할 필요 없음.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from qadmet.ensembles.registry import META_MODELS


class MetaModel(ABC):
    """meta-learner 의 추정기 계약.

    sklearn-like ``fit`` / ``predict`` 인터페이스. 분류기는 ``predict`` 가
    확률 (P(y=1)) 을 반환하도록 강제 — base 의 prob 와 같은 스케일.

    Note:
        Step 4.3 시점에서는 골격만 등록하고 ``fit`` / ``predict`` 까지 동작 가능
        하도록 구현한다. 단위 테스트는 lazy import 검증 + 회귀/분류 dispatch
        + 기본 인자 sanity 까지만 다룬다. 실제 ensemble 통합은 Step 4.4
        ``OOFXGBoostStrategy`` 에서.
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
    """공통 xgboost 인자 컨테이너 — 회귀/분류가 공유하는 기본값.

    옛 v4 hyperparameter 와 일치 (``n_estimators=100, max_depth=4,
    learning_rate=0.1, tree_method='hist'``). recipe ``meta_kwargs`` 로 override
    가능.

    추가 인자는 ``**extra`` 로 받아 그대로 xgb 생성자에 펼친다 — sklearn 류 API
    호환성 유지.
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
            raise ValueError(f"n_estimators 는 양수: {n_estimators}")
        if int(max_depth) <= 0:
            raise ValueError(f"max_depth 는 양수: {max_depth}")
        if float(learning_rate) <= 0:
            raise ValueError(f"learning_rate 는 양수: {learning_rate}")
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
    """XGBoost 회귀 meta-learner (옛 v4 ``xgb.XGBRegressor`` 등동).

    Why:
        boosted tree 는 base 예측 간 비선형 상호작용 (예: base A 가 강한 영역에서
        base B 의 잔차) 을 잡아내는 데 효과적이다. v4 OOF 에서 회귀 task 의
        meta-learner 로 채택됨.
    """

    is_classifier = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        import xgboost as xgb  # lazy

        self._impl = xgb.XGBRegressor(**self._build_kwargs())
        self._impl.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("XGBoostRegressor: fit 을 먼저 호출하세요.")
        return np.asarray(
            self._impl.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )


@META_MODELS.register("xgboost_classifier")
class XGBoostClassifier(_XGBoostBase):
    """XGBoost 분류 meta-learner — ``predict`` 는 ``P(y=1)`` 반환.

    옛 v4 ``XGBClassifier(use_label_encoder=False, eval_metric='logloss')`` 인자를
    기본값으로 채택. ``predict_proba(X)[:, 1]`` 동작.
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
                f"XGBoostClassifier: y 는 0/1 이진: {np.unique(y_arr)}"
            )
        self._impl = xgb.XGBClassifier(**self._build_kwargs())
        self._impl.fit(np.asarray(X, dtype=np.float64), y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("XGBoostClassifier: fit 을 먼저 호출하세요.")
        proba = self._impl.predict_proba(np.asarray(X, dtype=np.float64))
        # binary classifier → (N, 2). 양성 클래스 prob.
        return np.asarray(proba[:, 1], dtype=np.float64)


# ---------------------------------------------------------------------------- linear


@META_MODELS.register("ridge")
class RidgeMeta(MetaModel):
    """선형 회귀 (L2 reg) meta-learner — 옛 v4 ``Ridge(alpha=1.0)`` 등동.

    추가 인자는 ``**extra`` 로 받아 ``sklearn.linear_model.Ridge`` 에 그대로 전달.
    """

    is_classifier = False

    def __init__(self, *, alpha: float = 1.0, **extra: Any) -> None:
        if float(alpha) < 0:
            raise ValueError(f"alpha 는 음수가 아니어야 합니다: {alpha}")
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
            raise RuntimeError("RidgeMeta: fit 을 먼저 호출하세요.")
        return np.asarray(
            self._impl.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )


@META_MODELS.register("logistic_regression")
class LogRegMeta(MetaModel):
    """로지스틱 회귀 분류 meta-learner — ``predict`` 는 ``P(y=1)``.

    옛 v4 ``LogisticRegression(max_iter=1000)`` 등동. linear meta-learner 가
    필요한 가벼운 경우 (XGBoost 가 과적합) 의 fallback.
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
            raise ValueError(f"C 는 양수: {C}")
        if int(max_iter) <= 0:
            raise ValueError(f"max_iter 는 양수: {max_iter}")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.extra = dict(extra)
        self._impl: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegMeta":
        from sklearn.linear_model import LogisticRegression  # lazy

        y_arr = np.asarray(y).astype(int)
        if not set(np.unique(y_arr)).issubset({0, 1}):
            raise ValueError(
                f"LogRegMeta: y 는 0/1 이진: {np.unique(y_arr)}"
            )
        self._impl = LogisticRegression(
            C=self.C, max_iter=self.max_iter, **self.extra
        )
        self._impl.fit(np.asarray(X, dtype=np.float64), y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("LogRegMeta: fit 을 먼저 호출하세요.")
        proba = self._impl.predict_proba(np.asarray(X, dtype=np.float64))
        return np.asarray(proba[:, 1], dtype=np.float64)


__all__ = [
    "MetaModel",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "RidgeMeta",
    "LogRegMeta",
]
