"""``MetaLearner`` ABC + 등록된 strategy 들.

옛 ``run_stacking_v{1..6}.py`` 의 다양한 함수 시그니처를 ``fit/predict`` 인터페이스로
통합한다. ``runners.stack`` 은 strategy 와 무관하게 동일 호출 흐름을 사용.

현재 등록:
    - ``"average"`` — ``AverageStrategy`` (옛 v1, ``stack_simple_average``).
    - ``"inverse_variance"`` — ``InverseVarianceStrategy`` (옛 v2 단순화 버전).
    - ``"per_task_grid"`` — ``PerTaskGridStrategy`` (옛 v3 honest-holdout 버전).
    - ``"oof_xgboost"`` — ``OOFXGBoostStrategy`` (옛 v4 ``stack_one_task`` 동등).
    - ``"smart_fallback"`` — ``SmartFallbackStrategy`` (옛 v5 honest-holdout 변형).
    - ``"hybrid_blend"`` — ``HybridBlendStrategy`` (옛 v6 의 nested-strategy 일반화).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from qadmet.ensembles.meta_dataset import MetaDataset
from qadmet.ensembles.registry import META_MODELS, STRATEGIES


class MetaLearner(ABC):
    """ensemble strategy 의 공통 인터페이스.

    Why:
        옛 v1~v6 가 각자 다른 함수 시그니처로 흩어져 있었다. 동일한 ``fit``/
        ``predict`` 계약을 강제해 ``runners.stack`` 이 strategy 와 무관하게 호출.

    하위 클래스 책임:
        - ``__init__`` 으로 하이퍼파라미터 받기.
        - ``fit(meta_data)`` 로 학습 (학습 없는 strategy 는 no-op).
        - ``predict(meta_data)`` 로 ensemble 예측 long DataFrame 반환.

    Note:
        같은 ``MetaLearner`` 인스턴스에 fit 후 predict 호출이 일반적이지만,
        AverageStrategy 처럼 학습 없는 경우엔 fit 생략 가능.
    """

    @abstractmethod
    def fit(self, meta_data: MetaDataset) -> "MetaLearner":
        ...

    @abstractmethod
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        """ensemble 예측 long DataFrame ``[seed, task, smiles, y_true, y_pred]``."""
        ...


@STRATEGIES.register("average")
class AverageStrategy(MetaLearner):
    """가중 평균 strategy (옛 v1 의 ``stack_simple_average`` 동등).

    Args:
        weights: base 별 가중치. None 이면 균등(1/K). 길이가 base 수와 다르면
            ``predict`` 시점에 ValueError. 음수도 허용 (드물지만 boost-style).
        normalize_weights: True 면 ``weights`` 합이 1 이 되도록 재정규화.
            기본 False — 옛 v1 의 ``(0.7, 0.3)`` 같은 raw 가중을 그대로 쓴다.

    Note:
        분류 task 의 logit→prob 변환은 base 모델 책임 (이 strategy 는 raw 평균만).
        옛 v1 도 동일한 동작.
    """

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        *,
        normalize_weights: bool = False,
    ) -> None:
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        self.normalize_weights = bool(normalize_weights)

    # 학습 단계 없음.
    def fit(self, meta_data: MetaDataset) -> "AverageStrategy":  # noqa: ARG002
        return self

    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        if self.weights is None:
            w = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            if len(self.weights) != K:
                raise ValueError(
                    f"weights 길이 {len(self.weights)} ≠ base 수 {K} (bases={meta_data.bases})"
                )
            w = np.asarray(self.weights, dtype=np.float64)
            if self.normalize_weights:
                s = w.sum()
                if not np.isfinite(s) or s == 0:
                    raise ValueError(f"weights 합이 0 또는 비정상: {w}")
                w = w / s

        preds = meta_data.predictions_matrix()  # (N, K)
        y_pred = preds @ w  # (N,)

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        # 기존 컬럼 순서 통일.
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)


@STRATEGIES.register("inverse_variance")
class InverseVarianceStrategy(MetaLearner):
    """잔차 분산의 역수로 가중치를 결정하는 strategy (옛 v2 단순화 버전).

    파이프라인:
        1. ``fit(meta_val)`` — ``meta_val.split == "val"`` 가정. base 별 (task 별)
           ``MSE = mean((y_true - y_pred)^2)`` 를 계산해 ``weights = 1/MSE`` 정규화
           후 dict ``{task: (w_b1, w_b2, ...)}`` 로 보관한다.
        2. ``predict(meta_test)`` — ``meta_test.split == "test"`` 가 일반적. 학습된
           per-task 가중치를 곱해 weighted average 출력.

    옛 v2 (``run_stacking_v2_xgb.stack_inverse_var``) 와 차이:
        - 옛 v2 는 ``group.evaluate({task: preds})`` 로 leaderboard metric 자체를
          weight 로 썼다 (auroc → score, mae → 1/score). TDC 라이브러리 + 평가 셋
          호출이 필요해 무거웠다.
        - 신규는 잔차 분산 기반 — base 의 자체 val 예측만으로 self-contained 하게
          동작한다 (분류는 sigmoid 확률 기준 잔차, base 의 ``y_pred`` 가 prob 인지
          logit 인지는 base 책임으로 두되, ``MultitaskLitModule`` + ``runners.train``
          은 분류를 prob 으로 통일했다).

    Args:
        eps: ``MSE = 0`` 인 task (예: 분류 일치율 100%) 의 가중치 폭주 방지.
        global_fit: True 면 task 와 무관하게 전체 잔차 한 덩어리로 가중치를 산출한다
            (옛 v1 simple average 의 자동 가중치 버전). False (기본) 가 옛 v2 와 일관.
        normalize_weights: True 면 가중치 합이 1 이 되도록 재정규화. 기본 True
            (1/MSE 분자가 작으면 모든 base 가 거의 0 이 되는 케이스 방지).
        unfit_strategy: ``predict`` 가 ``fit`` 보다 먼저 호출되거나 fit 단계에서
            처음 보지 못한 task 가 test 에 등장할 때 동작.
            - ``"uniform"``: 균등 평균으로 fallback (안전).
            - ``"raise"``: ValueError.
            기본 ``"uniform"`` — runners.stack 이 task 부분집합으로 fit 하기 어려운
            경우를 대비.

    Raises:
        ValueError: ``fit`` 시 split 이 "val" 이 아니거나 잔차 계산이 불가능한 base
            task 조합 (모든 행이 NaN 등) 발견 시.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        global_fit: bool = False,
        normalize_weights: bool = True,
        unfit_strategy: str = "uniform",
    ) -> None:
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy 는 'uniform' | 'raise' 중 하나: {unfit_strategy!r}"
            )
        self.eps = float(eps)
        self.global_fit = bool(global_fit)
        self.normalize_weights = bool(normalize_weights)
        self.unfit_strategy = unfit_strategy
        # fit 산출물.
        self._task_weights: dict[str, np.ndarray] = {}
        self._global_weights: np.ndarray | None = None
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "InverseVarianceStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"InverseVarianceStrategy.fit 은 split='val' 필요 "
                f"(받음: {meta_data.split!r}). val 예측 base 가 필요합니다."
            )
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        self._bases = meta_data.bases

        # 전체(global) 가중치 — 모든 task 합쳐서 1/MSE.
        preds_all = meta_data.predictions_matrix()  # (N, K)
        y_all = meta_data.y_true()  # (N,)
        finite = np.isfinite(y_all) & np.all(np.isfinite(preds_all), axis=1)
        if finite.sum() == 0:
            raise ValueError("val 데이터에 finite y_true 행이 없습니다.")
        residuals_all = preds_all[finite] - y_all[finite, None]  # (M, K)
        mse_all = np.mean(residuals_all**2, axis=0)  # (K,)
        self._global_weights = self._mse_to_weights(mse_all)

        if not self.global_fit:
            for task, task_df in meta_data.iter_tasks():
                preds = meta_data.predictions_matrix(task_df)  # (Nt, K)
                y = meta_data.y_true(task_df)  # (Nt,)
                fin = np.isfinite(y) & np.all(np.isfinite(preds), axis=1)
                if fin.sum() == 0:
                    # task 전체가 NaN — 이 task 는 fit 에서 제외 (predict 시 fallback).
                    continue
                resid = preds[fin] - y[fin, None]
                mse = np.mean(resid**2, axis=0)
                self._task_weights[task] = self._mse_to_weights(mse)
        return self

    def _mse_to_weights(self, mse: np.ndarray) -> np.ndarray:
        """``1 / (mse + eps)`` 후 (옵션) 합=1 정규화."""
        inv = 1.0 / (mse + self.eps)
        if self.normalize_weights:
            s = inv.sum()
            if not np.isfinite(s) or s <= 0:
                return np.full_like(inv, 1.0 / len(inv))
            return inv / s
        return inv

    # ------------------------------------------------------------------ predict
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        if not self._bases:
            raise ValueError(
                "InverseVarianceStrategy: fit 을 먼저 호출하세요 (val 예측 필요)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"predict 시 bases 가 fit 과 다릅니다: fit={self._bases} "
                f"predict={meta_data.bases}"
            )

        preds = meta_data.predictions_matrix()  # (N, K)
        y_pred = np.empty(len(meta_data.df), dtype=np.float64)

        if self.global_fit:
            assert self._global_weights is not None
            y_pred = preds @ self._global_weights
        else:
            uniform = np.full(K, 1.0 / K, dtype=np.float64)
            for task in sorted(meta_data.df["task"].unique()):
                mask = (meta_data.df["task"] == task).to_numpy()
                if task in self._task_weights:
                    w = self._task_weights[task]
                elif self.unfit_strategy == "raise":
                    raise ValueError(
                        f"InverseVarianceStrategy: task {task!r} 는 fit 에서 본 적 없음."
                    )
                else:  # uniform
                    w = uniform
                y_pred[mask] = preds[mask] @ w

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)

    # ------------------------------------------------------------------ accessors
    @property
    def task_weights(self) -> dict[str, tuple[float, ...]]:
        """``{task: (w_b1, w_b2, ...)}`` 디버그/로깅용 (튜플 사본)."""
        return {t: tuple(float(x) for x in w) for t, w in self._task_weights.items()}

    @property
    def global_weights(self) -> tuple[float, ...] | None:
        if self._global_weights is None:
            return None
        return tuple(float(x) for x in self._global_weights)


@STRATEGIES.register("per_task_grid")
class PerTaskGridStrategy(MetaLearner):
    """Per-task 1D grid search 로 가중치를 결정하는 strategy (옛 v3 honest 변형).

    K=2 base 전제. 각 task 마다 ``w ∈ [0, 1]`` 후보를 ``grid_step`` 단위로 sweep
    하면서 ``(w * pred_a + (1-w) * pred_b)`` 의 잔차 MSE 가 최소가 되는 w 를 고른다.
    선택된 ``(w, 1-w)`` 를 task 별로 보관 → predict 단계에서 그대로 적용.

    옛 v3 (``run_stacking_v3_pertask.find_optimal_weight``) 와 차이:
        - 옛 v3 는 *test set 위에서* TDC ``group.evaluate`` 의 metric (auroc/mae)
          으로 sweep → optimistic bias (test 위에서 hyperparameter tuning).
        - 신규는 *val set 위에서* 잔차 MSE 로 sweep → honest holdout. ``InverseVariance``
          와 동일한 self-contained 패턴 (TDC 평가 함수 호출 없음, base 의 자체 val
          예측만으로 동작).

    K=2 제약:
        옛 v3 도 (P1.1, STL) 두 base 만 다뤘다. K≥3 은 simplex grid (계산량 폭주)
        가 필요해 별도 strategy 로 분리할 예정. 현재 ``fit`` 에서 K!=2 는 명시 reject.

    Args:
        grid_step: w sweep 의 격자 간격 (0 < grid_step ≤ 1). 기본 ``0.1`` → 11 후보.
            ``0.05`` 면 21 후보 등.
        global_fit: True 면 task 와 무관하게 전체 잔차 한 덩어리로 단일 w 산출
            (옛 v1 simple average + auto-tuning 효과). 기본 False (옛 v3 와 일관).
        unfit_strategy: ``"uniform"`` (기본) — fit 에 없던 task 등장 시 균등 평균,
            ``"raise"`` — ValueError. ``InverseVariance`` 와 동일 의미.
        tie_breaker: 동률 (같은 best MSE) 일 때 선택 규칙.
            - ``"smaller_w"``: 작은 w 우선 (기본, 첫 base 비중 보수적).
            - ``"larger_w"``: 큰 w 우선.
            - ``"closer_to_half"``: 0.5 에 가까운 w 우선.
            거의 항상 동률은 grid 가 거칠 때만 발생.

    Raises:
        ValueError: ``__init__`` 인자 비정상 또는 ``fit`` 시 K!=2 / split!='val' /
            잔차 계산 불가.
    """

    K_REQUIRED = 2

    def __init__(
        self,
        *,
        grid_step: float = 0.1,
        global_fit: bool = False,
        unfit_strategy: str = "uniform",
        tie_breaker: str = "smaller_w",
    ) -> None:
        if not (0.0 < float(grid_step) <= 1.0):
            raise ValueError(
                f"grid_step 은 (0, 1] 범위여야 합니다: {grid_step!r}"
            )
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy 는 'uniform' | 'raise' 중 하나: {unfit_strategy!r}"
            )
        if tie_breaker not in {"smaller_w", "larger_w", "closer_to_half"}:
            raise ValueError(
                f"tie_breaker 는 'smaller_w' | 'larger_w' | 'closer_to_half' 중 하나: "
                f"{tie_breaker!r}"
            )
        self.grid_step = float(grid_step)
        self.global_fit = bool(global_fit)
        self.unfit_strategy = unfit_strategy
        self.tie_breaker = tie_breaker
        # 격자 — 0/1 끝점 포함, 중복 없이 단조증가.
        n_points = int(round(1.0 / self.grid_step)) + 1
        self._grid: np.ndarray = np.linspace(0.0, 1.0, n_points)
        # fit 산출물.
        self._task_weights: dict[str, np.ndarray] = {}
        self._global_weights: np.ndarray | None = None
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "PerTaskGridStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"PerTaskGridStrategy.fit 은 split='val' 필요 "
                f"(받음: {meta_data.split!r}). val 예측 base 가 필요합니다."
            )
        K = len(meta_data.bases)
        if K != self.K_REQUIRED:
            raise ValueError(
                f"PerTaskGridStrategy 는 K=2 base 전제. 받음: K={K} "
                f"(bases={meta_data.bases}). K≥3 은 별도 simplex strategy 로 분리 예정."
            )
        self._bases = meta_data.bases

        # global fit: 모든 task 합쳐서 single w.
        preds_all = meta_data.predictions_matrix()  # (N, 2)
        y_all = meta_data.y_true()
        finite = np.isfinite(y_all) & np.all(np.isfinite(preds_all), axis=1)
        if finite.sum() == 0:
            raise ValueError("val 데이터에 finite y_true 행이 없습니다.")
        w_global = self._best_w(preds_all[finite], y_all[finite])
        self._global_weights = np.array([w_global, 1.0 - w_global], dtype=np.float64)

        if not self.global_fit:
            for task, task_df in meta_data.iter_tasks():
                preds = meta_data.predictions_matrix(task_df)  # (Nt, 2)
                y = meta_data.y_true(task_df)
                fin = np.isfinite(y) & np.all(np.isfinite(preds), axis=1)
                if fin.sum() == 0:
                    # task 전체가 NaN → fit 에서 제외. predict 시 fallback.
                    continue
                w = self._best_w(preds[fin], y[fin])
                self._task_weights[task] = np.array([w, 1.0 - w], dtype=np.float64)
        return self

    def _best_w(self, preds: np.ndarray, y: np.ndarray) -> float:
        """잔차 MSE 가 최소가 되는 w 격자값 반환.

        Args:
            preds: (M, 2) — finite 행만.
            y: (M,) — finite y_true.

        Returns:
            best w (float, 격자 후보 중 하나).
        """
        # mse(w) = mean((w*pred_a + (1-w)*pred_b - y)^2). 격자 후보 일괄 계산 (M, G).
        mse = np.empty(self._grid.shape[0], dtype=np.float64)
        for gi, w in enumerate(self._grid):
            combined = w * preds[:, 0] + (1.0 - w) * preds[:, 1]
            mse[gi] = float(np.mean((combined - y) ** 2))
        best_score = mse.min()
        # 동률 처리.
        ties = np.where(mse <= best_score + 1e-12)[0]
        if self.tie_breaker == "smaller_w":
            idx = ties[0]  # _grid 가 단조증가 → 가장 작은 w.
        elif self.tie_breaker == "larger_w":
            idx = ties[-1]
        else:  # closer_to_half
            cand_w = self._grid[ties]
            idx = ties[int(np.argmin(np.abs(cand_w - 0.5)))]
        return float(self._grid[idx])

    # ------------------------------------------------------------------ predict
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        if not self._bases:
            raise ValueError(
                "PerTaskGridStrategy: fit 을 먼저 호출하세요 (val 예측 필요)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"predict 시 bases 가 fit 과 다릅니다: fit={self._bases} "
                f"predict={meta_data.bases}"
            )

        preds = meta_data.predictions_matrix()  # (N, 2)
        y_pred = np.empty(len(meta_data.df), dtype=np.float64)
        uniform = np.array([0.5, 0.5], dtype=np.float64)

        if self.global_fit:
            assert self._global_weights is not None
            y_pred = preds @ self._global_weights
        else:
            for task in sorted(meta_data.df["task"].unique()):
                mask = (meta_data.df["task"] == task).to_numpy()
                if task in self._task_weights:
                    w = self._task_weights[task]
                elif self.unfit_strategy == "raise":
                    raise ValueError(
                        f"PerTaskGridStrategy: task {task!r} 는 fit 에서 본 적 없음."
                    )
                else:  # uniform
                    w = uniform
                y_pred[mask] = preds[mask] @ w

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)

    # ------------------------------------------------------------------ accessors
    @property
    def grid(self) -> tuple[float, ...]:
        """sweep 격자 후보 (튜플 사본)."""
        return tuple(float(x) for x in self._grid)

    @property
    def task_weights(self) -> dict[str, tuple[float, ...]]:
        """``{task: (w, 1-w)}`` — fit 에서 채워진 task 만 포함."""
        return {t: tuple(float(x) for x in w) for t, w in self._task_weights.items()}

    @property
    def global_weights(self) -> tuple[float, ...] | None:
        if self._global_weights is None:
            return None
        return tuple(float(x) for x in self._global_weights)


# ---------------------------------------------------------------------------- v4: OOF + meta-model

# meta-model family → (회귀 키, 분류 키). META_MODELS 에 등록된 키여야 한다.
_META_MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "xgboost": ("xgboost_regressor", "xgboost_classifier"),
    "linear": ("ridge", "logistic_regression"),
}


@STRATEGIES.register("oof_xgboost")
class OOFXGBoostStrategy(MetaLearner):
    """Out-of-fold + meta-model stacking (옛 v4 ``stack_one_task`` 동등).

    파이프라인:
        1. ``fit(meta_val)`` — ``split == "val"`` 강제. task 별로
           ``X = base 예측 (N_val, K)``, ``y = y_val``. seed 는 pool (옛 v4 와 동일).
           task type 에 따라 ``META_MODELS`` 에서 회귀/분류 추정기를 빌드해 fit.
           학습된 모델을 ``_task_models[task]`` 에 보관.
        2. ``predict(meta_test)`` — task 별 ``X_test`` 를 학습된 meta 에 통과시켜
           예측을 만든다. 분류는 ``predict_proba(X)[:, 1]`` (META_MODELS 의 분류기는
           ``predict`` 가 P(y=1) 을 반환하도록 통일되어 있음).

    옛 v4 (``run_stacking_v4_oof_xgb.stack_one_task``) 와 차이:
        - 옛 v4 는 ``meta_type ∈ {"xgb", "linear"}`` 인자로 두 추정기 페어를 직접
          호출. 신규는 ``meta_model_family`` (``"xgboost"`` | ``"linear"``) 로
          ``META_MODELS`` registry 를 lookup → 새 family 추가가 다른 strategy
          (HybridBlend 등) 와 공유 가능.
        - 옛 v4 는 task 위 모든 seed 의 val 을 concat 해서 학습 (seed-pooled).
          신규도 동일 — ``MetaDataset.from_predictionsets`` 결과가 자동으로
          모든 seed 를 포함하므로 task 별 한 번의 fit 으로 충분.

    Args:
        meta_model_family: ``"xgboost"`` | ``"linear"`` (또는 명시 family 키).
            ``EnsembleSpec.meta_model`` 이 이 값으로 매핑된다 (runner 가 전달).
            기본 ``"xgboost"``.
        meta_model_kwargs: 추정기 생성자에 전달할 추가 인자
            (예: ``{"n_estimators": 200, "max_depth": 6}``). 회귀/분류 모두 동일
            kwargs 를 그대로 사용 (옛 v4 동작과 동일). 기본 ``None``.
        task_types: ``{task_name: "regression" | "classification"}`` 매핑. None 이면
            ``task_type_default`` 사용. recipe 가 명시 dict 또는 ``task_set`` 키
            (runner 가 ``TASK_SETS`` 에서 도출) 둘 다 지원.
        task_type_default: ``task_types`` 에 없는 task 의 기본 type. None 이면
            그런 task 등장 시 ``fit`` 에서 ValueError. 기본 None — 의도하지 않은
            silent regression/classification 분류를 막는다.
        unfit_strategy: predict 에서 fit 단계에 못 본 task 를 만났을 때 동작.
            - ``"uniform"``: K 개 base 의 균등 평균 fallback (안전).
            - ``"raise"``: ValueError.
            기본 ``"uniform"``.

    Raises:
        ValueError: ``__init__`` 인자 비정상 (family 미지원, unfit_strategy 오타),
            ``fit`` 시 split 검증 실패 / task type 결정 실패 / 빈 base.
    """

    def __init__(
        self,
        *,
        meta_model_family: str = "xgboost",
        meta_model_kwargs: Mapping[str, Any] | None = None,
        task_types: Mapping[str, str] | None = None,
        task_type_default: str | None = None,
        unfit_strategy: str = "uniform",
    ) -> None:
        if meta_model_family not in _META_MODEL_FAMILIES:
            raise ValueError(
                f"meta_model_family 는 {sorted(_META_MODEL_FAMILIES)} 중 하나: "
                f"{meta_model_family!r}"
            )
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy 는 'uniform' | 'raise' 중 하나: {unfit_strategy!r}"
            )
        if task_type_default is not None and task_type_default not in (
            "regression",
            "classification",
        ):
            raise ValueError(
                f"task_type_default 는 'regression' | 'classification' | None: "
                f"{task_type_default!r}"
            )
        if task_types is not None:
            for t, tt in task_types.items():
                if tt not in ("regression", "classification"):
                    raise ValueError(
                        f"task_types[{t!r}] 는 'regression' | 'classification': {tt!r}"
                    )
        self.meta_model_family = meta_model_family
        self.meta_model_kwargs: dict[str, Any] = (
            {} if meta_model_kwargs is None else dict(meta_model_kwargs)
        )
        self.task_types: dict[str, str] = (
            {} if task_types is None else dict(task_types)
        )
        self.task_type_default = task_type_default
        self.unfit_strategy = unfit_strategy
        # fit 산출물.
        self._task_models: dict[str, Any] = {}
        self._task_type_used: dict[str, str] = {}
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ helpers
    def _resolve_task_type(self, task: str) -> str:
        if task in self.task_types:
            return self.task_types[task]
        if self.task_type_default is not None:
            return self.task_type_default
        raise ValueError(
            f"OOFXGBoostStrategy: task {task!r} 의 type 을 결정할 수 없습니다 — "
            f"task_types 매핑에 추가하거나 task_type_default 를 지정하세요."
        )

    def _build_meta(self, task_type: str) -> Any:
        reg_key, clf_key = _META_MODEL_FAMILIES[self.meta_model_family]
        key = reg_key if task_type == "regression" else clf_key
        return META_MODELS.build_with_kwargs(key, **self.meta_model_kwargs)

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "OOFXGBoostStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"OOFXGBoostStrategy.fit 은 split='val' 필요 "
                f"(받음: {meta_data.split!r}). val 예측 base 가 필요합니다."
            )
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        self._bases = meta_data.bases
        self._task_models = {}
        self._task_type_used = {}

        for task, task_df in meta_data.iter_tasks():
            X = meta_data.predictions_matrix(task_df)  # (Nt, K)
            y = meta_data.y_true(task_df)  # (Nt,)
            fin = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if fin.sum() == 0:
                # task 전체가 NaN — fit 에서 제외, predict 시 fallback.
                continue
            task_type = self._resolve_task_type(task)
            X_fin = X[fin]
            y_fin = y[fin]
            if task_type == "classification":
                # 옛 v4 와 동일하게 정수 캐스트 후 0/1 강제 (META_MODELS 분류기가
                # 다시 검증하지만, 미리 명시적으로 거른다).
                y_int = y_fin.astype(int)
                if not set(np.unique(y_int)).issubset({0, 1}):
                    raise ValueError(
                        f"task {task!r} 는 classification 인데 y 가 0/1 이 아닙니다: "
                        f"{np.unique(y_int)}"
                    )
                meta = self._build_meta(task_type)
                meta.fit(X_fin, y_int)
            else:
                meta = self._build_meta(task_type)
                meta.fit(X_fin, y_fin)
            self._task_models[task] = meta
            self._task_type_used[task] = task_type
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        if not self._bases:
            raise ValueError(
                "OOFXGBoostStrategy: fit 을 먼저 호출하세요 (val 예측 필요)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"predict 시 bases 가 fit 과 다릅니다: fit={self._bases} "
                f"predict={meta_data.bases}"
            )

        preds = meta_data.predictions_matrix()  # (N, K)
        y_pred = np.empty(len(meta_data.df), dtype=np.float64)
        uniform_w = np.full(K, 1.0 / K, dtype=np.float64)

        for task in sorted(meta_data.df["task"].unique()):
            mask = (meta_data.df["task"] == task).to_numpy()
            X_task = preds[mask]
            if task in self._task_models:
                meta = self._task_models[task]
                # META_MODELS.predict 는 회귀=값, 분류=P(y=1) 를 반환 (계약).
                y_pred[mask] = np.asarray(meta.predict(X_task), dtype=np.float64)
            elif self.unfit_strategy == "raise":
                raise ValueError(
                    f"OOFXGBoostStrategy: task {task!r} 는 fit 에서 본 적 없음."
                )
            else:  # uniform
                y_pred[mask] = X_task @ uniform_w

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)

    # ------------------------------------------------------------------ accessors
    @property
    def fit_tasks(self) -> tuple[str, ...]:
        """fit 단계에서 학습된 task 이름들 (정렬)."""
        return tuple(sorted(self._task_models))

    @property
    def task_type_used(self) -> dict[str, str]:
        """실제 fit 에 사용된 task type (디버깅용 사본)."""
        return dict(self._task_type_used)


# ---------------------------------------------------------------------------- v5: smart fallback


@STRATEGIES.register("smart_fallback")
class SmartFallbackStrategy(MetaLearner):
    """task 별 winner-takes-all 또는 fallback (옛 v5 의 honest-holdout 변형).

    파이프라인:
        1. ``fit(meta_val)`` — ``split == "val"`` 강제. task 별로 base 의 잔차 MSE
           ``mse_k = mean((y_pred_k - y_true)^2)`` 계산.
           - dominance = ``(best_mse + eps) / (second_best_mse + eps)``.
           - ``dominance < winner_threshold`` 이면 winner-takes-all
             (가장 낮은 MSE base 의 weight 만 1.0, 나머지 0.0).
           - 그 외엔 ``fallback`` 옵션에 따라 가중치 결정 — ``"average"`` 균등 평균
             또는 ``"inverse_variance"`` 1/MSE 정규화.
           산출: ``_task_weights[task] = ndarray(K,)``,
                 ``_task_decisions[task] ∈ {"winner", "fallback"}``.
        2. ``predict(meta_test)`` — task 별 가중치를 곱해 예측 결합.

    옛 v5 (``run_stacking_v5_smart``) 와 차이:
        - 옛 v5 의 "smart" 는 base 가용성 기반 (val/test size 불일치 시 base drop)
          후 Linear OOF 학습이었다. 이는 신규 ``MetaDataset.from_predictionsets``
          의 ``(seed, task, smiles)`` inner-join 으로 자동 처리되므로 동일 의미가
          이미 내재.
        - 신규 의도는 메모 #4.5 — *task 별 base 강점/약점* 평가 후 winner-takes-all
          또는 fallback. honest holdout (val 잔차 MSE) 기준 self-contained 동작.

    K=1 처리:
        ``runner-up`` 이 없으면 dominance 정의 불가 → ``weight = [1.0]`` 로 short-circuit.
        ``decisions[task] = "winner"`` 로 기록.

    Args:
        winner_threshold: dominance ratio 임계값. ``best_mse / second_best_mse``
            가 이 값보다 작으면 winner-takes-all. 0 < threshold ≤ 1. 기본 ``0.5``
            — winner 의 MSE 가 runner-up 의 절반 이하일 때만 단독 채택 (보수적).
        fallback: dominance ≥ threshold 일 때 사용할 가중 전략.
            - ``"average"`` — K base 균등 평균 (기본).
            - ``"inverse_variance"`` — ``1/(mse + eps)`` 정규화.
        eps: 분모 0 보호 + dominance 비율 안정화. 기본 ``1e-12``.
        global_fit: True 면 task 무관하게 전체 잔차로 단일 가중치. 기본 False.
        unfit_strategy: predict 에서 fit 단계에 못 본 task 만났을 때.
            - ``"uniform"``: 균등 평균 fallback (기본).
            - ``"raise"``: ValueError.

    Raises:
        ValueError: ``__init__`` 인자 비정상 또는 ``fit`` 시 split/잔차 검증 실패.
    """

    def __init__(
        self,
        *,
        winner_threshold: float = 0.5,
        fallback: str = "average",
        eps: float = 1e-12,
        global_fit: bool = False,
        unfit_strategy: str = "uniform",
    ) -> None:
        if not (0.0 < float(winner_threshold) <= 1.0):
            raise ValueError(
                f"winner_threshold 는 (0, 1] 범위여야 합니다: {winner_threshold!r}"
            )
        if fallback not in {"average", "inverse_variance"}:
            raise ValueError(
                f"fallback 은 'average' | 'inverse_variance' 중 하나: {fallback!r}"
            )
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy 는 'uniform' | 'raise' 중 하나: {unfit_strategy!r}"
            )
        if float(eps) < 0.0:
            raise ValueError(f"eps 는 음수일 수 없습니다: {eps!r}")
        self.winner_threshold = float(winner_threshold)
        self.fallback = fallback
        self.eps = float(eps)
        self.global_fit = bool(global_fit)
        self.unfit_strategy = unfit_strategy
        # fit 산출물.
        self._task_weights: dict[str, np.ndarray] = {}
        self._task_decisions: dict[str, str] = {}
        self._global_weights: np.ndarray | None = None
        self._global_decision: str | None = None
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ helpers
    def _decide_weights(self, mse: np.ndarray) -> tuple[np.ndarray, str]:
        """base MSE 벡터 → (가중치, 결정 라벨).

        Args:
            mse: shape (K,), 모든 값 ≥ 0. 비-finite 값 발생 시 fallback 으로 강제.

        Returns:
            (weights ndarray (K,), label ∈ {"winner", "fallback"}).
        """
        K = len(mse)
        if K == 1:
            return np.array([1.0], dtype=np.float64), "winner"

        finite = np.isfinite(mse)
        if not finite.all():
            # 비정상 MSE — winner 결정 위험 → fallback 강제.
            return self._fallback_weights(np.where(finite, mse, np.inf)), "fallback"

        order = np.argsort(mse, kind="stable")  # 가장 낮은 MSE 가 첫번째.
        best_idx = int(order[0])
        best = float(mse[order[0]])
        second_best = float(mse[order[1]])
        dominance = (best + self.eps) / (second_best + self.eps)

        if dominance < self.winner_threshold:
            w = np.zeros(K, dtype=np.float64)
            w[best_idx] = 1.0
            return w, "winner"
        return self._fallback_weights(mse), "fallback"

    def _fallback_weights(self, mse: np.ndarray) -> np.ndarray:
        """fallback 모드의 가중치 산출.

        ``"average"``: ``1/K`` 균등.
        ``"inverse_variance"``: ``1/(mse + eps)`` 정규화. inf MSE 항목은 0 가중.
        """
        K = len(mse)
        if self.fallback == "average":
            return np.full(K, 1.0 / K, dtype=np.float64)
        # inverse_variance.
        inv = 1.0 / (mse + self.eps)
        # inf MSE → inv = 0 자동 (1 / inf = 0).
        s = inv.sum()
        if not np.isfinite(s) or s <= 0.0:
            return np.full(K, 1.0 / K, dtype=np.float64)
        return inv / s

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "SmartFallbackStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"SmartFallbackStrategy.fit 은 split='val' 필요 "
                f"(받음: {meta_data.split!r}). val 예측 base 가 필요합니다."
            )
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        self._bases = meta_data.bases
        self._task_weights = {}
        self._task_decisions = {}

        # global fit: 모든 task 합쳐서 단일 결정.
        preds_all = meta_data.predictions_matrix()  # (N, K)
        y_all = meta_data.y_true()
        finite = np.isfinite(y_all) & np.all(np.isfinite(preds_all), axis=1)
        if finite.sum() == 0:
            raise ValueError("val 데이터에 finite y_true 행이 없습니다.")
        residuals_all = preds_all[finite] - y_all[finite, None]  # (M, K)
        mse_all = np.mean(residuals_all**2, axis=0)
        gw, gdec = self._decide_weights(mse_all)
        self._global_weights = gw
        self._global_decision = gdec

        if not self.global_fit:
            for task, task_df in meta_data.iter_tasks():
                preds = meta_data.predictions_matrix(task_df)
                y = meta_data.y_true(task_df)
                fin = np.isfinite(y) & np.all(np.isfinite(preds), axis=1)
                if fin.sum() == 0:
                    # task 전체 NaN — fit 에서 제외, predict 시 fallback.
                    continue
                resid = preds[fin] - y[fin, None]
                mse = np.mean(resid**2, axis=0)
                w, dec = self._decide_weights(mse)
                self._task_weights[task] = w
                self._task_decisions[task] = dec
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset 에 base 가 없습니다.")
        if not self._bases:
            raise ValueError(
                "SmartFallbackStrategy: fit 을 먼저 호출하세요 (val 예측 필요)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"predict 시 bases 가 fit 과 다릅니다: fit={self._bases} "
                f"predict={meta_data.bases}"
            )

        preds = meta_data.predictions_matrix()  # (N, K)
        y_pred = np.empty(len(meta_data.df), dtype=np.float64)

        if self.global_fit:
            assert self._global_weights is not None
            y_pred = preds @ self._global_weights
        else:
            uniform = np.full(K, 1.0 / K, dtype=np.float64)
            for task in sorted(meta_data.df["task"].unique()):
                mask = (meta_data.df["task"] == task).to_numpy()
                if task in self._task_weights:
                    w = self._task_weights[task]
                elif self.unfit_strategy == "raise":
                    raise ValueError(
                        f"SmartFallbackStrategy: task {task!r} 는 fit 에서 본 적 없음."
                    )
                else:  # uniform
                    w = uniform
                y_pred[mask] = preds[mask] @ w

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)

    # ------------------------------------------------------------------ accessors
    @property
    def task_weights(self) -> dict[str, tuple[float, ...]]:
        """``{task: (w_b1, w_b2, ...)}`` 디버그/로깅용 (튜플 사본)."""
        return {t: tuple(float(x) for x in w) for t, w in self._task_weights.items()}

    @property
    def task_decisions(self) -> dict[str, str]:
        """``{task: "winner" | "fallback"}`` — 어떤 결정 경로였는지."""
        return dict(self._task_decisions)

    @property
    def global_weights(self) -> tuple[float, ...] | None:
        if self._global_weights is None:
            return None
        return tuple(float(x) for x in self._global_weights)

    @property
    def global_decision(self) -> str | None:
        return self._global_decision


@STRATEGIES.register("hybrid_blend")
class HybridBlendStrategy(MetaLearner):
    """두 sub-strategy 의 예측을 ``alpha`` 로 블렌드 (옛 v6 의 일반화).

    옛 v6 (``run_stacking_v6_hybrid``) 는 ``alpha * v5_smart + (1 - alpha) * v1_avg(0.7, 0.3)``
    하드코딩이었다. 신규는 두 sub-strategy 를 registry 키로 받아 자유롭게 조합 가능
    하도록 일반화 — recipe 에서 ``primary``/``secondary`` 와 각각의 kwargs 를 명시.

    파이프라인:
        1. ``__init__`` — primary/secondary sub-strategy 를 registry lookup + 빌드.
        2. ``fit(meta)`` — 두 sub-strategy 모두 ``fit(meta)`` 호출.
            - ``AverageStrategy`` 처럼 fit no-op 인 경우도 안전 (default 구현).
            - ``InverseVariance`` / ``SmartFallback`` 등 ``split == "val"`` 강제하는
              sub-strategy 는 본 hybrid 의 fit 도 자동으로 val MetaDataset 을 요구.
        3. ``predict(meta)`` — 두 sub.predict(meta) 결과를 (seed, task, smiles, y_true)
            행 정렬 검증 후 ``alpha * primary + (1 - alpha) * secondary`` 출력.

    옛 v6 와 차이:
        - 사이즈 mismatch (옛 v6 가 ``len(v5p) != len(sp)`` 시 sp 만 사용했던 케이스):
          신규는 같은 MetaDataset 을 두 sub-strategy 에 넘기므로 row 수가 동일함을
          보장. 따라서 mismatch 처리 불필요. 입력 PredictionSet 자체에서 base 별
          val/test 누락이 있으면 ``MetaDataset.from_predictionsets`` 의 inner-join
          이 미리 처리.
        - sub-strategy 가 ``smart_fallback`` 외에도 자유롭게 (예: ``oof_xgboost`` +
          ``inverse_variance``) 결합 가능.

    Args:
        alpha: primary 가중치 (1 - alpha 는 secondary). ``[0, 1]`` 범위.
        primary: STRATEGIES registry 키 (기본 ``"smart_fallback"`` — 옛 v6 와 일관).
        secondary: STRATEGIES registry 키 (기본 ``"average"`` — 옛 v6 의 stack 70/30).
        primary_kwargs: primary 생성자 kwargs. 빈 dict 면 sub-strategy default.
        secondary_kwargs: secondary 생성자 kwargs.

    Raises:
        ValueError: alpha 범위 / primary == secondary 키 동일 / sub-strategy 빌드
            실패 / 예측 결과 row mismatch.

    Note:
        - sub-strategy 가 ``hybrid_blend`` 자체를 가리켜 재귀 lookup 도 등록 자체는
          가능하나, 무한 루프 방지는 사용자 책임 (기본 사용 사례엔 등장 안 함).
        - ``runners.stack`` 의 ``fit_split=val`` 옵션이 본 strategy 에도 그대로 적용
          — fit 단계는 val MetaDataset, predict 단계는 test MetaDataset 으로 자연
          dispatch.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        primary: str = "smart_fallback",
        secondary: str = "average",
        primary_kwargs: Mapping[str, Any] | None = None,
        secondary_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha 는 [0, 1] 범위여야 합니다: {alpha!r}")
        if primary == secondary and (
            (primary_kwargs or {}) == (secondary_kwargs or {})
        ):
            raise ValueError(
                f"primary 와 secondary 가 동일 strategy + 동일 kwargs 입니다 "
                f"({primary!r}). hybrid 효과가 없습니다."
            )
        # sub-strategy 빌드는 build_with_kwargs 에서 수행.
        primary_kw = dict(primary_kwargs) if primary_kwargs else {}
        secondary_kw = dict(secondary_kwargs) if secondary_kwargs else {}
        try:
            self._primary: MetaLearner = STRATEGIES.build_with_kwargs(
                primary, **primary_kw
            )
        except Exception as exc:
            raise ValueError(
                f"hybrid_blend: primary={primary!r} 빌드 실패: {exc}"
            ) from exc
        try:
            self._secondary: MetaLearner = STRATEGIES.build_with_kwargs(
                secondary, **secondary_kw
            )
        except Exception as exc:
            raise ValueError(
                f"hybrid_blend: secondary={secondary!r} 빌드 실패: {exc}"
            ) from exc

        self.alpha = float(alpha)
        self.primary_name = primary
        self.secondary_name = secondary
        self.primary_kwargs = primary_kw
        self.secondary_kwargs = secondary_kw
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "HybridBlendStrategy":
        self._primary.fit(meta_data)
        self._secondary.fit(meta_data)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        if not self._fitted:
            # average 등 무학습 sub-strategy 만 묶였다면 fit 호출이 noop 이라 위험 없음.
            # 하지만 일관성 위해 명시 안내 (val 강제 sub 가 섞이면 fit 필요).
            raise RuntimeError(
                "HybridBlendStrategy: fit 을 먼저 호출하세요. "
                "(sub-strategy 중 일부는 fit 단계에서 val 잔차로 가중치를 학습합니다.)"
            )
        primary_df = self._primary.predict(meta_data)
        secondary_df = self._secondary.predict(meta_data)
        if len(primary_df) != len(secondary_df):
            raise ValueError(
                f"hybrid_blend: primary/secondary predict 행 수 불일치 "
                f"(primary={len(primary_df)} secondary={len(secondary_df)})."
            )
        # row alignment 검증 — (seed, task, smiles) 가 동일 순서여야 안전 블렌드.
        keys = ["seed", "task", "smiles"]
        # MetaDataset.df 의 키 순서 그대로 두 sub 가 따랐다면 직접 인덱스 비교 가능.
        # AverageStrategy 등 reset_index(drop=True) 로 동일 순서 보장.
        if not primary_df[keys].reset_index(drop=True).equals(
            secondary_df[keys].reset_index(drop=True)
        ):
            raise ValueError(
                "hybrid_blend: primary/secondary predict 의 (seed, task, smiles) "
                "순서가 다릅니다. sub-strategy 출력 정렬을 점검하세요."
            )
        a = self.alpha
        y_pred = (
            a * primary_df["y_pred"].to_numpy(dtype=np.float64)
            + (1.0 - a) * secondary_df["y_pred"].to_numpy(dtype=np.float64)
        )
        out = primary_df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)

    # ------------------------------------------------------------------ accessors
    @property
    def primary(self) -> MetaLearner:
        """디버그용 — 학습된 primary sub-strategy 인스턴스."""
        return self._primary

    @property
    def secondary(self) -> MetaLearner:
        return self._secondary
