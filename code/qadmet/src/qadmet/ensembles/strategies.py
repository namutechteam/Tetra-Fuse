"""``MetaLearner`` ABC + registered strategies.

Unifies the diverse function signatures of the old ``run_stacking_v{1..6}.py`` into the ``fit/predict`` interface
``runners.stack`` uses the same call flow regardless of strategy.

Currently registered:
    - ``"average"`` — ``AverageStrategy`` (old v1, ``stack_simple_average``).
    - ``"inverse_variance"`` — ``InverseVarianceStrategy`` (simplified version of old v2).
    - ``"per_task_grid"`` — ``PerTaskGridStrategy`` (honest-holdout version of old v3).
    - ``"oof_xgboost"`` — ``OOFXGBoostStrategy`` (equivalent to old v4 ``stack_one_task``).
    - ``"smart_fallback"`` — ``SmartFallbackStrategy`` (honest-holdout variant of old v5).
    - ``"hybrid_blend"`` — ``HybridBlendStrategy`` (generalization of old v6's nested-strategy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from qadmet.ensembles.meta_dataset import MetaDataset
from qadmet.ensembles.registry import META_MODELS, STRATEGIES


class MetaLearner(ABC):
    """Common interface for ensemble strategies.

    Why:
        Old v1~v6 were scattered with different function signatures each. The same ``fit``/
        ``predict`` contract is enforced so that ``runners.stack`` can call regardless of strategy.

    Subclass responsibilities:
        - Receive hyperparameters via ``__init__``.
        - Train via ``fit(meta_data)`` (no-op for strategies without training).
        - Return the ensemble prediction long DataFrame via ``predict(meta_data)``.

    Note:
        Calling predict after fit on the same ``MetaLearner`` instance is typical, but
        for cases without training, like AverageStrategy, fit may be omitted.
    """

    @abstractmethod
    def fit(self, meta_data: MetaDataset) -> "MetaLearner":
        ...

    @abstractmethod
    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        """Ensemble prediction long DataFrame ``[seed, task, smiles, y_true, y_pred]``."""
        ...


@STRATEGIES.register("average")
class AverageStrategy(MetaLearner):
    """Weighted average strategy (equivalent to old v1's ``stack_simple_average``).

    Args:
        weights: per-base weight. If None, uniform (1/K). If the length differs from the number of bases,
            raises ValueError at ``predict`` time. Negative values are also allowed (rare, but boost-style).
        normalize_weights: If True, renormalizes so that ``weights`` sum to 1.
            Default False — uses raw weights like old v1's ``(0.7, 0.3)`` as-is.

    Note:
        Classification task's logit→prob conversion is the base model's responsibility (this strategy does raw averaging only).
        The old v1 behaves the same way.
    """

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        *,
        normalize_weights: bool = False,
    ) -> None:
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        self.normalize_weights = bool(normalize_weights)

    # No training step.
    def fit(self, meta_data: MetaDataset) -> "AverageStrategy":  # noqa: ARG002
        return self

    def predict(self, meta_data: MetaDataset) -> pd.DataFrame:
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset has no base.")
        if self.weights is None:
            w = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            if len(self.weights) != K:
                raise ValueError(
                    f"weights length {len(self.weights)} ≠ number of bases {K} (bases={meta_data.bases})"
                )
            w = np.asarray(self.weights, dtype=np.float64)
            if self.normalize_weights:
                s = w.sum()
                if not np.isfinite(s) or s == 0:
                    raise ValueError(f"weights sum is 0 or abnormal: {w}")
                w = w / s

        preds = meta_data.predictions_matrix()  # (N, K)
        y_pred = preds @ w  # (N,)

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        # Unify existing column order.
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)


@STRATEGIES.register("inverse_variance")
class InverseVarianceStrategy(MetaLearner):
    """Strategy that determines weights by inverse of residual variance (simplified version of old v2).

    Pipeline:
        1. ``fit(meta_val)`` — assumes ``meta_val.split == "val"``. per base (per task)
           computes ``MSE = mean((y_true - y_pred)^2)`` and normalizes ``weights = 1/MSE``
           then stores it as dict ``{task: (w_b1, w_b2, ...)}``.
        2. ``predict(meta_test)`` — ``meta_test.split == "test"`` is typical. the trained
           per-task weights are multiplied to output the weighted average.

    Differences from old v2 (``run_stacking_v2_xgb.stack_inverse_var``):
        - old v2 used ``group.evaluate({task: preds})`` to take the leaderboard metric itself
          as the weight (auroc → score, mae → 1/score). It required the TDC library + evaluation set
          calls, which was heavy.
        - the new version is based on residual variance — self-contained using only the base's own val predictions
          it operates (for classification, residual is based on sigmoid probability; whether base's ``y_pred`` is prob
          or logit is left to the base's responsibility, but ``MultitaskLitModule`` + ``runners.train``
          unified classification to prob).

    Args:
        eps: prevents weight blow-up for tasks where ``MSE = 0`` (e.g. classification agreement rate 100%).
        global_fit: if True, computes weights from all residuals combined regardless of task
            (an automatic-weight version of old v1 simple average). False (default) is consistent with old v2.
        normalize_weights: Renormalize so weights sum to 1 if True. Default True
            (Prevents case where all base weights become nearly 0 when the 1/MSE numerator is small).
        unfit_strategy: Behavior when ``predict`` is called before ``fit``, or at the fit stage
            when a task unseen during fit appears in test.
            - ``"uniform"``: fallback to uniform average (safe).
            - ``"raise"``: ValueError.
            Default ``"uniform"`` — since runners.stack may find it difficult to fit on a subset of tasks
            as a precaution for such cases.

    Raises:
        ValueError: Raised if, during ``fit``, split is not "val" or residual computation is impossible for the base
            task combination (e.g., all rows NaN).
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
                f"unfit_strategy must be one of 'uniform' | 'raise': {unfit_strategy!r}"
            )
        self.eps = float(eps)
        self.global_fit = bool(global_fit)
        self.normalize_weights = bool(normalize_weights)
        self.unfit_strategy = unfit_strategy
        # fit artifacts.
        self._task_weights: dict[str, np.ndarray] = {}
        self._global_weights: np.ndarray | None = None
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "InverseVarianceStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"InverseVarianceStrategy.fit requires split='val' "
                f"(got: {meta_data.split!r}). val prediction base is required."
            )
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset has no base.")
        self._bases = meta_data.bases

        # Global weight — 1/MSE combined across all tasks.
        preds_all = meta_data.predictions_matrix()  # (N, K)
        y_all = meta_data.y_true()  # (N,)
        finite = np.isfinite(y_all) & np.all(np.isfinite(preds_all), axis=1)
        if finite.sum() == 0:
            raise ValueError("No finite y_true rows in val data.")
        residuals_all = preds_all[finite] - y_all[finite, None]  # (M, K)
        mse_all = np.mean(residuals_all**2, axis=0)  # (K,)
        self._global_weights = self._mse_to_weights(mse_all)

        if not self.global_fit:
            for task, task_df in meta_data.iter_tasks():
                preds = meta_data.predictions_matrix(task_df)  # (Nt, K)
                y = meta_data.y_true(task_df)  # (Nt,)
                fin = np.isfinite(y) & np.all(np.isfinite(preds), axis=1)
                if fin.sum() == 0:
                    # task entirely NaN — exclude this task from fit (fallback at predict time).
                    continue
                resid = preds[fin] - y[fin, None]
                mse = np.mean(resid**2, axis=0)
                self._task_weights[task] = self._mse_to_weights(mse)
        return self

    def _mse_to_weights(self, mse: np.ndarray) -> np.ndarray:
        """``1 / (mse + eps)`` then (optionally) normalize to sum=1."""
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
            raise ValueError("MetaDataset has no base.")
        if not self._bases:
            raise ValueError(
                "InverseVarianceStrategy: call fit first (val prediction required)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"bases differ from fit at predict time: fit={self._bases} "
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
                        f"InverseVarianceStrategy: task {task!r} was never seen in fit."
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
        """``{task: (w_b1, w_b2, ...)}`` for debug/logging purposes (tuple copy)."""
        return {t: tuple(float(x) for x in w) for t, w in self._task_weights.items()}

    @property
    def global_weights(self) -> tuple[float, ...] | None:
        if self._global_weights is None:
            return None
        return tuple(float(x) for x in self._global_weights)


@STRATEGIES.register("per_task_grid")
class PerTaskGridStrategy(MetaLearner):
    """Strategy that determines weights via per-task 1D grid search (honest variant of old v3).

    Assumes K=2 bases. For each task, sweep ``w ∈ [0, 1]`` candidates in ``grid_step`` increments
    and pick the w that minimizes the residual MSE of ``(w * pred_a + (1-w) * pred_b)``.
    Store the chosen ``(w, 1-w)`` per task → apply as-is at the predict stage.

    Differences from old v3 (``run_stacking_v3_pertask.find_optimal_weight``):
        - Old v3 swept *on the test set* using TDC ``group.evaluate``'s metric (auroc/mae)
          → optimistic bias (hyperparameter tuning on the test set).
        - The new version sweeps *on the val set* via residual MSE → honest holdout. ``InverseVariance``
          has the same self-contained pattern (no TDC evaluation function calls, operates only on base's own val
          predictions).

    K=2 constraint:
        Old v3 also only handled two bases (P1.1, STL). K≥3 requires a simplex grid (computational blowup)
        so it will be split into a separate strategy. Currently ``fit`` explicitly rejects K!=2.

    Args:
        grid_step: grid spacing for w sweep (0 < grid_step ≤ 1). Default ``0.1`` → 11 candidates.
            ``0.05`` gives 21 candidates, etc.
        global_fit: If True, compute a single w from all residuals combined, regardless of task
            (old v1 simple average + auto-tuning effect). Default False (consistent with old v3).
        unfit_strategy: ``"uniform"`` (default) — equal averaging when a task not seen in fit appears,
            ``"raise"`` — ValueError. Same meaning as ``InverseVariance``.
        tie_breaker: selection rule for ties (same best MSE).
            - ``"smaller_w"``: prefer smaller w (default, conservative on first base's weight).
            - ``"larger_w"``: prefer larger w.
            - ``"closer_to_half"``: prefer w closer to 0.5.
            Ties almost always occur only when the grid is coarse.

    Raises:
        ValueError: ``__init__`` arguments invalid, or K!=2 / split!='val' / during ``fit``
            residual computation not possible.
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
                f"grid_step must be in range (0, 1]: {grid_step!r}"
            )
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy must be one of 'uniform' | 'raise': {unfit_strategy!r}"
            )
        if tie_breaker not in {"smaller_w", "larger_w", "closer_to_half"}:
            raise ValueError(
                f"tie_breaker must be one of 'smaller_w' | 'larger_w' | 'closer_to_half': "
                f"{tie_breaker!r}"
            )
        self.grid_step = float(grid_step)
        self.global_fit = bool(global_fit)
        self.unfit_strategy = unfit_strategy
        self.tie_breaker = tie_breaker
        # Grid — includes 0/1 endpoints, monotonically increasing without duplicates.
        n_points = int(round(1.0 / self.grid_step)) + 1
        self._grid: np.ndarray = np.linspace(0.0, 1.0, n_points)
        # fit artifacts.
        self._task_weights: dict[str, np.ndarray] = {}
        self._global_weights: np.ndarray | None = None
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "PerTaskGridStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"PerTaskGridStrategy.fit requires split='val' "
                f"(got: {meta_data.split!r}). val prediction base is required."
            )
        K = len(meta_data.bases)
        if K != self.K_REQUIRED:
            raise ValueError(
                f"PerTaskGridStrategy assumes K=2 bases. Got: K={K} "
                f"(bases={meta_data.bases}). K≥3 will be split into a separate simplex strategy."
            )
        self._bases = meta_data.bases

        # global fit: single w combining all tasks.
        preds_all = meta_data.predictions_matrix()  # (N, 2)
        y_all = meta_data.y_true()
        finite = np.isfinite(y_all) & np.all(np.isfinite(preds_all), axis=1)
        if finite.sum() == 0:
            raise ValueError("No finite y_true rows in val data.")
        w_global = self._best_w(preds_all[finite], y_all[finite])
        self._global_weights = np.array([w_global, 1.0 - w_global], dtype=np.float64)

        if not self.global_fit:
            for task, task_df in meta_data.iter_tasks():
                preds = meta_data.predictions_matrix(task_df)  # (Nt, 2)
                y = meta_data.y_true(task_df)
                fin = np.isfinite(y) & np.all(np.isfinite(preds), axis=1)
                if fin.sum() == 0:
                    # task entirely NaN → exclude from fit. Fallback at predict time.
                    continue
                w = self._best_w(preds[fin], y[fin])
                self._task_weights[task] = np.array([w, 1.0 - w], dtype=np.float64)
        return self

    def _best_w(self, preds: np.ndarray, y: np.ndarray) -> float:
        """Return the grid value of w that minimizes residual MSE.

        Args:
            preds: (M, 2) — finite rows only.
            y: (M,) — finite y_true.

        Returns:
            best w (float, one of the grid candidates).
        """
        # mse(w) = mean((w*pred_a + (1-w)*pred_b - y)^2). Batch compute over grid candidates (M, G).
        mse = np.empty(self._grid.shape[0], dtype=np.float64)
        for gi, w in enumerate(self._grid):
            combined = w * preds[:, 0] + (1.0 - w) * preds[:, 1]
            mse[gi] = float(np.mean((combined - y) ** 2))
        best_score = mse.min()
        # Tie handling.
        ties = np.where(mse <= best_score + 1e-12)[0]
        if self.tie_breaker == "smaller_w":
            idx = ties[0]  # _grid is monotonically increasing → smallest w.
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
            raise ValueError("MetaDataset has no base.")
        if not self._bases:
            raise ValueError(
                "PerTaskGridStrategy: call fit first (val prediction required)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"bases differ from fit at predict time: fit={self._bases} "
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
                        f"PerTaskGridStrategy: task {task!r} was never seen in fit."
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
        """Sweep grid candidates (tuple copy)."""
        return tuple(float(x) for x in self._grid)

    @property
    def task_weights(self) -> dict[str, tuple[float, ...]]:
        """``{task: (w, 1-w)}`` — only includes tasks populated during fit."""
        return {t: tuple(float(x) for x in w) for t, w in self._task_weights.items()}

    @property
    def global_weights(self) -> tuple[float, ...] | None:
        if self._global_weights is None:
            return None
        return tuple(float(x) for x in self._global_weights)


# ---------------------------------------------------------------------------- v4: OOF + meta-model

# meta-model family → (regression key, classification key). Must be a key registered in META_MODELS.
_META_MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "xgboost": ("xgboost_regressor", "xgboost_classifier"),
    "linear": ("ridge", "logistic_regression"),
}


@STRATEGIES.register("oof_xgboost")
class OOFXGBoostStrategy(MetaLearner):
    """Out-of-fold + meta-model stacking (equivalent to legacy v4 ``stack_one_task``).

    Pipeline:
        1. ``fit(meta_val)`` — enforces ``split == "val"``. Per task,
           ``X = base predictions (N_val, K)``, ``y = y_val``. seed is pooled (same as legacy v4).
           builds and fits a regression/classification estimator from ``META_MODELS`` depending on task type.
           Stores the trained model in ``_task_models[task]``.
        2. ``predict(meta_test)`` — passes per-task ``X_test`` through the trained meta model to
           produce predictions. Classification uses ``predict_proba(X)[:, 1]`` (classifiers in META_MODELS are
           unified so that ``predict`` returns P(y=1)).

    Differences from legacy v4 (``run_stacking_v4_oof_xgb.stack_one_task``):
        - legacy v4 directly called one of two estimator pairs via a ``meta_type ∈ {"xgb", "linear"}`` argument to
          invoke them. The new version uses ``meta_model_family`` (``"xgboost"`` | ``"linear"``) to
          lookup the ``META_MODELS`` registry → adding a new family can be shared with other strategies
          (such as HybridBlend).
        - legacy v4 trained by concatenating val across all seeds per task (seed-pooled).
          The new version is the same — the result of ``MetaDataset.from_predictionsets`` automatically
          includes all seeds, so a single fit per task is sufficient.

    Args:
        meta_model_family: ``"xgboost"`` | ``"linear"`` (or an explicit family key).
            ``EnsembleSpec.meta_model`` is mapped to this value (passed by the runner).
            Default ``"xgboost"``.
        meta_model_kwargs: additional arguments passed to the estimator constructor
            (e.g., ``{"n_estimators": 200, "max_depth": 6}``). Same for both regression/classification —
            kwargs used as-is (same behavior as legacy v4). Default ``None``.
        task_types: ``{task_name: "regression" | "classification"}`` mapping. If None,
            uses ``task_type_default``. Supports both an explicit dict from the recipe or a ``task_set`` key
            (derived by the runner from ``TASK_SETS``).
        task_type_default: default type for tasks not in ``task_types``. If None,
            raises ValueError in ``fit`` when such a task appears. Default None — prevents unintended
            silent regression/classification misclassification.
        unfit_strategy: behavior when predict encounters a task not seen during fit.
            - ``"uniform"``: fallback to uniform average of K bases (safe).
            - ``"raise"``: ValueError.
            Default ``"uniform"``.

    Raises:
        ValueError: invalid ``__init__`` argument (unsupported family, typo in unfit_strategy),
            split validation failure / task type determination failure / empty base during ``fit``.
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
                f"meta_model_family must be one of {sorted(_META_MODEL_FAMILIES)}: "
                f"{meta_model_family!r}"
            )
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy must be one of 'uniform' | 'raise': {unfit_strategy!r}"
            )
        if task_type_default is not None and task_type_default not in (
            "regression",
            "classification",
        ):
            raise ValueError(
                f"task_type_default must be 'regression' | 'classification' | None: "
                f"{task_type_default!r}"
            )
        if task_types is not None:
            for t, tt in task_types.items():
                if tt not in ("regression", "classification"):
                    raise ValueError(
                        f"task_types[{t!r}] must be 'regression' | 'classification': {tt!r}"
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
        # fit artifacts.
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
            f"OOFXGBoostStrategy: cannot determine type for task {task!r} — "
            f"add it to the task_types mapping or specify task_type_default."
        )

    def _build_meta(self, task_type: str) -> Any:
        reg_key, clf_key = _META_MODEL_FAMILIES[self.meta_model_family]
        key = reg_key if task_type == "regression" else clf_key
        return META_MODELS.build_with_kwargs(key, **self.meta_model_kwargs)

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "OOFXGBoostStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"OOFXGBoostStrategy.fit requires split='val' "
                f"(received: {meta_data.split!r}). val prediction bases are required."
            )
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset has no base.")
        self._bases = meta_data.bases
        self._task_models = {}
        self._task_type_used = {}

        for task, task_df in meta_data.iter_tasks():
            X = meta_data.predictions_matrix(task_df)  # (Nt, K)
            y = meta_data.y_true(task_df)  # (Nt,)
            fin = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if fin.sum() == 0:
                # entire task is NaN — excluded from fit, fallback at predict time.
                continue
            task_type = self._resolve_task_type(task)
            X_fin = X[fin]
            y_fin = y[fin]
            if task_type == "classification":
                # same as legacy v4: cast to int then enforce 0/1 (META_MODELS classifier
                # re-validates it, but we filter explicitly beforehand).
                y_int = y_fin.astype(int)
                if not set(np.unique(y_int)).issubset({0, 1}):
                    raise ValueError(
                        f"task {task!r} is classification but y is not 0/1: "
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
            raise ValueError("MetaDataset has no base.")
        if not self._bases:
            raise ValueError(
                "OOFXGBoostStrategy: call fit first (val predictions required)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"bases differ from fit during predict: fit={self._bases} "
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
                # META_MODELS.predict returns regression=value, classification=P(y=1) (contract).
                y_pred[mask] = np.asarray(meta.predict(X_task), dtype=np.float64)
            elif self.unfit_strategy == "raise":
                raise ValueError(
                    f"OOFXGBoostStrategy: task {task!r} was never seen in fit."
                )
            else:  # uniform
                y_pred[mask] = X_task @ uniform_w

        out = meta_data.df[["seed", "task", "smiles", "y_true"]].copy()
        out["y_pred"] = y_pred
        return out[["seed", "task", "smiles", "y_true", "y_pred"]].reset_index(drop=True)

    # ------------------------------------------------------------------ accessors
    @property
    def fit_tasks(self) -> tuple[str, ...]:
        """Task names trained during fit (sorted)."""
        return tuple(sorted(self._task_models))

    @property
    def task_type_used(self) -> dict[str, str]:
        """Task type actually used in fit (copy for debugging)."""
        return dict(self._task_type_used)


# ---------------------------------------------------------------------------- v5: smart fallback


@STRATEGIES.register("smart_fallback")
class SmartFallbackStrategy(MetaLearner):
    """Per-task winner-takes-all or fallback (variant of legacy v5's honest-holdout).

    Pipeline:
        1. ``fit(meta_val)`` — ``split == "val"`` enforced. Compute per-task residual MSE of base
           Compute ``mse_k = mean((y_pred_k - y_true)^2)``.
           - dominance = ``(best_mse + eps) / (second_best_mse + eps)``.
           - If ``dominance < winner_threshold``, winner-takes-all
             (only the base with the lowest MSE gets weight 1.0, rest 0.0).
           - Otherwise, determine weights according to the ``fallback`` option — ``"average"`` uniform average
             or ``"inverse_variance"`` 1/MSE normalization.
           Output: ``_task_weights[task] = ndarray(K,)``,
                 ``_task_decisions[task] ∈ {"winner", "fallback"}``.
        2. ``predict(meta_test)`` — combine predictions weighted per task.

    Differences from old v5 (``run_stacking_v5_smart``):
        - The "smart" in old v5 was based on base availability (dropping bases on val/test size mismatch)
          followed by Linear OOF training. This is handled automatically by the new ``MetaDataset.from_predictionsets``
          via its ``(seed, task, smiles)`` inner-join, so the same meaning is
          already inherent.
        - New intent is memo #4.5 — *per-task base strength/weakness* evaluation, then winner-takes-all
          or fallback. Self-contained operation based on honest holdout (val residual MSE).

    K=1 handling:
        If there is no ``runner-up``, dominance cannot be defined → short-circuit with ``weight = [1.0]``.
        Recorded as ``decisions[task] = "winner"``.

    Args:
        winner_threshold: dominance ratio threshold. ``best_mse / second_best_mse``
            if this is smaller than this value, winner-takes-all. 0 < threshold ≤ 1. Default ``0.5``
            — adopted alone only when the winner's MSE is at most half of the runner-up's (conservative).
        fallback: weighting strategy used when dominance ≥ threshold.
            - ``"average"`` — uniform average of K bases (default).
            - ``"inverse_variance"`` — ``1/(mse + eps)`` normalization.
        eps: protects against zero denominator + stabilizes dominance ratio. Default ``1e-12``.
        global_fit: if True, a single weight from overall residuals regardless of task. Default False.
        unfit_strategy: when predict encounters a task not seen during fit.
            - ``"uniform"``: uniform average fallback (default).
            - ``"raise"``: ValueError.

    Raises:
        ValueError: ``__init__`` arguments invalid or split/residual validation failure during ``fit``.
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
                f"winner_threshold must be in the range (0, 1]: {winner_threshold!r}"
            )
        if fallback not in {"average", "inverse_variance"}:
            raise ValueError(
                f"fallback must be one of 'average' | 'inverse_variance': {fallback!r}"
            )
        if unfit_strategy not in {"uniform", "raise"}:
            raise ValueError(
                f"unfit_strategy must be one of 'uniform' | 'raise': {unfit_strategy!r}"
            )
        if float(eps) < 0.0:
            raise ValueError(f"eps cannot be negative: {eps!r}")
        self.winner_threshold = float(winner_threshold)
        self.fallback = fallback
        self.eps = float(eps)
        self.global_fit = bool(global_fit)
        self.unfit_strategy = unfit_strategy
        # fit outputs.
        self._task_weights: dict[str, np.ndarray] = {}
        self._task_decisions: dict[str, str] = {}
        self._global_weights: np.ndarray | None = None
        self._global_decision: str | None = None
        self._bases: tuple[str, ...] = ()

    # ------------------------------------------------------------------ helpers
    def _decide_weights(self, mse: np.ndarray) -> tuple[np.ndarray, str]:
        """Base MSE vector → (weight, decision label).

        Args:
            mse: shape (K,), all values ≥ 0. Forced to fallback if non-finite values occur.

        Returns:
            (weights ndarray (K,), label ∈ {"winner", "fallback"}).
        """
        K = len(mse)
        if K == 1:
            return np.array([1.0], dtype=np.float64), "winner"

        finite = np.isfinite(mse)
        if not finite.all():
            # Abnormal MSE — risky to decide winner → force fallback.
            return self._fallback_weights(np.where(finite, mse, np.inf)), "fallback"

        order = np.argsort(mse, kind="stable")  # lowest MSE first.
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
        """Weight computation for fallback mode.

        ``"average"``: ``1/K`` uniform.
        ``"inverse_variance"``: ``1/(mse + eps)`` normalization. inf MSE entries get 0 weight.
        """
        K = len(mse)
        if self.fallback == "average":
            return np.full(K, 1.0 / K, dtype=np.float64)
        # inverse_variance.
        inv = 1.0 / (mse + self.eps)
        # inf MSE → inv = 0 automatically (1 / inf = 0).
        s = inv.sum()
        if not np.isfinite(s) or s <= 0.0:
            return np.full(K, 1.0 / K, dtype=np.float64)
        return inv / s

    # ------------------------------------------------------------------ fit
    def fit(self, meta_data: MetaDataset) -> "SmartFallbackStrategy":
        if meta_data.split != "val":
            raise ValueError(
                f"SmartFallbackStrategy.fit requires split='val' "
                f"(received: {meta_data.split!r}). val prediction base is required."
            )
        K = len(meta_data.bases)
        if K == 0:
            raise ValueError("MetaDataset has no base.")
        self._bases = meta_data.bases
        self._task_weights = {}
        self._task_decisions = {}

        # global fit: single decision combining all tasks.
        preds_all = meta_data.predictions_matrix()  # (N, K)
        y_all = meta_data.y_true()
        finite = np.isfinite(y_all) & np.all(np.isfinite(preds_all), axis=1)
        if finite.sum() == 0:
            raise ValueError("No finite y_true rows in val data.")
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
                    # task entirely NaN — excluded from fit, fallback at predict.
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
            raise ValueError("MetaDataset has no base.")
        if not self._bases:
            raise ValueError(
                "SmartFallbackStrategy: call fit first (val prediction required)."
            )
        if meta_data.bases != self._bases:
            raise ValueError(
                f"bases differ from fit during predict: fit={self._bases} "
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
                        f"SmartFallbackStrategy: task {task!r} was never seen during fit."
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
        """``{task: (w_b1, w_b2, ...)}`` for debug/logging (tuple copy)."""
        return {t: tuple(float(x) for x in w) for t, w in self._task_weights.items()}

    @property
    def task_decisions(self) -> dict[str, str]:
        """``{task: "winner" | "fallback"}`` — which decision path was taken."""
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
    """Blend predictions of two sub-strategies with ``alpha`` (generalization of old v6).

    Old v6 (``run_stacking_v6_hybrid``) was ``alpha * v5_smart + (1 - alpha) * v1_avg(0.7, 0.3)``
    hardcoded. The new version takes two sub-strategies as registry keys allowing free combination
    generalized — specify ``primary``/``secondary`` and their respective kwargs in the recipe.

    Pipeline:
        1. ``__init__`` — registry lookup + build the primary/secondary sub-strategy.
        2. ``fit(meta)`` — calls ``fit(meta)`` on both sub-strategies.
            - Safe even when fit is a no-op, as with ``AverageStrategy`` (default implementation).
            - For sub-strategies like ``InverseVariance`` / ``SmartFallback`` that force ``split == "val"``,
              this hybrid's fit also automatically requires a val MetaDataset.
        3. ``predict(meta)`` — validates row alignment of both sub.predict(meta) results by (seed, task, smiles, y_true)
            then outputs ``alpha * primary + (1 - alpha) * secondary``.

    Differences from old v6:
        - Size mismatch (the case where old v6 used only sp when ``len(v5p) != len(sp)``):
          the new version passes the same MetaDataset to both sub-strategies, guaranteeing
          an identical row count. Therefore mismatch handling is unnecessary. If the input PredictionSet itself is missing
          val/test per base, the inner-join of ``MetaDataset.from_predictionsets``
          already handles it beforehand.
        - sub-strategies can also be combined freely beyond ``smart_fallback`` (e.g. ``oof_xgboost`` +
          ``inverse_variance``).

    Args:
        alpha: primary weight (1 - alpha is secondary). Range ``[0, 1]``.
        primary: STRATEGIES registry key (default ``"smart_fallback"`` — consistent with old v6).
        secondary: STRATEGIES registry key (default ``"average"`` — old v6's stack 70/30).
        primary_kwargs: constructor kwargs for primary. If an empty dict, uses the sub-strategy default.
        secondary_kwargs: constructor kwargs for secondary.

    Raises:
        ValueError: alpha out of range / primary == secondary same key / sub-strategy build
            failure / prediction result row mismatch.

    Note:
        - Even registering a sub-strategy that points to ``hybrid_blend`` itself for recursive lookup is
          possible, but preventing infinite loops is the user's responsibility (does not occur in default use cases).
        - The ``fit_split=val`` option of ``runners.stack`` also applies as-is to this strategy
          — naturally resulting in the fit stage using the val MetaDataset and the predict stage using the test MetaDataset
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
            raise ValueError(f"alpha must be in range [0, 1]: {alpha!r}")
        if primary == secondary and (
            (primary_kwargs or {}) == (secondary_kwargs or {})
        ):
            raise ValueError(
                f"primary and secondary are the same strategy + same kwargs "
                f"({primary!r}). There is no hybrid effect."
            )
        # sub-strategy build is performed in build_with_kwargs.
        primary_kw = dict(primary_kwargs) if primary_kwargs else {}
        secondary_kw = dict(secondary_kwargs) if secondary_kwargs else {}
        try:
            self._primary: MetaLearner = STRATEGIES.build_with_kwargs(
                primary, **primary_kw
            )
        except Exception as exc:
            raise ValueError(
                f"hybrid_blend: primary={primary!r} build failed: {exc}"
            ) from exc
        try:
            self._secondary: MetaLearner = STRATEGIES.build_with_kwargs(
                secondary, **secondary_kw
            )
        except Exception as exc:
            raise ValueError(
                f"hybrid_blend: secondary={secondary!r} build failed: {exc}"
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
            # If only untrained sub-strategies like average are combined, fit calls are a noop, so there is no risk.
            # But for consistency, an explicit notice is given (fit is required if a val-forcing sub is mixed in).
            raise RuntimeError(
                "HybridBlendStrategy: call fit first. "
                "(Some sub-strategies learn weights from val residuals during the fit stage.)"
            )
        primary_df = self._primary.predict(meta_data)
        secondary_df = self._secondary.predict(meta_data)
        if len(primary_df) != len(secondary_df):
            raise ValueError(
                f"hybrid_blend: primary/secondary predict row count mismatch "
                f"(primary={len(primary_df)} secondary={len(secondary_df)})."
            )
        # row alignment validation — (seed, task, smiles) must be in the same order for a safe blend.
        keys = ["seed", "task", "smiles"]
        # If both subs followed the key order of MetaDataset.df exactly, direct index comparison is possible.
        # AverageStrategy and others guarantee the same order via reset_index(drop=True).
        if not primary_df[keys].reset_index(drop=True).equals(
            secondary_df[keys].reset_index(drop=True)
        ):
            raise ValueError(
                "hybrid_blend: (seed, task, smiles) of primary/secondary predict "
                "order differs. Check the sub-strategy output ordering."
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
        """For debugging — the trained primary sub-strategy instance."""
        return self._primary

    @property
    def secondary(self) -> MetaLearner:
        return self._secondary
