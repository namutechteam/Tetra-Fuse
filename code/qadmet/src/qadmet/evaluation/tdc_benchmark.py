"""신규 artifact 의 ``predictions.parquet`` → 옛 phase result CSV 와 같은
``[dataset, metric, mean, std, ensemble_mean, ensemble_median, n_seeds, n_test]``
표를 만든다.

옛 ``pipeline_mtl.py:1800-1900`` 이 ``tdc.benchmark_group.admet_group.evaluate_many``
+ prediction-level ensemble (mean/median) 으로 만들던 CSV 와 같은 모양을, 신규
long-format parquet 한 장에서 만든다. TDC 인터넷 의존성이 없는 self-contained
재현이 목적.

- ``mean`` / ``std`` = seed 별 metric 의 평균/표준편차 (sklearn metric 기준)
- ``ensemble_mean`` / ``ensemble_median`` = seed 차원으로 prediction 을 평균/중앙값
  취한 뒤의 metric (옛 "Step 1 신규" prediction-level ensemble)
- ``n_seeds`` / ``n_test`` = 진단용 (seed 수, ensemble SMILES 교집합 크기)

NaN-마스킹·sigmoid 등은 신규 train/stack runner 가 이미 적용한 상태 (분류는
prob, 회귀는 unscaled 원 단위). 본 모듈은 추가 변환을 하지 않는다.

History:
    Step 3.7 시점에는 ``parity._metrics`` 에서 살았으나 Step 5 에서 evaluation
    런너가 동일 함수를 사용하면서 본 위치로 이동. ``parity._metrics`` 는 본
    모듈의 함수를 re-export 하는 호환 shim 이 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from qadmet.tasks.metrics import compute_metric
from qadmet.tasks.registry import TASK_REGISTRY, TASK_SETS

if TYPE_CHECKING:  # pragma: no cover
    from qadmet.core.artifacts import ArtifactStore


REQUIRED_COLUMNS = {"seed", "task", "split", "smiles", "y_true", "y_pred"}


def load_predictions(parquet_path: Path, *, split: str = "test") -> pd.DataFrame:
    """``predictions.parquet`` 로드 + split 필터 + 필수 컬럼 검증."""
    df = pd.read_parquet(parquet_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions.parquet 필수 컬럼 누락: {sorted(missing)} — got {list(df.columns)}"
        )
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"split={split!r} 행 없음 (전체 split: {sorted(df['split'].unique())})")
    return sub


def _resolve_tasks(predictions: pd.DataFrame, task_set: str | None) -> list[str]:
    """비교 대상 task 목록 결정.

    task_set 이 주어지면 ``TASK_SETS`` 의 순서를 그대로 사용하고, 없으면 parquet
    에 등장한 task 중 ``TASK_REGISTRY`` 에 등록된 것만 정렬해서 사용한다.
    """
    if task_set is not None:
        ts = TASK_SETS.get(task_set)
        return list(ts.tasks)
    seen = sorted(predictions["task"].unique())
    return [t for t in seen if t in TASK_REGISTRY._items]  # type: ignore[attr-defined]


def per_task_metric_df(
    predictions: pd.DataFrame,
    *,
    task_set: str | None = None,
) -> pd.DataFrame:
    """seed 별 metric 평균/표준편차 + prediction-level ensemble metric.

    Args:
        predictions: ``predictions.parquet`` 의 한 split 부분 (load_predictions 결과).
        task_set: ``TASK_SETS`` 키 (예: "tdc_admet_22"). None 이면 parquet 의 task 자동.

    Returns:
        ``[dataset, metric, mean, std, ensemble_mean, ensemble_median, n_seeds, n_test]``
        — 옛 phase result CSV 와 컬럼 모양 일치 (+ 진단용 두 컬럼 추가).

    Note:
        같은 (seed, task, smiles) 행이 여럿이면 마지막 값을 채택 (groupby last).
        seed 마다 SMILES 집합이 달라도 prediction-level ensemble 은 SMILES 교집합에서만
        평균한다 (옛 코드와 동일 — 옛 코드는 같은 fold 라 항상 일치한 가정).
    """
    rows: list[dict] = []
    tasks = _resolve_tasks(predictions, task_set)

    for task_name in tasks:
        meta = TASK_REGISTRY.get(task_name)
        metric = meta.metric
        sub = predictions[predictions["task"] == task_name]
        if sub.empty:
            rows.append({
                "dataset": task_name, "metric": metric,
                "mean": float("nan"), "std": float("nan"),
                "ensemble_mean": float("nan"), "ensemble_median": float("nan"),
                "n_seeds": 0, "n_test": 0,
            })
            continue

        seed_ids = sorted(sub["seed"].unique())
        per_seed_score: list[float] = []
        # SMILES 별 (seed → y_pred) 누적. y_true 는 seed 간 동일 가정.
        y_true_by_smi: dict[str, float] = {}
        seed_pred_by_smi: dict[int, dict[str, float]] = {s: {} for s in seed_ids}

        for s in seed_ids:
            seed_sub = sub[sub["seed"] == s]
            # 같은 (seed, smiles) 가 중복이면 마지막 행 채택.
            seed_sub = seed_sub.drop_duplicates(subset=["smiles"], keep="last")
            yt = seed_sub["y_true"].to_numpy()
            yp = seed_sub["y_pred"].to_numpy()
            score = compute_metric(metric, yt, yp)
            if np.isfinite(score):
                per_seed_score.append(float(score))
            for smi, t, p in zip(seed_sub["smiles"], yt, yp):
                y_true_by_smi.setdefault(smi, float(t))
                seed_pred_by_smi[s][smi] = float(p)

        # 모든 seed 가 공통으로 가진 SMILES 만 ensemble.
        common = sorted(set.intersection(*(set(d) for d in seed_pred_by_smi.values()))) \
            if seed_pred_by_smi else []
        if common:
            stack = np.stack([
                np.array([seed_pred_by_smi[s][smi] for smi in common], dtype=np.float64)
                for s in seed_ids
            ])
            yt_common = np.array([y_true_by_smi[smi] for smi in common], dtype=np.float64)
            ens_mean = compute_metric(metric, yt_common, stack.mean(axis=0))
            ens_med = compute_metric(metric, yt_common, np.median(stack, axis=0))
        else:
            ens_mean = float("nan")
            ens_med = float("nan")

        if per_seed_score:
            mean_v = float(np.mean(per_seed_score))
            std_v = float(np.std(per_seed_score, ddof=0))
        else:
            mean_v = float("nan")
            std_v = float("nan")

        rows.append({
            "dataset": task_name,
            "metric": metric,
            "mean": mean_v,
            "std": std_v,
            "ensemble_mean": float(ens_mean) if np.isfinite(ens_mean) else float("nan"),
            "ensemble_median": float(ens_med) if np.isfinite(ens_med) else float("nan"),
            "n_seeds": len(seed_ids),
            "n_test": len(common),
        })

    return pd.DataFrame(rows)


def metric_direction(metric: str) -> int:
    """+1 = 클수록 좋음, -1 = 작을수록 좋음. 미등록 metric 은 +1."""
    return -1 if metric.lower() == "mae" else +1


def evaluate_artifact(
    name: str,
    *,
    kind: str = "train",
    store: "ArtifactStore | None" = None,
    run_id: str | None = None,
    split: str = "test",
    task_set: str | None = None,
) -> pd.DataFrame:
    """artifact 의 ``predictions.parquet`` → 7-컬럼 metric DataFrame.

    train / stack / predict 어떤 kind 든 ``predictions.parquet`` 만 있으면 동작.

    Args:
        name: artifact 이름 (recipe ``name``).
        kind: artifact kind (``"train"`` | ``"stack"`` | ``"predict"`` 등).
        store: ArtifactStore. 미지정 시 default store 사용.
        run_id: 명시 시 해당 run, 미지정 시 latest.
        split: ``predictions.parquet`` 의 split 컬럼 값 (보통 ``"test"``).
        task_set: 비교 task 묶음 (``TASK_SETS`` 키). None 이면 parquet 의 등장 task.

    Returns:
        per_task_metric_df 결과 DataFrame.

    Raises:
        FileNotFoundError: artifact 또는 ``predictions.parquet`` 없음.
        ValueError: split 행 없음·필수 컬럼 누락.
    """
    from qadmet.core.artifacts import ArtifactStore

    if store is None:
        store = ArtifactStore()
    if run_id is not None:
        artifact = store.load(kind, name, run_id)
    else:
        artifact = store.latest(kind, name)
        if artifact is None:
            raise FileNotFoundError(f"artifact 없음: {kind}/{name}")
    parquet = artifact.data_dir / "predictions.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"predictions.parquet 없음: {parquet}")
    preds = load_predictions(parquet, split=split)
    return per_task_metric_df(preds, task_set=task_set)
