"""New artifact's ``predictions.parquet`` → same as old phase result CSV
``[dataset, metric, mean, std, ensemble_mean, ensemble_median, n_seeds, n_test]``
produces the table.

Old ``pipeline_mtl.py:1800-1900`` used ``tdc.benchmark_group.admet_group.evaluate_many``
+ prediction-level ensemble (mean/median) to build a CSV of the same shape; from the new
single long-format parquet. Self-contained without TDC internet dependency,
reproduction is the goal.

- ``mean`` / ``std`` = mean/std of the metric per seed (based on sklearn metric)
- ``ensemble_mean`` / ``ensemble_median`` = averaging/median of predictions along the seed dimension,
  metric after taking that (old "Step 1 new" prediction-level ensemble)
- ``n_seeds`` / ``n_test`` = for diagnostics (number of seeds, size of ensemble SMILES intersection)

NaN-masking, sigmoid, etc. have already been applied by the new train/stack runner (classification is
prob, regression is unscaled original units). This module performs no additional transformation.

History:
    Lived in ``parity._metrics`` as of Step 3.7, but as of Step 5, the evaluation
    runner moved it to this location while using the same function. ``parity._metrics`` is this
    module's compatibility shim that re-exports its functions.
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
    """Load ``predictions.parquet`` + filter split + validate required columns."""
    df = pd.read_parquet(parquet_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions.parquet missing required columns: {sorted(missing)} — got {list(df.columns)}"
        )
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"split={split!r} no rows (all splits: {sorted(df['split'].unique())})")
    return sub


def _resolve_tasks(predictions: pd.DataFrame, task_set: str | None) -> list[str]:
    """Determine the list of tasks to compare.

    If task_set is given, use the order of ``TASK_SETS`` as is; otherwise, from the parquet
    use only the tasks that appear and are registered in ``TASK_REGISTRY``, sorted.
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
    """Mean/std of metric per seed + prediction-level ensemble metric.

    Args:
        predictions: a single-split portion of ``predictions.parquet`` (result of load_predictions).
        task_set: ``TASK_SETS`` key (e.g. "tdc_admet_22"). None means auto-detect task from parquet.

    Returns:
        ``[dataset, metric, mean, std, ensemble_mean, ensemble_median, n_seeds, n_test]``
        — matches the column shape of the old phase result CSV (+ two additional diagnostic columns).

    Note:
        If there are multiple rows with the same (seed, task, smiles), take the last value (groupby last).
        Even if the SMILES set differs per seed, prediction-level ensemble averages only over the SMILES
        intersection (same as the old code — the old code assumed they always matched since it was the same fold).
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
        # Accumulate (seed → y_pred) per SMILES. Assumes y_true is identical across seeds.
        y_true_by_smi: dict[str, float] = {}
        seed_pred_by_smi: dict[int, dict[str, float]] = {s: {} for s in seed_ids}

        for s in seed_ids:
            seed_sub = sub[sub["seed"] == s]
            # If (seed, smiles) is duplicated, take the last row.
            seed_sub = seed_sub.drop_duplicates(subset=["smiles"], keep="last")
            yt = seed_sub["y_true"].to_numpy()
            yp = seed_sub["y_pred"].to_numpy()
            score = compute_metric(metric, yt, yp)
            if np.isfinite(score):
                per_seed_score.append(float(score))
            for smi, t, p in zip(seed_sub["smiles"], yt, yp):
                y_true_by_smi.setdefault(smi, float(t))
                seed_pred_by_smi[s][smi] = float(p)

        # Ensemble only the SMILES common to all seeds.
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
    """+1 = higher is better, -1 = lower is better. Unregistered metrics default to +1."""
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
    """artifact's ``predictions.parquet`` → 7-column metric DataFrame.

    Works for any kind — train / stack / predict — as long as ``predictions.parquet`` exists.

    Args:
        name: artifact name (recipe ``name``).
        kind: artifact kind (``"train"`` | ``"stack"`` | ``"predict"`` etc.).
        store: ArtifactStore. Uses the default store if unspecified.
        run_id: the specified run if given, otherwise latest.
        split: value of the split column in ``predictions.parquet`` (usually ``"test"``).
        task_set: comparison task group (``TASK_SETS`` key). None means tasks appearing in the parquet.

    Returns:
        per_task_metric_df result DataFrame.

    Raises:
        FileNotFoundError: artifact or ``predictions.parquet`` not found.
        ValueError: no rows for split · missing required columns.
    """
    from qadmet.core.artifacts import ArtifactStore

    if store is None:
        store = ArtifactStore()
    if run_id is not None:
        artifact = store.load(kind, name, run_id)
    else:
        artifact = store.latest(kind, name)
        if artifact is None:
            raise FileNotFoundError(f"artifact not found: {kind}/{name}")
    parquet = artifact.data_dir / "predictions.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"predictions.parquet not found: {parquet}")
    preds = load_predictions(parquet, split=split)
    return per_task_metric_df(preds, task_set=task_set)
