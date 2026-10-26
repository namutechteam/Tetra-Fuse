"""Merge multiple results (new artifact + old phase CSV) into a wide pivot and compute W/L/T
summary.

Absorbs the responsibilities of the old ``compare_experiments.py`` / ``compare_all_experiments.py``.
Supported inputs:

- New artifact: ``per_task_metric_df`` output ``(dataset, metric, mean, std,
  ensemble_mean, ensemble_median, ...)``.
- Old phase result CSV: same column shape (`load_legacy_csv` normalizes to 7 columns).

Two main outputs:

- ``build_pivot``: wide DataFrame ``(dataset, metric, [series1, series2, ...])``.
  one column per series. The ``column`` argument selects which metric column to compare
  (``"mean"`` | ``"ensemble_mean"`` | ``"ensemble_median"``).
- ``compute_wlt``: per comparison target ``{name: {"win": w, "loss": l, "tie": t,
  "missing": m, "net": n}}``. metric direction (mae=-1, auroc/auprc/spearman=+1)
  reflected.

The report is rendered as a markdown table by ``format_pivot_md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from qadmet.evaluation.tdc_benchmark import metric_direction

CANONICAL_COLUMNS = ["dataset", "metric", "mean", "std", "ensemble_mean", "ensemble_median"]
VALID_COMPARISON_COLUMNS = ("mean", "ensemble_mean", "ensemble_median")


# ---------------------------------------------------------------------------- legacy CSV


def load_legacy_csv(path: Path) -> pd.DataFrame:
    """Old phase result CSV → standard 7-column DataFrame.

    Note:
        Missing ``ensemble_median`` (e.g. ``phase4_stack_v6_hybrid_a0.5.csv``) becomes NaN.
        Other auxiliary columns (best_w1, bases_used) are ignored.
    """
    df = pd.read_csv(path)
    if "dataset" not in df.columns or "metric" not in df.columns or "mean" not in df.columns:
        raise ValueError(
            f"Missing required columns in old CSV ({path}): {[c for c in CANONICAL_COLUMNS if c not in df.columns]} "
            f"— got {list(df.columns)}"
        )
    out = df.reindex(columns=CANONICAL_COLUMNS).copy()
    return out


# ---------------------------------------------------------------------------- pivot


def _validate_input(df: pd.DataFrame, *, label: str, column: str) -> pd.DataFrame:
    """Validate comparison input DataFrame + return only the two required columns."""
    if column not in VALID_COMPARISON_COLUMNS:
        raise ValueError(
            f"column must be one of {VALID_COMPARISON_COLUMNS}: got {column!r}"
        )
    missing = {"dataset", "metric", column} - set(df.columns)
    if missing:
        raise ValueError(
            f"{label!r} DataFrame missing required columns: {sorted(missing)} — got {list(df.columns)}"
        )
    return df[["dataset", "metric", column]].copy()


def build_pivot(
    series: Mapping[str, pd.DataFrame],
    *,
    column: str = "ensemble_mean",
) -> pd.DataFrame:
    """Merge multiple result DataFrames into a wide table of task rows × series columns.

    Args:
        series: ``{label: DataFrame}``. Each DataFrame is a ``per_task_metric_df``
            output or ``load_legacy_csv`` output shape.
        column: name of the metric column to use for comparison.

    Returns:
        ``[dataset, metric, *labels]`` wide DataFrame. If a series
        does not have a particular task, NaN. metric may not match across series
        (NaN if another series does not have a particular task), the metric of the first series
        takes priority → if absent, fill in subsequent order.
    """
    if not series:
        raise ValueError("series is empty — please provide at least one input to compare.")

    # task → metric (first series it appears in takes priority)
    task_metric: dict[str, str] = {}
    long_rows: list[dict] = []
    for label, df in series.items():
        sub = _validate_input(df, label=label, column=column)
        for _, row in sub.iterrows():
            ds = str(row["dataset"])
            mt = str(row["metric"])
            if ds not in task_metric:
                task_metric[ds] = mt
            long_rows.append({
                "dataset": ds,
                "metric": mt,
                "label": label,
                "value": float(row[column]) if pd.notna(row[column]) else float("nan"),
            })

    long_df = pd.DataFrame(long_rows)
    pivot = long_df.pivot_table(
        index=["dataset", "metric"],
        columns="label",
        values="value",
        aggfunc="last",
    ).reset_index()

    # Sort columns in the input order of the series dict.
    ordered = ["dataset", "metric"] + [k for k in series if k in pivot.columns]
    extras = [c for c in pivot.columns if c not in ordered]
    pivot = pivot.reindex(columns=ordered + extras)
    pivot.columns.name = None
    return pivot


# ---------------------------------------------------------------------------- W/L/T


@dataclass
class TaskDelta:
    """Result for one task / one comparison target."""

    dataset: str
    metric: str
    target: float
    other: float
    delta: float          # signed: target_advantage = (target - other) * direction
    verdict: str          # "win" | "loss" | "tie" | "missing"


def _classify(target: float, other: float, *, direction: int, tie_eps: float) -> tuple[float, str]:
    """Determine whether target is better than other.

    Returns:
        (signed_advantage, verdict). advantage > 0 = target is ahead.
        ``direction=+1`` (auroc, etc.) → advantage = target - other.
        ``direction=-1`` (mae) → advantage = other - target (lower is better).
    """
    if not (np.isfinite(target) and np.isfinite(other)):
        return float("nan"), "missing"
    advantage = (target - other) if direction > 0 else (other - target)
    if abs(advantage) <= tie_eps:
        return float(advantage), "tie"
    return float(advantage), ("win" if advantage > 0 else "loss")


def compute_wlt(
    pivot: pd.DataFrame,
    *,
    target: str,
    others: list[str] | None = None,
    tie_eps: float = 0.005,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Compare other columns against the target column per task.

    Args:
        pivot: result of ``build_pivot``.
        target: reference column (usually the new artifact label).
        others: columns to compare against. If None, all columns in pivot except target.
        tie_eps: tie if |advantage| <= tie_eps.

    Returns:
        (long DataFrame ``[dataset, metric, target, other_label, delta, verdict]``,
         summary ``{other_label: {"win","loss","tie","missing","net"}}``).
    """
    if "dataset" not in pivot.columns or "metric" not in pivot.columns:
        raise ValueError("pivot must be the result of build_pivot (dataset/metric columns required).")
    if target not in pivot.columns:
        raise ValueError(f"target column not found in pivot: {target!r} (available: {list(pivot.columns)})")
    if others is None:
        others = [c for c in pivot.columns if c not in {"dataset", "metric", target}]
    else:
        unknown = [c for c in others if c not in pivot.columns]
        if unknown:
            raise ValueError(f"comparison columns not found in pivot: {unknown}")

    rows: list[TaskDelta] = []
    summary: dict[str, dict[str, int]] = {}
    for other in others:
        s = {"win": 0, "loss": 0, "tie": 0, "missing": 0, "net": 0}
        for _, r in pivot.iterrows():
            metric = str(r["metric"])
            direction = metric_direction(metric)
            tval = float(r[target]) if pd.notna(r[target]) else float("nan")
            oval = float(r[other]) if pd.notna(r[other]) else float("nan")
            delta, verdict = _classify(tval, oval, direction=direction, tie_eps=tie_eps)
            rows.append(TaskDelta(
                dataset=str(r["dataset"]), metric=metric,
                target=tval, other=oval, delta=delta, verdict=verdict,
            ))
            s[verdict] = s.get(verdict, 0) + 1
            # For record-keeping, one row per other comparison target as well (long format).
        s["net"] = s["win"] - s["loss"]
        summary[other] = s

    long_columns = ["dataset", "metric", target, "other_label", "other", "delta", "verdict"]
    long_rows = []
    for other in others:
        for _, r in pivot.iterrows():
            metric = str(r["metric"])
            direction = metric_direction(metric)
            tval = float(r[target]) if pd.notna(r[target]) else float("nan")
            oval = float(r[other]) if pd.notna(r[other]) else float("nan")
            delta, verdict = _classify(tval, oval, direction=direction, tie_eps=tie_eps)
            long_rows.append({
                "dataset": str(r["dataset"]),
                "metric": metric,
                target: tval,
                "other_label": other,
                "other": oval,
                "delta": delta,
                "verdict": verdict,
            })
    long_df = pd.DataFrame(long_rows, columns=long_columns)
    return long_df, summary


# ---------------------------------------------------------------------------- markdown


def format_pivot_md(
    pivot: pd.DataFrame,
    *,
    column: str,
    target: str | None = None,
    summary: dict[str, dict[str, int]] | None = None,
) -> str:
    """Render the pivot table + (optional) W/L/T summary as markdown."""
    lines: list[str] = []
    lines.append(f"# Comparison ({column})")
    lines.append("")
    if target is not None:
        lines.append(f"Target: **{target}**")
        lines.append("")
    if summary:
        lines.append("## W/L/T")
        lines.append("")
        lines.append("| Other | W | L | T | missing | net |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for other, s in summary.items():
            lines.append(
                f"| {other} | {s['win']} | {s['loss']} | {s['tie']} | "
                f"{s['missing']} | **{s['net']:+d}** |"
            )
        lines.append("")
    lines.append("## Per-task")
    lines.append("")
    cols = list(pivot.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, r in pivot.iterrows():
        cells: list[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.notna(v):
                cells.append(f"{float(v):.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
