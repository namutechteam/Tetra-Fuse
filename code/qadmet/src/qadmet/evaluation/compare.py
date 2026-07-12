"""다수 결과 (신규 artifact + 옛 phase CSV) 를 wide pivot 으로 합치고 W/L/T
요약을 계산한다.

옛 ``compare_experiments.py`` / ``compare_all_experiments.py`` 의 책임을 흡수.
지원하는 입력:

- 신규 artifact: ``per_task_metric_df`` 출력 ``(dataset, metric, mean, std,
  ensemble_mean, ensemble_median, ...)``.
- 옛 phase result CSV: 같은 컬럼 모양 (`load_legacy_csv` 가 7-컬럼 정규화).

핵심 출력 두 가지:

- ``build_pivot``: wide DataFrame ``(dataset, metric, [series1, series2, ...])``.
  series 마다 한 컬럼. ``column`` 인자로 어떤 metric 컬럼을 비교할지 선택
  (``"mean"`` | ``"ensemble_mean"`` | ``"ensemble_median"``).
- ``compute_wlt``: 비교 대상 별 ``{name: {"win": w, "loss": l, "tie": t,
  "missing": m, "net": n}}``. metric direction (mae=-1, auroc/auprc/spearman=+1)
  반영.

리포트는 ``format_pivot_md`` 가 markdown 표로 출력.
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
    """옛 phase result CSV → 표준 7 컬럼 DataFrame.

    Note:
        ``ensemble_median`` 누락 (예: ``phase4_stack_v6_hybrid_a0.5.csv``) 은 NaN.
        그 외 보조 컬럼 (best_w1, bases_used) 은 무시.
    """
    df = pd.read_csv(path)
    if "dataset" not in df.columns or "metric" not in df.columns or "mean" not in df.columns:
        raise ValueError(
            f"옛 CSV 필수 컬럼 누락 ({path}): {[c for c in CANONICAL_COLUMNS if c not in df.columns]} "
            f"— got {list(df.columns)}"
        )
    out = df.reindex(columns=CANONICAL_COLUMNS).copy()
    return out


# ---------------------------------------------------------------------------- pivot


def _validate_input(df: pd.DataFrame, *, label: str, column: str) -> pd.DataFrame:
    """비교 입력 DataFrame 유효성 검증 + 필요한 두 컬럼만 반환."""
    if column not in VALID_COMPARISON_COLUMNS:
        raise ValueError(
            f"column 은 {VALID_COMPARISON_COLUMNS} 중 하나여야 합니다: got {column!r}"
        )
    missing = {"dataset", "metric", column} - set(df.columns)
    if missing:
        raise ValueError(
            f"{label!r} DataFrame 필수 컬럼 누락: {sorted(missing)} — got {list(df.columns)}"
        )
    return df[["dataset", "metric", column]].copy()


def build_pivot(
    series: Mapping[str, pd.DataFrame],
    *,
    column: str = "ensemble_mean",
) -> pd.DataFrame:
    """여러 결과 DataFrame 을 task 행 × series 컬럼 wide 표로 합친다.

    Args:
        series: ``{label: DataFrame}``. 각 DataFrame 은 ``per_task_metric_df``
            출력 또는 ``load_legacy_csv`` 출력 형태.
        column: 비교에 사용할 metric 컬럼 이름.

    Returns:
        ``[dataset, metric, *labels]`` wide DataFrame. 한 series 가 특정 task
        를 보유하지 않으면 NaN. metric 은 series 간 일치하지 않을 수 있음
        (다른 series 가 특정 task 를 못 보면 NaN), 첫 번째 series 의 metric 을
        우선시 → 없으면 다음 순서로 채움.
    """
    if not series:
        raise ValueError("series 가 비어있습니다 — 비교할 입력을 하나 이상 주세요.")

    # task → metric (등장한 첫 series 우선)
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

    # series dict 의 입력 순서대로 컬럼 정렬.
    ordered = ["dataset", "metric"] + [k for k in series if k in pivot.columns]
    extras = [c for c in pivot.columns if c not in ordered]
    pivot = pivot.reindex(columns=ordered + extras)
    pivot.columns.name = None
    return pivot


# ---------------------------------------------------------------------------- W/L/T


@dataclass
class TaskDelta:
    """한 task / 한 비교 대상 결과."""

    dataset: str
    metric: str
    target: float
    other: float
    delta: float          # signed: target_advantage = (target - other) * direction
    verdict: str          # "win" | "loss" | "tie" | "missing"


def _classify(target: float, other: float, *, direction: int, tie_eps: float) -> tuple[float, str]:
    """target 이 other 대비 더 좋은지 판정.

    Returns:
        (signed_advantage, verdict). advantage > 0 = target 이 우세.
        ``direction=+1`` (auroc 등) → advantage = target - other.
        ``direction=-1`` (mae) → advantage = other - target (낮은 쪽이 좋음).
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
    """target 컬럼을 기준으로 다른 컬럼들과 task 별 비교.

    Args:
        pivot: ``build_pivot`` 결과.
        target: 기준 컬럼 (보통 신규 artifact label).
        others: 비교 대상 컬럼들. None 이면 pivot 의 target 외 모든 컬럼.
        tie_eps: |advantage| <= tie_eps 면 tie.

    Returns:
        (long DataFrame ``[dataset, metric, target, other_label, delta, verdict]``,
         summary ``{other_label: {"win","loss","tie","missing","net"}}``).
    """
    if "dataset" not in pivot.columns or "metric" not in pivot.columns:
        raise ValueError("pivot 은 build_pivot 결과여야 합니다 (dataset/metric 컬럼 필요).")
    if target not in pivot.columns:
        raise ValueError(f"pivot 에 target 컬럼 없음: {target!r} (가용: {list(pivot.columns)})")
    if others is None:
        others = [c for c in pivot.columns if c not in {"dataset", "metric", target}]
    else:
        unknown = [c for c in others if c not in pivot.columns]
        if unknown:
            raise ValueError(f"pivot 에 없는 비교 컬럼: {unknown}")

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
            # 기록을 위해 다른 비교 대상 별 행도 한 줄씩 (long format).
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
    """pivot 표 + (선택) W/L/T 요약을 markdown 으로 렌더링."""
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
