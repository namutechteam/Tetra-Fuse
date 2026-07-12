"""evaluate runner: ExperimentSpec(kind="evaluate") → metric + comparison artifact.

흐름:
    1. ``spec.evaluation.kwargs`` 에서 runner 메타 옵션 추출 (target / split /
       task_set / column / tie_eps / legacy_csvs / leaderboard / 등).
    2. target artifact 로드 → ``evaluate_artifact`` 로 7-컬럼 metric DataFrame.
    3. ``compare_to`` 의 다른 신규 artifact 들 + ``legacy_csvs`` 의 옛 phase CSV
       들을 같은 column 으로 wide pivot 합친 뒤 W/L/T 계산.
    4. 산출물:
       - ``metrics.csv``: target 자체 7-컬럼 metric 표.
       - ``comparison.csv``: long ``[dataset, metric, target, other_label, other,
         delta, verdict]``. ``compute_wlt`` 결과를 행별로 평탄화.
       - ``comparison.md``: ``format_pivot_md`` 가 만든 markdown 표.
       - (선택) ``leaderboard.csv``: ``compare_to_leaderboard`` 결과.
    5. manifest.inputs 에 target + 모든 비교 artifact 의 (kind/name/run_id) 등록.
       manifest.metrics 에 wlt summary + target 메타.

옛 ``compare_experiments.py`` / ``compare_all_experiments.py`` / ``eval_stl_ensemble.py``
의 책임을 단일 recipe-driven 진입점으로 통합한다.

Why:
    base prediction artifact, stack artifact, evaluate artifact 를 하나의
    lineage 그래프 위에서 추적할 수 있다. ``qadmet lineage show evaluate
    leaderboard_diff`` 가 옛 phase CSV 까지 메타로 잇는다 (legacy CSV 는
    artifact 가 아니므로 inputs 에는 미등록 — manifest.metrics.legacy_csvs 로만
    기록).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qadmet.core.artifacts import Artifact, ArtifactStore, InputRef
from qadmet.core.experiment import RUNNER_REGISTRY
from qadmet.core.logging import get_logger
from qadmet.core.specs import ExperimentSpec
from qadmet.evaluation.compare import (
    build_pivot,
    compute_wlt,
    format_pivot_md,
    load_legacy_csv,
)
from qadmet.evaluation.tdc_benchmark import evaluate_artifact

logger = get_logger("qadmet.runners.evaluate")


VALID_COMPARISON_COLUMNS = ("mean", "ensemble_mean", "ensemble_median")


def _resolve_artifact(
    name: str,
    *,
    kind: str,
    store: ArtifactStore,
    run_id: str | None,
) -> Artifact:
    """latest 또는 run_id 핀으로 artifact 한 건 로드 (FileNotFoundError on miss)."""
    if run_id is not None:
        return store.load(kind, name, run_id)
    art = store.latest(kind, name)
    if art is None:
        raise FileNotFoundError(f"artifact 없음: {kind}/{name}")
    return art


@RUNNER_REGISTRY.register("evaluate")
def run_evaluate(*, spec: ExperimentSpec, store: ArtifactStore, repo_root: Path) -> Artifact:
    """evaluate recipe 한 건을 실행하고 artifact 를 반환한다.

    Args:
        spec: ExperimentSpec(kind="evaluate") — evaluation 필수.
        store: ArtifactStore.
        repo_root: git/환경 메타 추출 + legacy_csvs 의 상대경로 해석 기준.

    Returns:
        생성된 evaluate Artifact (data/ 에 metrics.csv, comparison.csv,
        comparison.md, (선택) leaderboard.csv 포함).

    Raises:
        ValueError: target 결정 불가, column 미지원, kwargs 형식 오류.
        FileNotFoundError: target/compare_to artifact 또는 legacy CSV 미존재.
    """
    spec.validate()
    if spec.evaluation is None:
        raise ValueError("evaluate recipe 에 evaluation 섹션이 필요합니다.")

    eval_spec = spec.evaluation
    kwargs: dict[str, Any] = dict(eval_spec.kwargs)

    target: str | None = kwargs.pop("target", None)
    target_kind: str = str(kwargs.pop("target_kind", "train"))
    target_run_id: str | None = kwargs.pop("target_run_id", None)
    split: str = str(kwargs.pop("split", "test"))
    task_set = kwargs.pop("task_set", None)
    if task_set is not None:
        task_set = str(task_set)
    column: str = str(kwargs.pop("column", "ensemble_mean"))
    if column not in VALID_COMPARISON_COLUMNS:
        raise ValueError(
            f"evaluation.kwargs.column 은 {VALID_COMPARISON_COLUMNS} 중 하나: got {column!r}"
        )
    tie_eps: float = float(kwargs.pop("tie_eps", 0.005))

    # compare_to artifact 의 kind 매핑 (이름 → kind, 기본 "train").
    compare_kinds_raw = kwargs.pop("compare_kinds", {}) or {}
    if not isinstance(compare_kinds_raw, dict):
        raise TypeError(
            f"evaluation.kwargs.compare_kinds 는 dict 여야 합니다: {type(compare_kinds_raw)}"
        )
    compare_kinds: dict[str, str] = {str(k): str(v) for k, v in compare_kinds_raw.items()}

    base_run_ids_raw = kwargs.pop("base_run_ids", {}) or {}
    if not isinstance(base_run_ids_raw, dict):
        raise TypeError(
            f"evaluation.kwargs.base_run_ids 는 dict 여야 합니다: {type(base_run_ids_raw)}"
        )
    base_run_ids: dict[str, str] = {str(k): str(v) for k, v in base_run_ids_raw.items()}

    legacy_csvs_raw = kwargs.pop("legacy_csvs", {}) or {}
    if not isinstance(legacy_csvs_raw, dict):
        raise TypeError(
            f"evaluation.kwargs.legacy_csvs 는 dict 여야 합니다: {type(legacy_csvs_raw)}"
        )
    legacy_paths: dict[str, Path] = {}
    for label, p in legacy_csvs_raw.items():
        path = Path(str(p))
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        legacy_paths[str(label)] = path

    leaderboard_opts = kwargs.pop("leaderboard", None)
    if leaderboard_opts is not None and not isinstance(leaderboard_opts, dict):
        raise TypeError(
            f"evaluation.kwargs.leaderboard 는 dict (또는 미지정) 여야 합니다: {type(leaderboard_opts)}"
        )

    if kwargs:
        # 알 수 없는 키는 명시적으로 거부 (디버깅 친화).
        raise ValueError(
            f"evaluation.kwargs 에 알 수 없는 키: {sorted(kwargs)}. "
            f"지원: target, target_kind, target_run_id, split, task_set, column, tie_eps, "
            f"compare_kinds, base_run_ids, legacy_csvs, leaderboard."
        )

    bases = list(eval_spec.compare_to)
    if target is None:
        if not bases:
            raise ValueError(
                "evaluate recipe 에 target 또는 compare_to 가 필요합니다 — "
                "evaluation.kwargs.target (권장) 또는 evaluation.compare_to[0] 으로 지정하세요."
            )
        target = bases[0]
    target = str(target)

    # ----- artifact 로드 -----
    target_artifact = _resolve_artifact(
        target, kind=target_kind, store=store, run_id=target_run_id
    )
    inputs: list[InputRef] = [InputRef(
        kind=target_artifact.kind, name=target_artifact.name, run_id=target_artifact.run_id,
    )]
    target_df = evaluate_artifact(
        target,
        kind=target_kind,
        store=store,
        run_id=target_artifact.run_id,
        split=split,
        task_set=task_set,
    )

    other_dfs: dict[str, pd.DataFrame] = {}
    for nm in bases:
        if nm == target:
            continue
        kk = compare_kinds.get(nm, "train")
        rid = base_run_ids.get(nm)
        a = _resolve_artifact(nm, kind=kk, store=store, run_id=rid)
        df = evaluate_artifact(
            nm, kind=kk, store=store, run_id=a.run_id, split=split, task_set=task_set,
        )
        other_dfs[nm] = df
        inputs.append(InputRef(kind=a.kind, name=a.name, run_id=a.run_id))

    legacy_dfs: dict[str, pd.DataFrame] = {}
    for label, path in legacy_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"legacy CSV 없음: {label}={path}")
        legacy_dfs[label] = load_legacy_csv(path)

    # ----- artifact 생성 -----
    artifact = store.create(spec, repo_root=repo_root, inputs=inputs)

    # 1) target metric 자체 표.
    metrics_csv = artifact.data_dir / "metrics.csv"
    target_df.to_csv(metrics_csv, index=False)
    store.register_output(artifact, "metrics.csv")

    # 2) wide pivot + W/L/T.
    series: dict[str, pd.DataFrame] = {target: target_df, **other_dfs, **legacy_dfs}
    pivot = build_pivot(series, column=column)
    long_df, summary = compute_wlt(
        pivot, target=target, others=list(other_dfs) + list(legacy_dfs), tie_eps=tie_eps,
    )
    comp_csv = artifact.data_dir / "comparison.csv"
    long_df.to_csv(comp_csv, index=False)
    store.register_output(artifact, "comparison.csv")

    md_path = artifact.data_dir / "comparison.md"
    md_path.write_text(format_pivot_md(pivot, column=column, target=target, summary=summary))
    store.register_output(artifact, "comparison.md")

    # 3) leaderboard (선택).
    leaderboard_meta: dict[str, Any] | None = None
    if leaderboard_opts is not None:
        from qadmet.evaluation.leaderboard import compare_to_leaderboard

        methods = list(leaderboard_opts.get("methods") or [])
        use_cache = bool(leaderboard_opts.get("use_cache", True))
        cache_path_opt = leaderboard_opts.get("cache_path")
        cache_path = Path(cache_path_opt) if cache_path_opt else None
        try:
            lb_df = compare_to_leaderboard(
                target_df, methods=methods, use_cache=use_cache, cache_path=cache_path,
            )
            lb_csv = artifact.data_dir / "leaderboard.csv"
            lb_df.to_csv(lb_csv, index=False)
            store.register_output(artifact, "leaderboard.csv")
            leaderboard_meta = {
                "methods": methods,
                "use_cache": use_cache,
                "n_tasks": int(len(lb_df)),
                "n_with_rank": int(lb_df["rank"].notna().sum()) if "rank" in lb_df.columns else 0,
            }
        except Exception as e:  # noqa: BLE001  — 네트워크 실패는 leaderboard 만 스킵.
            logger.warning(f"leaderboard 호출 실패: {e}")
            leaderboard_meta = {"error": repr(e), "methods": methods}

    # ----- manifest.metrics -----
    metrics_payload: dict[str, Any] = {
        "target": target,
        "target_kind": target_kind,
        "target_run_id": target_artifact.run_id,
        "split": split,
        "column": column,
        "task_set": task_set,
        "n_compare_to": len(other_dfs),
        "n_legacy": len(legacy_dfs),
        "wlt": {k: dict(v) for k, v in summary.items()},
    }
    if legacy_paths:
        metrics_payload["legacy_csvs"] = {k: str(v) for k, v in legacy_paths.items()}
    if leaderboard_meta is not None:
        metrics_payload["leaderboard"] = leaderboard_meta
    store.set_metrics(artifact, metrics_payload)

    logger.info(
        "evaluate done",
        extra={"payload": {
            "target": target,
            "n_compare_to": len(other_dfs),
            "n_legacy": len(legacy_dfs),
            "wlt": metrics_payload["wlt"],
        }},
    )
    return artifact
