"""evaluate runner: ExperimentSpec(kind="evaluate") → metric + comparison artifact.

Flow:
    1. Extract runner meta options from ``spec.evaluation.kwargs`` (target / split /
       task_set / column / tie_eps / legacy_csvs / leaderboard / etc.).
    2. Load target artifact → 7-column metric DataFrame via ``evaluate_artifact``.
    3. Other new artifacts from ``compare_to`` + old phase CSVs from ``legacy_csvs``
       are merged via wide pivot on the same column, then W/L/T is computed.
    4. Outputs:
       - ``metrics.csv``: target's own 7-column metric table.
       - ``comparison.csv``: long ``[dataset, metric, target, other_label, other,
         delta, verdict]``. Flattens ``compute_wlt`` results row by row.
       - ``comparison.md``: markdown table produced by ``format_pivot_md``.
       - (optional) ``leaderboard.csv``: result of ``compare_to_leaderboard``.
    5. Register (kind/name/run_id) of target + all comparison artifacts in manifest.inputs.
       wlt summary + target meta in manifest.metrics.

Unifies the responsibility of the old ``compare_experiments.py`` / ``compare_all_experiments.py`` / ``eval_stl_ensemble.py``
into a single recipe-driven entry point.

Why:
    base prediction artifact, stack artifact, and evaluate artifact can be traced on a single
    lineage graph. ``qadmet lineage show evaluate
    leaderboard_diff`` links even old phase CSVs as metadata (legacy CSV is
    not an artifact, so it is not registered in inputs — only via manifest.metrics.legacy_csvs
    recorded).
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
    """Load one artifact by latest or run_id pin (FileNotFoundError on miss)."""
    if run_id is not None:
        return store.load(kind, name, run_id)
    art = store.latest(kind, name)
    if art is None:
        raise FileNotFoundError(f"artifact not found: {kind}/{name}")
    return art


@RUNNER_REGISTRY.register("evaluate")
def run_evaluate(*, spec: ExperimentSpec, store: ArtifactStore, repo_root: Path) -> Artifact:
    """Executes one evaluate recipe and returns the artifact.

    Args:
        spec: ExperimentSpec(kind="evaluate") — evaluation is required.
        store: ArtifactStore.
        repo_root: basis for extracting git/env meta + resolving relative paths of legacy_csvs.

    Returns:
        Generated evaluate Artifact (data/ contains metrics.csv, comparison.csv,
        comparison.md, (optional) leaderboard.csv).

    Raises:
        ValueError: target cannot be determined, column unsupported, kwargs format error.
        FileNotFoundError: target/compare_to artifact or legacy CSV does not exist.
    """
    spec.validate()
    if spec.evaluation is None:
        raise ValueError("evaluate recipe requires an evaluation section.")

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
            f"evaluation.kwargs.column must be one of {VALID_COMPARISON_COLUMNS}: got {column!r}"
        )
    tie_eps: float = float(kwargs.pop("tie_eps", 0.005))

    # kind mapping for compare_to artifacts (name → kind, default "train").
    compare_kinds_raw = kwargs.pop("compare_kinds", {}) or {}
    if not isinstance(compare_kinds_raw, dict):
        raise TypeError(
            f"evaluation.kwargs.compare_kinds must be a dict: {type(compare_kinds_raw)}"
        )
    compare_kinds: dict[str, str] = {str(k): str(v) for k, v in compare_kinds_raw.items()}

    base_run_ids_raw = kwargs.pop("base_run_ids", {}) or {}
    if not isinstance(base_run_ids_raw, dict):
        raise TypeError(
            f"evaluation.kwargs.base_run_ids must be a dict: {type(base_run_ids_raw)}"
        )
    base_run_ids: dict[str, str] = {str(k): str(v) for k, v in base_run_ids_raw.items()}

    legacy_csvs_raw = kwargs.pop("legacy_csvs", {}) or {}
    if not isinstance(legacy_csvs_raw, dict):
        raise TypeError(
            f"evaluation.kwargs.legacy_csvs must be a dict: {type(legacy_csvs_raw)}"
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
            f"evaluation.kwargs.leaderboard must be a dict (or unspecified): {type(leaderboard_opts)}"
        )

    if kwargs:
        # Explicitly reject unknown keys (debugging-friendly).
        raise ValueError(
            f"Unknown key in evaluation.kwargs: {sorted(kwargs)}. "
            f"supported: target, target_kind, target_run_id, split, task_set, column, tie_eps, "
            f"compare_kinds, base_run_ids, legacy_csvs, leaderboard."
        )

    bases = list(eval_spec.compare_to)
    if target is None:
        if not bases:
            raise ValueError(
                "evaluate recipe requires target or compare_to — "
                "specify via evaluation.kwargs.target (recommended) or evaluation.compare_to[0]."
            )
        target = bases[0]
    target = str(target)

    # ----- load artifact -----
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
            raise FileNotFoundError(f"legacy CSV not found: {label}={path}")
        legacy_dfs[label] = load_legacy_csv(path)

    # ----- create artifact -----
    artifact = store.create(spec, repo_root=repo_root, inputs=inputs)

    # 1) target metric's own table.
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

    # 3) leaderboard (optional).
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
        except Exception as e:  # noqa: BLE001  — network failure only skips leaderboard.
            logger.warning(f"leaderboard call failed: {e}")
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
