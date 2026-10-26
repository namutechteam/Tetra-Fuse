"""stack runner: ExperimentSpec(kind="stack") → ensemble prediction artifact.

Flow:
    1. ``spec.ensemble.bases`` for each name, the most recent train artifact's
       ``predictions.parquet`` loaded as a ``PredictionSet`` (lineage input).
    2. Bundle the input base artifacts as ``InputRef`` and pass them to ``store.create``,
       creating this stack artifact (manifest.inputs recorded automatically).
    3. Inner-join on (seed, task, smiles) via ``MetaDataset.from_predictionsets``.
    4. Build a strategy instance from ``STRATEGIES`` → ``fit`` → ``predict``.
    5. Attach a split column to the resulting long DataFrame, then save as ``predictions.parquet``.

The legacy ``run_stacking_v1.stack_simple_average`` is equivalently expressed by this runner + AverageStrategy's
``weights=(0.7, 0.3)`` combination.

Why:
    Collects the lineage of the base prediction artifact and the stack artifact in one manifest
    location. The next-stage evaluate runner takes the ``stack`` artifact as input
    and performs the leaderboard comparison on the same graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qadmet.core.artifacts import Artifact, ArtifactStore, InputRef
from qadmet.core.experiment import RUNNER_REGISTRY
from qadmet.core.logging import get_logger
from qadmet.core.specs import ExperimentSpec

logger = get_logger("qadmet.runners.stack")


# ---------------------------------------------------------------------------- helpers


def _resolve_base_predictionsets(
    bases: tuple[str, ...],
    store: ArtifactStore,
    *,
    base_kind: str,
    base_run_ids: dict[str, str] | None,
):
    """Base names → (list of PredictionSet, list of InputRef).

    Args:
        bases: tuple of train artifact names (recipe ``name``).
        store: ArtifactStore.
        base_kind: defaults to ``"train"``. Use ``ensemble.kwargs.base_kind`` to point to ``predict`` or
            other kinds.
        base_run_ids: name → run_id mapping. For specified bases, use that run instead of the latest.

    Raises:
        FileNotFoundError: raised if any base's artifact or predictions.parquet does not exist.
    """
    from qadmet.ensembles.prediction_set import PredictionSet

    pred_sets = []
    inputs: list[InputRef] = []
    for name in bases:
        run_id = (base_run_ids or {}).get(name)
        ps = PredictionSet.from_artifact(name, store, run_id=run_id, kind=base_kind)
        if ps.artifact is None:  # pragma: no cover  — from_artifact always fills this in.
            raise RuntimeError(f"PredictionSet has no artifact: {name!r}")
        pred_sets.append(ps)
        inputs.append(
            InputRef(kind=ps.artifact.kind, name=ps.artifact.name, run_id=ps.artifact.run_id)
        )
    return pred_sets, inputs


# ---------------------------------------------------------------------------- runner


@RUNNER_REGISTRY.register("stack")
def run_stack(*, spec: ExperimentSpec, store: ArtifactStore, repo_root: Path) -> Artifact:
    """Execute a single stack recipe and return the artifact.

    Outputs (relative to artifact.data_dir):
        - ``predictions.parquet``: ``[seed, task, split, smiles, y_true, y_pred]`` long.
          defaults to ``split="test"``. Use ``ensemble.kwargs.split`` to specify a different split.
        - ``manifest.yaml`` (artifact root): records base artifact (kind/name/run_id) in inputs,
          and stores base count / strategy / split metadata in metrics.

    Fit/predict separation:
        If ``ensemble.kwargs.fit_split`` is specified, first run ``strategy.fit`` with a
        separate MetaDataset (for that split), then call ``predict`` with the main ``split``
        MetaDataset. The standard use case is a pattern like ``InverseVarianceStrategy``, which
        trains on val residuals and then applies to test. If ``fit_split`` is not specified, a single MetaDataset
        is used for both fit + predict (compatible with the legacy ``AverageStrategy``).

    Args:
        spec: ExperimentSpec(kind="stack") — ensemble is required.
        store: ArtifactStore.
        repo_root: repo root used to extract git info and environment metadata.

    Returns:
        The generated stack Artifact.
    """
    spec.validate()
    if spec.ensemble is None:
        raise ValueError("stack recipe requires an ensemble section.")
    if not spec.ensemble.bases:
        raise ValueError("ensemble.bases is empty — specify a base to combine.")

    # import ensembles package — triggers STRATEGIES/META_MODELS registration.
    import qadmet.ensembles  # noqa: F401
    from qadmet.ensembles.meta_dataset import MetaDataset
    from qadmet.ensembles.registry import STRATEGIES

    kwargs_raw: dict[str, Any] = dict(spec.ensemble.kwargs)
    # runner meta options are extracted from kwargs and not passed to the strategy constructor.
    split: str = str(kwargs_raw.pop("split", "test"))
    fit_split_raw = kwargs_raw.pop("fit_split", None)
    fit_split: str | None = None if fit_split_raw is None else str(fit_split_raw)
    base_kind: str = str(kwargs_raw.pop("base_kind", "train"))
    base_run_ids: dict[str, str] | None = kwargs_raw.pop("base_run_ids", None)
    if base_run_ids is not None and not isinstance(base_run_ids, dict):
        raise TypeError(
            f"ensemble.kwargs.base_run_ids must be a dict: {type(base_run_ids)}"
        )

    # task type derivation: kwargs.task_set (TASK_SETS lookup) → kwargs.task_types (explicit dict).
    # both are merged into task_types and passed to the strategy constructor (if absent, the strategy applies defaults).
    task_set_name = kwargs_raw.pop("task_set", None)
    task_types_explicit = kwargs_raw.pop("task_types", None)
    if task_types_explicit is not None and not isinstance(task_types_explicit, dict):
        raise TypeError(
            f"ensemble.kwargs.task_types must be a dict: {type(task_types_explicit)}"
        )
    resolved_task_types: dict[str, str] = {}
    if task_set_name is not None:
        from qadmet.tasks import TASK_SETS

        task_set = TASK_SETS.get(str(task_set_name))
        resolved_task_types.update(
            {t: tt for t, tt in zip(task_set.tasks, task_set.task_types())}
        )
    if task_types_explicit:
        resolved_task_types.update({str(k): str(v) for k, v in task_types_explicit.items()})

    # pass spec.ensemble.meta_model to the strategy as the meta_model_family argument
    # (currently only OOFXGBoostStrategy supports this, but the same pattern will apply to HybridBlend etc. in the future).
    if spec.ensemble.meta_model is not None:
        kwargs_raw.setdefault("meta_model_family", spec.ensemble.meta_model)
    if resolved_task_types:
        kwargs_raw.setdefault("task_types", resolved_task_types)

    pred_sets, inputs = _resolve_base_predictionsets(
        spec.ensemble.bases,
        store,
        base_kind=base_kind,
        base_run_ids=base_run_ids,
    )
    logger.info(
        "stack base resolved",
        extra={
            "payload": {
                "bases": list(spec.ensemble.bases),
                "base_kind": base_kind,
                "split": split,
                "n_runs": len(pred_sets),
            }
        },
    )

    artifact = store.create(spec, repo_root=repo_root, inputs=inputs)

    meta = MetaDataset.from_predictionsets(pred_sets, split=split)
    strategy = STRATEGIES.build_with_kwargs(spec.ensemble.strategy, **kwargs_raw)
    if fit_split is not None and fit_split != split:
        fit_meta = MetaDataset.from_predictionsets(pred_sets, split=fit_split)
        strategy.fit(fit_meta)
    else:
        strategy.fit(meta)
    pred_df = strategy.predict(meta)

    # insert the split column at the same position as in the base train runner schema.
    if "split" in pred_df.columns:
        pred_df = pred_df.drop(columns=["split"])
    pred_df.insert(2, "split", split)
    pred_df = pred_df[["seed", "task", "split", "smiles", "y_true", "y_pred"]]

    out_pq = artifact.data_dir / "predictions.parquet"
    pred_df.to_parquet(out_pq, index=False)
    store.register_output(artifact, "predictions.parquet")

    store.set_metrics(
        artifact,
        {
            "n_bases": len(spec.ensemble.bases),
            "bases": list(spec.ensemble.bases),
            "strategy": spec.ensemble.strategy,
            "split": split,
            "fit_split": fit_split,
            "n_predictions": int(len(pred_df)),
        },
    )
    return artifact
