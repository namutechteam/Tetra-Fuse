"""stack runner: ExperimentSpec(kind="stack") → ensemble 예측 artifact.

흐름:
    1. ``spec.ensemble.bases`` 의 각 이름으로 가장 최근 train artifact 의
       ``predictions.parquet`` 를 ``PredictionSet`` 으로 로드한다 (lineage 입력).
    2. 입력 base artifact 들을 ``InputRef`` 로 묶어 ``store.create`` 에 넘기고
       이번 stack artifact 를 만든다 (manifest.inputs 자동 기록).
    3. ``MetaDataset.from_predictionsets`` 로 (seed, task, smiles) inner-join.
    4. ``STRATEGIES`` 에서 strategy 인스턴스 빌드 → ``fit`` → ``predict``.
    5. 결과 long DataFrame 에 split 컬럼 부착 후 ``predictions.parquet`` 로 저장.

옛 ``run_stacking_v1.stack_simple_average`` 가 이 runner + AverageStrategy 의
``weights=(0.7, 0.3)`` 조합으로 동등 표현된다.

Why:
    base prediction artifact 와 stack artifact 의 lineage 를 manifest 한 곳에
    모은다. 다음 단계인 evaluate runner 가 ``stack`` artifact 를 입력으로 받아
    leaderboard 비교까지 동일한 그래프 위에서 수행하게 된다.
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
    """base 이름들 → (PredictionSet 리스트, InputRef 리스트).

    Args:
        bases: train artifact 이름 (recipe ``name``) 의 튜플.
        store: ArtifactStore.
        base_kind: 기본 ``"train"``. ``ensemble.kwargs.base_kind`` 로 ``predict`` 등
            다른 kind 를 가리킬 수 있다.
        base_run_ids: 이름 → run_id 매핑. 명시된 base 는 latest 대신 해당 run 사용.

    Raises:
        FileNotFoundError: 어느 한 base 라도 artifact 또는 predictions.parquet 미존재.
    """
    from qadmet.ensembles.prediction_set import PredictionSet

    pred_sets = []
    inputs: list[InputRef] = []
    for name in bases:
        run_id = (base_run_ids or {}).get(name)
        ps = PredictionSet.from_artifact(name, store, run_id=run_id, kind=base_kind)
        if ps.artifact is None:  # pragma: no cover  — from_artifact 는 항상 채움.
            raise RuntimeError(f"PredictionSet 에 artifact 가 없습니다: {name!r}")
        pred_sets.append(ps)
        inputs.append(
            InputRef(kind=ps.artifact.kind, name=ps.artifact.name, run_id=ps.artifact.run_id)
        )
    return pred_sets, inputs


# ---------------------------------------------------------------------------- runner


@RUNNER_REGISTRY.register("stack")
def run_stack(*, spec: ExperimentSpec, store: ArtifactStore, repo_root: Path) -> Artifact:
    """stack recipe 한 건을 실행하고 artifact 를 반환한다.

    산출물 (artifact.data_dir 기준):
        - ``predictions.parquet``: ``[seed, task, split, smiles, y_true, y_pred]`` long.
          기본 ``split="test"``. ``ensemble.kwargs.split`` 로 다른 split 도 지정 가능.
        - ``manifest.yaml`` (artifact root): inputs 에 base artifact (kind/name/run_id)
          기록, metrics 에 base 수·strategy·split 메타 저장.

    Fit/predict 분리:
        ``ensemble.kwargs.fit_split`` 가 명시되면 별도 MetaDataset (해당 split) 으로
        ``strategy.fit`` 을 먼저 수행한 뒤 본 ``split`` MetaDataset 으로 ``predict``
        를 호출한다. ``InverseVarianceStrategy`` 처럼 val 잔차로 학습 후 test 에
        적용하는 패턴이 표준 사용 사례. ``fit_split`` 미지정이면 단일 MetaDataset
        으로 fit + predict 둘 다 (옛 ``AverageStrategy`` 호환).

    Args:
        spec: ExperimentSpec(kind="stack") — ensemble 필수.
        store: ArtifactStore.
        repo_root: git 정보·환경 메타 추출용 repo 루트.

    Returns:
        생성된 stack Artifact.
    """
    spec.validate()
    if spec.ensemble is None:
        raise ValueError("stack recipe 에 ensemble 섹션이 필요합니다.")
    if not spec.ensemble.bases:
        raise ValueError("ensemble.bases 가 비어있습니다 — 결합할 base 를 지정하세요.")

    # ensembles 패키지 import — STRATEGIES/META_MODELS 등록 트리거.
    import qadmet.ensembles  # noqa: F401
    from qadmet.ensembles.meta_dataset import MetaDataset
    from qadmet.ensembles.registry import STRATEGIES

    kwargs_raw: dict[str, Any] = dict(spec.ensemble.kwargs)
    # runner 메타 옵션은 kwargs 에서 추출 후 strategy 생성자에 넘기지 않는다.
    split: str = str(kwargs_raw.pop("split", "test"))
    fit_split_raw = kwargs_raw.pop("fit_split", None)
    fit_split: str | None = None if fit_split_raw is None else str(fit_split_raw)
    base_kind: str = str(kwargs_raw.pop("base_kind", "train"))
    base_run_ids: dict[str, str] | None = kwargs_raw.pop("base_run_ids", None)
    if base_run_ids is not None and not isinstance(base_run_ids, dict):
        raise TypeError(
            f"ensemble.kwargs.base_run_ids 는 dict 여야 합니다: {type(base_run_ids)}"
        )

    # task type 도출: kwargs.task_set (TASK_SETS lookup) → kwargs.task_types (명시 dict).
    # 둘 다 strategy 생성자에 task_types 로 합쳐 넘긴다 (없으면 strategy 가 기본 처리).
    task_set_name = kwargs_raw.pop("task_set", None)
    task_types_explicit = kwargs_raw.pop("task_types", None)
    if task_types_explicit is not None and not isinstance(task_types_explicit, dict):
        raise TypeError(
            f"ensemble.kwargs.task_types 는 dict 여야 합니다: {type(task_types_explicit)}"
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

    # spec.ensemble.meta_model 을 strategy 인자 meta_model_family 로 전달
    # (현재는 OOFXGBoostStrategy 만 지원하지만 추후 HybridBlend 등도 동일 패턴).
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

    # split 컬럼을 base train runner schema 와 동일한 자리에 삽입.
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
