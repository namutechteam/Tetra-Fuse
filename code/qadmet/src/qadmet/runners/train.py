"""train runner: ExperimentSpec(kind="train") → 모델 학습 + test 예측 artifact.

흐름:
    1. spec.dataset.name → TaskSet → task_names + task_types.
    2. spec.trainer.seeds 별로:
         a. ``load_fold(seed)`` → TdcAdmetFold.
         b. ``prepare_fold_loaders(fold, spec.features)`` → 모달리티 stack 된 DataLoader.
         c. inverse-freq task weights (옛 ``compute_inverse_freq_weights``).
         d. ``MultitaskLitModule`` 빌드 (spec.model + 위 메타).
         e. ``build_trainer`` + ``trainer.fit``.
         f. task 별 ``trainer.predict(ckpt_path="best")`` → predictions parquet.
    3. seed 결과를 artifact 에 저장 + manifest metrics 갱신.

본 runner 는 actual TDC fold + chemprop dataloader 를 구동한다 (인터넷 + 사전 계산
parquet + (옵션) GPU 필요). 단위 테스트는 mock fold + 더미 parquet 으로.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from qadmet.core.artifacts import Artifact, ArtifactStore
from qadmet.core.experiment import RUNNER_REGISTRY
from qadmet.core.logging import get_logger
from qadmet.core.specs import BackboneSpec, ExperimentSpec, FeatureSpec, ModelSpec, TrainerSpec

logger = get_logger("qadmet.runners.train")


# ---------------------------------------------------------------------------- helpers


def _compute_inverse_freq_weights(
    samples_per_task: Mapping[str, int], task_names: Sequence[str]
):
    """옛 ``pipeline_mtl.compute_inverse_freq_weights`` 와 수치 동등.

    inverse-freq + mean=1 정규화. 결과 텐서는 LitModule.task_weights buffer 로.
    """
    import torch

    counts = np.array([samples_per_task[t] for t in task_names], dtype=np.float32)
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _resolve_parquet_root(spec: ExperimentSpec, repo_root: Path) -> Path:
    """recipe dataset.kwargs.parquet_root → 절대경로. default ``data/embeddings``."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    raw = kw.get("parquet_root", "data/embeddings")
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _resolve_parquet_paths(
    spec: ExperimentSpec, repo_root: Path, modalities: Sequence[str]
) -> dict[str, Path]:
    """modality 이름 → parquet 절대경로. dataset.kwargs.parquet_paths 명시 우선.

    LMDB modality (``lmdb_paths`` 에 등록된) + synthetic modality (collate 단계에서
    SMILES 즉석 변환 — e.g. ``graphmvp_data``) 는 결과에서 제외 — parquet 존재 검증
    대상에서 빠진다.
    """
    from qadmet.datasets.loaders import SYNTHETIC_MODALITY_KEYS

    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    explicit: dict[str, str] = dict(kw.get("parquet_paths", {}))
    lmdb_keys = set((kw.get("lmdb_paths") or {}).keys())
    root = _resolve_parquet_root(spec, repo_root)
    out: dict[str, Path] = {}
    for name in modalities:
        if name in lmdb_keys or name in SYNTHETIC_MODALITY_KEYS:
            continue
        if name in explicit:
            p = Path(explicit[name])
            out[name] = p if p.is_absolute() else repo_root / p
        else:
            out[name] = root / f"{name}.parquet"
    return out


def _resolve_lmdb_paths(
    spec: ExperimentSpec, repo_root: Path, modalities: Sequence[str]
) -> dict[str, Path]:
    """modality 이름 → LMDB 절대경로. dataset.kwargs.lmdb_paths 명시만 (default 없음)."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    explicit: dict[str, str] = dict(kw.get("lmdb_paths") or {})
    out: dict[str, Path] = {}
    for name in modalities:
        if name in explicit:
            p = Path(explicit[name])
            out[name] = p if p.is_absolute() else repo_root / p
    return out


def _inject_d_xd(spec: BackboneSpec, descriptor_dim_total: int) -> BackboneSpec:
    """DMPNN backbone 의 ``d_xd`` 를 descriptor 합으로 주입한 새 BackboneSpec 반환.

    Why:
        chemprop ``MPNN.fingerprint`` 가 ``X_d`` 를 fp 뒤에 concat 한다 →
        backbone output_dim = ``d_h + d_xd``. 신규 코드는 descriptor 를 X_d 로 흐르게
        하므로, runner 가 modality dim 합을 알아낸 시점에 backbone 에 자동 전달한다.
        recipe 에서 d_xd 를 직접 적지 않아도 동작하도록.

    Args:
        spec: 원본 BackboneSpec.
        descriptor_dim_total: descriptors 차원 합. 0 이면 spec 그대로 반환 (no-op).

    Returns:
        type/consumes 보존, kwargs 에 ``d_xd`` 추가된 새 BackboneSpec.
        DMPNN 이 아닌 backbone 에 대해서는 spec 그대로 반환 (X_d 미사용).
        recipe 가 이미 ``d_xd`` 를 명시했다면 사용자 의도 우선 — 덮어쓰지 않는다.
    """
    if spec.type != "dmpnn" or descriptor_dim_total <= 0:
        return spec
    if "d_xd" in spec.kwargs:
        return spec
    new_kwargs = {**dict(spec.kwargs), "d_xd": descriptor_dim_total}
    return dataclasses.replace(spec, kwargs=new_kwargs)


def _build_lit_module(
    *,
    model_spec: ModelSpec,
    loss_spec,
    n_tasks: int,
    task_types: list[str],
    ordered_tasks: list[str],
    embedding_dims: Mapping[str, int],
    descriptors: tuple[str, ...],
    embeddings: tuple[str, ...],
    descriptor_dim_total: int,
    task_set,
    grouping_override=None,
    task_weights,
    lr: float,
    init_lr: float = 1e-4,
    final_lr: float = 1e-4,
    warmup_epochs: int = 2,
    lr_lora: float | None = None,
    lr_fusion: float | None = None,
    head_warmup_epochs: int = 0,
):
    """ModelSpec + 메타 → ``MultitaskLitModule`` 인스턴스.

    Args:
        embedding_dims: ``{name: D}`` embeddings 의 차원만 (descriptors 는 X_d 흐름).
        descriptor_dim_total: descriptors 차원 합 — DMPNN backbone 의 d_xd 자동 주입.

    Note:
        head.routing 에 따라 ``task_to_group`` / ``groups`` 를 자동 산출한다
        (routing 값과 무관하게 ``task_set.initial_grouping`` 을 그대로 사용 —
        그룹 수는 TaskSet 이 결정한다).
    """
    from qadmet.modules.multitask.lit_module import MultitaskLitModule

    head_spec = model_spec.head
    task_to_group: dict[str, str] | None = None
    groups: list[str] | None = None
    if head_spec.type in ("grouped", "grouped_routed"):
        # routing 은 "groups 를 TaskSet.initial_grouping 에서 그대로 가져온다" 는 뜻일 뿐,
        # 그룹 구성/수는 routing 값이 아니라 TaskSet 이 결정한다 (causal6=6, mech=7 등).
        # "admet_7_group" 은 구 manifest/recipe 하위호환용 레거시 별칭.
        routing = head_spec.routing or "taskset_grouping"
        if routing not in ("taskset_grouping", "admet_7_group"):
            raise NotImplementedError(
                f"head.routing={routing!r} 미지원 — 'taskset_grouping' 만 (레거시 별칭: 'admet_7_group')."
            )
        # grouping_override (그룹 탐색용) 가 있으면 TaskSet 기본 grouping 대신 사용.
        grouping = grouping_override or task_set.initial_grouping
        if not grouping:
            raise ValueError(f"TaskSet({task_set.name}) 에 initial_grouping 이 비어있음")
        task_to_group = {t: g for g, ts in grouping.items() for t in ts}
        groups = list(grouping.keys())
        if grouping_override is not None:
            # override 는 task_set 의 모든 task 를 정확히 한 번씩 덮어야 한다.
            covered = sorted(task_to_group)
            expected = sorted(task_set.tasks)
            if covered != expected:
                missing = sorted(set(expected) - set(covered))
                extra = sorted(set(covered) - set(expected))
                raise ValueError(
                    f"grouping_override 가 task_set({task_set.name}) 를 정확히 덮지 못함 — "
                    f"missing={missing}, extra={extra}"
                )

    primary_spec = _inject_d_xd(model_spec.backbone, descriptor_dim_total)

    return MultitaskLitModule(
        backbone_spec=primary_spec,
        extra_backbone_specs=model_spec.extra_backbones,
        head_spec=head_spec,
        loss_spec=loss_spec,
        n_tasks=n_tasks,
        task_types=task_types,
        fusion_spec=model_spec.fusion,
        ordered_tasks=ordered_tasks,
        modality_dims=embedding_dims,
        descriptors=descriptors,
        embeddings=embeddings,
        task_to_group=task_to_group,
        groups=groups,
        task_weights=task_weights,
        lr=lr,
        init_lr=init_lr,
        final_lr=final_lr,
        warmup_epochs=warmup_epochs,
        lr_lora=lr_lora,
        lr_fusion=lr_fusion,
        aux_loss_spec=model_spec.aux_loss,
        head_warmup_epochs=head_warmup_epochs,
    )


def _print_task_diagnostics(
    model,
    task_names: Sequence[str],
    seed: int,
) -> None:
    """학습 후 per-task σ / inverse-freq coefficient / loss 통계를 로그에 출력한다."""
    import numpy as np

    lm = model.loss_module
    has_kendall = hasattr(lm, "log_vars")
    has_gradnorm = hasattr(lm, "w") and hasattr(lm, "alpha")
    if not (has_kendall or has_gradnorm):
        return

    if has_kendall:
        col1_name = "sigma"
        col1 = (lm.log_vars / 2).exp().detach().cpu().numpy()  # (K,)
    else:
        col1_name = "w(gn)"
        col1 = lm.w.detach().cpu().numpy()                      # (K,)
    tw = (
        lm.task_weights_buf.detach().cpu().numpy()
        if hasattr(lm, "task_weights_buf")
        else np.ones(len(task_names))
    )

    history = getattr(model, "_task_loss_epoch_means", [])
    if history:
        arr = np.array(history)  # (epochs, K)
        loss_mean = arr.mean(axis=0)
        loss_min = arr.min(axis=0)
        loss_max = arr.max(axis=0)
    else:
        loss_mean = loss_min = loss_max = np.full(len(task_names), float("nan"))

    col_w = max(len(n) for n in task_names)
    header = (
        f"{'Task':<{col_w}}  {col1_name:>7}  {'coeff':>7}  "
        f"{'loss_mean':>10}  {'loss_min':>10}  {'loss_max':>10}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        f"=== Task Diagnostics (seed={seed}, epochs={len(history)}) ===",
        header,
        sep,
    ]
    for i, name in enumerate(task_names):
        lines.append(
            f"{name:<{col_w}}  {col1[i]:>7.4f}  {tw[i]:>7.4f}"
            f"  {loss_mean[i]:>10.4f}  {loss_min[i]:>10.4f}  {loss_max[i]:>10.4f}"
        )
    lines.append(sep)
    logger.info("\n".join(lines))


def _log_seed_test_metrics(
    pred_rows_for_seed: list[dict[str, Any]],
    *,
    seed: int,
    task_set_name: str,
) -> None:
    """시드 한 건의 test 예측 → TDC leaderboard metric 표 stdout 출력.

    학습이 오래 걸릴 때 시드 끝날 때마다 진단용으로 즉시 결과 확인.
    final ``predictions.parquet`` + ``per_task_metric_df`` 와 동일 메트릭 사용.
    """
    from qadmet.evaluation.tdc_benchmark import per_task_metric_df

    if not pred_rows_for_seed:
        logger.warning("seed test metrics 미출력 — pred_rows 비어있음", extra={"payload": {"seed": seed}})
        return
    df = pd.DataFrame(pred_rows_for_seed)
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        return
    metric_df = per_task_metric_df(test_df, task_set=task_set_name)

    col_w = max(len(str(d)) for d in metric_df["dataset"]) if not metric_df.empty else 8
    header = f"{'Task':<{col_w}}  {'metric':>8}  {'score':>8}  {'n_test':>6}"
    sep = "-" * len(header)
    lines = [
        "",
        f"=== TDC Leaderboard Metrics (seed={seed}, split=test) ===",
        header,
        sep,
    ]
    for _, row in metric_df.iterrows():
        score = row["mean"]  # n_seeds=1 이므로 mean == 그 시드의 score
        score_str = f"{score:>8.4f}" if pd.notna(score) else f"{'NaN':>8}"
        lines.append(
            f"{str(row['dataset']):<{col_w}}  {str(row['metric']):>8}  {score_str}  {int(row['n_test']):>6}"
        )
    lines.append(sep)
    logger.info("\n".join(lines))


def _build_predict_trainer(runtime: Mapping[str, Any]) -> Any:
    """Predict 전용 single-device Trainer. DDP fit 후 rank-0 에서만 호출된다.

    callbacks 를 비워 ModelCheckpoint 재생성으로 best.ckpt 덮어쓰기 방지.
    logger=False 로 CSVLogger 도 막아 추가 디렉토리 생성 없음.
    """
    import lightning as L

    accelerator = runtime.get("accelerator")
    if accelerator is None:
        try:
            import torch

            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            accelerator = "cpu"
    return L.Trainer(
        devices=1,
        num_nodes=1,
        accelerator=accelerator,
        strategy="auto",
        logger=False,
        callbacks=[],
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def _predict_per_task(
    *,
    trainer,
    model,
    loader,
    task_idx: int,
    task_name: str,
    task_type: str,
    log_tasks,
    output_scaler,
    reg_task_idx_in_scaler: Mapping[int, int],
    ckpt_path: str | None = "best",
) -> np.ndarray:
    """task 한 개의 예측 → 원본 스케일 ``numpy (N,)``.

    Args:
        loader: per-task DataLoader (val_loaders 또는 test_loaders 의 한 entry).
        task_idx: 모델 출력 ``(N, n_tasks)`` 에서 해당 task 의 컬럼.
        task_type: ``"regression"`` 또는 ``"classification"``.
        log_tasks: ``log1p`` 적용된 regression task 이름 set.
        output_scaler: ``(StandardScaler, tuple(reg_idx_in_n_tasks))`` 또는 None.
        reg_task_idx_in_scaler: ``{task_idx: scaler_column_idx}``. regression task 만.
        ckpt_path: lightning ``predict`` 의 ckpt_path. ``"best"`` 는 학습 trainer 가
            관리하는 best 모델, 절대경로 문자열은 명시 ckpt (DDP 분리 predict 에 사용).
            None 이면 현재 model 가중치 그대로.

    후처리:
        - regression: ``out_unscaled = out * scale_[j] + mean_[j]`` (scaler 가 fit 됐을 때).
            ``log_tasks`` 면 추가 ``expm1``.
        - classification: ``sigmoid`` (raw logit → prob, STL runner 와 동일).
    """
    import torch

    preds_list = trainer.predict(model, loader, ckpt_path=ckpt_path)
    # bf16-mixed precision 학습 시 preds 가 bf16 이라 numpy 직접 변환 불가 → float() 로 fp32 cast.
    preds = torch.cat(preds_list, dim=0).float().cpu().numpy()  # (N, n_tasks)
    out = preds[:, task_idx].astype(np.float64)
    if task_type == "regression":
        if output_scaler is not None and task_idx in reg_task_idx_in_scaler:
            scaler, _ = output_scaler
            j = reg_task_idx_in_scaler[task_idx]
            out = out * float(scaler.scale_[j]) + float(scaler.mean_[j])
        if task_name in log_tasks:
            out = np.expm1(out)
    elif task_type == "classification":
        out = 1.0 / (1.0 + np.exp(-out))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------- runner


@RUNNER_REGISTRY.register("train")
def run_train(
    *,
    spec: ExperimentSpec,
    store: ArtifactStore,
    repo_root: Path,
    runtime: dict | None = None,
    force: bool = False,
) -> Artifact:
    """train recipe 한 건을 실행하고 artifact 를 반환한다.

    Dispatch:
        - ``spec.model.type == "multitask"``: 본 함수의 MTL 경로.
        - ``spec.model.type == "singletask"``: ``runners.train_stl.run_train_stl`` 위임.

    Args:
        runtime: DDP 등 실행 환경 메타 (`devices`, `strategy`, `num_nodes`,
            `accelerator`, `find_unused_parameters`, `world_size`, `cli_args` 등).
            None 이면 single-device 학습. manifest.runtime 으로 기록 (해시 영향 없음).

    산출물 (artifact.data_dir 기준, MTL 경로):
        - ``seed_{N}/best.ckpt``: 시드별 lightning 체크포인트.
        - ``predictions.parquet``: ``[seed, task, split, smiles, y_true, y_pred]`` long 포맷.
          ``split ∈ {"val", "test"}`` 둘 다 작성한다. val 예측은 OOF stacking
          (Step 4.2 InverseVariance 등) 의 입력으로 사용된다.
        - ``manifest.yaml`` (artifact root): metrics 에 시드별 train/val loss 저장.

    DDP 동작:
        - 모든 rank 가 동일 spec 으로 진입. `store.create` 는 rank-0 만 호출하고
          다른 rank 는 barrier 후 동일 경로의 artifact 를 `store.load` 로 회수.
        - `trainer.fit` 은 모든 rank 참여, `trainer.predict` 는 rank-0 의 별도
          single-device Trainer 로만 (DistributedSampler 의 padding/reorder 회피).
        - 시드 간/시드 후 barrier 로 동기화. 종료 시 process group 정리.
    """
    spec.validate()
    if spec.dataset is None or spec.features is None or spec.model is None or spec.loss is None or spec.trainer is None:
        raise ValueError("train recipe: dataset/features/model/loss/trainer 모두 필요")

    # STL 분기 — model.type 으로 dispatch (recipe 의 kind 는 동일하게 "train").
    if spec.model.type == "singletask":
        from qadmet.runners.train_stl import run_train_stl

        return run_train_stl(spec=spec, store=store, repo_root=repo_root, force=force)
    if spec.model.type != "multitask":
        raise ValueError(
            f"model.type 은 'multitask' 또는 'singletask': {spec.model.type!r}"
        )

    # 의존 import 는 run 시점에만.
    from qadmet.core.distributed import (
        barrier_if_distributed,
        cleanup_process_group,
        init_process_group_if_torchrun,
        is_rank_zero,
    )
    from qadmet.datasets.loaders import prepare_fold_loaders
    from qadmet.datasets.tdc_admet import load_fold
    from qadmet.tasks import TASK_SETS
    from qadmet.training import build_trainer, extract_best_val_loss, run_seed_loop, SeedRun
    import qadmet.modules  # noqa: F401  — registry 등록 트리거 (backbones / fusion / heads / losses).

    runtime = runtime or {}
    # rank-zero artifact 생성 전에 process group 을 미리 init — fit 이전 barrier 가
    # 실제로 동작하려면 필수 (Lightning 은 already-initialized 면 재사용).
    pg_inited = init_process_group_if_torchrun()
    rank0 = is_rank_zero()
    logger.info(
        "train runner start",
        extra={"payload": {"rank0": rank0, "runtime": runtime, "pg_inited_early": pg_inited}},
    )

    task_set = TASK_SETS.get(spec.dataset.name)
    task_names: list[str] = list(task_set.tasks)
    task_types: list[str] = task_set.task_types()
    n_tasks = len(task_names)
    log_tasks_cfg = frozenset(task_set.log_transform_tasks())

    # artifact 생성 — rank-0 만 디스크 쓰기. 다른 rank 는 barrier 후 같은 path 로
    # load (run_id 가 deterministic 이라 rank-0 가 만든 manifest 가 같은 위치에 있음).
    if rank0:
        artifact = store.create(spec, repo_root=repo_root, runtime=runtime, force=force)
    barrier_if_distributed()
    if not rank0:
        # rank-0 가 만든 manifest.yaml 을 동일 경로에서 회수.
        from qadmet.core.artifacts import compute_run_id, git_info

        gi = git_info(repo_root)
        run_id = compute_run_id(spec, gi["sha"])
        artifact = store.load(spec.kind, spec.name, run_id)

    feature_spec: FeatureSpec = spec.features
    trainer_spec: TrainerSpec = spec.trainer

    from qadmet.datasets.loaders import expand_descriptors
    modality_keys = expand_descriptors(feature_spec.descriptors) + list(feature_spec.embeddings)
    parquet_paths = _resolve_parquet_paths(spec, repo_root, modality_keys)
    lmdb_paths = _resolve_lmdb_paths(spec, repo_root, modality_keys)
    # 학습 시작 전 누락 parquet/LMDB 즉시 fail.
    for name, path in parquet_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"modality {name!r} parquet 누락: {path}\n"
                f"  recipe.dataset.kwargs.parquet_paths 또는 parquet_root 를 확인하세요."
            )
    for name, path in lmdb_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"modality {name!r} LMDB 누락: {path}\n"
                f"  recipe.dataset.kwargs.lmdb_paths 를 확인하세요."
            )

    pred_rows: list[dict[str, Any]] = []
    seed_metrics: dict[int, dict[str, Any]] = {}

    parquet_root = _resolve_parquet_root(spec, repo_root)

    def fit_one(seed: int, seed_dir: Path) -> SeedRun:
        fold = load_fold(spec.dataset.name, seed=seed)
        loaders = prepare_fold_loaders(
            fold,
            feature_spec,
            task_names=task_names,
            task_types=task_types,
            parquet_root=parquet_root,
            parquet_paths=parquet_paths,
            lmdb_paths=lmdb_paths,
            batch_size=trainer_spec.batch_size,
            num_workers=trainer_spec.num_workers,
            seed=seed,
            apply_log_transform=True,
        )

        if spec.loss.task_weighting in (None, "uniform"):
            task_weights = None
        else:
            task_weights = _compute_inverse_freq_weights(fold.samples_per_task, task_names)
        # GradNorm 은 동적 task weight 를 학습 → inverse-freq prior 와 곱하면 해석 모호.
        # recipe 가 task_weighting!=null 로 지정해 둔 경우를 명시적으로 차단.
        if spec.loss.type == "mixed_task_gradnorm" and task_weights is not None:
            raise ValueError(
                "loss.type='mixed_task_gradnorm' 은 task_weighting=null (uniform) 만 허용합니다. "
                f"받음: task_weighting={spec.loss.task_weighting!r}."
            )
        # Kendall UW + PCGrad: 사용자가 GradNorm 대비 ablation 으로 쓰는 경우가 많고
        # PCGrad 는 weighted gradient 기반이므로 task_weighting=null 강제 (Kendall σ
        # 가 이미 reweighting 을 담당). recipe 에서 inverse-freq prior 도 켜 두면 두
        # 가중치가 곱해져 해석 모호. 명시적으로 차단.
        if spec.loss.type == "mixed_task_uncertainty_pcgrad" and task_weights is not None:
            raise ValueError(
                "loss.type='mixed_task_uncertainty_pcgrad' 은 task_weighting=null 만 허용합니다. "
                f"Kendall σ 가 이미 reweighting 을 담당. 받음: task_weighting={spec.loss.task_weighting!r}."
            )
        # descriptors 는 backbone X_d 로 흐르므로 LitModule 의 modality_dims 에서 제외.
        embedding_dims = {
            k: loaders.modality_dims[k]
            for k in feature_spec.embeddings
            if k in loaders.modality_dims
        }
        model = _build_lit_module(
            model_spec=spec.model,
            loss_spec=spec.loss,
            n_tasks=n_tasks,
            task_types=task_types,
            ordered_tasks=task_names,
            embedding_dims=embedding_dims,
            descriptors=feature_spec.descriptors,
            embeddings=feature_spec.embeddings,
            descriptor_dim_total=loaders.descriptor_dim_total,
            task_set=task_set,
            grouping_override=spec.dataset.grouping_override,
            task_weights=task_weights,
            lr=trainer_spec.lr,
            init_lr=trainer_spec.init_lr,
            final_lr=trainer_spec.final_lr,
            warmup_epochs=trainer_spec.warmup_epochs,
            lr_lora=trainer_spec.lr_lora,
            lr_fusion=trainer_spec.lr_fusion,
            head_warmup_epochs=trainer_spec.head_warmup_epochs,
        )
        # predict-only: fit + fit-trainer 생략 (기존 best.ckpt 로 예측만 재생성).
        # CSVLogger/ModelCheckpoint 를 만드는 fit-trainer 자체를 건너뛰어 기존
        # csv_logs / best.ckpt 를 건드리지 않는다. 단일 프로세스 실행 전제.
        predict_only = bool(runtime.get("predict_only", False))
        if predict_only:
            trainer = None
            logger.info(
                "predict-only: fit 생략, 기존 ckpt 로 예측 재생성",
                extra={"payload": {"seed": seed, "ckpt_dir": str(seed_dir)}},
            )
        else:
            trainer = build_trainer(
                trainer_spec,
                ckpt_dir=seed_dir,
                log_dir=seed_dir / "csv_logs",
                monitor="val_loss",
                mode="min",
                devices=runtime.get("devices", 1),
                strategy=runtime.get("strategy"),
                num_nodes=runtime.get("num_nodes", 1),
                accelerator=runtime.get("accelerator"),
                find_unused_parameters=runtime.get("find_unused_parameters", False),
                ddp_static_graph=runtime.get("ddp_static_graph", True),
                save_last=(trainer_spec.predict_ckpt == "last"),
            )
            trainer.fit(model, loaders.train_loader, loaders.val_loader)
            barrier_if_distributed()  # 모든 rank 가 fit 끝낼 때까지 sync

        # predict_ckpt='last' 면 last.ckpt 우선, 없으면 best.ckpt 로 fallback.
        # 본 옵션은 head warmup + gate-first 학습 시 val_loss 가 epoch 1 에 잡혀
        # best.ckpt 가 untrained gate 시점을 가리키는 문제를 회피하기 위한 용도.
        use_last = trainer_spec.predict_ckpt == "last"
        last_candidate = seed_dir / "last.ckpt"
        best_candidate = seed_dir / "best.ckpt"
        if use_last and last_candidate.exists():
            ckpt_candidate = last_candidate
        else:
            ckpt_candidate = best_candidate
        ckpt_path: Path | None = ckpt_candidate if ckpt_candidate.exists() else None

        # rank-0 외 rank 는 predict 스킵하고 barrier 에서 다음 seed 대기.
        # predict-only(trainer=None) 는 단일 프로세스 → 항상 global zero.
        if trainer is not None and not trainer.is_global_zero:
            # 진단 출력은 rank-0 의 _last_task_loss / log_vars 에만 의미 있음.
            # 다른 rank 는 즉시 다음 seed 로 넘어가도록 placeholder 반환.
            barrier_if_distributed()
            return SeedRun(seed=seed, ckpt_path=None, metrics={})

        _print_task_diagnostics(model, task_names, seed)

        # output_scaler: regression task 의 (scaler 컬럼 인덱스) 매핑.
        reg_task_idx_in_scaler: dict[int, int] = {}
        if loaders.output_scaler is not None:
            _, reg_idx_tuple = loaders.output_scaler
            reg_task_idx_in_scaler = {ti: j for j, ti in enumerate(reg_idx_tuple)}

        log_tasks_combined = log_tasks_cfg | loaders.log_tasks

        # ---- Predict: rank-0 의 single-device Trainer 로만 ----
        # DistributedSampler 의 padding/reorder 가 SMILES↔prediction 매핑을 깨뜨리는
        # 것을 회피한다. fit 의 DDP process group 은 살아있어도 새 Trainer 의
        # SingleDeviceStrategy 가 자체 dataloader 를 sampler 변조 없이 사용한다.
        predict_trainer = _build_predict_trainer(runtime)
        predict_ckpt = str(ckpt_candidate) if ckpt_candidate.exists() else None

        # task 별 val + test 예측.
        for task_idx, task_name in enumerate(task_names):
            task_type = task_types[task_idx]

            # val 예측 (OOF stacking 입력) — val_loaders 에 있는 task 만.
            if task_name in loaders.val_loaders:
                val_loader, val_smis, val_y_true = loaders.val_loaders[task_name]
                val_pred = _predict_per_task(
                    trainer=predict_trainer, model=model, loader=val_loader,
                    task_idx=task_idx, task_name=task_name, task_type=task_type,
                    log_tasks=log_tasks_combined,
                    output_scaler=loaders.output_scaler,
                    reg_task_idx_in_scaler=reg_task_idx_in_scaler,
                    ckpt_path=predict_ckpt,
                )
                for smi, yt, yp in zip(val_smis, val_y_true, val_pred):
                    pred_rows.append(
                        {
                            "seed": seed,
                            "task": task_name,
                            "split": "val",
                            "smiles": smi,
                            "y_true": float(yt),
                            "y_pred": float(yp),
                        }
                    )

            # test 예측.
            test_loader, smis, y_true = loaders.test_loaders[task_name]
            y_pred = _predict_per_task(
                trainer=predict_trainer, model=model, loader=test_loader,
                task_idx=task_idx, task_name=task_name, task_type=task_type,
                log_tasks=log_tasks_combined,
                output_scaler=loaders.output_scaler,
                reg_task_idx_in_scaler=reg_task_idx_in_scaler,
                ckpt_path=predict_ckpt,
            )
            for smi, yt, yp in zip(smis, y_true, y_pred):
                pred_rows.append(
                    {
                        "seed": seed,
                        "task": task_name,
                        "split": "test",
                        "smiles": smi,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    }
                )

        # predict-only 는 fit trainer 가 없어 best metric 을 추출할 수 없음 (NaN).
        metrics = {
            "best_val_loss": extract_best_val_loss(trainer)
            if trainer is not None
            else float("nan")
        }
        seed_metrics[seed] = metrics

        # 시드 종료 직후 TDC leaderboard metric stdout 출력 (긴 학습용 진단).
        seed_pred_rows = [r for r in pred_rows if r.get("seed") == seed]
        try:
            _log_seed_test_metrics(seed_pred_rows, seed=seed, task_set_name=spec.dataset.name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"seed metric 출력 실패 (학습은 계속): {e}", extra={"payload": {"seed": seed}})

        barrier_if_distributed()  # rank-0 predict 끝, 다른 rank 동기 해제
        return SeedRun(
            seed=seed, ckpt_path=ckpt_path, metrics=metrics,
        )

    runs = run_seed_loop(
        seeds=trainer_spec.seeds,
        fit_one=fit_one,
        ckpt_root=artifact.data_dir,
    )
    logger.info("all seeds done", extra={"payload": {"n_seeds": len(runs)}})

    # 최종 산출물 쓰기 — rank-0 만. non-rank-0 은 pred_rows / seed_metrics 가 비어 있음.
    if rank0:
        pred_df = pd.DataFrame(pred_rows)
        out_pq = artifact.data_dir / "predictions.parquet"
        pred_df.to_parquet(out_pq, index=False)
        store.register_output(artifact, "predictions.parquet")

        store.set_metrics(
            artifact,
            {
                "n_seeds": len(runs),
                "seeds": list(trainer_spec.seeds),
                "per_seed": seed_metrics,
            },
        )

    barrier_if_distributed()  # 모든 rank 가 종료 동기화 후 process group cleanup
    cleanup_process_group()
    return artifact
