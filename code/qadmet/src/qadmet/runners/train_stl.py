"""STL train runner — task 별 독립 모델 + per-seed 학습.

옛 ``run_stl_dmpnn_rdkit.py`` / ``run_stl_molformer_lora.py`` /
``run_stl_unimol2_lora.py`` 세 변종을 ``ExperimentSpec(kind="train",
model.type="singletask")`` 단일 진입점으로 통합한다 (+ chemeleon / graphmvp
adaptor STL 추가).

실행 흐름:
    1. spec.dataset.name → TaskSet → task_names + task_types.
    2. backbone type 으로 fold loader 분기:
        - ``dmpnn`` / ``chemeleon``: ``prepare_fold_loaders`` (task_names=[t])
          — chemprop 경로 (둘 다 ``MPNN.fingerprint(bmg, V_d, X_d)`` 인터페이스).
        - ``molformer_lora``: ``prepare_stl_molformer_loaders`` — token parquet.
        - ``unimol2_lora``: ``prepare_stl_unimol2_loaders`` — LMDB.
        - ``graphmvp``: ``prepare_stl_graphmvp_loaders`` — synthetic
          (collate 가 SMILES → PyG Batch).
    3. (task_name × seed) 별로 ``MultitaskLitModule(n_tasks=1, loss=single_task)``
       빌드 → fit → test 예측.
    4. predictions long parquet + per-(task,seed) metrics 누적.

Why (다른 collate/loader path):
    LoRA STL backbone (MoLFormer/UniMol2) 은 chemprop 그래프가 필요 없다 — 토큰 또는
    원시 3D feature dict 를 직접 입력으로 받는다. 옛 코드와 동일한 방식으로 dataset
    구성을 분기한다.

Lazy import:
    chemprop / lightning / transformers / peft / lmdb 모두 본 모듈 호출 시점에만
    import 된다 (모두 ``run_train_stl`` 내부 또는 호출되는 sub-loader 안에서).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qadmet.core.artifacts import Artifact, ArtifactStore
from qadmet.core.logging import get_logger
from qadmet.core.specs import ExperimentSpec, FeatureSpec, TrainerSpec

logger = get_logger("qadmet.runners.train_stl")


# ---------------------------------------------------------------------------- helpers


def _resolve_lmdb_path(spec: ExperimentSpec, repo_root: Path) -> Path:
    """unimol2 LMDB 경로 — recipe.dataset.kwargs.lmdb_path 우선, 기본 옛 경로."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    raw = kw.get("lmdb_path", "data/embeddings/unimol2_features.lmdb")
    p = Path(raw)
    return p if p.is_absolute() else repo_root / p


def _resolve_tokens_parquet(spec: ExperimentSpec, repo_root: Path) -> Path:
    """molformer_tokens parquet 경로 — recipe.dataset.kwargs.parquet_paths 또는 컨벤션."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    explicit = dict(kw.get("parquet_paths", {}))
    if "molformer_tokens" in explicit:
        p = Path(explicit["molformer_tokens"])
        return p if p.is_absolute() else repo_root / p
    root = Path(kw.get("parquet_root", "data/embeddings"))
    if not root.is_absolute():
        root = repo_root / root
    return root / "molformer_tokens.parquet"


def _build_stl_lit_module(
    *,
    spec: ExperimentSpec,
    task_type: str,
    feature_spec: FeatureSpec,
    modality_dims: dict[str, int],
    descriptor_dim_total: int,
    lr: float,
    lr_lora: float | None = None,
):
    """STL LitModule 빌드 — n_tasks=1, loss=single_task.

    Args:
        modality_dims: embeddings 만의 ``{name: D}`` (descriptors 는 backbone X_d 로
            흐르므로 LitModule 의 modality 검증 대상에서 제외).
        descriptor_dim_total: descriptors 합 차원 — DMPNN backbone 의 d_xd 자동 주입.
            non-DMPNN STL (LoRA) 은 0.

    Why:
        MTL ``_build_lit_module`` (runners/train.py) 와 분리한다. STL 은
        head=linear / inverse_freq 미사용 / kendall 미사용으로 메타가 다르다.
    """
    from qadmet.modules.multitask.lit_module import MultitaskLitModule

    if spec.loss is None:
        raise ValueError("STL: loss spec 필수")
    if spec.loss.type != "single_task":
        raise ValueError(
            f"STL: loss.type='single_task' 이어야 함 (받음: {spec.loss.type!r})"
        )
    if spec.model is None:
        raise ValueError("STL: model spec 필수")
    if spec.model.head.type not in {"linear", "bottleneck"}:
        raise ValueError(
            f"STL: head.type ∈ {{'linear','bottleneck'}} 만 지원 (받음: {spec.model.head.type!r})"
        )

    from qadmet.runners.train import _inject_d_xd

    primary_spec = _inject_d_xd(spec.model.backbone, descriptor_dim_total)

    return MultitaskLitModule(
        backbone_spec=primary_spec,
        extra_backbone_specs=spec.model.extra_backbones,
        head_spec=spec.model.head,
        loss_spec=spec.loss,
        n_tasks=1,
        task_types=[task_type],
        fusion_spec=spec.model.fusion,
        ordered_tasks=None,
        modality_dims=modality_dims,
        descriptors=feature_spec.descriptors,
        embeddings=feature_spec.embeddings,
        task_to_group=None,
        groups=None,
        task_weights=None,
        lr=lr,
        lr_lora=lr_lora,
    )


def _predict_loader(trainer, model, loader) -> np.ndarray:
    """trainer.predict → numpy (N,)."""
    import torch

    out = trainer.predict(model, loader, ckpt_path="best")
    out = [t for t in out if t is not None]
    if not out:
        return np.zeros(0, dtype=np.float32)
    # bf16-mixed precision 시 predict 출력이 BFloat16 → numpy 미지원이라 float 로 캐스트.
    arr = torch.cat(out, dim=0).float().cpu().numpy()
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(np.float32)


def _unscale_regression(
    preds: np.ndarray, output_scaler: tuple | None, log_task: bool
) -> np.ndarray:
    """regression 예측을 원본 스케일로 복원.

    1. StandardScaler.inverse_transform (학습 시 정규화 했으므로).
    2. log1p 적용 task 면 expm1.
    """
    if output_scaler is not None:
        scaler, _ = output_scaler
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten().astype(np.float32)
    if log_task:
        preds = np.expm1(preds)
    return preds.astype(np.float32)


# ---------------------------------------------------------------------------- runner


def run_train_stl(
    *, spec: ExperimentSpec, store: ArtifactStore, repo_root: Path, force: bool = False,
) -> Artifact:
    """STL train recipe 한 건 실행 → artifact.

    산출물 (artifact.data_dir):
        - ``{task}/seed_{N}/best.ckpt``: per-(task,seed) 체크포인트.
        - ``predictions.parquet``: long 포맷
            ``[task, seed, split, smiles, y_true, y_pred]`` (split ∈ {"val", "test"}).
            val 예측은 OOF stacking base 로 사용 가능.
        - ``manifest.yaml``: metrics 에 task 별 best_val_loss 누적.
    """
    spec.validate()
    if (
        spec.dataset is None
        or spec.features is None
        or spec.model is None
        or spec.loss is None
        or spec.trainer is None
    ):
        raise ValueError("STL train recipe: dataset/features/model/loss/trainer 모두 필요")

    from qadmet.datasets.loaders import (
        prepare_fold_loaders,
        prepare_stl_graphmvp_loaders,
        prepare_stl_molformer_loaders,
        prepare_stl_unimol2_loaders,
        filter_fold_to_task,
    )
    from qadmet.datasets.tdc_admet import load_fold
    from qadmet.runners.train import (
        _resolve_parquet_paths,
        _resolve_parquet_root,
    )
    from qadmet.tasks import TASK_SETS
    from qadmet.training import build_trainer, extract_best_val_loss
    import qadmet.modules  # noqa: F401  — registry 등록 트리거.

    task_set = TASK_SETS.get(spec.dataset.name)
    task_names: list[str] = list(task_set.tasks)
    task_types_all: list[str] = task_set.task_types()
    task_type_map = dict(zip(task_names, task_types_all))

    # 선택적 ``dataset.kwargs.task_filter``: resume / GPU 분할 실행을 위한 subset 지정.
    # 원본 ALL_TASKS 순서를 유지한 채 지정 task 만 학습. spec 해시에 포함되므로 filter 별
    # 별도 artifact dir 가 생성된다.
    _task_filter = spec.dataset.kwargs.get("task_filter") if spec.dataset.kwargs else None
    if _task_filter is not None:
        wanted = set(_task_filter)
        unknown = wanted - set(task_names)
        if unknown:
            raise ValueError(f"task_filter: 알 수 없는 task {sorted(unknown)}")
        task_names = [t for t in task_names if t in wanted]
        if not task_names:
            raise ValueError("task_filter 가 모든 task 를 제외함")

    artifact = store.create(spec, repo_root=repo_root, force=force)
    feature_spec: FeatureSpec = spec.features
    trainer_spec: TrainerSpec = spec.trainer

    backbone_type = spec.model.backbone.type
    SUPPORTED_STL_BACKBONES = (
        "dmpnn", "chemeleon", "graphmvp", "molformer_lora", "unimol2_lora",
    )
    if backbone_type not in SUPPORTED_STL_BACKBONES:
        raise NotImplementedError(
            f"STL backbone {backbone_type!r} — 지원: {SUPPORTED_STL_BACKBONES}"
        )

    # 백본별 사전 자원 경로 (없으면 fail).
    # chemeleon 은 dmpnn 과 동일 chemprop 그래프 경로를 사용 → parquet 경로 처리도 공유.
    parquet_root: Path | None = None
    parquet_paths: dict[str, Path] = {}
    tokens_parquet: Path | None = None
    lmdb_path: Path | None = None

    if backbone_type in ("dmpnn", "chemeleon"):
        modality_keys = list(feature_spec.descriptors) + list(feature_spec.embeddings)
        parquet_root = _resolve_parquet_root(spec, repo_root)
        parquet_paths = _resolve_parquet_paths(spec, repo_root, modality_keys)
        for name, path in parquet_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"modality {name!r} parquet 누락: {path}\n"
                    f"  recipe.dataset.kwargs.parquet_paths 또는 parquet_root 확인."
                )
    elif backbone_type == "molformer_lora":
        tokens_parquet = _resolve_tokens_parquet(spec, repo_root)
        if not tokens_parquet.exists():
            raise FileNotFoundError(
                f"molformer_tokens parquet 누락: {tokens_parquet}\n"
                f"  qadmet precompute --recipe recipes/precompute/molformer_tokens.yaml"
            )
    elif backbone_type == "unimol2_lora":
        lmdb_path = _resolve_lmdb_path(spec, repo_root)
        if not lmdb_path.exists():
            raise FileNotFoundError(
                f"unimol2 LMDB 누락: {lmdb_path}\n"
                f"  옛 unimol2_features LMDB (atomic feature dict) 가 필요합니다."
            )

    seeds = list(trainer_spec.seeds)
    pred_rows: list[dict[str, Any]] = []
    per_task_metrics: dict[str, dict[int, dict[str, float]]] = {}

    for task_name in task_names:
        task_type = task_type_map[task_name]
        task_dir = artifact.data_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        per_task_metrics[task_name] = {}

        for seed in seeds:
            from qadmet.core.seed import seed_everything

            seed_everything(seed)
            seed_dir = task_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            fold = load_fold(spec.dataset.name, seed=seed)

            # ---- backbone 별 loader 빌드 ----
            if backbone_type in ("dmpnn", "chemeleon"):
                # chemeleon 은 chemprop ``MPNN.fingerprint`` 를 그대로 사용 →
                # dmpnn 과 동일 loader 경로. chemeleon adapter 도 batch['bmg']/V_d/X_d
                # 를 받는다.
                sub_fold = filter_fold_to_task(fold, task_name)
                loaders = prepare_fold_loaders(
                    sub_fold, feature_spec,
                    task_names=[task_name],
                    task_types=[task_type],
                    parquet_root=parquet_root,  # type: ignore[arg-type]
                    parquet_paths=parquet_paths,
                    batch_size=trainer_spec.batch_size,
                    num_workers=trainer_spec.num_workers,
                    seed=seed,
                    apply_log_transform=True,
                )
                train_loader = loaders.train_loader
                val_loader = loaders.val_loader
                test_loader, test_smis, test_y_true = loaders.test_loaders[task_name]
                # val OOF stacking 용: prepare_fold_loaders 가 task 별 val_loaders dict
                # 를 만든다 (test_loaders 와 동일한 single-task 패턴, SMILES + y_true
                # 보존). 학습용 ``val_loader`` 와 별개의 predict 전용 loader.
                val_loader_for_predict, val_smis, val_y_true = loaders.val_loaders[task_name]
                output_scaler = loaders.output_scaler
                log_task = task_name in loaders.log_tasks
                # embeddings 만 LitModule 검증 대상 (descriptors 는 X_d 흐름).
                modality_dims = {
                    k: loaders.modality_dims[k]
                    for k in feature_spec.embeddings
                    if k in loaders.modality_dims
                }
                descriptor_dim_total = loaders.descriptor_dim_total
            elif backbone_type == "graphmvp":
                stl_loaders = prepare_stl_graphmvp_loaders(
                    fold,
                    task_name=task_name,
                    task_type=task_type,
                    batch_size=trainer_spec.batch_size,
                    num_workers=trainer_spec.num_workers,
                    seed=seed,
                )
                train_loader = stl_loaders.train_loader
                val_loader = stl_loaders.val_loader
                test_loader = stl_loaders.test_loader
                test_smis = stl_loaders.test_smiles
                test_y_true = stl_loaders.test_y_true
                val_loader_for_predict = stl_loaders.val_loader
                val_smis = stl_loaders.val_smiles
                val_y_true = stl_loaders.val_y_true
                output_scaler = stl_loaders.output_scaler
                log_task = stl_loaders.log_task
                modality_dims = {}
                descriptor_dim_total = 0
            elif backbone_type == "molformer_lora":
                stl_loaders = prepare_stl_molformer_loaders(
                    fold,
                    task_name=task_name,
                    task_type=task_type,
                    tokens_parquet=tokens_parquet,  # type: ignore[arg-type]
                    batch_size=trainer_spec.batch_size,
                    num_workers=trainer_spec.num_workers,
                    seed=seed,
                )
                train_loader = stl_loaders.train_loader
                val_loader = stl_loaders.val_loader
                test_loader = stl_loaders.test_loader
                test_smis = stl_loaders.test_smiles
                test_y_true = stl_loaders.test_y_true
                val_loader_for_predict = stl_loaders.val_loader
                val_smis = stl_loaders.val_smiles
                val_y_true = stl_loaders.val_y_true
                output_scaler = stl_loaders.output_scaler
                log_task = stl_loaders.log_task
                modality_dims = {}
                descriptor_dim_total = 0
            else:  # unimol2_lora
                stl_loaders = prepare_stl_unimol2_loaders(
                    fold,
                    task_name=task_name,
                    task_type=task_type,
                    lmdb_path=lmdb_path,  # type: ignore[arg-type]
                    batch_size=trainer_spec.batch_size,
                    num_workers=trainer_spec.num_workers,
                    seed=seed,
                )
                train_loader = stl_loaders.train_loader
                val_loader = stl_loaders.val_loader
                test_loader = stl_loaders.test_loader
                test_smis = stl_loaders.test_smiles
                test_y_true = stl_loaders.test_y_true
                val_loader_for_predict = stl_loaders.val_loader
                val_smis = stl_loaders.val_smiles
                val_y_true = stl_loaders.val_y_true
                output_scaler = stl_loaders.output_scaler
                log_task = stl_loaders.log_task
                modality_dims = {}
                descriptor_dim_total = 0

            # ---- model + trainer ----
            model = _build_stl_lit_module(
                spec=spec,
                task_type=task_type,
                feature_spec=feature_spec,
                modality_dims=modality_dims,
                descriptor_dim_total=descriptor_dim_total,
                lr=trainer_spec.lr,
                lr_lora=trainer_spec.lr_lora,
            )
            # CSVLogger 활성화 — epoch 별 train_loss/val_loss(및 alpha/peft 진단)을
            # seed_dir/metrics/version_*/metrics.csv 로 영구 저장. log_dir=None 이면
            # logger 가 꺼져 self.log() 값이 progress bar 에만 남고 파일로 안 남는다.
            trainer = build_trainer(
                trainer_spec, ckpt_dir=seed_dir, log_dir=seed_dir / "metrics",
                monitor="val_loss", mode="min",
            )
            trainer.fit(model, train_loader, val_loader)

            # ---- predictions (val + test) ----
            test_preds = _predict_loader(trainer, model, test_loader)
            if task_type == "regression":
                test_preds = _unscale_regression(test_preds, output_scaler, log_task)
            elif task_type == "classification":
                # head 가 logits 출력 — sigmoid 로 prob 변환 (단, predict_step 미커스텀 시
                # logit 그대로 나올 수 있음. SingleTaskLoss 학습 시 BCE-with-logits 이므로
                # 사용 측이 sigmoid 적용 필요 — 일관 위해 여기서 변환).
                test_preds = 1.0 / (1.0 + np.exp(-test_preds.astype(np.float64))).astype(
                    np.float32,
                )

            for smi, yt, yp in zip(test_smis, test_y_true, test_preds):
                pred_rows.append({
                    "task": task_name,
                    "seed": seed,
                    "split": "test",
                    "smiles": smi,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                })

            # val 예측 — STL 도 OOF stacking base 로 쓸 수 있도록 SMILES + y_true 와
            # 함께 저장한다 (DMPNN/LoRA 모두 동일 경로). val_loader_for_predict 는
            # 학습에 사용된 ``val_loader`` 와 별개의 predict 전용 loader.
            if len(val_smis) > 0:
                val_preds = _predict_loader(trainer, model, val_loader_for_predict)
                if task_type == "regression":
                    val_preds = _unscale_regression(val_preds, output_scaler, log_task)
                elif task_type == "classification":
                    val_preds = 1.0 / (1.0 + np.exp(-val_preds.astype(np.float64))).astype(
                        np.float32,
                    )
                for smi, yt, yp in zip(val_smis, val_y_true, val_preds):
                    pred_rows.append({
                        "task": task_name,
                        "seed": seed,
                        "split": "val",
                        "smiles": smi,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    })

            metrics = {"best_val_loss": extract_best_val_loss(trainer)}
            per_task_metrics[task_name][seed] = metrics
            logger.info(
                "STL done",
                extra={"payload": {"task": task_name, "seed": seed, "metrics": metrics}},
            )

    # ---- predictions.parquet 저장 ----
    pred_df = pd.DataFrame(pred_rows)
    out_pq = artifact.data_dir / "predictions.parquet"
    pred_df.to_parquet(out_pq, index=False)
    store.register_output(artifact, "predictions.parquet")

    store.set_metrics(
        artifact,
        {
            "kind": "stl",
            "n_tasks": len(task_names),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "per_task": per_task_metrics,
        },
    )
    return artifact
