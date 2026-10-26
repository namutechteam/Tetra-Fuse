"""STL train runner — per-task independent model + per-seed training.

Legacy ``run_stl_dmpnn_rdkit.py`` / ``run_stl_molformer_lora.py`` /
``run_stl_unimol2_lora.py`` three variants into ``ExperimentSpec(kind="train",
model.type="singletask")`` consolidated into a single entry point (+ chemeleon / graphmvp
adaptor STL added).

Execution flow:
    1. spec.dataset.name → TaskSet → task_names + task_types.
    2. Branch fold loader by backbone type:
        - ``dmpnn`` / ``chemeleon``: ``prepare_fold_loaders`` (task_names=[t])
          — chemprop path (both use the ``MPNN.fingerprint(bmg, V_d, X_d)`` interface).
        - ``molformer_lora``: ``prepare_stl_molformer_loaders`` — token parquet.
        - ``unimol2_lora``: ``prepare_stl_unimol2_loaders`` — LMDB.
        - ``graphmvp``: ``prepare_stl_graphmvp_loaders`` — synthetic
          (collate converts SMILES → PyG Batch).
    3. For each (task_name × seed), build ``MultitaskLitModule(n_tasks=1, loss=single_task)``
       → fit → test prediction.
    4. Accumulate predictions long parquet + per-(task,seed) metrics.

Why (different collate/loader path):
    LoRA STL backbone (MoLFormer/UniMol2) does not need a chemprop graph — it takes tokens or
    raw 3D feature dict directly as input. In the same way as the legacy code, dataset
    construction is branched.

Lazy import:
    chemprop / lightning / transformers / peft / lmdb are all imported only at the point this module is called
    (all within ``run_train_stl`` or inside the sub-loader it calls).
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
    """unimol2 LMDB path — recipe.dataset.kwargs.lmdb_path takes priority, default legacy path."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    raw = kw.get("lmdb_path", "data/embeddings/unimol2_features.lmdb")
    p = Path(raw)
    return p if p.is_absolute() else repo_root / p


def _resolve_tokens_parquet(spec: ExperimentSpec, repo_root: Path) -> Path:
    """molformer_tokens parquet path — recipe.dataset.kwargs.parquet_paths or convention."""
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
    """Build STL LitModule — n_tasks=1, loss=single_task.

    Args:
        modality_dims: ``{name: D}`` for embeddings only (descriptors flow through backbone X_d,
            so they're excluded from LitModule's modality validation target).
        descriptor_dim_total: sum dimension of descriptors — auto-injected into DMPNN backbone's d_xd.
            non-DMPNN STL (LoRA) is 0.

    Why:
        Separated from MTL ``_build_lit_module`` (runners/train.py). STL
        has different metadata with head=linear / inverse_freq unused / kendall unused.
    """
    from qadmet.modules.multitask.lit_module import MultitaskLitModule

    if spec.loss is None:
        raise ValueError("STL: loss spec required")
    if spec.loss.type != "single_task":
        raise ValueError(
            f"STL: loss.type must be 'single_task' (received: {spec.loss.type!r})"
        )
    if spec.model is None:
        raise ValueError("STL: model spec required")
    if spec.model.head.type not in {"linear", "bottleneck"}:
        raise ValueError(
            f"STL: head.type ∈ {{'linear','bottleneck'}} only supported (received: {spec.model.head.type!r})"
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
    # Under bf16-mixed precision, predict output is BFloat16 → cast to float since numpy doesn't support it.
    arr = torch.cat(out, dim=0).float().cpu().numpy()
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(np.float32)


def _unscale_regression(
    preds: np.ndarray, output_scaler: tuple | None, log_task: bool
) -> np.ndarray:
    """Restore regression predictions to the original scale.

    1. StandardScaler.inverse_transform (since normalization was applied during training).
    2. expm1 if the task had log1p applied.
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
    """Execute a single STL train recipe → artifact.

    Outputs (artifact.data_dir):
        - ``{task}/seed_{N}/best.ckpt``: per-(task,seed) checkpoint.
        - ``predictions.parquet``: long format
            ``[task, seed, split, smiles, y_true, y_pred]`` (split ∈ {"val", "test"}).
            val predictions can be used as an OOF stacking base.
        - ``manifest.yaml``: accumulates per-task best_val_loss in metrics.
    """
    spec.validate()
    if (
        spec.dataset is None
        or spec.features is None
        or spec.model is None
        or spec.loss is None
        or spec.trainer is None
    ):
        raise ValueError("STL train recipe: dataset/features/model/loss/trainer all required")

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
    import qadmet.modules  # noqa: F401  — triggers registry registration.

    task_set = TASK_SETS.get(spec.dataset.name)
    task_names: list[str] = list(task_set.tasks)
    task_types_all: list[str] = task_set.task_types()
    task_type_map = dict(zip(task_names, task_types_all))

    # Optional ``dataset.kwargs.task_filter``: specifies a subset for resume / split-GPU runs.
    # Trains only the specified tasks while preserving the original ALL_TASKS order. Since it's included in the spec hash, a separate
    # artifact dir is created per filter.
    _task_filter = spec.dataset.kwargs.get("task_filter") if spec.dataset.kwargs else None
    if _task_filter is not None:
        wanted = set(_task_filter)
        unknown = wanted - set(task_names)
        if unknown:
            raise ValueError(f"task_filter: unknown task {sorted(unknown)}")
        task_names = [t for t in task_names if t in wanted]
        if not task_names:
            raise ValueError("task_filter excludes all tasks")

    artifact = store.create(spec, repo_root=repo_root, force=force)
    feature_spec: FeatureSpec = spec.features
    trainer_spec: TrainerSpec = spec.trainer

    backbone_type = spec.model.backbone.type
    SUPPORTED_STL_BACKBONES = (
        "dmpnn", "chemeleon", "graphmvp", "molformer_lora", "unimol2_lora",
    )
    if backbone_type not in SUPPORTED_STL_BACKBONES:
        raise NotImplementedError(
            f"STL backbone {backbone_type!r} — supported: {SUPPORTED_STL_BACKBONES}"
        )

    # Per-backbone prerequisite resource path (fails if missing).
    # chemeleon uses the same chemprop graph path as dmpnn → parquet path handling is also shared.
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
                    f"modality {name!r} parquet missing: {path}\n"
                    f"  check recipe.dataset.kwargs.parquet_paths or parquet_root."
                )
    elif backbone_type == "molformer_lora":
        tokens_parquet = _resolve_tokens_parquet(spec, repo_root)
        if not tokens_parquet.exists():
            raise FileNotFoundError(
                f"molformer_tokens parquet missing: {tokens_parquet}\n"
                f"  qadmet precompute --recipe recipes/precompute/molformer_tokens.yaml"
            )
    elif backbone_type == "unimol2_lora":
        lmdb_path = _resolve_lmdb_path(spec, repo_root)
        if not lmdb_path.exists():
            raise FileNotFoundError(
                f"unimol2 LMDB missing: {lmdb_path}\n"
                f"  legacy unimol2_features LMDB (atomic feature dict) is required."
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

            # ---- build loader per backbone ----
            if backbone_type in ("dmpnn", "chemeleon"):
                # chemeleon uses chemprop ``MPNN.fingerprint`` as-is →
                # Same loader path as dmpnn. chemeleon adapter also takes batch['bmg']/V_d/X_d
                # as input.
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
                # For val OOF stacking: prepare_fold_loaders builds a per-task val_loaders dict
                # (same single-task pattern as test_loaders, preserving SMILES + y_true
                # ). A predict-only loader separate from the training ``val_loader``.
                val_loader_for_predict, val_smis, val_y_true = loaders.val_loaders[task_name]
                output_scaler = loaders.output_scaler
                log_task = task_name in loaders.log_tasks
                # Only embeddings are the LitModule validation target (descriptors flow via X_d).
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
            # Enable CSVLogger — persists per-epoch train_loss/val_loss (and alpha/peft diagnostics)
            # to seed_dir/metrics/version_*/metrics.csv. If log_dir=None,
            # the logger is disabled, so self.log() values remain only in the progress bar and are not saved to a file.
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
                # head outputs logits — converted to prob via sigmoid (however, if predict_step isn't customized,
                # the logit may come out as-is. Since SingleTaskLoss trains with BCE-with-logits,
                # the consumer needs to apply sigmoid — converted here for consistency).
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

            # val prediction — saved together with SMILES + y_true so STL can also serve as an OOF stacking base
            # (same path for both DMPNN/LoRA). val_loader_for_predict is
            # a predict-only loader separate from the ``val_loader`` used for training.
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

    # ---- save predictions.parquet ----
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
