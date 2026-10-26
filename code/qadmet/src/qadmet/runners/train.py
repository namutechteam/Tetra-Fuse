"""train runner: ExperimentSpec(kind="train") → model training + test prediction artifact.

Flow:
    1. spec.dataset.name → TaskSet → task_names + task_types.
    2. For each spec.trainer.seeds:
         a. ``load_fold(seed)`` → TdcAdmetFold.
         b. ``prepare_fold_loaders(fold, spec.features)`` → modality-stacked DataLoader.
         c. inverse-freq task weights (formerly ``compute_inverse_freq_weights``).
         d. Build ``MultitaskLitModule`` (spec.model + above metadata).
         e. ``build_trainer`` + ``trainer.fit``.
         f. Per task, ``trainer.predict(ckpt_path="best")`` → predictions parquet.
    3. Save seed results to artifact + update manifest metrics.

This runner drives actual TDC fold + chemprop dataloader (requires internet + precomputed
parquet + (optional) GPU). Unit tests use mock fold + dummy parquet.
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
    """Numerically equivalent to legacy ``pipeline_mtl.compute_inverse_freq_weights``.

    inverse-freq + mean=1 normalization. Resulting tensor becomes LitModule.task_weights buffer.
    """
    import torch

    counts = np.array([samples_per_task[t] for t in task_names], dtype=np.float32)
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _resolve_parquet_root(spec: ExperimentSpec, repo_root: Path) -> Path:
    """recipe dataset.kwargs.parquet_root → absolute path. default ``data/embeddings``."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    raw = kw.get("parquet_root", "data/embeddings")
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _resolve_parquet_paths(
    spec: ExperimentSpec, repo_root: Path, modalities: Sequence[str]
) -> dict[str, Path]:
    """modality name → parquet absolute path. dataset.kwargs.parquet_paths takes precedence if specified.

    LMDB modality (registered in ``lmdb_paths``) + synthetic modality (converted on-the-fly from
    SMILES during collate — e.g. ``graphmvp_data``) are excluded from the result — parquet existence check
    is skipped for these.
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
    """modality name → LMDB absolute path. Only dataset.kwargs.lmdb_paths specified (no default)."""
    kw = dict(spec.dataset.kwargs) if spec.dataset is not None else {}
    explicit: dict[str, str] = dict(kw.get("lmdb_paths") or {})
    out: dict[str, Path] = {}
    for name in modalities:
        if name in explicit:
            p = Path(explicit[name])
            out[name] = p if p.is_absolute() else repo_root / p
    return out


def _inject_d_xd(spec: BackboneSpec, descriptor_dim_total: int) -> BackboneSpec:
    """Return a new BackboneSpec with DMPNN backbone's ``d_xd`` injected as the descriptor sum.

    Why:
        chemprop ``MPNN.fingerprint`` concatenates ``X_d`` after the fp →
        backbone output_dim = ``d_h + d_xd``. New code routes descriptors through X_d,
        so the runner automatically forwards it to the backbone once the modality dim sum is known.
        so it works without the recipe explicitly specifying d_xd.

    Args:
        spec: original BackboneSpec.
        descriptor_dim_total: sum of descriptor dimensions. If 0, returns spec unchanged (no-op).

    Returns:
        New BackboneSpec preserving type/consumes, with ``d_xd`` added to kwargs.
        For non-DMPNN backbones, returns spec unchanged (X_d unused).
        If recipe already specifies ``d_xd``, user intent takes precedence — not overwritten.
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
    """ModelSpec + metadata → ``MultitaskLitModule`` instance.

    Args:
        embedding_dims: ``{name: D}`` dimensions of embeddings only (descriptors flow through X_d).
        descriptor_dim_total: sum of descriptor dimensions — auto-injected as d_xd for DMPNN backbone.

    Note:
        Automatically derives ``task_to_group`` / ``groups`` based on head.routing
        (uses ``task_set.initial_grouping`` as-is regardless of routing value —
        the number of groups is determined by TaskSet).
    """
    from qadmet.modules.multitask.lit_module import MultitaskLitModule

    head_spec = model_spec.head
    task_to_group: dict[str, str] | None = None
    groups: list[str] | None = None
    if head_spec.type in ("grouped", "grouped_routed"):
        # routing merely means "take groups as-is from TaskSet.initial_grouping",
        # the group composition/count is determined by TaskSet, not the routing value (causal6=6, mech=7, etc.).
        # "admet_7_group" is a legacy alias for backward compatibility with old manifest/recipe.
        routing = head_spec.routing or "taskset_grouping"
        if routing not in ("taskset_grouping", "admet_7_group"):
            raise NotImplementedError(
                f"head.routing={routing!r} unsupported — only 'taskset_grouping' (legacy alias: 'admet_7_group')."
            )
        # If grouping_override (for group search) is present, use it instead of TaskSet's default grouping.
        grouping = grouping_override or task_set.initial_grouping
        if not grouping:
            raise ValueError(f"TaskSet({task_set.name}) has empty initial_grouping")
        task_to_group = {t: g for g, ts in grouping.items() for t in ts}
        groups = list(grouping.keys())
        if grouping_override is not None:
            # override must cover every task in task_set exactly once.
            covered = sorted(task_to_group)
            expected = sorted(task_set.tasks)
            if covered != expected:
                missing = sorted(set(expected) - set(covered))
                extra = sorted(set(covered) - set(expected))
                raise ValueError(
                    f"grouping_override does not exactly cover task_set({task_set.name}) — "
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
    """After training, log per-task σ / inverse-freq coefficient / loss statistics."""
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
    """Test prediction for a single seed → print TDC leaderboard metric table to stdout.

    When training takes long, check results immediately after each seed finishes, for diagnostics.
    Uses the same metric as final ``predictions.parquet`` + ``per_task_metric_df``.
    """
    from qadmet.evaluation.tdc_benchmark import per_task_metric_df

    if not pred_rows_for_seed:
        logger.warning("seed test metrics not printed — pred_rows empty", extra={"payload": {"seed": seed}})
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
        score = row["mean"]  # n_seeds=1, so mean == that seed's score
        score_str = f"{score:>8.4f}" if pd.notna(score) else f"{'NaN':>8}"
        lines.append(
            f"{str(row['dataset']):<{col_w}}  {str(row['metric']):>8}  {score_str}  {int(row['n_test']):>6}"
        )
    lines.append(sep)
    logger.info("\n".join(lines))


def _build_predict_trainer(runtime: Mapping[str, Any]) -> Any:
    """Predict-only single-device Trainer. Called only on rank-0 after DDP fit.

    Empty callbacks to prevent ModelCheckpoint re-creation from overwriting best.ckpt.
    logger=False also blocks CSVLogger, so no extra directory is created.
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
    """Prediction for a single task → original-scale ``numpy (N,)``.

    Args:
        loader: per-task DataLoader (one entry from val_loaders or test_loaders).
        task_idx: the column for this task in model output ``(N, n_tasks)``.
        task_type: ``"regression"`` or ``"classification"``.
        log_tasks: set of regression task names with ``log1p`` applied.
        output_scaler: ``(StandardScaler, tuple(reg_idx_in_n_tasks))`` or None.
        reg_task_idx_in_scaler: ``{task_idx: scaler_column_idx}``. Regression tasks only.
        ckpt_path: ckpt_path for lightning ``predict``. ``"best"`` refers to the model managed by the
            training trainer's best model; an absolute path string is an explicit ckpt (used for DDP-separated predict).
            If None, use current model weights as-is.

    Post-processing:
        - regression: ``out_unscaled = out * scale_[j] + mean_[j]`` (when scaler was fit).
            additionally ``expm1`` if ``log_tasks``.
        - classification: ``sigmoid`` (raw logit → prob, same as STL runner).
    """
    import torch

    preds_list = trainer.predict(model, loader, ckpt_path=ckpt_path)
    # With bf16-mixed precision training, preds are bf16 so direct numpy conversion fails → cast to fp32 via float().
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
    """Execute a single train recipe and return the artifact.

    Dispatch:
        - ``spec.model.type == "multitask"``: MTL path of this function.
        - ``spec.model.type == "singletask"``: delegated to ``runners.train_stl.run_train_stl``.

    Args:
        runtime: execution environment metadata such as DDP (`devices`, `strategy`, `num_nodes`,
            `accelerator`, `find_unused_parameters`, `world_size`, `cli_args`, etc.).
            If None, single-device training. Recorded as manifest.runtime (does not affect hash).

    Outputs (relative to artifact.data_dir, MTL path):
        - ``seed_{N}/best.ckpt``: lightning checkpoint per seed.
        - ``predictions.parquet``: long format ``[seed, task, split, smiles, y_true, y_pred]``.
          Both ``split ∈ {"val", "test"}`` are written. val predictions are used as input for OOF stacking
          (Step 4.2 InverseVariance, etc.).
        - ``manifest.yaml`` (artifact root): stores per-seed train/val loss in metrics.

    DDP behavior:
        - All ranks enter with the same spec. `store.create` is called only by rank-0, and
          other ranks retrieve the artifact at the same path via `store.load` after the barrier.
        - `trainer.fit` involves all ranks; `trainer.predict` is run only via a separate
          single-device Trainer on rank-0 (to avoid DistributedSampler's padding/reorder).
        - Synchronized via barrier between/after seeds. Process group cleaned up on exit.
    """
    spec.validate()
    if spec.dataset is None or spec.features is None or spec.model is None or spec.loss is None or spec.trainer is None:
        raise ValueError("train recipe: dataset/features/model/loss/trainer all required")

    # STL branch — dispatch by model.type (recipe's kind remains "train").
    if spec.model.type == "singletask":
        from qadmet.runners.train_stl import run_train_stl

        return run_train_stl(spec=spec, store=store, repo_root=repo_root, force=force)
    if spec.model.type != "multitask":
        raise ValueError(
            f"model.type must be 'multitask' or 'singletask': {spec.model.type!r}"
        )

    # Dependent imports only at run time.
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
    import qadmet.modules  # noqa: F401  — triggers registry registration (backbones / fusion / heads / losses).

    runtime = runtime or {}
    # Pre-init process group before rank-zero artifact creation — needed for the barrier before fit
    # to actually work (Lightning reuses it if already-initialized).
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

    # Artifact creation — only rank-0 writes to disk. Other ranks go to the same path after barrier
    # to load (since run_id is deterministic, rank-0's manifest is at the same location).
    if rank0:
        artifact = store.create(spec, repo_root=repo_root, runtime=runtime, force=force)
    barrier_if_distributed()
    if not rank0:
        # Retrieve manifest.yaml created by rank-0 from the same path.
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
    # Fail immediately for missing parquet/LMDB before training starts.
    for name, path in parquet_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"modality {name!r} parquet missing: {path}\n"
                f"  Check recipe.dataset.kwargs.parquet_paths or parquet_root."
            )
    for name, path in lmdb_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"modality {name!r} LMDB missing: {path}\n"
                f"  Check recipe.dataset.kwargs.lmdb_paths."
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
        # GradNorm learns dynamic task weights → multiplying with inverse-freq prior makes interpretation ambiguous.
        # Explicitly block the case where recipe specifies task_weighting!=null.
        if spec.loss.type == "mixed_task_gradnorm" and task_weights is not None:
            raise ValueError(
                "loss.type='mixed_task_gradnorm' allows only task_weighting=null (uniform). "
                f"received: task_weighting={spec.loss.task_weighting!r}."
            )
        # Kendall UW + PCGrad: often used by users as an ablation against GradNorm, and
        # since PCGrad is based on weighted gradients, task_weighting=null is enforced (Kendall σ
        # already handles reweighting). If the recipe also enables inverse-freq prior, the two
        # weights get multiplied, making interpretation ambiguous. Explicitly blocked.
        if spec.loss.type == "mixed_task_uncertainty_pcgrad" and task_weights is not None:
            raise ValueError(
                "loss.type='mixed_task_uncertainty_pcgrad' allows only task_weighting=null. "
                f"Kendall σ already handles reweighting. received: task_weighting={spec.loss.task_weighting!r}."
            )
        # Descriptors flow through backbone X_d, so excluded from LitModule's modality_dims.
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
        # predict-only: skip fit + fit-trainer (only regenerate predictions with existing best.ckpt).
        # Skip the fit-trainer itself, which creates CSVLogger/ModelCheckpoint, to leave existing
        # csv_logs / best.ckpt untouched. Assumes single-process execution.
        predict_only = bool(runtime.get("predict_only", False))
        if predict_only:
            trainer = None
            logger.info(
                "predict-only: skipping fit, regenerating predictions from existing ckpt",
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
            barrier_if_distributed()  # sync until all ranks finish fit

        # If predict_ckpt='last', prefer last.ckpt, falling back to best.ckpt if absent.
        # This option avoids the issue where, during head warmup + gate-first training, val_loss gets
        # locked in at epoch 1, causing best.ckpt to point to the untrained gate stage.
        use_last = trainer_spec.predict_ckpt == "last"
        last_candidate = seed_dir / "last.ckpt"
        best_candidate = seed_dir / "best.ckpt"
        if use_last and last_candidate.exists():
            ckpt_candidate = last_candidate
        else:
            ckpt_candidate = best_candidate
        ckpt_path: Path | None = ckpt_candidate if ckpt_candidate.exists() else None

        # Ranks other than rank-0 skip predict and wait for the next seed at the barrier.
        # predict-only (trainer=None) is single-process → always global zero.
        if trainer is not None and not trainer.is_global_zero:
            # Diagnostic output is only meaningful for rank-0's _last_task_loss / log_vars.
            # Other ranks return a placeholder to move on to the next seed immediately.
            barrier_if_distributed()
            return SeedRun(seed=seed, ckpt_path=None, metrics={})

        _print_task_diagnostics(model, task_names, seed)

        # output_scaler: (scaler column index) mapping for regression tasks.
        reg_task_idx_in_scaler: dict[int, int] = {}
        if loaders.output_scaler is not None:
            _, reg_idx_tuple = loaders.output_scaler
            reg_task_idx_in_scaler = {ti: j for j, ti in enumerate(reg_idx_tuple)}

        log_tasks_combined = log_tasks_cfg | loaders.log_tasks

        # ---- Predict: only via rank-0's single-device Trainer ----
        # Avoids DistributedSampler's padding/reorder breaking the SMILES↔prediction mapping.
        # Even if fit's DDP process group is alive, the new Trainer's
        # SingleDeviceStrategy uses its own dataloader without sampler modification.
        predict_trainer = _build_predict_trainer(runtime)
        predict_ckpt = str(ckpt_candidate) if ckpt_candidate.exists() else None

        # Per-task val + test prediction.
        for task_idx, task_name in enumerate(task_names):
            task_type = task_types[task_idx]

            # val prediction (OOF stacking input) — only tasks present in val_loaders.
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

            # test prediction.
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

        # predict-only has no fit trainer, so best metric cannot be extracted (NaN).
        metrics = {
            "best_val_loss": extract_best_val_loss(trainer)
            if trainer is not None
            else float("nan")
        }
        seed_metrics[seed] = metrics

        # Print TDC leaderboard metric to stdout right after seed finishes (diagnostics for long training).
        seed_pred_rows = [r for r in pred_rows if r.get("seed") == seed]
        try:
            _log_seed_test_metrics(seed_pred_rows, seed=seed, task_set_name=spec.dataset.name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"seed metric output failed (training continues): {e}", extra={"payload": {"seed": seed}})

        barrier_if_distributed()  # rank-0 predict done, releasing other ranks' sync
        return SeedRun(
            seed=seed, ckpt_path=ckpt_path, metrics=metrics,
        )

    runs = run_seed_loop(
        seeds=trainer_spec.seeds,
        fit_one=fit_one,
        ckpt_root=artifact.data_dir,
    )
    logger.info("all seeds done", extra={"payload": {"n_seeds": len(runs)}})

    # Writing final outputs — rank-0 only. non-rank-0 has empty pred_rows / seed_metrics.
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

    barrier_if_distributed()  # process group cleanup after all ranks sync at exit
    cleanup_process_group()
    return artifact
