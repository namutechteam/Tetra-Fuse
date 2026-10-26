"""``TrainerSpec`` → lightning ``Trainer`` builder.

Trainer arguments from the old ``train_mtl_mpnn`` (scripts-old/pipeline_mtl.py:1409-1413)
combination brought over as-is::

    pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1,
        callbacks=callbacks, enable_progress_bar=True,
        logger=False, enable_model_summary=False,
    )

Why dedicated builder:
    Gathers Trainer arguments (precision, deterministic, gradient_clip, etc.) in one place
    to respond consistently to spec changes. In the old code, arguments were scattered inside the function,
    causing hard-coded values to leak during experiment comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qadmet.core.specs import TrainerSpec
from qadmet.training.callbacks import build_callbacks


def build_trainer(
    spec: TrainerSpec,
    *,
    ckpt_dir: Path | str,
    log_dir: Path | str | None = None,
    monitor: str = "val_loss",
    mode: str = "min",
    accelerator: str | None = None,
    devices: int = 1,
    strategy: str | None = None,
    num_nodes: int = 1,
    find_unused_parameters: bool = False,
    ddp_static_graph: bool = True,
    deterministic: bool = False,
    save_last: bool = False,
) -> Any:
    """``TrainerSpec`` → lightning ``Trainer`` instance.

    Args:
        spec: TrainerSpec (max_epochs / patience / precision / early_stopping, etc.).
        ckpt_dir: ModelCheckpoint save directory (should differ per seed).
        log_dir: CSVLogger output directory. If None, logger is disabled.
        monitor / mode: common monitor for callbacks/checkpoints.
        accelerator: ``"gpu"`` / ``"cpu"`` / None (if None, auto-detects CUDA).
        devices: number of devices to use.
        strategy: ``"ddp"`` / None. If None, Lightning auto (single → SingleDevice).
            If ``"ddp"``, builds a DDPStrategy instance (reflecting find_unused_parameters).
        num_nodes: number of multi-nodes (usually 1).
        find_unused_parameters: DDPStrategy option. Default is False (performance priority).
            Set True only when the forward path touches different parameters per batch.
        ddp_static_graph: DDPStrategy's ``static_graph`` option. Default is True
            (used to avoid NCCL hangs when combined with find_unused_parameters=True — caches
            the used-param set of the first iteration to ensure consistency across ranks).
            PyTorch official recommendation: when using ``find_unused_parameters=True`` alone,
            the grad-ready order can diverge per rank, causing NCCL allreduce spin-wait deadlock
            possible. This model (GroupedHead) guarantees that all trunk + head calls occur in forward,
            satisfying the static_graph assumption.
        deterministic: if True, strengthens reproducibility via ``Trainer(deterministic=True)``.

    Returns:
        lightning.pytorch.Trainer instance.

    Note:
        Trainer argument priority: defaults < ``spec.kwargs`` < explicit arguments of this function.
        Values flowed in from the CLI, such as ``devices/strategy/num_nodes``, override the recipe's
        ``trainer.kwargs`` values.
    """
    import lightning as L

    if accelerator is None:
        try:
            import torch

            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            accelerator = "cpu"

    callbacks = build_callbacks(
        ckpt_dir=ckpt_dir,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        early_stopping=spec.early_stopping,
        patience=spec.patience,
        extra=spec.extra_callbacks,
        save_last=save_last,
    )

    pl_logger: Any = False
    if log_dir is not None:
        from lightning.pytorch.loggers import CSVLogger

        log_path = Path(log_dir)
        pl_logger = CSVLogger(save_dir=str(log_path.parent), name=log_path.name)

    # Handle as DDPStrategy instance or string depending on strategy value.
    resolved_strategy: Any = "auto"
    if strategy is not None:
        if strategy == "ddp":
            from lightning.pytorch.strategies import DDPStrategy

            resolved_strategy = DDPStrategy(
                find_unused_parameters=find_unused_parameters,
                static_graph=ddp_static_graph,
            )
        else:
            # ddp_spawn / fsdp / deepspeed, etc. — pass string as-is, Lightning interprets it.
            resolved_strategy = strategy

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": spec.max_epochs,
        "accelerator": accelerator,
        "devices": devices,
        "callbacks": callbacks,
        "logger": pl_logger,
        "precision": spec.precision,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "deterministic": deterministic,
    }
    # Priority: defaults < spec.kwargs < explicit arguments (CLI).
    trainer_kwargs.update(dict(spec.kwargs))
    trainer_kwargs["devices"] = devices
    trainer_kwargs["num_nodes"] = num_nodes
    trainer_kwargs["strategy"] = resolved_strategy
    return L.Trainer(**trainer_kwargs)
