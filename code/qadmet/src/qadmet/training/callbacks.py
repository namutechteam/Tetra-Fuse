"""Standard Lightning callback builder.

Follows the callback composition of the old ``train_mtl_mpnn`` (scripts-old/pipeline_mtl.py:1386-1388)
``ModelCheckpoint(monitor=val_loss, mode=min, save_top_k=1)`` +
``EarlyStopping(monitor=val_loss, mode=min, patience=...)`` exactly.

Lazy import:
    lightning callbacks are imported only at call time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_callbacks(
    *,
    ckpt_dir: Path | str,
    monitor: str = "val_loss",
    mode: str = "min",
    save_top_k: int = 1,
    early_stopping: bool = True,
    patience: int = 20,
    extra: tuple[str, ...] = (),
    save_last: bool = False,
) -> list[Any]:
    """Builds the standard callback composition.

    Args:
        ckpt_dir: ModelCheckpoint save directory. Automatically created including parent directories.
        monitor: Name of the metric to monitor (the key logged via ``self.log`` in LitModule).
        mode: ``"min"`` or ``"max"``.
        save_top_k: Number of top-k checkpoints to save via ModelCheckpoint.
        early_stopping: If True, adds EarlyStopping.
        patience: EarlyStopping patience.
        extra: Names of registered additional callbacks (currently unsupported, room for a future registry).

    Returns:
        A list that can be passed directly to the Trainer ``callbacks`` argument.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename="best",
            save_last=save_last,
            # Save decision only right after validation epoch end (manual_optimization + DDP
            # combination, where saving right after train_epoch_end captured a stale metric,
            # blocking the issue where best.ckpt saved weights 1 epoch behind the true best).
            save_on_train_epoch_end=False,
        )
    ]
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor, mode=mode, patience=patience, check_finite=True,
            )
        )

    if extra:
        # TODO(research direction): additional callback registry — alpha diagnostics, gradient norm logging, etc.
        raise NotImplementedError(
            f"extra_callbacks={extra} is not yet supported (Step 3.6 stage). "
            "Will be enabled once the callback registry is introduced."
        )

    return callbacks


def extract_best_val_loss(trainer: Any, monitor: str = "val_loss") -> float:
    """Extracts the true best metric from the Trainer's EarlyStopping / ModelCheckpoint.

    ``trainer.callback_metrics[monitor]`` is the *last* validation at the end of training
    value, so it is not the best. EarlyStopping.best_score holds the monotonic minimum of monitor
    as-is, so it takes priority. If that is unavailable, ModelCheckpoint.best_model_score,
    and as the last fallback, the last value of callback_metrics.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    for cb in getattr(trainer, "callbacks", []) or []:
        if (
            isinstance(cb, EarlyStopping)
            and cb.monitor == monitor
            and cb.best_score is not None
        ):
            return float(cb.best_score)
    for cb in getattr(trainer, "callbacks", []) or []:
        if (
            isinstance(cb, ModelCheckpoint)
            and cb.monitor == monitor
            and cb.best_model_score is not None
        ):
            return float(cb.best_model_score)
    return float(trainer.callback_metrics.get(monitor, float("nan")))
