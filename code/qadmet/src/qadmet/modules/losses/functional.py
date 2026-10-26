"""Per-task functional loss: BCE / MSE / MAE (mask-aware, reduction=none).

Why:
    The old ``_grouped_loss`` used reg columns as ``(preds-targets)**2``, and clf columns
    ``F.binary_cross_entropy_with_logits(..., reduction='none')`` directly
    called. This has been registered in the PER_TASK_LOSSES registry so that LossSpec
    can be swapped via the ``classification`` / ``regression`` keys.

Function signature:
    fn(preds, targets, mask) -> Tensor (B, n_tasks)
        - mask: ``targets.isfinite()`` (Bool, B×n). NaN positions are set to 0.
        - The return value is unreduced. The upper-level loss (e.g. Kendall) handles mask·sample_weight multiplication and summation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .registry import PER_TASK_LOSSES


@PER_TASK_LOSSES.register("masked_mse")
def masked_mse(preds: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """Element-wise (preds-targets)**2.

    Args:
        preds: (B, n_tasks).
        targets: (B, n_tasks). NaN possible. Replaced with NaN→0 before returning.
        mask: Not used (compatibility signature). Masking happens in the upper-level loss.

    Returns:
        (B, n_tasks) per-element squared error.
    """
    safe_targets = targets.nan_to_num(nan=0.0)
    return (preds - safe_targets) ** 2


@PER_TASK_LOSSES.register("masked_mae")
def masked_mae(preds: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """Element-wise |preds-targets|."""
    safe_targets = targets.nan_to_num(nan=0.0)
    return (preds - safe_targets).abs()


@PER_TASK_LOSSES.register("bce_with_logits")
def bce_with_logits(preds: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """logits → ``F.binary_cross_entropy_with_logits`` (reduction=none).

    Note:
        NaN positions in targets will be masked out by mask, but F.bce_with_logits
        produces NaN gradients on NaN input. Replaced with 0 for safety.
    """
    safe_targets = targets.nan_to_num(nan=0.0)
    return F.binary_cross_entropy_with_logits(preds, safe_targets, reduction="none")


def per_task_loss_dispatch(
    preds: Tensor,
    targets: Tensor,
    task_types: list[str],
    *,
    classification_key: str = "bce_with_logits",
    regression_key: str = "masked_mse",
) -> Tensor:
    """Computes reg/clf columns separately per task_type, then scatters into the original order.

    Why:
        When training with mixed tasks, MSE must be applied to reg columns and BCE to clf columns.
        Looks up PER_TASK_LOSSES by key → applies per column, then combines and returns (B, n_tasks).

    Args:
        preds: (B, n_tasks). reg is in scaled space, clf is logits.
        targets: (B, n_tasks). NaN mask possible.
        task_types: length n_tasks. Each entry is "regression" or "classification".
        classification_key, regression_key: PER_TASK_LOSSES registry keys.

    Returns:
        (B, n_tasks) per-task loss tensor.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}")
    if preds.size(1) != len(task_types):
        raise ValueError(
            f"task_types length({len(task_types)}) != preds.size(1)({preds.size(1)})"
        )

    reg_idx = [i for i, t in enumerate(task_types) if t == "regression"]
    clf_idx = [i for i, t in enumerate(task_types) if t == "classification"]
    other = [i for i, t in enumerate(task_types) if t not in ("regression", "classification")]
    if other:
        raise ValueError(f"Unknown values in task_types: {[task_types[i] for i in other]}")

    reg_fn = PER_TASK_LOSSES.get(regression_key)
    clf_fn = PER_TASK_LOSSES.get(classification_key)

    out = torch.zeros_like(preds)
    # mixed precision compatibility: casts per-task fn results (e.g. BCE promoted to fp32 by autocast)
    # to out dtype. No-op in 32-true.
    if reg_idx:
        ridx = torch.as_tensor(reg_idx, dtype=torch.long, device=preds.device)
        out.index_copy_(
            1, ridx,
            reg_fn(preds.index_select(1, ridx), targets.index_select(1, ridx)).to(out.dtype),
        )
    if clf_idx:
        cidx = torch.as_tensor(clf_idx, dtype=torch.long, device=preds.device)
        out.index_copy_(
            1, cidx,
            clf_fn(preds.index_select(1, cidx), targets.index_select(1, cidx)).to(out.dtype),
        )
    return out
