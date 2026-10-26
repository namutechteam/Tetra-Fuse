"""Single-task loss — STL training thin wrapper.

Legacy STL code (``run_stl_dmpnn_rdkit.py``, ``run_stl_molformer_lora.py``,
``run_stl_unimol2_lora.py``) called, per task, ``F.binary_cross_entropy_with_logits``
or ``F.mse_loss`` directly. This module folds that behavior into the ``LOSSES`` registry
entry ``"single_task"``, allowing the STL runner to look at a single ``LossSpec(type="single_task")``
and build identically.

Why:
    The ``MultitaskLitModule`` interface identical to the MTL Kendall/Mixed loss is also
    used in STL. Enforces n_tasks=1 + NaN-safe mask + sample_weights support.
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import LOSSES, PER_TASK_LOSSES


@LOSSES.register("single_task")
class SingleTaskLoss(nn.Module):
    """Single-task training thin loss wrapper.

    Args:
        task_type: ``"regression"`` | ``"classification"``.
        classification: PER_TASK_LOSSES clf key (default ``"bce_with_logits"``).
        regression: PER_TASK_LOSSES reg key (default ``"masked_mse"`` — equivalent to legacy STL MSE).

    Forward:
        preds (B, 1): regression is scaled space, classification is logits.
        targets (B, 1): NaN possible — finite mask automatically applied.
        sample_weights (B,) | None.
        Returns: scalar mean loss.
    """

    def __init__(
        self,
        task_type: str,
        classification: str = "bce_with_logits",
        regression: str = "masked_mse",
    ) -> None:
        super().__init__()
        if task_type not in ("regression", "classification"):
            raise ValueError(
                f"task_type must be 'regression' or 'classification': {task_type!r}"
            )
        self.task_type = task_type
        self.classification_key = classification
        self.regression_key = regression
        self._fn_key = regression if task_type == "regression" else classification
        # Fail fast on missing registration.
        _ = PER_TASK_LOSSES.get(self._fn_key)

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        if preds.shape != targets.shape:
            raise ValueError(
                f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}"
            )
        if preds.size(1) != 1:
            raise ValueError(
                f"single_task supports only n_tasks=1 (got: {preds.size(1)})"
            )
        fn = PER_TASK_LOSSES.get(self._fn_key)
        per_elem = fn(preds, targets)  # (B, 1), reduction=none
        mask = targets.isfinite()
        per_elem = per_elem * mask.to(per_elem.dtype)
        if sample_weights is not None:
            per_elem = per_elem * sample_weights.to(per_elem.dtype).view(-1, 1)
        denom = mask.sum().clamp(min=1)
        return per_elem.sum() / denom
