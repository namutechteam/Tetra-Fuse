"""Mixed-task (regression + classification) Kendall uncertainty loss — standard formula.

Formula (Kendall 2018 standard, per-task generalization):

    L_total = Σ_k 1[has_data_k] · (reg_var_k / σ²_k) · w_k · L̄_k
              + 0.5 · Σ_k 1[has_data_k] · log σ²_k

    - L̄_k = per-task **mean** loss over task k's valid samples (+ sample_weight).
    - reg_var_k = 0.5 for "regression", 1.0 for "classification" (Kendall 2018).
    - L_{i,k} = MSE (reg) or BCE-with-logits (clf), reduction=none. PER_TASK_LOSSES
      replaces NaN target cells with 0 before computing, so the mask multiplication is handled in this class.

Why (difference from the old ``_grouped_loss`` / ``_mixed_loss``):

    The old code (a) ``weighted.sum() / n_valid`` — divides by the total valid cell count, so per-task
    valid sample count differences dominate, (b) ``reg_term / n_active_tasks`` — a task average not in the Kendall 2018
    formula. This corrects the above two non-standard reductions to the standard (per-task mean + reg term
    simple sum) formula.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .functional import per_task_loss_dispatch
from .registry import LOSSES


@LOSSES.register("mixed_task_uncertainty")
class MixedTaskUncertaintyLoss(nn.Module):
    """Mixed-task Kendall weighted loss (standard formula).

    Args:
        n_tasks: number of tasks (== len(task_types)).
        task_types: "regression" | "classification" label for each task.
        task_weights: (n_tasks,) inverse-freq weight. 1 if None.
        classification: clf functional key in PER_TASK_LOSSES.
        regression: reg functional key in PER_TASK_LOSSES.

    Forward:
        preds (B, n_tasks): reg columns are scaled, clf columns are logits.
        targets (B, n_tasks): NaN allowed (mask handled automatically).
        sample_weights (B,) or None: per-sample weight — applied together to the numerator/denominator of the per-task mean.
        Returns: scalar loss.

    Note:
        ``log_vars`` is indexed by ``task_has_data`` so that the dead task (all rows NaN)'s
        gradient is automatically excluded. ``task_denom = valid.sum(dim=0).clamp(min=1)`` ensures
        safe division. ``log_var_min`` floor prevents σ→0 blow-up (precision divergence).
    """

    def __init__(
        self,
        n_tasks: int,
        task_types: list[str],
        task_weights: Tensor | None = None,
        classification: str = "bce_with_logits",
        regression: str = "masked_mse",
        log_var_min: float = -1.0,
        freeze_log_vars: bool = False,
    ) -> None:
        super().__init__()
        if len(task_types) != n_tasks:
            raise ValueError(f"len(task_types)={len(task_types)} != n_tasks={n_tasks}")
        self.n_tasks = n_tasks
        self.task_types = list(task_types)
        self.classification_key = classification
        self.regression_key = regression
        self.log_var_min = float(log_var_min)
        self.freeze_log_vars = bool(freeze_log_vars)

        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        if self.freeze_log_vars:
            # Disable Kendall σ learning freedom — fix σ=1 (precision=1).
            # The formula simplifies to an inverse-freq weighted sum.
            self.log_vars.requires_grad_(False)
        reg_var = torch.tensor(
            [0.5 if t == "regression" else 1.0 for t in task_types],
            dtype=torch.float32,
        )
        self.register_buffer("reg_var_per_task", reg_var)

        if task_weights is None:
            tw = torch.ones(n_tasks, dtype=torch.float32)
        else:
            tw = (
                task_weights
                if isinstance(task_weights, Tensor)
                else torch.as_tensor(task_weights, dtype=torch.float32)
            )
        self.register_buffer("task_weights_buf", tw.to(torch.float32))

    # ---- public components (shared by PCGrad / diagnostic path) ---------------------------------

    def _compute_components(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Internal step of forward (reused in the PCGrad path).

        Returns:
            task_loss: (K,) per-task **mean** loss (graph preserved, unweighted).
            weighted_task: (K,) per-task loss after Kendall weighting (graph preserved).
                = reg_var · (1/σ²) · task_weight · L̄_k.
            log_var_reg: scalar 0.5 Σ log σ² (includes the learnable ``log_vars`` graph).
            task_has_data: (K,) bool — False for dead tasks where all rows are NaN.
        """
        if preds.shape != targets.shape:
            raise ValueError(f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}")
        if preds.size(1) != self.n_tasks:
            raise ValueError(f"preds.size(1)={preds.size(1)} != n_tasks={self.n_tasks}")

        mask = targets.isfinite()                                    # (B, K)
        per_task = per_task_loss_dispatch(                           # (B, K) — NaN→0 handled internally
            preds, targets, self.task_types,
            classification_key=self.classification_key,
            regression_key=self.regression_key,
        )

        valid = mask.to(per_task.dtype)                              # (B, K)
        if sample_weights is not None:
            valid = valid * sample_weights.to(per_task.dtype).view(-1, 1)

        task_has_data = mask.any(dim=0)                              # (K,)
        task_denom = valid.sum(dim=0).clamp(min=1)                   # (K,)
        task_loss = (per_task * valid).sum(dim=0) / task_denom       # (K,) per-task mean
        self._last_task_loss = task_loss.detach()                    # side-car for diagnostics

        # log_var floor: prevents σ→0 blow-up.
        safe_log_vars = self.log_vars.clamp(min=self.log_var_min)
        precision = torch.exp(-safe_log_vars)                        # (K,) 1/σ²
        weighted_task = (
            self.reg_var_per_task * precision
            * self.task_weights_buf * task_loss
        )                                                            # (K,)
        # Zero out the weighted term for dead tasks — ensures safety for autograd.grad calls.
        weighted_task = weighted_task * task_has_data.to(weighted_task.dtype)
        log_var_reg = 0.5 * (safe_log_vars * task_has_data.to(safe_log_vars.dtype)).sum()
        return task_loss, weighted_task, log_var_reg, task_has_data

    def weighted_per_task_losses(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """For the PCGrad path: Kendall weighted per-task loss vector + log_var regularizer.

        Why:
            PCGrad needs separate per-task gradients, so immediately reducing via ``forward``'s
            ``weighted[task_has_data].sum()`` must not be done.
            This method separates the (K,) vector + (scalar) regularizer + dead-task mask,
            for use in ``MultitaskLitModule._pcgrad_training_step`` calling ``autograd.grad``
            K times.

        Returns:
            weighted_task: (K,) — dead tasks are masked to 0.
            log_var_reg: 0.5 Σ log σ² (includes the learnable ``log_vars`` graph).
            task_has_data: (K,) bool — used to determine autograd.grad skipping.
        """
        _, weighted_task, log_var_reg, task_has_data = self._compute_components(
            preds, targets, sample_weights
        )
        return weighted_task, log_var_reg, task_has_data

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        _, weighted_task, log_var_reg, _task_has_data = self._compute_components(
            preds, targets, sample_weights
        )
        return weighted_task.sum() + log_var_reg


@LOSSES.register("mixed_task_uncertainty_pcgrad")
class MixedTaskUncertaintyPCGradLoss(MixedTaskUncertaintyLoss):
    """Kendall UW + PCGrad sentinel.

    Loss-level definition is 100% identical to ``MixedTaskUncertaintyLoss``.
    This subclass exists to leave a PCGrad activation signal in the manifest via ``LossSpec.type``
    marker — ``MultitaskLitModule`` branches via isinstance / type-name check
    to the PCGrad training_step.

    Why subclass (instead of an option kwarg):
        - Just looking at ``loss.type`` in the manifest immediately reveals "PCGrad on/off".
        - Keeps LossSpec's frozen data model as-is (no need to add/validate kwargs keys).
        - Consistent with the pattern where two algorithms, like Kendall UW + GradNorm,
          have a manual_optimization branch.

    Kwargs (flows in via kwargs from LossSpec.kwargs):
        log_var_min: σ→0 floor (same as Kendall, default -1.0).
        freeze_log_vars: turn off σ learning (same as Kendall, default False).
        pcgrad_scope: "trunk" (default) — backbone + fusion + extra_backbones
            apply PCGrad projection only to parameters. "all" — everything including the head.
            LitModule reads this value to determine the shared-params set.
        pcgrad_eps: prevents division by 0 (default 1e-12).
        uniform_weight: if True, completely removes Kendall UW and switches to **PCGrad-only** mode,
            operation. Flattens even reg_var (0.5 reg / 1.0 clf) all to 1.0, and fixes σ
            to 1 (forcing freeze_log_vars), making log_var regularizer=0 → weighted_L_k = L̄_k.
            i.e., applies gradient surgery only to the equally-weighted per-task loss across all tasks.
            For "Kendall UW + PCGrad" vs "PCGrad only" ablation (default False).
    """

    def __init__(
        self,
        n_tasks: int,
        task_types: list[str],
        task_weights: Tensor | None = None,
        classification: str = "bce_with_logits",
        regression: str = "masked_mae",
        log_var_min: float = -1.0,
        freeze_log_vars: bool = False,
        pcgrad_scope: str = "trunk",
        pcgrad_eps: float = 1e-12,
        uniform_weight: bool = False,
    ) -> None:
        if uniform_weight:
            # PCGrad only: fix σ=1 (precision=1, log_var_reg=0). Remove learning freedom.
            freeze_log_vars = True
        super().__init__(
            n_tasks=n_tasks,
            task_types=task_types,
            task_weights=task_weights,
            classification=classification,
            regression=regression,
            log_var_min=log_var_min,
            freeze_log_vars=freeze_log_vars,
        )
        if pcgrad_scope not in ("trunk", "all"):
            raise ValueError(
                f"pcgrad_scope must be 'trunk' | 'all' — got: {pcgrad_scope!r}"
            )
        self.pcgrad_scope = pcgrad_scope
        self.pcgrad_eps = float(pcgrad_eps)
        self.uniform_weight = bool(uniform_weight)
        if uniform_weight:
            # Flatten reg_var(0.5/1.0) → equal weighting for all tasks. Keep buffer in-place.
            self.reg_var_per_task.fill_(1.0)
