"""Mixed-task GradNorm loss (Chen et al. 2018, ICML).

Formulas:

    L_main(t) = Σ_k w_k(t) · L̄_k(t)                             # main backward (update W only)
    G_W^(k)(t) = w_k(t) · ||∇_W L̄_k(t)||_2                       # shared layer gradient norm
    L̃_k(t) = L̄_k(t) / L̄_k(0)                                    # relative training progress
    r_k(t) = L̃_k / mean_k L̃                                     # inverse training rate
    target_k(t) = mean_k G_W · r_k^α                             # constant (no grad)
    L_grad(t) = Σ_k |G_W^(k) - target_k|                         # aux backward (update w only)

At each step:
    1) Compute preds, L̄_k (preserve graph)
    2) Compute ||∇_W L̄_k||_2 via ``autograd.grad(..., create_graph=False)`` → scalar constant.
    3) G_W^(k) = w_k · ||∇_W L̄_k||_2 (differentiable only with respect to w, blocks W gradient flow).
    4) L_grad backward → accumulate w.grad.
    5) L_main = (w.detach() · L̄_k).sum() backward → accumulate W, head, backbone .grad.
       (w is excluded from main backward via detach, only L_grad trains w.)
    6) opt_main.step(), opt_w.step().
    7) Renormalize w to positive + sum=K.

This class **computes only the per-task loss**. The shared-layer gradient and w update are
handled by ``MultitaskLitModule._gradnorm_training_step`` via manual optimization.

Since multiplying ``task_weighting`` (inverse-freq prior) by GradNorm's dynamic w makes the interpretation ambiguous,
``MixedTaskGradNormLoss`` accepts a task_weights argument but only validates it, forcing the buffer to 1
(the runner also enforces task_weighting=null). The initial value of w is all 1 (GradNorm standard).

Note:
    - No log_var (intended to replace Kendall σ).
    - per-task NaN mask handling is identical to ``mixed_task_uncertainty``.
    - For a dead task (all rows NaN), G automatically becomes 0 → target also shrinks to r=0, making it self-consistent.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .functional import per_task_loss_dispatch
from .registry import LOSSES


@LOSSES.register("mixed_task_gradnorm")
class MixedTaskGradNormLoss(nn.Module):
    """GradNorm dynamic task weighting loss (mixed reg + clf).

    Args:
        n_tasks: number of tasks.
        task_types: ``"regression"`` | ``"classification"`` labels (length n_tasks).
        task_weights: GradNorm is not compatible with inverse-freq prior. None or all-ones
            raises ``ValueError`` for any other input. The runner enforces ``task_weighting=null``.
        classification: PER_TASK_LOSSES clf key.
        regression: PER_TASK_LOSSES reg key.
        alpha: GradNorm restoring force (Chen et al. 2018). Recommended value 1.5 for ADMET asymmetric.
        shared_layer: LitModule attribute path for measuring the GradNorm gradient norm
            (e.g. ``"backbone.proj.0"``). If None, the LitModule auto-detects the last
            ``nn.Linear`` of the backbone.

    Forward:
        preds (B, n_tasks), targets (B, n_tasks), sample_weights (B,) | None.
        Returns: the weighted sum ``(w · L̄_k).sum()`` (scalar). During training, the LitModule
        ignores this value and uses ``per_task_losses`` + manual backward.

    Sidecars:
        ``_last_task_loss``: (n_tasks,) detached. For per-epoch average accumulation (compatible with existing LitModule
            logging path).
    """

    def __init__(
        self,
        n_tasks: int,
        task_types: list[str],
        task_weights: Tensor | None = None,
        classification: str = "bce_with_logits",
        regression: str = "masked_mae",
        alpha: float = 1.5,
        shared_layer: str | None = None,
    ) -> None:
        super().__init__()
        if len(task_types) != n_tasks:
            raise ValueError(f"len(task_types)={len(task_types)} != n_tasks={n_tasks}")
        if task_weights is not None:
            tw = (
                task_weights
                if isinstance(task_weights, Tensor)
                else torch.as_tensor(task_weights, dtype=torch.float32)
            )
            if not torch.allclose(tw, torch.ones_like(tw)):
                raise ValueError(
                    "mixed_task_gradnorm: task_weights is not uniform. "
                    "Since GradNorm dynamically learns task weight, combining with inverse-freq prior "
                    "makes interpretation ambiguous. Set task_weighting=null in the recipe."
                )
        self.n_tasks = n_tasks
        self.task_types = list(task_types)
        self.classification_key = classification
        self.regression_key = regression
        self.alpha = float(alpha)
        self.shared_layer_name: str | None = shared_layer

        # GradNorm dynamic weight (Chen et al. 2018). Initial value 1, renormalize after training → sum=K.
        self.w = nn.Parameter(torch.ones(n_tasks))

    # ---- public API for LitModule ----

    def per_task_losses(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        """(n_tasks,) per-task **mean** loss with graph (no w weighting applied).

        Why:
            The GradNorm algorithm uses raw L̄_k and G_W^(k) as separate flows.
            If ``forward`` returns only the weighted sum, the LitModule would need to recompute L̄_k
            so both are exposed via a single forward path.
        """
        if preds.shape != targets.shape:
            raise ValueError(f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}")
        if preds.size(1) != self.n_tasks:
            raise ValueError(f"preds.size(1)={preds.size(1)} != n_tasks={self.n_tasks}")

        mask = targets.isfinite()                                    # (B, K)
        per_task = per_task_loss_dispatch(
            preds, targets, self.task_types,
            classification_key=self.classification_key,
            regression_key=self.regression_key,
        )
        valid = mask.to(per_task.dtype)
        if sample_weights is not None:
            valid = valid * sample_weights.to(per_task.dtype).view(-1, 1)
        task_denom = valid.sum(dim=0).clamp(min=1)
        task_loss = (per_task * valid).sum(dim=0) / task_denom        # (K,)
        self._last_task_loss = task_loss.detach()
        return task_loss

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        """For the validation / fallback path. LitModule training_step does not use this function.

        Returns ``(w · L̄_k).sum()``. Since w is not detached, a normal backward call
        could incorrectly update w as well via ∂L/∂w_k = L̄_k — in the LitModule manual path,
        call ``per_task_losses`` directly.
        """
        task_loss = self.per_task_losses(preds, targets, sample_weights)
        return (self.w * task_loss).sum()
