"""Mixed-task GradNorm loss (Chen et al. 2018, ICML).

수식:

    L_main(t) = Σ_k w_k(t) · L̄_k(t)                             # main backward (W 만 업데이트)
    G_W^(k)(t) = w_k(t) · ||∇_W L̄_k(t)||_2                       # shared layer 그라디언트 norm
    L̃_k(t) = L̄_k(t) / L̄_k(0)                                    # 상대 학습 진척
    r_k(t) = L̃_k / mean_k L̃                                     # inverse training rate
    target_k(t) = mean_k G_W · r_k^α                             # constant (no grad)
    L_grad(t) = Σ_k |G_W^(k) - target_k|                         # aux backward (w 만 업데이트)

각 step 마다:
    1) preds, L̄_k 계산 (graph 보존)
    2) ||∇_W L̄_k||_2 를 ``autograd.grad(..., create_graph=False)`` 로 산출 → 스칼라 상수.
    3) G_W^(k) = w_k · ||∇_W L̄_k||_2 (w 에 대해서만 미분가능, W 그라디언트 흐름 차단).
    4) L_grad backward → w.grad 누적.
    5) L_main = (w.detach() · L̄_k).sum() backward → W, head, backbone .grad 누적.
       (w 는 detach 로 main backward 에서 제외, 오직 L_grad 만 w 를 학습.)
    6) opt_main.step(), opt_w.step().
    7) w 를 양수+합=K 로 renormalize.

본 클래스는 **per-task loss 만 계산**한다. shared-layer 그라디언트와 w 업데이트는
``MultitaskLitModule._gradnorm_training_step`` 가 manual optimization 으로 처리한다.

``task_weighting`` (inverse-freq prior) 는 GradNorm 의 동적 w 와 곱하면 해석이 모호해지므로
``MixedTaskGradNormLoss`` 는 task_weights 인자를 받지만 검증만 하고 buffer 에 1 로
강제한다 (runner 도 task_weighting=null 강제). w 초기값은 모두 1 (GradNorm 표준).

Note:
    - log_var 없음 (Kendall σ 대체가 목적).
    - per-task NaN mask 처리는 ``mixed_task_uncertainty`` 와 동일.
    - dead task (모든 row NaN) 의 G 는 자동으로 0 → target 도 r=0 으로 작아져 self-consistent.
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
        n_tasks: task 수.
        task_types: ``"regression"`` | ``"classification"`` 라벨 (길이 n_tasks).
        task_weights: GradNorm 은 inverse-freq prior 와 호환 안 됨. None 또는 all-ones
            이외 입력 시 ``ValueError``. runner 에서 ``task_weighting=null`` 강제.
        classification: PER_TASK_LOSSES clf 키.
        regression: PER_TASK_LOSSES reg 키.
        alpha: GradNorm restoring force (Chen et al. 2018). ADMET asymmetric 권장값 1.5.
        shared_layer: GradNorm 그라디언트 norm 을 측정할 LitModule attribute path
            (예: ``"backbone.proj.0"``). None 이면 LitModule 이 backbone 의 마지막
            ``nn.Linear`` 를 auto-detect.

    Forward:
        preds (B, n_tasks), targets (B, n_tasks), sample_weights (B,) | None.
        Returns: 가중합 ``(w · L̄_k).sum()`` (scalar). 학습 시엔 LitModule 이 이 값을
        무시하고 ``per_task_losses`` + 수동 backward 를 사용한다.

    Sidecars:
        ``_last_task_loss``: (n_tasks,) detached. 에폭별 평균 누적용 (기존 LitModule
            로깅 경로와 호환).
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
                    "mixed_task_gradnorm: task_weights 가 uniform 이 아님. "
                    "GradNorm 은 동적으로 task weight 를 학습하므로 inverse-freq prior 와 "
                    "함께 쓰면 해석이 모호함. recipe 에서 task_weighting=null 로 설정하세요."
                )
        self.n_tasks = n_tasks
        self.task_types = list(task_types)
        self.classification_key = classification
        self.regression_key = regression
        self.alpha = float(alpha)
        self.shared_layer_name: str | None = shared_layer

        # GradNorm 동적 weight (Chen et al. 2018). 초기값 1, 학습 후 renormalize → sum=K.
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
            GradNorm 알고리즘은 raw L̄_k 와 G_W^(k) 를 별도 흐름으로 사용한다.
            ``forward`` 가 weighted sum 만 반환하면 LitModule 이 L̄_k 를 다시 계산해야
            하므로, 한 forward path 로 둘 다 노출.
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
        """Validation / fallback 경로용. LitModule training_step 은 본 함수 미사용.

        ``(w · L̄_k).sum()`` 을 반환. w 가 detach 안 된 상태이므로 일반적 backward 호출
        시 w 도 ∂L/∂w_k = L̄_k 로 잘못 업데이트될 수 있다 — LitModule manual path 에서는
        ``per_task_losses`` 를 직접 호출.
        """
        task_loss = self.per_task_losses(preds, targets, sample_weights)
        return (self.w * task_loss).sum()
