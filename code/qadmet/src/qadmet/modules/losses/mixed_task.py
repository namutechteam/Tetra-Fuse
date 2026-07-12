"""Mixed-task (regression + classification) Kendall uncertainty loss — 표준 식.

수식 (Kendall 2018 표준, per-task 일반화):

    L_total = Σ_k 1[has_data_k] · (reg_var_k / σ²_k) · w_k · L̄_k
              + 0.5 · Σ_k 1[has_data_k] · log σ²_k

    - L̄_k = task k 의 valid 샘플(+ sample_weight)에 대한 per-task **평균** loss.
    - reg_var_k = "regression" 이면 0.5, "classification" 이면 1.0 (Kendall 2018).
    - L_{i,k} = MSE (reg) 또는 BCE-with-logits (clf), reduction=none. PER_TASK_LOSSES
      가 NaN target 셀을 0 으로 치환 후 계산하므로 mask 곱은 본 클래스에서 처리.

Why (옛 ``_grouped_loss`` / ``_mixed_loss`` 와의 차이):

    옛 코드는 (a) ``weighted.sum() / n_valid`` — 전체 valid 셀 수로 나눠 task 별
    valid 샘플 수 차이가 dominate, (b) ``reg_term / n_active_tasks`` — Kendall 2018
    식엔 없는 task 평균. 위 두 비표준 reduction 을 표준 (per-task mean + reg term
    단순 합) 으로 정정한 식.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .functional import per_task_loss_dispatch
from .registry import LOSSES


@LOSSES.register("mixed_task_uncertainty")
class MixedTaskUncertaintyLoss(nn.Module):
    """혼합 task Kendall 가중 loss (표준 식).

    Args:
        n_tasks: task 수 (== len(task_types)).
        task_types: 각 task 의 "regression" | "classification" 라벨.
        task_weights: (n_tasks,) inverse-freq weight. None 이면 1.
        classification: PER_TASK_LOSSES 의 clf functional 키.
        regression: PER_TASK_LOSSES 의 reg functional 키.

    Forward:
        preds (B, n_tasks): reg 컬럼은 scaled, clf 컬럼은 logits.
        targets (B, n_tasks): NaN 가능 (mask 자동 처리).
        sample_weights (B,) or None: 샘플별 가중치 — task-별 평균 분자/분모에 함께 적용.
        Returns: scalar loss.

    Note:
        ``log_vars`` 는 ``task_has_data`` 인덱싱으로 dead task (모든 row NaN) 의
        gradient 가 자동 제외된다. ``task_denom = valid.sum(dim=0).clamp(min=1)`` 로
        division 안전. ``log_var_min`` floor 로 σ→0 폭주 (precision 발산) 차단.
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
            # Kendall σ 학습 자유도 무력화 — σ=1 고정 (precision=1).
            # 식이 inverse-freq weighted sum 으로 단순화.
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

    # ---- public components (PCGrad / 진단 path 공용) ---------------------------------

    def _compute_components(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """forward 의 내부 단계 (PCGrad path 에서 재사용).

        Returns:
            task_loss: (K,) per-task **mean** loss (그래프 보존, 가중치 미적용).
            weighted_task: (K,) Kendall 가중 후 per-task loss (그래프 보존).
                = reg_var · (1/σ²) · task_weight · L̄_k.
            log_var_reg: scalar 0.5 Σ log σ² (학습 가능한 ``log_vars`` 그래프 포함).
            task_has_data: (K,) bool — 모든 row NaN 인 dead task 는 False.
        """
        if preds.shape != targets.shape:
            raise ValueError(f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}")
        if preds.size(1) != self.n_tasks:
            raise ValueError(f"preds.size(1)={preds.size(1)} != n_tasks={self.n_tasks}")

        mask = targets.isfinite()                                    # (B, K)
        per_task = per_task_loss_dispatch(                           # (B, K) — NaN→0 내부 처리
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
        self._last_task_loss = task_loss.detach()                    # 진단용 side-car

        # log_var floor: σ→0 폭주 차단.
        safe_log_vars = self.log_vars.clamp(min=self.log_var_min)
        precision = torch.exp(-safe_log_vars)                        # (K,) 1/σ²
        weighted_task = (
            self.reg_var_per_task * precision
            * self.task_weights_buf * task_loss
        )                                                            # (K,)
        # dead task 의 weighted 항을 0 으로 — autograd.grad 호출 안전성 확보.
        weighted_task = weighted_task * task_has_data.to(weighted_task.dtype)
        log_var_reg = 0.5 * (safe_log_vars * task_has_data.to(safe_log_vars.dtype)).sum()
        return task_loss, weighted_task, log_var_reg, task_has_data

    def weighted_per_task_losses(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """PCGrad path 용: Kendall 가중 per-task loss 벡터 + log_var regularizer.

        Why:
            PCGrad 는 task-별 그라디언트가 따로 필요하므로 ``forward`` 의
            ``weighted[task_has_data].sum()`` 으로 즉시 reduce 하면 안 된다.
            본 메서드는 (K,) 벡터 + (scalar) regularizer + dead-task mask 를 분리해
            ``MultitaskLitModule._pcgrad_training_step`` 에서 K 번 ``autograd.grad``
            호출에 사용한다.

        Returns:
            weighted_task: (K,) — dead task 는 0 으로 마스킹됨.
            log_var_reg: 0.5 Σ log σ² (학습 가능한 ``log_vars`` 그래프 포함).
            task_has_data: (K,) bool — autograd.grad 스킵 판단용.
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

    Loss-level 정의는 ``MixedTaskUncertaintyLoss`` 와 100% 동일하다.
    본 subclass 는 ``LossSpec.type`` 으로 PCGrad 활성화 신호를 manifest 에 남기기
    위한 marker — ``MultitaskLitModule`` 가 isinstance / type-name 검사로
    PCGrad training_step 으로 분기한다.

    Why subclass (옵션 kwarg 대신):
        - manifest 의 ``loss.type`` 만 보면 "PCGrad on/off" 가 즉시 드러난다.
        - LossSpec 의 frozen 데이터모델을 그대로 유지 (kwargs 키 추가/검증 불필요).
        - Kendall UW + GradNorm 처럼 두 algorithm 이 manual_optimization 분기를
          가지는 패턴과 일관됨.

    Kwargs (kwargs 로 LossSpec.kwargs 에서 흘러옴):
        log_var_min: σ→0 floor (Kendall 과 동일, default -1.0).
        freeze_log_vars: σ 학습 끄기 (Kendall 과 동일, default False).
        pcgrad_scope: "trunk" (default) — backbone + fusion + extra_backbones
            파라미터에만 PCGrad projection 적용. "all" — head 포함 전체.
            LitModule 가 본 값을 읽어 shared-params 집합을 결정.
        pcgrad_eps: 0 으로 나눔 방지 (default 1e-12).
        uniform_weight: True 이면 Kendall UW 를 완전히 제거하고 **PCGrad 단독** 모드로
            동작한다. reg_var(0.5 reg / 1.0 clf) 까지 모두 1.0 으로 평탄화하고 σ 를
            1 로 고정(freeze_log_vars 강제)해 log_var regularizer=0 → weighted_L_k = L̄_k.
            즉 모든 task 를 균등 가중한 per-task loss 에 gradient surgery 만 적용.
            "Kendall UW + PCGrad" vs "PCGrad only" ablation 용 (default False).
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
            # PCGrad 단독: σ=1 고정 (precision=1, log_var_reg=0). 학습 자유도 제거.
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
                f"pcgrad_scope 는 'trunk' | 'all' — 받음: {pcgrad_scope!r}"
            )
        self.pcgrad_scope = pcgrad_scope
        self.pcgrad_eps = float(pcgrad_eps)
        self.uniform_weight = bool(uniform_weight)
        if uniform_weight:
            # reg_var(0.5/1.0) 평탄화 → 모든 task 균등 가중. buffer in-place 유지.
            self.reg_var_per_task.fill_(1.0)
