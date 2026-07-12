"""Single-task loss — STL 학습용 thin wrapper.

옛 STL 코드 (``run_stl_dmpnn_rdkit.py``, ``run_stl_molformer_lora.py``,
``run_stl_unimol2_lora.py``) 는 task 별 ``F.binary_cross_entropy_with_logits``
또는 ``F.mse_loss`` 를 직접 호출했다. 본 모듈은 그 동작을 ``LOSSES`` registry
항목 ``"single_task"`` 로 편입시켜 STL runner 가 ``LossSpec(type="single_task")``
하나만 보고 동일하게 빌드할 수 있게 한다.

Why:
    MTL Kendall/Mixed loss 와 동일한 ``MultitaskLitModule`` 인터페이스를 STL 에서도
    사용한다. n_tasks=1 강제 + NaN-safe mask + sample_weights 지원.
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import LOSSES, PER_TASK_LOSSES


@LOSSES.register("single_task")
class SingleTaskLoss(nn.Module):
    """단일 task 학습용 thin loss wrapper.

    Args:
        task_type: ``"regression"`` | ``"classification"``.
        classification: PER_TASK_LOSSES clf 키 (기본 ``"bce_with_logits"``).
        regression: PER_TASK_LOSSES reg 키 (기본 ``"masked_mse"`` — 옛 STL MSE 동등).

    Forward:
        preds (B, 1): regression 은 scaled space, classification 은 logits.
        targets (B, 1): NaN 가능 — finite mask 자동 적용.
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
                f"task_type 은 'regression' 또는 'classification': {task_type!r}"
            )
        self.task_type = task_type
        self.classification_key = classification
        self.regression_key = regression
        self._fn_key = regression if task_type == "regression" else classification
        # 등록 누락 빠른 fail.
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
                f"single_task 는 n_tasks=1 만 지원 (받음: {preds.size(1)})"
            )
        fn = PER_TASK_LOSSES.get(self._fn_key)
        per_elem = fn(preds, targets)  # (B, 1), reduction=none
        mask = targets.isfinite()
        per_elem = per_elem * mask.to(per_elem.dtype)
        if sample_weights is not None:
            per_elem = per_elem * sample_weights.to(per_elem.dtype).view(-1, 1)
        denom = mask.sum().clamp(min=1)
        return per_elem.sum() / denom
