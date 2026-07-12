"""Per-task functional loss: BCE / MSE / MAE (mask 인지, reduction=none).

Why:
    옛 ``_grouped_loss`` 가 reg 컬럼은 ``(preds-targets)**2``, clf 컬럼은
    ``F.binary_cross_entropy_with_logits(..., reduction='none')`` 를 직접
    호출했다. 이를 PER_TASK_LOSSES registry 에 등록해서 LossSpec
    ``classification`` / ``regression`` 키로 교체 가능하게 한다.

함수 시그니처:
    fn(preds, targets, mask) -> Tensor (B, n_tasks)
        - mask: ``targets.isfinite()`` (Bool, B×n). NaN 위치는 0 으로 만든다.
        - 반환은 unreduced. 상위 loss(Kendall 등) 가 mask·sample_weight 곱·합 처리.
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
        targets: (B, n_tasks). NaN 가능. 반환 전에 NaN→0 으로 치환.
        mask: 사용하지 않음 (호환 시그니처). 마스킹은 상위 loss 에서.

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
        targets 의 NaN 자리는 mask 로 가려질 예정이지만, F.bce_with_logits
        는 NaN 입력에서 NaN gradient 를 만든다. 안전을 위해 0 으로 치환.
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
    """task_type 별로 reg/clf 컬럼을 분리 계산해 원본 순서로 scatter.

    Why:
        혼합 task 학습 시 reg 컬럼은 MSE, clf 컬럼은 BCE 를 적용해야 한다.
        PER_TASK_LOSSES 에서 키로 lookup → 컬럼별 적용 후 합쳐 (B, n_tasks) 반환.

    Args:
        preds: (B, n_tasks). reg 는 scaled space, clf 는 logits.
        targets: (B, n_tasks). NaN mask 가능.
        task_types: 길이 n_tasks. 각 항목은 "regression" 또는 "classification".
        classification_key, regression_key: PER_TASK_LOSSES registry 키.

    Returns:
        (B, n_tasks) per-task loss tensor.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"preds/targets shape mismatch: {preds.shape} vs {targets.shape}")
    if preds.size(1) != len(task_types):
        raise ValueError(
            f"task_types 길이({len(task_types)}) != preds.size(1)({preds.size(1)})"
        )

    reg_idx = [i for i, t in enumerate(task_types) if t == "regression"]
    clf_idx = [i for i, t in enumerate(task_types) if t == "classification"]
    other = [i for i, t in enumerate(task_types) if t not in ("regression", "classification")]
    if other:
        raise ValueError(f"task_types 에 알 수 없는 값: {[task_types[i] for i in other]}")

    reg_fn = PER_TASK_LOSSES.get(regression_key)
    clf_fn = PER_TASK_LOSSES.get(classification_key)

    out = torch.zeros_like(preds)
    # mixed precision 호환: per-task fn 결과(autocast 가 fp32 promote 한 BCE 등)를
    # out dtype 으로 캐스팅. 32-true 에서는 no-op.
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
