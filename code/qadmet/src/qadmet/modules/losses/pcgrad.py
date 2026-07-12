"""PCGrad gradient projection (Yu et al. 2020, NeurIPS).

표준 알고리즘 — Kendall UW 가 정한 loss-level 가중치로 계산된 task 별
그라디언트를 받아 task 간 충돌 (negative cosine) 방향을 제거한다::

    for k in 1..K:
        g_k ← flatten(∇_θ L_k^weighted)
    for k in random_order:
        for j ≠ k in random_order:
            if g_k · g_j < 0:
                g_k ← g_k - (g_k · g_j / ‖g_j‖²) g_j   # j 의 정상평면으로 사영
    g_total ← Σ_k g_k

본 모듈은 Kendall UW path 와 직교하게 동작 — Kendall 이 결정한 task weight 가
이미 입력 그라디언트에 반영되어 있다고 가정한다 (lit_module 가 weighted_per_task_losses
로부터 K 개의 ``autograd.grad`` 를 호출해 ``grads_per_task`` 를 만든다).

Why pure-function:
    PCGrad 는 stateless 알고리즘 (학습 가능 파라미터 없음). class 로 묶을 이유가
    없고, ``MultitaskLitModule`` 가 manual optimization 안에서 직접 호출한다.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _flatten_grads(
    grads_per_task: list[list[Tensor | None]],
    shapes: list[torch.Size],
) -> list[Tensor]:
    """task 별 (param list of tensors) → task 별 (D,) flattened tensor.

    ``None`` 그라디언트 (autograd.grad allow_unused=True 결과 또는 dead task)
    는 0 텐서로 치환되어 동일 layout 을 유지한다.
    """
    flat_per_task: list[Tensor] = []
    for grads in grads_per_task:
        parts: list[Tensor] = []
        for g, shape in zip(grads, shapes):
            if g is None:
                # dead task / requires_grad=False: 동일 shape 의 0 으로 채워 layout 유지.
                parts.append(torch.zeros(int(torch.tensor(shape).prod()), device=_first_device(grads_per_task)))
            else:
                parts.append(g.reshape(-1))
        flat_per_task.append(torch.cat(parts, dim=0) if parts else torch.zeros(0))
    return flat_per_task


def _first_device(grads_per_task: list[list[Tensor | None]]) -> torch.device:
    for grads in grads_per_task:
        for g in grads:
            if g is not None:
                return g.device
    return torch.device("cpu")


def _unflatten_to_shapes(
    flat: Tensor, shapes: list[torch.Size]
) -> list[Tensor]:
    out: list[Tensor] = []
    offset = 0
    for shape in shapes:
        n = int(torch.tensor(shape).prod()) if len(shape) > 0 else 1
        out.append(flat[offset : offset + n].reshape(shape))
        offset += n
    return out


def project_conflicting_gradients(
    grads_per_task: list[list[Tensor | None]],
    shapes: list[torch.Size],
    *,
    valid_task: list[bool] | None = None,
    eps: float = 1e-12,
    generator: torch.Generator | None = None,
) -> list[Tensor]:
    """PCGrad projection 후 합산된 shared-params 그라디언트 (param shape 별 list).

    Args:
        grads_per_task: ``grads_per_task[k][i]`` = task k 의 shared param i 그라디언트.
            ``None`` 허용 (autograd.grad allow_unused, dead task).
        shapes: shared params 각각의 ``shape``. ``None`` 을 0 텐서로 복원하는 데 사용.
        valid_task: 길이 K bool — False 인 task 는 그라디언트 0 으로 간주, 다른 task
            의 projection 에도 참여하지 않음. None 이면 grad norm > 0 인 task 만 사용.
        eps: 분모 클램프.
        generator: ``torch.randperm`` 시드. None 이면 글로벌 RNG.

    Returns:
        list[Tensor] — shared params 별 PCGrad 그라디언트 (원본 shape).
    """
    K = len(grads_per_task)
    if K == 0:
        return [torch.zeros(shape) for shape in shapes]

    flat = _flatten_grads(grads_per_task, shapes)
    # validity: 명시된 valid_task 또는 ‖g_k‖>0.
    if valid_task is None:
        valid_task = [bool(g.pow(2).sum().sqrt().item() > eps) for g in flat]
    valid_idx = [k for k in range(K) if valid_task[k]]
    if not valid_idx:
        # 모든 task 가 dead — 0 그라디언트.
        return [torch.zeros(shape, device=flat[0].device, dtype=flat[0].dtype) for shape in shapes]

    device = flat[0].device

    # PCGrad: 각 task k 의 g_k 를 모든 j (j∈valid_idx, j≠k) 와 비교, 충돌(<0) 사영 제거.
    # j 순서는 random — Yu et al. 표준.
    pc_flat: list[Tensor] = [flat[k].clone() for k in range(K)]
    for k in valid_idx:
        # j 순서 셔플 (k 자신 제외)
        others = torch.tensor([j for j in valid_idx if j != k], device=device)
        if others.numel() == 0:
            continue
        perm = torch.randperm(others.numel(), generator=generator, device=device)
        for pos in perm.tolist():
            j = int(others[pos].item())
            g_j = flat[j]              # 원본 (다른 task projection 영향 X — 표준)
            dot = (pc_flat[k] * g_j).sum()
            if dot.item() < 0.0:
                denom = g_j.pow(2).sum().clamp_min(eps)
                pc_flat[k] = pc_flat[k] - (dot / denom) * g_j

    # 합산 (Yu et al. default — mean 이 아닌 sum: Kendall 가중치가 이미 스케일 결정).
    total_flat = torch.zeros_like(pc_flat[0])
    for k in valid_idx:
        total_flat = total_flat + pc_flat[k]

    return _unflatten_to_shapes(total_flat, shapes)


def cosine_conflict_rate(
    grads_per_task: list[list[Tensor | None]],
    shapes: list[torch.Size],
    *,
    valid_task: list[bool] | None = None,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """진단 신호: 모든 (k, j) pair 중 cos < 0 비율 + 평균 cosine.

    PCGrad 가 실제로 얼마나 많은 충돌을 사영했는지를 epoch-level 로 로깅 가능.
    """
    flat = _flatten_grads(grads_per_task, shapes)
    K = len(flat)
    if valid_task is None:
        valid_task = [bool(g.pow(2).sum().sqrt().item() > eps) for g in flat]
    valid_idx = [k for k in range(K) if valid_task[k]]
    if len(valid_idx) < 2:
        return 0.0, 0.0
    cosines: list[float] = []
    n_conflict = 0
    for i in range(len(valid_idx)):
        for j in range(i + 1, len(valid_idx)):
            a, b = flat[valid_idx[i]], flat[valid_idx[j]]
            na = a.pow(2).sum().sqrt().clamp_min(eps)
            nb = b.pow(2).sum().sqrt().clamp_min(eps)
            c = float(((a * b).sum() / (na * nb)).item())
            cosines.append(c)
            if c < 0.0:
                n_conflict += 1
    return n_conflict / len(cosines), sum(cosines) / len(cosines)
