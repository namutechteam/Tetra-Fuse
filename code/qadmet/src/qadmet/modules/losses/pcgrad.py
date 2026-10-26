"""PCGrad gradient projection (Yu et al. 2020, NeurIPS).

Standard algorithm — per-task gradients, computed with loss-level weights determined by Kendall UW
receives gradients and removes conflicting (negative cosine) directions between tasks::

    for k in 1..K:
        g_k ← flatten(∇_θ L_k^weighted)
    for k in random_order:
        for j ≠ k in random_order:
            if g_k · g_j < 0:
                g_k ← g_k - (g_k · g_j / ‖g_j‖²) g_j   # projected onto the normal plane of j
    g_total ← Σ_k g_k

This module operates orthogonally to the Kendall UW path — the task weights determined by Kendall
are assumed to already be reflected in the input gradients (lit_module calls weighted_per_task_losses
and calls ``autograd.grad`` K times from it to build ``grads_per_task``).

Why pure-function:
    PCGrad is a stateless algorithm (no learnable parameters). There's no reason to wrap it in a
    class, and ``MultitaskLitModule`` calls it directly within manual optimization.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _flatten_grads(
    grads_per_task: list[list[Tensor | None]],
    shapes: list[torch.Size],
) -> list[Tensor]:
    """Per-task (param list of tensors) → per-task (D,) flattened tensor.

    ``None`` gradients (result of autograd.grad allow_unused=True or a dead task)
    are replaced with a 0 tensor to preserve the same layout.
    """
    flat_per_task: list[Tensor] = []
    for grads in grads_per_task:
        parts: list[Tensor] = []
        for g, shape in zip(grads, shapes):
            if g is None:
                # dead task / requires_grad=False: fill with 0 of the same shape to preserve layout.
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
    """Summed shared-params gradient after PCGrad projection (list per param shape).

    Args:
        grads_per_task: ``grads_per_task[k][i]`` = gradient of shared param i for task k.
            ``None`` allowed (autograd.grad allow_unused, dead task).
        shapes: ``shape`` of each shared param. Used to restore ``None`` as a 0 tensor.
        valid_task: length-K bool — tasks with False are treated as having 0 gradient, and do not participate in
            the projection of other tasks either. If None, only tasks with grad norm > 0 are used.
        eps: denominator clamp.
        generator: seed for ``torch.randperm``. If None, uses the global RNG.

    Returns:
        list[Tensor] — PCGrad gradient per shared param (original shape).
    """
    K = len(grads_per_task)
    if K == 0:
        return [torch.zeros(shape) for shape in shapes]

    flat = _flatten_grads(grads_per_task, shapes)
    # validity: explicit valid_task or ‖g_k‖>0.
    if valid_task is None:
        valid_task = [bool(g.pow(2).sum().sqrt().item() > eps) for g in flat]
    valid_idx = [k for k in range(K) if valid_task[k]]
    if not valid_idx:
        # all tasks are dead — 0 gradient.
        return [torch.zeros(shape, device=flat[0].device, dtype=flat[0].dtype) for shape in shapes]

    device = flat[0].device

    # PCGrad: compare each task k's g_k against all j (j∈valid_idx, j≠k), removing conflicting (<0) projections.
    # order of j is random — standard per Yu et al.
    pc_flat: list[Tensor] = [flat[k].clone() for k in range(K)]
    for k in valid_idx:
        # shuffle order of j (excluding k itself)
        others = torch.tensor([j for j in valid_idx if j != k], device=device)
        if others.numel() == 0:
            continue
        perm = torch.randperm(others.numel(), generator=generator, device=device)
        for pos in perm.tolist():
            j = int(others[pos].item())
            g_j = flat[j]              # original (not affected by other task projections — standard)
            dot = (pc_flat[k] * g_j).sum()
            if dot.item() < 0.0:
                denom = g_j.pow(2).sum().clamp_min(eps)
                pc_flat[k] = pc_flat[k] - (dot / denom) * g_j

    # sum (Yu et al. default — sum, not mean: Kendall weights already determine the scale).
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
    """Diagnostic signal: fraction of cos < 0 among all (k, j) pairs + mean cosine.

    Can be logged at epoch-level to see how many conflicts PCGrad actually projected.
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
