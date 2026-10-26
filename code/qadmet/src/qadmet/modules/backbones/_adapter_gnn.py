"""AdapterGNNBlock — paper-faithful AdapterGNN (Li et al., AAAI'24).

Paper Eq. (7):
    A(x) = BN(W_up(ReLU(W_down(x))))

This implementation additionally multiplies by a learnable scalar ``s`` (=γ, init=0) before returning:

    forward(x) = s · BN(W_up(ReLU(W_down(x))))

i.e., **no internal residual**. Sum of parallel branches per paper Eq. (8):

    h_l = BN(MLP(MP(x_l))) + s1·A1(x_l) + s2·A2(MP(x_l))

is implemented in the backbone-side forward by adding the frozen path + two adapter outputs element-wise
implemented. The adapter only returns ``s·A(x)`` (= delta); the residual sum is done by the caller.

Design points:
    - BN1d(d) — paper: BN is advantageous for mitigating graph over-smoothing (vs. LN).
    - ``s`` (=γ, learnable scalar, init=0) — preserves backbone-only distribution early in training,
      the model dynamically gates the task knowledge ratio (mitigates catastrophic forgetting).
    - ``up.weight`` zero-init is not used (γ=0 alone guarantees zero initial output).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdapterGNNBlock(nn.Module):
    """Paper-faithful AdapterGNN bottleneck (NO internal residual).

    Args:
        d: input/output feature dim (frozen backbone hidden dim).
        r: bottleneck dim (typically much smaller than d).
        dropout: dropout probability immediately after BN+scale.

    Attributes:
        d, r, gamma (learnable scalar ``s``, init=0).
    """

    def __init__(self, d: int, r: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d = d
        self.r = r

        self.down = nn.Linear(d, r)
        self.act = nn.ReLU()
        self.up = nn.Linear(r, d)
        self.bn = nn.BatchNorm1d(d)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``s · BN(W_up(ReLU(W_down(x))))`` (NO residual added).

        Args:
            x: ``(*, d)`` — arbitrary shape whose last axis is ``d``. For ``BatchNorm1d`` compatibility,
                reshape internally to ``(N, d)`` then restore the result to its original shape.

        Returns:
            ``s · A(x)`` (same shape as ``x``). The caller sums it element-wise with the frozen path.
        """
        orig_shape = x.shape
        if x.dim() != 2:
            x_flat = x.reshape(-1, self.d)
        else:
            x_flat = x

        z = self.down(x_flat)
        z = self.act(z)
        z = self.up(z)
        z = self.bn(z)
        z = self.gamma * z
        z = self.drop(z)

        if z.shape != orig_shape:
            z = z.reshape(orig_shape)
        return z
