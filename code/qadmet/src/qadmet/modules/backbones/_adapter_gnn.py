"""AdapterGNNBlock — paper-faithful AdapterGNN (Li et al., AAAI'24).

논문 Eq. (7):
    A(x) = BN(W_up(ReLU(W_down(x))))

본 구현은 학습 가능한 scalar ``s`` (=γ, init=0) 를 추가로 곱해서 반환:

    forward(x) = s · BN(W_up(ReLU(W_down(x))))

즉 **internal residual 없음**. 논문 Eq. (8) 의 parallel branch 합산:

    h_l = BN(MLP(MP(x_l))) + s1·A1(x_l) + s2·A2(MP(x_l))

은 backbone 측 forward 에서 frozen 경로 + 두 어댑터 출력을 element-wise 더해
구현. 어댑터는 ``s·A(x)`` (= delta) 만 반환하고 residual 합산은 caller 가 함.

설계 포인트:
    - BN1d(d) — 논문: graph over-smoothing 완화에 BN 유리 (LN 대비).
    - ``s`` (=γ, learnable scalar, init=0) — 학습 초기에 backbone-only 분포 보존,
      task knowledge 비율을 모델이 동적으로 게이팅 (catastrophic forgetting 완화).
    - ``up.weight`` zero-init 은 사용 안 함 (γ=0 만으로 초기 출력 0 이 보장됨).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdapterGNNBlock(nn.Module):
    """Paper-faithful AdapterGNN bottleneck (NO internal residual).

    Args:
        d: input/output feature dim (frozen backbone hidden dim).
        r: bottleneck dim (보통 d 보다 훨씬 작음).
        dropout: BN+scale 직후의 dropout 확률.

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
            x: ``(*, d)`` — 마지막 축이 ``d`` 인 임의 shape. ``BatchNorm1d`` 호환을 위해
                내부에서 ``(N, d)`` 로 reshape 후 결과를 원형 복원.

        Returns:
            ``s · A(x)`` (same shape as ``x``). caller 가 frozen path 와 element-wise 합산.
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
