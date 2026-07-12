"""LinearHead: Z (B, input_dim) → (B, n_tasks). 단순 다중 태스크 head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import HEADS


@HEADS.register("linear")
class LinearHead(nn.Module):
    """공유 trunk 위 단일 Linear projection.

    Args:
        input_dim: Z 차원.
        n_tasks: task 수 (= 출력 차원).
        bias: nn.Linear bias 사용 여부.
    """

    def __init__(self, input_dim: int, n_tasks: int, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_tasks = n_tasks
        self.proj = nn.Linear(input_dim, n_tasks, bias=bias)

    def forward(self, Z: Tensor) -> Tensor:
        return self.proj(Z)
