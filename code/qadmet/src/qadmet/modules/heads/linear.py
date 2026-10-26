"""LinearHead: Z (B, input_dim) → (B, n_tasks). Simple multi-task head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import HEADS


@HEADS.register("linear")
class LinearHead(nn.Module):
    """Single Linear projection on top of the shared trunk.

    Args:
        input_dim: Z dimension.
        n_tasks: number of tasks (= output dimension).
        bias: whether to use nn.Linear bias.
    """

    def __init__(self, input_dim: int, n_tasks: int, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_tasks = n_tasks
        self.proj = nn.Linear(input_dim, n_tasks, bias=bias)

    def forward(self, Z: Tensor) -> Tensor:
        return self.proj(Z)
