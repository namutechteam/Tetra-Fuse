"""BottleneckHead: Z (B, input_dim) → (B, n_tasks). reuses trunk from grouped.py."""

from __future__ import annotations

from torch import Tensor, nn

from .registry import HEADS


@HEADS.register("bottleneck")
class BottleneckHead(nn.Module):
    """Single input → bottleneck MLP → n_tasks projection.

    Args:
        input_dim: Z dimension.
        n_tasks: output dimension.
        trunk_hidden: first hidden size. Second is //2.
        dropout: dropout probability after first ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        trunk_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_tasks = n_tasks
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden // 2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(trunk_hidden // 2, n_tasks)

    def forward(self, Z: Tensor) -> Tensor:
        return self.proj(self.trunk(Z))
