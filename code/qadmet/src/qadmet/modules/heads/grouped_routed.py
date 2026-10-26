"""GroupedRoutedHead — per-group fused Z (B, n_groups, D) → group trunk → per-task head.

Difference from ``grouped``:
    - Input Z is 3D (B, n_groups, input_dim). Directly receives the output of GR-MF (group_routed fusion).
    - The group axis must match the order of ``groups`` in fusion (lit_module injects the same
      list into both).
    - The trunk structure is numerically equivalent to grouped.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .registry import HEADS


@HEADS.register("grouped_routed")
class GroupedRoutedHead(nn.Module):
    """Z (B, n_groups, D) → per-group trunk → per-task linear.

    Args:
        input_dim: dimension D of the per-group fused vector.
        ordered_tasks: output column order.
        task_to_group: task name → group name mapping.
        groups: list of group names in the same order as fusion. 1:1 correspondence with group axis index.
        trunk_hidden: hidden size of the first Linear.
        dropout: dropout probability after the first ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        ordered_tasks: list[str],
        task_to_group: Mapping[str, str],
        groups: list[str],
        trunk_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.ordered_tasks = list(ordered_tasks)
        self.task_to_group = dict(task_to_group)
        self.groups = list(groups)
        self.group_idx = {g: i for i, g in enumerate(self.groups)}

        missing = [t for t in self.ordered_tasks if t not in self.task_to_group]
        if missing:
            raise ValueError(f"task not in task_to_group: {missing}")
        unknown_groups = [g for g in self.task_to_group.values() if g not in self.groups]
        if unknown_groups:
            raise ValueError(
                f"Value of task_to_group not in groups: {sorted(set(unknown_groups))}"
            )

        self.trunks = nn.ModuleDict({
            g: nn.Sequential(
                nn.Linear(input_dim, trunk_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(trunk_hidden, trunk_hidden // 2),
                nn.ReLU(),
            )
            for g in self.groups
        })
        self.heads = nn.ModuleDict({
            t: nn.Linear(trunk_hidden // 2, 1) for t in self.ordered_tasks
        })

    def forward(self, Z: Tensor) -> Tensor:
        if Z.dim() != 3:
            raise ValueError(
                f"GroupedRoutedHead requires (B, n_groups, D) 3D input, got {Z.dim()}D"
            )
        if Z.size(1) != len(self.groups):
            raise ValueError(
                f"Z group axis ({Z.size(1)}) != head groups ({len(self.groups)}). "
                "Check that the groups order matches between fusion and head."
            )
        group_outs = {
            g: self.trunks[g](Z[:, self.group_idx[g], :]) for g in self.groups
        }
        outs = [
            self.heads[t](group_outs[self.task_to_group[t]]).squeeze(-1)
            for t in self.ordered_tasks
        ]
        return torch.stack(outs, dim=1)  # (B, n_tasks)
