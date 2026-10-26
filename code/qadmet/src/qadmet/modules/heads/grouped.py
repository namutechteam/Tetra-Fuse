"""GroupedHead — group trunk based on task affinity + per-task head.

Preserves numerical equivalence with the old ``pipeline_mtl.py:GroupedHead`` (line ~835-867).

Structure:
    Z (B, input_dim)
        ↓ (group trunk)  Linear → ReLU → Dropout → Linear → ReLU
    h_g (B, trunk_hidden//2)  per group
        ↓ (per-task head) Linear(trunk_hidden//2, 1)
    out (B, n_tasks)
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .registry import HEADS


@HEADS.register("grouped")
class GroupedHead(nn.Module):
    """Z → group trunk → per-task head.

    Args:
        input_dim: Z dimension.
        ordered_tasks: output column order (list of task names).
        task_to_group: mapping from task name to group name.
        groups: list of group names that appear (determines trunk creation order).
        trunk_hidden: hidden size of the first trunk linear. The second is //2.
        dropout: dropout probability after the first ReLU.

    Why:
        Mitigates negative transfer via 7-group routing. Trunk is shared at the group level, so
        tasks within the same group share the representation.
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

        missing = [t for t in self.ordered_tasks if t not in self.task_to_group]
        if missing:
            raise ValueError(f"task not in task_to_group: {missing}")
        unknown_groups = [g for g in self.task_to_group.values() if g not in self.groups]
        if unknown_groups:
            raise ValueError(
                f"task_to_group values not in groups: {sorted(set(unknown_groups))}"
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
        # compute group trunk only once (shared Z)
        group_outs = {g: self.trunks[g](Z) for g in self.groups}
        outs = [
            self.heads[t](group_outs[self.task_to_group[t]]).squeeze(-1)
            for t in self.ordered_tasks
        ]
        return torch.stack(outs, dim=1)  # (B, n_tasks)
