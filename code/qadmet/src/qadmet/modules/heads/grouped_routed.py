"""GroupedRoutedHead — per-group fused Z (B, n_groups, D) → group trunk → per-task head.

``grouped`` 와의 차이:
    - 입력 Z 가 3D (B, n_groups, input_dim). GR-MF (group_routed fusion) 의 출력을 직접 받는다.
    - group axis 는 fusion 의 ``groups`` 순서와 동일해야 한다 (lit_module 이 동일
      list 를 양쪽에 주입).
    - trunk 구조는 grouped 와 수치 동등.
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
        input_dim: per-group fused vector 의 차원 D.
        ordered_tasks: 출력 컬럼 순서.
        task_to_group: task name → group name 매핑.
        groups: fusion 과 동일한 순서의 group name 리스트. group axis index 와 1:1 대응.
        trunk_hidden: 첫 Linear 의 hidden size.
        dropout: 첫 ReLU 뒤 dropout 확률.
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
            raise ValueError(f"task_to_group 에 없는 task: {missing}")
        unknown_groups = [g for g in self.task_to_group.values() if g not in self.groups]
        if unknown_groups:
            raise ValueError(
                f"task_to_group 의 값이 groups 에 없음: {sorted(set(unknown_groups))}"
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
                f"GroupedRoutedHead 는 (B, n_groups, D) 3D 입력 필요, got {Z.dim()}D"
            )
        if Z.size(1) != len(self.groups):
            raise ValueError(
                f"Z group axis ({Z.size(1)}) != head groups ({len(self.groups)}). "
                "fusion 과 head 의 groups 순서가 동일한지 확인."
            )
        group_outs = {
            g: self.trunks[g](Z[:, self.group_idx[g], :]) for g in self.groups
        }
        outs = [
            self.heads[t](group_outs[self.task_to_group[t]]).squeeze(-1)
            for t in self.ordered_tasks
        ]
        return torch.stack(outs, dim=1)  # (B, n_tasks)
