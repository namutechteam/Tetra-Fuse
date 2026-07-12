"""GroupedHead — task affinity 기반 group trunk + per-task head.

옛 ``pipeline_mtl.py:GroupedHead`` (line ~835-867) 와 수치 동등성 보존.

구조:
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
        input_dim: Z 차원.
        ordered_tasks: 출력 컬럼 순서 (task name 리스트).
        task_to_group: task name → group name 매핑.
        groups: 등장하는 group name 리스트 (trunk 생성 순서 결정).
        trunk_hidden: 첫 번째 trunk linear 의 hidden size. 두 번째는 //2.
        dropout: 첫 번째 ReLU 뒤 dropout 확률.

    Why:
        7-group routing 으로 negative transfer 완화. group 단위로 trunk 를
        공유하므로 같은 group 안의 task 들이 representation 을 공유.
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
        # group trunk 한 번만 계산 (같은 Z 공유)
        group_outs = {g: self.trunks[g](Z) for g in self.groups}
        outs = [
            self.heads[t](group_outs[self.task_to_group[t]]).squeeze(-1)
            for t in self.ordered_tasks
        ]
        return torch.stack(outs, dim=1)  # (B, n_tasks)
