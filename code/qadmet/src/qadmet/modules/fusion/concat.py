"""ConcatFusion — 모달리티 임베딩을 그대로 concat 후 linear projection.

K=1 일 때는 단일 projection. alpha 는 의미 없으나 형식 통일을 위해 1/K uniform 반환.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("concat")
class ConcatFusion(nn.Module):
    """concat → Linear(sum(input_dims), output_dim).

    Why:
        가장 단순한 baseline. AttentionFusion 의 K=1 동작과 비슷하지만, 다중
        모달리티 시 projection 을 분리하지 않고 합쳐서 한 번에 적용한다.
    """

    def __init__(self, input_dims: list[int], output_dim: int = 300) -> None:
        super().__init__()
        self.K = len(input_dims)
        self.output_dim = output_dim
        if self.K == 0:
            raise ValueError("ConcatFusion 은 최소 1 개 input 이 필요합니다.")
        self.proj = nn.Linear(sum(input_dims), output_dim)

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")
        cat = torch.cat(embeddings, dim=-1)
        z = self.proj(cat)
        alpha = torch.full((z.size(0), self.K), 1.0 / self.K, device=z.device)
        return z, alpha
