"""ConcatFusion — concatenate modality embeddings as-is, then apply linear projection.

Single projection when K=1. alpha has no real meaning but returns 1/K uniform for format consistency.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("concat")
class ConcatFusion(nn.Module):
    """concat → Linear(sum(input_dims), output_dim).

    Why:
        Simplest baseline. Similar to AttentionFusion's K=1 behavior, but for multiple
        modalities, applies projection jointly instead of separately.
    """

    def __init__(self, input_dims: list[int], output_dim: int = 300) -> None:
        super().__init__()
        self.K = len(input_dims)
        self.output_dim = output_dim
        if self.K == 0:
            raise ValueError("ConcatFusion requires at least 1 input.")
        self.proj = nn.Linear(sum(input_dims), output_dim)

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")
        cat = torch.cat(embeddings, dim=-1)
        z = self.proj(cat)
        alpha = torch.full((z.size(0), self.K), 1.0 / self.K, device=z.device)
        return z, alpha
