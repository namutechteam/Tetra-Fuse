"""GatedFusion — independently controls per-modality contribution via a sigmoid gate.

Why:
    The old AttentionFusion's softmax causing weight to collapse to 1 for a single modality
    "alpha-collapse" phenomenon occurred (per user feedback in CLAUDE.md). The sigmoid gate
    does not sum to 1, so each modality can independently survive or die, making it
    more conducive to mitigating collapse.

Formula:
    proj_k = Linear(D_k, output_dim)(emb_k)        for k in 1..K
    gate   = sigmoid(Linear(K * output_dim, K)(cat([proj_1, ..., proj_K])))   # (B, K)
    z      = Σ_k gate[:, k:k+1] * proj_k

When K=1, trivial (gate ≡ 1).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("gated")
class GatedFusion(nn.Module):
    """sigmoid gate fusion."""

    def __init__(self, input_dims: list[int], output_dim: int = 300) -> None:
        super().__init__()
        self.K = len(input_dims)
        self.output_dim = output_dim
        if self.K == 0:
            raise ValueError("GatedFusion requires at least 1 input.")
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
        if self.K > 1:
            self.gate = nn.Linear(output_dim * self.K, self.K)

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")
        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]
        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha
        cat = torch.cat(projected, dim=-1)
        alpha = torch.sigmoid(self.gate(cat))  # (B, K), sum != 1
        stacked = torch.stack(projected, dim=1)  # (B, K, D)
        z = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        return z, alpha
