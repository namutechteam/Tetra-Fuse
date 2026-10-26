"""AttentionFusion — softmax gate fusion (old pipeline_mtl.py:AttentionFusion).

Numerical equivalence:
    K=1 : trivial pass-through (projection → output, alpha=ones).
    K>1 : softmax gate learning.
        gate(cat) = Linear(K*D, D) → ReLU → Linear(D, K)
        alpha     = softmax(gate(cat), dim=-1)
        z_fused   = Σ_k alpha[:, k:k+1] * proj_k(emb_k)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("attention")
class AttentionFusion(nn.Module):
    """Combine K embeddings via learned softmax α.

    Args:
        input_dims: dimension of each modality embedding (length K).
        output_dim: common projection dimension.
    """

    def __init__(self, input_dims: list[int], output_dim: int = 300) -> None:
        super().__init__()
        self.K = len(input_dims)
        self.output_dim = output_dim
        if self.K == 0:
            raise ValueError("AttentionFusion requires at least 1 input.")
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
        if self.K > 1:
            self.gate = nn.Sequential(
                nn.Linear(output_dim * self.K, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, self.K),
            )

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received count: {len(embeddings)}")
        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]
        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha
        cat = torch.cat(projected, dim=-1)
        alpha = torch.softmax(self.gate(cat), dim=-1)  # (B, K)
        stacked = torch.stack(projected, dim=1)  # (B, K, D)
        z_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        return z_fused, alpha
