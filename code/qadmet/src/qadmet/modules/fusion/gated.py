"""GatedFusion — sigmoid 게이트로 모달리티별 contribution 을 독립적으로 조절.

Why:
    옛 AttentionFusion 의 softmax 가 한 모달리티에 weight 가 1 로 collapse 하는
    "alpha-collapse" 현상이 있었다 (CLAUDE.md 의 사용자 의견). sigmoid 게이트는
    합이 1 이 아니므로 각 모달리티가 독립적으로 살아남거나 죽을 수 있어 collapse
    완화에 더 친화적.

수식:
    proj_k = Linear(D_k, output_dim)(emb_k)        for k in 1..K
    gate   = sigmoid(Linear(K * output_dim, K)(cat([proj_1, ..., proj_K])))   # (B, K)
    z      = Σ_k gate[:, k:k+1] * proj_k

K=1 일 때는 trivial (gate ≡ 1).
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
            raise ValueError("GatedFusion 은 최소 1 개 input 이 필요합니다.")
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
        if self.K > 1:
            self.gate = nn.Linear(output_dim * self.K, self.K)

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")
        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]
        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha
        cat = torch.cat(projected, dim=-1)
        alpha = torch.sigmoid(self.gate(cat))  # (B, K), 합 != 1
        stacked = torch.stack(projected, dim=1)  # (B, K, D)
        z = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        return z, alpha
