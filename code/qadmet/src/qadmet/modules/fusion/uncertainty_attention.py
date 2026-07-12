"""UncertaintyAttentionFusion — RobustAttention + task uncertainty coupling.

설계:
    - RobustAttention 을 상속.
    - 추가로 modality 별 learned precision (log_precision_k) 파라미터.
    - gate logits 에 precision bias 추가: logits += log_precision.
    - MixedTaskUncertaintyLoss 의 sigma 신호를 받아 tau 에 반영
      (sigma 높으면 tau 올려 균등 분포 유도, low-uncertainty modality 자동 집중).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS
from .robust_attention import RobustAttentionFusion


@FUSIONS.register("uncertainty_attention")
class UncertaintyAttentionFusion(RobustAttentionFusion):
    """RobustAttention + learned modality precision (MixedTaskUncertaintyLoss 연동).

    부모 인자는 RobustAttentionFusion 과 동일.
    추가 인자:
        None (precision 은 자동 초기화).
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int = 300,
        tau_init: float = 2.0,
        tau_min: float = 0.5,
        anneal_epochs: int = 50,
        modality_dropout: float = 0.1,
        residual: bool = True,
        log_precision_min: float = -2.0,
        log_precision_max: float = 2.0,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dim=output_dim,
            tau_init=tau_init,
            tau_min=tau_min,
            anneal_epochs=anneal_epochs,
            modality_dropout=modality_dropout,
            residual=residual,
        )
        # modality 별 learned precision (log scale).
        # gate logits 폭주(=alpha collapse) 방어를 위해 [-2, 2] clamp 적용.
        self.log_precision = nn.Parameter(torch.zeros(self.K))
        self.log_precision_min = float(log_precision_min)
        self.log_precision_max = float(log_precision_max)

    def set_sigma_signal(self, sigma_mean: float) -> None:
        """Task 평균 uncertainty 신호를 받아 tau 에 반영.

        sigma 가 높으면 (고 불확실도) tau 를 올려 modality 간 균등 분포 유도.
        sigma 가 낮으면 (저 불확실도) tau 를 내려 winner-take-all 선호.

        Args:
            sigma_mean: MixedTaskUncertaintyLoss log_vars.exp().sqrt().mean() 값.
        """
        # sigma_mean ∈ [0.5, 3.0] 범위로 clamp
        sigma_signal = torch.tensor(sigma_mean, dtype=torch.float32)
        sigma_clamped = torch.clamp(sigma_signal, 0.5, 3.0)
        # tau_effective = tau_init * sigma_clamped
        tau_effective = self.tau_init * sigma_clamped.item()
        self.tau = tau_effective

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        """K 개 embedding 을 fusion (precision bias 포함).

        Args:
            embeddings: 길이 K 의 tensor 리스트.

        Returns:
            z_fused: (B, output_dim)
            alpha: (B, K) softmax weights (precision bias 포함)
        """
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")

        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]

        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha

        stacked = torch.stack(projected, dim=1)  # (B, K, D)
        B, K, D = stacked.shape

        # gate logits + precision bias (clamp 로 폭주 차단)
        cat = torch.cat(projected, dim=-1)  # (B, K*D)
        lp = self.log_precision.clamp(min=self.log_precision_min, max=self.log_precision_max)
        logits = self.gate(cat) + lp  # (B, K) + (K,) → broadcast

        # modality dropout
        if self.training and self.modality_dropout > 0:
            mask = torch.rand(B, K, device=logits.device) > self.modality_dropout
            any_active = mask.any(dim=-1, keepdim=True)
            mask = mask | ~any_active
            logits = logits.masked_fill(~mask, -1e9)

        # softmax with current temperature
        alpha = torch.softmax(logits / self.tau, dim=-1)  # (B, K)

        # weighted sum
        z_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)  # (B, D)

        # residual
        if self.residual:
            z_fused = z_fused + stacked.mean(dim=1)

        return z_fused, alpha
