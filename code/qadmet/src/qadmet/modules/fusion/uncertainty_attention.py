"""UncertaintyAttentionFusion — RobustAttention + task uncertainty coupling.

Design:
    - Inherits from RobustAttention.
    - Additionally, per-modality learned precision (log_precision_k) parameter.
    - Add precision bias to gate logits: logits += log_precision.
    - Receives sigma signal from MixedTaskUncertaintyLoss and reflects it in tau
      (if sigma is high, raise tau to induce uniform distribution; automatically focus on low-uncertainty modality).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS
from .robust_attention import RobustAttentionFusion


@FUSIONS.register("uncertainty_attention")
class UncertaintyAttentionFusion(RobustAttentionFusion):
    """RobustAttention + learned modality precision (integrates with MixedTaskUncertaintyLoss).

    Parent arguments are identical to RobustAttentionFusion.
    Additional arguments:
        None (precision is auto-initialized).
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
        # per-modality learned precision (log scale).
        # Apply [-2, 2] clamp to guard against gate logits blow-up (=alpha collapse).
        self.log_precision = nn.Parameter(torch.zeros(self.K))
        self.log_precision_min = float(log_precision_min)
        self.log_precision_max = float(log_precision_max)

    def set_sigma_signal(self, sigma_mean: float) -> None:
        """Receives task-averaged uncertainty signal and reflects it in tau.

        If sigma is high (high uncertainty), raise tau to induce uniform distribution across modalities.
        If sigma is low (low uncertainty), lower tau to favor winner-take-all.

        Args:
            sigma_mean: MixedTaskUncertaintyLoss log_vars.exp().sqrt().mean() value.
        """
        # clamp sigma_mean to the range ∈ [0.5, 3.0]
        sigma_signal = torch.tensor(sigma_mean, dtype=torch.float32)
        sigma_clamped = torch.clamp(sigma_signal, 0.5, 3.0)
        # tau_effective = tau_init * sigma_clamped
        tau_effective = self.tau_init * sigma_clamped.item()
        self.tau = tau_effective

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Fuse K embeddings (including precision bias).

        Args:
            embeddings: tensor list of length K.

        Returns:
            z_fused: (B, output_dim)
            alpha: (B, K) softmax weights (including precision bias)
        """
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")

        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]

        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha

        stacked = torch.stack(projected, dim=1)  # (B, K, D)
        B, K, D = stacked.shape

        # gate logits + precision bias (blow-up blocked via clamp)
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
