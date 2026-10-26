"""RobustAttentionFusion — anti-collapse modality fusion (tau annealing + dropout + residual).

Design:
    1. Map K modalities to the shared projection space.
    2. Learn softmax alpha via gate network.
    3. modality dropout (training only): guarantee at least 1 active.
    4. temperature annealing (tau): soften(sharp) softmax as training progresses → harden.
    5. residual: weighted sum + mean(projected) residual → prevent information washout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("robust_attention")
class RobustAttentionFusion(nn.Module):
    """Anti-collapse modality fusion with temperature annealing, dropout, residual.

    Args:
        input_dims: embedding dimension per modality (length K).
        output_dim: common projection dimension.
        tau_init: initial softmax temperature (higher = more uniform distribution).
        tau_min: annealing lower bound.
        anneal_epochs: number of epochs for tau to reach tau_min.
        modality_dropout: per-modality zero-out probability during training (guarantees at least 1 active).
        residual: whether to add mean(projected) residual to the weighted sum.
        warmup_epochs: warmup period (number of epochs). If > 0, during the first N epochs, gate
            is frozen + alpha=1/K uniform hard-coded → backbones' pretrained features
            secure time to adapt to the task. lit_module's ``on_train_epoch_start``
            toggles via ``set_warmup_active``.
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
        pre_norm: bool = False,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.K = len(input_dims)
        self.output_dim = output_dim
        # self.tau_init = tau_init
        # self.tau_min = tau_min
        # self.anneal_epochs = max(1, anneal_epochs)
        self.modality_dropout = modality_dropout
        self.residual = residual
        self.pre_norm = pre_norm
        # self.warmup_epochs = warmup_epochs
        self.tau = tau_init  # current temperature
        self._warmup_active = False  # toggled from lit_module

        if self.K == 0:
            raise ValueError("RobustAttentionFusion requires at least 1 input.")

        # pre-LayerNorm: absorbs norm differences across modality inputs.
        # Recommended for scale-asymmetric cases like CheMeleon(2048) vs MoLFormer/UniMol2(768).
        if pre_norm:
            self.input_norms = nn.ModuleList([nn.LayerNorm(d) for d in input_dims])
        else:
            self.input_norms = None

        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])

        if self.K > 1:
            self.gate = nn.Sequential(
                nn.Linear(output_dim * self.K, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, self.K),
            )

    def set_tau(self, tau: float) -> None:
        """Set the temperature for the current epoch."""
        self.tau = tau

    def set_warmup_active(self, active: bool) -> None:
        """Toggle warmup mode. If active=True, gate is frozen and alpha=1/K is forced.

        Why: early in training, gate (lr=1e-3) overwhelmingly, compared to backbone (lr_lora=5e-5),
        adapts faster → alpha collapses toward the modality with strong pretrained features. During the warmup period
        if gate output is ignored, after backbones sufficiently adapt to the task, gate
        decides the fusion.
        """
        self._warmup_active = active
        if hasattr(self, "gate"):
            for p in self.gate.parameters():
                p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """per-sample modality presence mask applied only to the next single forward.

        Args:
            mask: ``(B, K)`` bool — if True, use that sample's corresponding modality; if False,
                block fusion alpha (softmax logit -inf). The caller (LitModule.encode)
                calls this right before forward. After forward ends, it automatically resets to None.
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Fuse K embeddings using learned alpha.

        Args:
            embeddings: list of K tensors, each (B, d_k).

        Returns:
            z_fused: (B, output_dim)
            alpha: (B, K) softmax weights
        """
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]
        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]

        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha

        # stack: (B, K, D)
        stacked = torch.stack(projected, dim=1)
        B, K, D = stacked.shape

        # warmup: bypass gate → alpha=1/K uniform (so only backbone is trained).
        if self._warmup_active:
            alpha = torch.full((B, K), 1.0 / K, device=stacked.device, dtype=stacked.dtype)
            z_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
            if self.residual:
                z_fused = z_fused + stacked.mean(dim=1)
            return z_fused, alpha

        # gate logits
        cat = torch.cat(projected, dim=-1)  # (B, K*D)
        logits = self.gate(cat)  # (B, K)

        # modality dropout: independently drop modality per sample during training
        if self.training and self.modality_dropout > 0:
            mask = torch.rand(B, K, device=logits.device) > self.modality_dropout
            # if all are dropped → activate all (guarantee at least 1 modality)
            any_active = mask.any(dim=-1, keepdim=True)
            mask = mask | ~any_active
            logits = logits.masked_fill(~mask, -1e9)

        # per-sample presence mask: missing modality gets alpha 0 (softmax logit -inf).
        # the case where all modalities of a sample are missing is pre-removed by collate, but
        # defensively, fully missing rows fall back to uniform distribution (to prevent NaN).
        presence = getattr(self, "_presence_mask", None)
        if presence is not None:
            presence = presence.to(device=logits.device)
            any_present = presence.any(dim=-1, keepdim=True)
            safe_presence = presence | ~any_present  # only rows that are all False become all True
            logits = logits.masked_fill(~safe_presence, -1e9)
            self._presence_mask = None  # reset until next forward

        # softmax with current temperature
        alpha = torch.softmax(logits / self.tau, dim=-1)  # (B, K)

        # weighted sum
        z_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)  # (B, D)

        # residual: add mean(projected) → prevent information washout
        if self.residual:
            z_fused = z_fused + stacked.mean(dim=1)

        return z_fused, alpha
