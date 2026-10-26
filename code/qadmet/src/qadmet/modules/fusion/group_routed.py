"""Group-Routed Modality Fusion (GR-MF) — per-group static gate over modality projections.

Why:
    robust_attention's per-sample dynamic gate depends on norm/init at early training, so modality
    collapse occurs (weak modality survives, strong one dies). GR-MF fixes the gate
    as a *per-group static weight*, structurally removing the RGR (rich-get-richer) dynamic
    and carries stacking's per-task modality routing insight directly into the fusion stage
    .

Design:
    1. per-modality LayerNorm + Linear (d_m → output_dim) → z'_m  (B, D)
    2. [optional] cross-modal enrichment: stack → (B, K, D)
       → TransformerEncoderLayer × cross_modal_layers (modality-as-token self-attn)
       → z''_m. If presence_mask exists, block missing modality via key_padding_mask.
    3. per-group static gate W ∈ R^{n_groups × K} (learnable, gate_init_std Gaussian).
       If presence exists, masked_fill missing modality logit with -inf then softmax.
       α ∈ (B, n_groups, K).
    4. weighted sum: z_g[b] = Σ_m α[b,g,m] · z''[b,m,:]  ∈ R^D.
    5. shared output_proj (Linear D→D, identity-init) — GradNorm shared_layer candidate.

Returns:
    z: (B, n_groups, output_dim).  ← Used by GroupedRoutedHead, sliced per group.
    alpha: (B, K)  ← group average (for lit_module logging compatibility). Diagnostics via ``last_group_alpha``.

Note:
    fusion.output_dim means the *per-group* dim (D). head takes (B, n_groups, D) and
    slices it directly by group_idx.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("group_routed")
class GroupRoutedFusion(nn.Module):
    """Per-group static-gate fusion over K modality projections.

    Args:
        input_dims: length K. Input dimension of each modality.
        groups: group name order from causal grouping. The fusion output's group axis follows this order.
            Must share the exact same order as head (injected by lit_module).
        output_dim: dimension D of the per-group fused vector (head's input_dim).
        cross_modal_layers: number of modality-as-token transformer layers.
            If 0, pure projection + gate with no enrichment.
        cross_modal_heads: nhead of TransformerEncoderLayer. Requires output_dim % heads == 0.
        cross_modal_ffn_mult: FFN hidden = output_dim * mult.
        cross_modal_dropout: TransformerEncoderLayer dropout.
        gate_temperature: τ in softmax(W / τ).
        gate_init_std: N(0, std) initialization of group_gate (n_groups, K). Keeping it small yields initial
            α≈1/K uniform. If an STL OOF affinity prior exists, overwritten via ``init_gate_prior``.
        pre_norm: whether to apply per-modality LayerNorm.
        use_output_proj: adds one Linear(D, D, identity-init) stage at the end. GradNorm
            shared_layer candidate (all tasks' forward pass through it, always trainable).
    """

    def __init__(
        self,
        input_dims: list[int],
        groups: list[str],
        output_dim: int = 256,
        cross_modal_layers: int = 1,
        cross_modal_heads: int = 4,
        cross_modal_ffn_mult: int = 4,
        cross_modal_dropout: float = 0.1,
        gate_temperature: float = 1.0,
        gate_init_std: float = 0.02,
        pre_norm: bool = True,
        use_output_proj: bool = True,
    ) -> None:
        super().__init__()
        self.K = len(input_dims)
        if self.K < 2:
            raise ValueError(
                f"GroupRoutedFusion requires K>=2 modality, K={self.K}"
            )
        self.groups = list(groups)
        self.n_groups = len(self.groups)
        if self.n_groups < 1:
            raise ValueError("GroupRoutedFusion requires at least 1 group")
        if output_dim % cross_modal_heads != 0 and cross_modal_layers > 0:
            raise ValueError(
                f"output_dim({output_dim}) and cross_modal_heads({cross_modal_heads}) "
                "are not evenly divisible."
            )
        self.output_dim = output_dim
        self.gate_temperature = gate_temperature
        self._presence_mask: Tensor | None = None
        self._last_group_alpha: Tensor | None = None

        if pre_norm:
            self.input_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.LayerNorm(d) for d in input_dims]
            )
        else:
            self.input_norms = None

        self.modality_projs = nn.ModuleList(
            [nn.Linear(d, output_dim) for d in input_dims]
        )

        if cross_modal_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=cross_modal_heads,
                dim_feedforward=output_dim * cross_modal_ffn_mult,
                dropout=cross_modal_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cross_modal: nn.Module | None = nn.TransformerEncoder(
                enc_layer, num_layers=cross_modal_layers
            )
        else:
            self.cross_modal = None

        # per-group static gate: small Gaussian → initial α≈1/K. Learns per-group modality preference through training.
        self.group_gate = nn.Parameter(
            torch.randn(self.n_groups, self.K) * gate_init_std
        )

        self.output_proj: nn.Linear | None
        if use_output_proj:
            self.output_proj = nn.Linear(output_dim, output_dim)
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = None

    def set_presence_mask(self, mask: Tensor) -> None:
        """Apply modality presence (B, K) bool for the next forward call only."""
        self._presence_mask = mask

    def init_gate_prior(self, prior: Tensor) -> None:
        """Overwrites group_gate with a prior such as STL OOF affinity.

        Args:
            prior: (n_groups, K) logit before softmax. Usually log(affinity) or
                z-scored corr block-mean, etc.
        """
        if prior.shape != self.group_gate.shape:
            raise ValueError(
                f"prior shape {tuple(prior.shape)} != "
                f"group_gate shape {tuple(self.group_gate.shape)}"
            )
        with torch.no_grad():
            self.group_gate.copy_(prior.to(device=self.group_gate.device))

    @property
    def last_group_alpha(self) -> Tensor | None:
        """Gate distribution (B, n_groups, K) from the previous forward. For diagnostics/logging."""
        return self._last_group_alpha

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(
                f"K={self.K} embeddings required, received: {len(embeddings)}"
            )

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]

        projected = [proj(e) for proj, e in zip(self.modality_projs, embeddings)]
        z = torch.stack(projected, dim=1)  # (B, K, D)
        B, K, D = z.shape

        presence = self._presence_mask
        if presence is not None:
            presence = presence.to(device=z.device, dtype=torch.bool)
            # Row where all are missing → fallback to all present (prevents NaN).
            any_present = presence.any(dim=-1, keepdim=True)
            presence = presence | ~any_present
            self._presence_mask = None

        if self.cross_modal is not None:
            kpm = ~presence if presence is not None else None
            z = self.cross_modal(z, src_key_padding_mask=kpm)

        # per-group gate logits, block missing modality if presence exists.
        logits = self.group_gate.unsqueeze(0).expand(B, -1, -1)  # (B, n_groups, K)
        if presence is not None:
            mask_broadcast = presence.unsqueeze(1).expand(B, self.n_groups, K)
            logits = logits.masked_fill(~mask_broadcast, float("-inf"))
        alpha = torch.softmax(logits / self.gate_temperature, dim=-1)

        # weighted sum across modalities per group.
        z_group = torch.einsum("bgk,bkd->bgd", alpha, z)

        if self.output_proj is not None:
            z_group = self.output_proj(z_group)

        # During train, keep the live tensor as-is (KL-to-uniform reg must backward through it).
        # Eval/predict has no graph, so the same tensor is naturally no-grad.
        self._last_group_alpha = alpha

        # Compatible with lit_module's alpha logging: (B, K) — n_groups average.
        alpha_compat = alpha.mean(dim=1)
        return z_group, alpha_compat
