"""Cross-attention fusion — asymmetric primary injects attention into aux modalities.

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017 — multi-head attention.

Why (asymmetric primary + cross-attention):
    multi_film's symmetric (K primary) fusion was displaced by robust_attention/GR-MF(asymmetric)
    redesign adopted afterward. Molecular representation literature (MuMo arXiv:2510.23640, Asymmetric Multi-layer
    Fusion arXiv:2108.05009)'s winning formula = "privilege one stream (asymmetric), but
    inject cross-modal information via attention, not concat". robust_attention's
    expressiveness↑ vs simple weighted sum, cross-modal correlation capture↑ vs film's concat-FiLM.

Design:
    - ``embeddings[primary_idx]`` = primary (query), the rest = aux (key/value).
    - project primary to ``output_dim`` → x (query, length-1 token).
    - project each of the K-1 aux to ``output_dim`` → stack → memory (B, K-1, D).
    - ``nn.TransformerDecoderLayer`` × N (tgt=primary, memory=aux):
        self-attn(query) [length-1 → effectively no-op] + cross-attn(query→aux) + FFN.
        built-in residual always preserves primary(tgt) → structurally blocks washout (asymmetric
        's inherent advantage, no separate defense like film's γ-clamp needed).
    - (optional) output_proj (Linear D→D, identity-init).

Anti-collapse mechanisms:
    - modality dropout: mask aux tokens per-sample during training (guarantee at least 1 active).
    - presence mask: missing aux modality → blocked via memory_key_padding_mask.
    - warmup: freeze decoder/aux_proj for first N epochs → z = primary only.

Note:
    primary_idx follows lit_module's embeddings order (= [primary_backbone, *extra_backbones,
    *non_consumed_modalities]). Keeping backbone config fixed, changing only primary_idx
    enables primary ablation. alpha returns uniform (B, K) for format consistency (cross-attn
    weight is not directly exposed from nn.TransformerDecoder).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("cross_attention")
class CrossAttentionFusion(nn.Module):
    """Asymmetric primary-query cross-attention fusion.

    Args:
        input_dims: length K. Input dim per modality. K ≥ 2 required.
        output_dim: projection / attention dim (= head input). ``output_dim %
            num_heads == 0`` required.
        primary_idx: embeddings index of the modality to use as query (the rest are K/V).
        num_layers: number of TransformerDecoderLayer.
        num_heads: number of multi-head attention heads.
        ffn_mult: FFN hidden = output_dim * ffn_mult.
        dropout: decoder dropout.
        modality_dropout: sample-wise mask probability for each aux token during training
            (guarantee at least 1 active).
        pre_norm: per-modality input LayerNorm (absorbs modality scale asymmetry).
        warmup_epochs: freeze decoder/aux_proj for first N epochs + primary only.
            lit_module ``on_train_epoch_start`` toggles ``set_warmup_active``.
        use_output_proj: one final Linear(D, D, identity-init) stage. GradNorm
            shared_layer candidate (passed through by all task forwards, always trainable).
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int = 300,
        primary_idx: int = 0,
        num_layers: int = 1,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.1,
        pre_norm: bool = True,
        warmup_epochs: int = 0,
        use_output_proj: bool = True,
    ) -> None:
        super().__init__()
        self.K = len(input_dims)
        if self.K < 2:
            raise ValueError(
                f"CrossAttentionFusion requires primary + ≥1 aux (K>=2), K={self.K}."
            )
        if not 0 <= primary_idx < self.K:
            raise ValueError(f"primary_idx={primary_idx} is outside range [0, {self.K}).")
        if output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim({output_dim}) is not divisible by num_heads({num_heads}) "
                "evenly."
            )
        self.output_dim = output_dim
        self.primary_idx = primary_idx
        self.modality_dropout = modality_dropout
        self.warmup_epochs = warmup_epochs
        self._warmup_active = False
        self._presence_mask: Tensor | None = None

        # aux index (excluding primary).
        self._aux_idx: list[int] = [j for j in range(self.K) if j != primary_idx]

        if pre_norm:
            self.input_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.LayerNorm(d) for d in input_dims]
            )
        else:
            self.input_norms = None

        self.primary_proj = nn.Linear(input_dims[primary_idx], output_dim)
        self.aux_projs = nn.ModuleList(
            [nn.Linear(input_dims[j], output_dim) for j in self._aux_idx]
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.output_proj: nn.Linear | None
        if use_output_proj:
            self.output_proj = nn.Linear(output_dim, output_dim)
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = None

    def set_warmup_active(self, active: bool) -> None:
        """Warmup toggle. active=True → decoder/aux_proj freeze + primary only."""
        self._warmup_active = active
        for m in (self.decoder, self.aux_projs):
            for p in m.parameters():
                p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """Applies modality presence (B, K) bool for the next forward call only.

        Samples with missing aux modality block that token via memory_key_padding_mask.
        Missing primary is ignored (handled upstream via mask_token).
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]

        x = self.primary_proj(embeddings[self.primary_idx])  # (B, D)
        B, D = x.shape

        presence = self._presence_mask
        if presence is not None:
            presence = presence.to(device=x.device, dtype=torch.bool)
            self._presence_mask = None

        if self._warmup_active:
            z = x
            if self.output_proj is not None:
                z = self.output_proj(z)
            alpha = torch.full((B, self.K), 1.0 / self.K, device=z.device, dtype=z.dtype)
            return z, alpha

        aux = torch.stack(
            [ap(embeddings[j]) for ap, j in zip(self.aux_projs, self._aux_idx)],
            dim=1,
        )  # (B, K-1, D)
        n_aux = aux.size(1)

        # memory_key_padding_mask: True = ignore that aux token.
        kpm: Tensor | None = None
        if presence is not None:
            aux_presence = presence[:, self._aux_idx]  # (B, K-1)
            kpm = ~aux_presence
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(B, n_aux, device=aux.device) < self.modality_dropout
            all_drop = drop.all(dim=-1, keepdim=True)
            drop = drop & ~all_drop  # fully-dropped row → keep all
            kpm = drop if kpm is None else (kpm | drop)
        if kpm is not None:
            # guard against fully-padded row (all -inf → softmax NaN) → unmask all.
            all_pad = kpm.all(dim=-1, keepdim=True)
            kpm = kpm & ~all_pad

        attended = self.decoder(
            tgt=x.unsqueeze(1),  # (B, 1, D)
            memory=aux,  # (B, K-1, D)
            memory_key_padding_mask=kpm,
        )  # (B, 1, D)
        z = attended.squeeze(1)

        if self.output_proj is not None:
            z = self.output_proj(z)

        alpha = torch.full((B, self.K), 1.0 / self.K, device=z.device, dtype=z.dtype)
        return z, alpha
