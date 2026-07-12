"""Cross-attention fusion — asymmetric primary 를 aux modality 들로 attention 주입.

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017 — multi-head attention.

Why (asymmetric primary + cross-attention):
    multi_film 의 대칭(K개 primary) 융합이 robust_attention/GR-MF(asymmetric)에 밀린
    뒤 채택한 재설계. 분자표현 문헌(MuMo arXiv:2510.23640, Asymmetric Multi-layer
    Fusion arXiv:2108.05009)의 승리 공식 = "한 stream 을 특권화(asymmetric)하되,
    concat 이 아니라 attention 으로 cross-modal 정보를 주입". robust_attention 의
    단순 가중합보다 표현력↑, film 의 concat-FiLM 보다 cross-modal 연관 포착↑.

설계:
    - ``embeddings[primary_idx]`` = primary (query), 나머지 = aux (key/value).
    - primary 를 ``output_dim`` 으로 project → x (query, length-1 token).
    - aux K-1 개를 각각 ``output_dim`` 으로 project → stack → memory (B, K-1, D).
    - ``nn.TransformerDecoderLayer`` × N (tgt=primary, memory=aux):
        self-attn(query) [length-1 → 사실상 no-op] + cross-attn(query→aux) + FFN.
        내장 residual 이 primary(tgt)를 항상 보존 → washout 구조적 차단 (asymmetric
        의 본질적 장점, film 의 γ-clamp 같은 별도 방어 불필요).
    - (optional) output_proj (Linear D→D, identity-init).

Anti-collapse 장치:
    - modality dropout: training 중 aux token 을 sample-별 mask (최소 1개 active 보장).
    - presence mask: 누락 aux modality → memory_key_padding_mask 로 차단.
    - warmup: 처음 N epoch decoder/aux_proj freeze → z = primary 단독.

Note:
    primary_idx 는 lit_module 의 embeddings 순서 (= [primary_backbone, *extra_backbones,
    *non_consumed_modalities]) 기준. backbone 구성을 고정한 채 primary_idx 만 바꿔
    primary ablation 가능. alpha 는 형식 통일용 uniform (B, K) 반환 (cross-attn
    weight 는 nn.TransformerDecoder 에서 직접 노출 안 됨).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("cross_attention")
class CrossAttentionFusion(nn.Module):
    """Asymmetric primary-query cross-attention fusion.

    Args:
        input_dims: 길이 K. 각 modality 입력 차원. K ≥ 2 필수.
        output_dim: projection / attention 차원 (= head 입력). ``output_dim %
            num_heads == 0`` 필요.
        primary_idx: query 로 쓸 modality 의 embeddings 인덱스 (나머지는 K/V).
        num_layers: TransformerDecoderLayer 수.
        num_heads: multi-head attention head 수.
        ffn_mult: FFN hidden = output_dim * ffn_mult.
        dropout: decoder dropout.
        modality_dropout: training 중 각 aux token 의 sample-wise mask 확률
            (최소 1개 active 보장).
        pre_norm: per-modality 입력 LayerNorm (modality scale 비대칭 흡수).
        warmup_epochs: 처음 N epoch decoder/aux_proj freeze + primary 단독.
            lit_module ``on_train_epoch_start`` 가 ``set_warmup_active`` 토글.
        use_output_proj: 마지막 Linear(D, D, identity-init) 1 단. GradNorm
            shared_layer 후보 (모든 task forward 통과, 항상 trainable).
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
                f"CrossAttentionFusion 은 primary + ≥1 aux 필요 (K>=2), K={self.K}."
            )
        if not 0 <= primary_idx < self.K:
            raise ValueError(f"primary_idx={primary_idx} 가 [0, {self.K}) 범위 밖.")
        if output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim({output_dim}) 가 num_heads({num_heads}) 로 "
                "나눠떨어지지 않음."
            )
        self.output_dim = output_dim
        self.primary_idx = primary_idx
        self.modality_dropout = modality_dropout
        self.warmup_epochs = warmup_epochs
        self._warmup_active = False
        self._presence_mask: Tensor | None = None

        # aux index (primary 제외).
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
        """Warmup 토글. active=True → decoder/aux_proj freeze + primary 단독."""
        self._warmup_active = active
        for m in (self.decoder, self.aux_projs):
            for p in m.parameters():
                p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """다음 forward 1회에 한해 modality presence (B, K) bool 적용.

        aux modality 가 누락된 sample 은 그 token 을 memory_key_padding_mask 로 차단.
        primary 누락은 무시 (mask_token 으로 upstream 처리).
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")

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

        # memory_key_padding_mask: True = 그 aux token 무시.
        kpm: Tensor | None = None
        if presence is not None:
            aux_presence = presence[:, self._aux_idx]  # (B, K-1)
            kpm = ~aux_presence
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(B, n_aux, device=aux.device) < self.modality_dropout
            all_drop = drop.all(dim=-1, keepdim=True)
            drop = drop & ~all_drop  # 전부 drop 된 row → 전부 keep
            kpm = drop if kpm is None else (kpm | drop)
        if kpm is not None:
            # 전부 pad 된 row 방어 (all -inf → softmax NaN) → 전부 unmask.
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
