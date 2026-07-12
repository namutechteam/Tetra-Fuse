"""Group-Routed Modality Fusion (GR-MF) — per-group static gate over modality projections.

Why:
    robust_attention 의 per-sample 동적 gate 는 학습 초기 norm/init 에 따라 modality
    collapse 가 일어남 (weak modality 가 살고 strong 이 죽음). GR-MF 는 gate 를
    *per-group 정적 weight* 로 두어 RGR (rich-get-richer) 다이내믹을 구조적으로
    제거하고, stacking 의 per-task modality routing 인사이트를 fusion 단에 직접
    옮긴다.

설계:
    1. per-modality LayerNorm + Linear (d_m → output_dim) → z'_m  (B, D)
    2. [optional] cross-modal enrichment: stack → (B, K, D)
       → TransformerEncoderLayer × cross_modal_layers (modality-as-token self-attn)
       → z''_m. presence_mask 가 있으면 key_padding_mask 로 missing modality 차단.
    3. per-group static gate W ∈ R^{n_groups × K} (learnable, gate_init_std 가우시안).
       presence 가 있으면 missing modality logit 을 -inf 로 masked_fill 후 softmax.
       α ∈ (B, n_groups, K).
    4. weighted sum: z_g[b] = Σ_m α[b,g,m] · z''[b,m,:]  ∈ R^D.
    5. shared output_proj (Linear D→D, identity-init) — GradNorm shared_layer 후보.

Returns:
    z: (B, n_groups, output_dim).  ← GroupedRoutedHead 에서 group 별 slice 해서 사용.
    alpha: (B, K)  ← group 평균 (lit_module 의 logging 호환). 진단은 ``last_group_alpha``.

Note:
    fusion.output_dim 은 *group 단위* dim (D) 을 의미. head 는 (B, n_groups, D) 를
    그대로 받아 group_idx 로 slice.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("group_routed")
class GroupRoutedFusion(nn.Module):
    """Per-group static-gate fusion over K modality projections.

    Args:
        input_dims: 길이 K. 각 modality 의 입력 차원.
        groups: causal grouping 의 group name 순서. fusion 출력의 group axis 가 이 순서.
            반드시 head 와 동일한 순서를 공유해야 한다 (lit_module 이 주입).
        output_dim: per-group fused vector 의 차원 D (head 의 input_dim).
        cross_modal_layers: modality-as-token transformer layer 수.
            0 이면 enrichment 없이 순수 projection + gate.
        cross_modal_heads: TransformerEncoderLayer 의 nhead. output_dim % heads == 0 필요.
        cross_modal_ffn_mult: FFN hidden = output_dim * mult.
        cross_modal_dropout: TransformerEncoderLayer dropout.
        gate_temperature: softmax(W / τ) 의 τ.
        gate_init_std: group_gate (n_groups, K) 의 N(0, std) 초기화. 작게 두면 초기
            α≈1/K 균등. STL OOF affinity prior 가 있으면 ``init_gate_prior`` 로 덮어쓴다.
        pre_norm: per-modality LayerNorm 적용 여부.
        use_output_proj: 마지막에 Linear(D, D, identity-init) 1 단 추가. GradNorm
            shared_layer 후보 (모든 task 의 forward 가 통과, 항상 trainable).
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
                f"GroupRoutedFusion 은 K>=2 modality 필요, K={self.K}"
            )
        self.groups = list(groups)
        self.n_groups = len(self.groups)
        if self.n_groups < 1:
            raise ValueError("GroupRoutedFusion 은 최소 1 group 필요")
        if output_dim % cross_modal_heads != 0 and cross_modal_layers > 0:
            raise ValueError(
                f"output_dim({output_dim}) 가 cross_modal_heads({cross_modal_heads}) "
                "로 나눠떨어지지 않음."
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

        # per-group static gate: 작은 가우시안 → 초기 α≈1/K. 학습으로 group 별 modality 선호 발견.
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
        """다음 forward 1회에 한해 modality presence (B, K) bool 적용."""
        self._presence_mask = mask

    def init_gate_prior(self, prior: Tensor) -> None:
        """STL OOF affinity 등의 prior 로 group_gate 를 덮어쓴다.

        Args:
            prior: (n_groups, K) softmax 이전 logit. 보통 log(affinity) 또는
                z-scored corr block-mean 등.
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
        """직전 forward 의 (B, n_groups, K) gate 분포. 진단/로깅용."""
        return self._last_group_alpha

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(
                f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}"
            )

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]

        projected = [proj(e) for proj, e in zip(self.modality_projs, embeddings)]
        z = torch.stack(projected, dim=1)  # (B, K, D)
        B, K, D = z.shape

        presence = self._presence_mask
        if presence is not None:
            presence = presence.to(device=z.device, dtype=torch.bool)
            # 전부 missing 인 row → 모두 present 로 fallback (NaN 방지).
            any_present = presence.any(dim=-1, keepdim=True)
            presence = presence | ~any_present
            self._presence_mask = None

        if self.cross_modal is not None:
            kpm = ~presence if presence is not None else None
            z = self.cross_modal(z, src_key_padding_mask=kpm)

        # per-group gate logits, presence 가 있으면 missing modality 차단.
        logits = self.group_gate.unsqueeze(0).expand(B, -1, -1)  # (B, n_groups, K)
        if presence is not None:
            mask_broadcast = presence.unsqueeze(1).expand(B, self.n_groups, K)
            logits = logits.masked_fill(~mask_broadcast, float("-inf"))
        alpha = torch.softmax(logits / self.gate_temperature, dim=-1)

        # weighted sum across modalities per group.
        z_group = torch.einsum("bgk,bkd->bgd", alpha, z)

        if self.output_proj is not None:
            z_group = self.output_proj(z_group)

        # Train 시에는 live tensor 그대로 (KL-to-uniform reg 가 backward 해야 함).
        # Eval/predict 는 graph 가 없으므로 같은 tensor 가 자연스럽게 no-grad.
        self._last_group_alpha = alpha

        # lit_module 의 alpha logging 호환: (B, K) — n_groups 평균.
        alpha_compat = alpha.mean(dim=1)
        return z_group, alpha_compat
