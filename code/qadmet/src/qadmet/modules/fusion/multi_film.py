"""Multi-primary FiLM + cross-attention late fusion (symmetric FiLM).

Why:
    단일-primary ``film`` (CheMeleon main + 보조 concat → γ/β) 은 primary 선택이
    결과를 좌우하고 보조 modality 가 1급 시민이 아니다. ``multi_film`` 은 K 개
    modality 를 *각각 한 번씩 primary* 로 두고 (자기 self-anchor residual + 나머지
    modality 들로 affine modulation) K 개 branch 표현을 만든 뒤, modality-as-token
    cross-attention 으로 **늦게(late)** 한 번 섞는다.

    ⚠ 완전 대칭(모든 layer 상호 조건화) 융합은 분자표현 문헌에서 modality collapse·
    불안정 최적화로 알려져 있다 (MuMo, arXiv:2510.23640 등은 의도적으로 asymmetric
    선택). 본 모듈은 그 위험을 줄이기 위해 "약한 대칭"으로 설계:
      - 각 branch 가 self-residual(``+x_i``) 로 자기 primary 를 앵커 (washout 방지).
      - 상호작용은 마지막 cross-attention 한 군데로 late fusion (layer 마다 상호
        조건화하지 않음).
      - identity-init + γ clamp + condition dropout + branch dropout 으로
        변조 폭주·branch collapse 차단.

설계:
    1. branch i (각 modality 가 primary):
         x_i = primary_proj_i(norm_i(e_i))                         ∈ R^D
         각 conditioner j≠i 에 대해 (γ_ij, β_ij) = film_gen_ij(cond_proj_ij(e_j))
         γ_ij = clamp(1 + γ_pre_ij, [gamma_min, gamma_max])
         delta_i = mean_{j≠i}(γ_ij ⊙ x_i + β_ij)        ← 병렬 합산(평균: scale 안정)
         z_i = x_i + delta_i                              ← self-anchor residual
    2. stack(z_0..z_{K-1}) → (B, K, D)
    3. cross-attention mixer: TransformerEncoderLayer × N (norm_first/gelu,
       modality-as-token self-attn). GR-MF(group_routed) 와 동일 패턴.
       presence/branch-dropout 는 ``src_key_padding_mask`` 로 차단.
    4. masked mean pool over branch tokens → (B, D).
    5. (optional) output_proj (Linear D→D, identity-init).

Anti-collapse 장치:
    - condition dropout: training 중 각 conditioner 를 sample-별 zero-out.
    - branch dropout: training 중 mixer 입력 branch token 을 sample-별 mask
      (최소 1개 active 보장) — 한 branch 로의 attention collapse 방어.
    - presence mask: 누락 modality → 그 conditioner contribution 0 + 그 modality 의
      branch token mask.
    - warmup: 처음 N epoch film_gen/cond_proj freeze + modulation identity →
      mixer 가 raw primary 들만 섞음 (backbone 적응 시간 확보).
    - identity-init: film_gen 마지막 Linear=0 → γ≈1, β≈0 시작.

Note:
    forward 시그니처 통일을 위해 alpha 는 uniform ``(B, K)`` 반환 (cross-attention
    attn weight 는 nn.TransformerEncoder 에서 직접 노출되지 않음 — 진단 의미 약함).
    entropy_weight 는 0 으로 두는 게 자연스럽다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("multi_film")
class MultiFilmFusion(nn.Module):
    """Multi-primary FiLM branches + cross-attention late fusion.

    Args:
        input_dims: 길이 K. 각 modality 의 입력 차원. K 개 modality 가 각각 한 번씩
            primary 가 되어 K 개 branch 를 만든다. K ≥ 2 필수.
        output_dim: branch projection / modulation 차원 (= head 입력 차원).
        cond_hidden: 각 conditioner projection 차원 (film_gen 입력).
        condition_dropout: training 중 각 conditioner 의 sample-wise zero-out 확률.
        film_init_identity: film_gen 마지막 Linear 0-init → γ≈1, β≈0 시작.
        pre_norm: per-modality 입력 LayerNorm (modality scale 비대칭 흡수).
        gamma_min / gamma_max: γ clamp 범위. residual 과 함께면 x 계수 하한 보장 →
            primary washout 차단. 권장 0.3 / 3.0. 둘 다 None 이면 clamp 미적용.
        warmup_epochs: 처음 N epoch film_gen/cond_proj freeze + modulation identity.
            lit_module ``on_train_epoch_start`` 가 ``set_warmup_active`` 토글.
        cross_modal_layers: mixer TransformerEncoderLayer 수. 0 이면 mixer 없이
            branch 평균 pool.
        cross_modal_heads: mixer nhead. ``output_dim % heads == 0`` 필요.
        cross_modal_ffn_mult: mixer FFN hidden = output_dim * mult.
        cross_modal_dropout: mixer dropout.
        branch_dropout: training 중 mixer 입력 branch token 의 sample-wise mask 확률
            (최소 1개 active 보장). branch collapse 방어.
        use_output_proj: 마지막 Linear(D, D, identity-init) 1 단. GradNorm
            shared_layer 후보 (모든 task forward 통과, 항상 trainable).
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int = 300,
        cond_hidden: int = 256,
        condition_dropout: float = 0.2,
        film_init_identity: bool = True,
        pre_norm: bool = True,
        gamma_min: float | None = None,
        gamma_max: float | None = None,
        warmup_epochs: int = 0,
        cross_modal_layers: int = 1,
        cross_modal_heads: int = 4,
        cross_modal_ffn_mult: int = 4,
        cross_modal_dropout: float = 0.1,
        branch_dropout: float = 0.1,
        use_output_proj: bool = True,
    ) -> None:
        super().__init__()
        self.K = len(input_dims)
        if self.K < 2:
            raise ValueError(
                f"MultiFilmFusion 은 K>=2 modality 필요 (각 modality 가 primary 1회), "
                f"K={self.K}."
            )
        if cross_modal_layers > 0 and output_dim % cross_modal_heads != 0:
            raise ValueError(
                f"output_dim({output_dim}) 가 cross_modal_heads({cross_modal_heads}) "
                "로 나눠떨어지지 않음."
            )
        self.output_dim = output_dim
        self.condition_dropout = condition_dropout
        self.branch_dropout = branch_dropout
        self.warmup_epochs = warmup_epochs
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self._warmup_active = False
        self._presence_mask: Tensor | None = None

        # 각 branch i 의 conditioner index (자기 제외).
        self._cond_idx: list[list[int]] = [
            [j for j in range(self.K) if j != i] for i in range(self.K)
        ]

        if pre_norm:
            self.input_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.LayerNorm(d) for d in input_dims]
            )
        else:
            self.input_norms = None

        # branch i 의 primary projection: d_i → output_dim
        self.primary_projs = nn.ModuleList(
            [nn.Linear(input_dims[i], output_dim) for i in range(self.K)]
        )

        # branch i 의 conditioner j 별 (cond_proj, film_gen).
        # 중첩 ModuleList: cond_projs[i][local_j], film_gens[i][local_j].
        self.cond_projs = nn.ModuleList()
        self.film_gens = nn.ModuleList()
        for i in range(self.K):
            cps = nn.ModuleList(
                [nn.Linear(input_dims[j], cond_hidden) for j in self._cond_idx[i]]
            )
            fgs = nn.ModuleList()
            for _ in self._cond_idx[i]:
                gen = nn.Sequential(
                    nn.Linear(cond_hidden, output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, 2 * output_dim),
                )
                if film_init_identity:
                    nn.init.zeros_(gen[-1].weight)
                    nn.init.zeros_(gen[-1].bias)  # γ_pre=0 → γ=1, β=0
                fgs.append(gen)
            self.cond_projs.append(cps)
            self.film_gens.append(fgs)

        # cross-attention mixer (modality-as-token). GR-MF 와 동일 패턴.
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

        self.output_proj: nn.Linear | None
        if use_output_proj:
            self.output_proj = nn.Linear(output_dim, output_dim)
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = None

    def set_warmup_active(self, active: bool) -> None:
        """Warmup 토글. active=True → 모든 film_gen/cond_proj freeze + modulation identity."""
        self._warmup_active = active
        for mlist in (self.film_gens, self.cond_projs):
            for branch in mlist:
                for m in branch:
                    for p in m.parameters():
                        p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """다음 forward 1회에 한해 modality presence (B, K) bool 적용.

        ``mask[:, i]`` False → modality i 가 그 sample 에서 누락. branch i 의
        primary 로서의 token 은 mixer 에서 mask 되고, 다른 branch 의 conditioner
        로서의 contribution 은 0 이 된다.
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]

        x = [self.primary_projs[i](embeddings[i]) for i in range(self.K)]  # 각 (B, D)
        B, D = x[0].shape

        presence = self._presence_mask
        if presence is not None:
            presence = presence.to(device=x[0].device, dtype=torch.bool)
            # 전부 missing 인 row → 모두 present 로 fallback (NaN 방지).
            any_present = presence.any(dim=-1, keepdim=True)
            presence = presence | ~any_present
            self._presence_mask = None

        branch_out: list[Tensor] = []
        for i in range(self.K):
            xi = x[i]
            if self._warmup_active:
                branch_out.append(xi + xi)  # modulation identity + residual (film.py 일치)
                continue
            deltas: list[Tensor] = []
            for local_j, j in enumerate(self._cond_idx[i]):
                c = self.cond_projs[i][local_j](embeddings[j])  # (B, cond_hidden)
                if presence is not None:
                    c = c * presence[:, j].unsqueeze(-1).to(c.dtype)
                if self.training and self.condition_dropout > 0:
                    drop = torch.rand(B, 1, device=c.device) < self.condition_dropout
                    c = c.masked_fill(drop, 0.0)
                gamma_beta = self.film_gens[i][local_j](c)  # (B, 2D)
                gamma_pre, beta = gamma_beta.chunk(2, dim=-1)
                gamma = 1.0 + gamma_pre
                if self.gamma_min is not None or self.gamma_max is not None:
                    gamma = gamma.clamp(min=self.gamma_min, max=self.gamma_max)
                deltas.append(gamma * xi + beta)
            delta = torch.stack(deltas, dim=0).mean(dim=0)  # 병렬 합산(평균)
            branch_out.append(xi + delta)  # self-anchor residual

        stacked = torch.stack(branch_out, dim=1)  # (B, K, D)

        # mixer key_padding_mask: True = 그 branch token 무시.
        kpm: Tensor | None = None
        if presence is not None:
            kpm = ~presence  # (B, K)
        if self.training and self.branch_dropout > 0:
            keep = torch.rand(B, self.K, device=stacked.device) > self.branch_dropout
            any_keep = keep.any(dim=-1, keepdim=True)
            keep = keep | ~any_keep  # 전부 drop 된 row → 전부 keep
            drop = ~keep
            kpm = drop if kpm is None else (kpm | drop)
        if kpm is not None:
            # 전부 pad 된 row 방어 (presence + dropout 동시) → 전부 unmask.
            all_pad = kpm.all(dim=-1, keepdim=True)
            kpm = kpm & ~all_pad

        if self.cross_modal is not None:
            fused = self.cross_modal(stacked, src_key_padding_mask=kpm)  # (B, K, D)
        else:
            fused = stacked

        # masked mean pool over branch tokens.
        if kpm is not None:
            valid = (~kpm).unsqueeze(-1).to(fused.dtype)  # (B, K, 1)
            z = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            z = fused.mean(dim=1)

        if self.output_proj is not None:
            z = self.output_proj(z)

        # alpha sidecar (진단용 uniform — cross-attn weight 미노출).
        alpha = torch.full((B, self.K), 1.0 / self.K, device=z.device, dtype=z.dtype)
        return z, alpha
