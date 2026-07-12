"""FiLM fusion — primary modality 를 보조 modality 들로 affine modulation.

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.

설계:
    - ``embeddings[0]`` = primary (CheMeleon 가정), ``embeddings[1:]`` = 보조 conditions.
    - primary 를 ``output_dim`` 으로 project → ``x``.
    - 보조 K-1 개를 각각 ``cond_hidden`` 으로 project → concat → MLP → ``(γ_pre, β)``.
    - ``γ = 1 + γ_pre`` (identity init: γ≈1, β≈0 → 학습 초기 primary 보존).
    - (optional) ``γ`` 를 ``[gamma_min, gamma_max]`` 로 clamp — residual 과 함께면
      x 의 effective 계수가 ``γ+1`` 이므로 lower bound 가 primary washout
      (``γ→-1`` 에서 계수 0) 을 구조적으로 차단. GradNorm ``w_k`` clamp 와 동일 철학.
    - ``z = γ ⊙ x + β`` (+ optional residual ``+x``).
    - (optional) ``z = output_proj(z)`` — fused z 위에 직접 작용하는 post-fusion
      Linear. GradNorm shared_layer 후보 (모든 22 task 의 forward 가 거치고
      fused multimodal signal 을 직접 보는 단일 W tensor). 항상 trainable
      (warmup 영향 없음). identity-init (weight=I, bias=0) 으로 활성화 시
      기존 FiLM forward 동작 보존.

Anti-collapse 장치:
    - condition dropout: training 중 각 보조 condition 을 sample-별 zero-out.
    - presence mask: 누락된 보조 modality (예: UniMol2 ETKDG 실패 sample) 의
      contribution 을 zero out.
    - warmup: 처음 N epoch 동안 film_gen freeze + (γ=1, β=0) → primary 만 단독 학습.
    - identity-init: 마지막 Linear weight=0 + bias=0 → 학습 시작점에서 modulation 무력.

Note:
    fusion 시그니처 통일을 위해 alpha 는 uniform ``(B, K)`` 로 반환 (FiLM 은 attention
    이 아니므로 진단 의미 약함). entropy_weight 는 0 으로 두는 게 자연스럽다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("film")
class FilmFusion(nn.Module):
    """Primary modality 를 보조 modality 들로 affine modulation 하는 fusion.

    Args:
        input_dims: 길이 K. ``input_dims[0]`` = primary, 나머지 = 보조 conditions.
            K ≥ 2 필수.
        output_dim: primary projection / modulation 차원 (head 입력 차원).
        cond_hidden: 각 보조 condition 의 projection 차원 (concat 전).
        condition_dropout: training 중 각 condition 의 sample-wise zero-out 확률.
        residual: ``z = γ⊙x + β + x`` 형태로 primary 잔차 추가 (information washout 방지).
        pre_norm: 입력 LayerNorm (각 modality scale 비대칭 흡수).
        film_init_identity: 마지막 Linear 를 0 으로 초기화 → γ≈1, β≈0 시작.
        warmup_epochs: 처음 N epoch 동안 film_gen freeze + identity modulation.
            lit_module ``on_train_epoch_start`` 가 ``set_warmup_active`` 토글.
        use_output_proj: ``True`` 면 forward 끝에 ``nn.Linear(output_dim, output_dim)``
            을 추가해 fused ``z`` 를 한 번 더 project. identity-init (weight=I,
            bias=0) 이라 활성화 시점의 forward 출력은 비활성화 때와 동일.
            GradNorm shared_layer 후보로 사용 가능 — fused signal 위에 직접
            작용하는 단일 W (모든 22 task 의 forward 가 거치고 항상 trainable).
        gamma_min: 설정 시 ``γ`` 의 lower bound. residual 과 함께면 x 계수가
            ``γ+1`` 이라 ``gamma_min > -1`` 이면 primary washout 차단. 권장 0.3.
        gamma_max: 설정 시 ``γ`` 의 upper bound (modulation 폭주 억제). 권장 3.0.
            ``gamma_min``/``gamma_max`` 둘 다 ``None`` (기본) 이면 clamp 미적용 —
            기존 recipe 동작 보존, 사용 시 recipe 에서 opt-in.
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int = 768,
        cond_hidden: int = 256,
        condition_dropout: float = 0.1,
        residual: bool = True,
        pre_norm: bool = True,
        film_init_identity: bool = True,
        warmup_epochs: int = 0,
        use_output_proj: bool = False,
        gamma_min: float | None = None,
        gamma_max: float | None = None,
    ) -> None:
        super().__init__()
        self.K = len(input_dims)
        if self.K < 2:
            raise ValueError(
                "FilmFusion 은 primary + ≥1 보조 modality 가 필요합니다 "
                f"(K={self.K}). 단일 modality 라면 fusion 자체를 비우세요."
            )
        self.output_dim = output_dim
        self.condition_dropout = condition_dropout
        self.residual = residual
        self.warmup_epochs = warmup_epochs
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self._warmup_active = False
        self._presence_mask: Tensor | None = None

        if pre_norm:
            self.input_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.LayerNorm(d) for d in input_dims]
            )
        else:
            self.input_norms = None

        # primary projection: d_primary → output_dim
        self.primary_proj = nn.Linear(input_dims[0], output_dim)

        # condition projections: 각 d_aux_i → cond_hidden
        self.cond_projs = nn.ModuleList(
            [nn.Linear(d, cond_hidden) for d in input_dims[1:]]
        )

        # FiLM generator: concat(K-1 conds) → (γ_pre, β) ∈ R^{2·output_dim}
        n_cond = self.K - 1
        self.film_gen = nn.Sequential(
            nn.Linear(cond_hidden * n_cond, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2 * output_dim),
        )
        if film_init_identity:
            last = self.film_gen[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)  # γ_pre=0 → γ=1, β=0

        # Optional post-fusion projection. fused z 위에 직접 작용하는 단일 Linear
        # — GradNorm shared_layer 후보 (모든 task forward 통과, 항상 trainable).
        # identity-init 로 forward 출력은 비활성화 케이스와 동일하게 시작.
        self.output_proj: nn.Linear | None
        if use_output_proj:
            self.output_proj = nn.Linear(output_dim, output_dim)
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = None

    def set_warmup_active(self, active: bool) -> None:
        """Warmup 모드 토글. active=True → film_gen freeze + (γ=1, β=0) hard-code."""
        self._warmup_active = active
        for p in self.film_gen.parameters():
            p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """다음 forward 한 번에만 적용되는 per-sample modality presence mask (B, K).

        ``mask[:, 0]`` (primary) 은 무시 — primary 는 항상 존재.
        ``mask[:, 1:]`` 가 False 인 sample 은 그 보조 condition 의 projected 값을 0 으로.
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]

        x = self.primary_proj(embeddings[0])  # (B, output_dim)
        B, D = x.shape

        c_projected: list[Tensor] = [
            cp(e) for cp, e in zip(self.cond_projs, embeddings[1:])
        ]

        presence = self._presence_mask
        if presence is not None:
            presence = presence.to(device=x.device)
            for i in range(len(c_projected)):
                # presence[:, 0] = primary, presence[:, i+1] = i 번째 보조
                p = presence[:, i + 1].to(dtype=c_projected[i].dtype).unsqueeze(-1)
                c_projected[i] = c_projected[i] * p
            self._presence_mask = None  # 다음 forward 까지 reset

        if self.training and self.condition_dropout > 0:
            for i in range(len(c_projected)):
                drop = (
                    torch.rand(B, 1, device=x.device) < self.condition_dropout
                )
                c_projected[i] = c_projected[i].masked_fill(drop, 0.0)

        if self._warmup_active:
            z = x + x if self.residual else x
            if self.output_proj is not None:
                z = self.output_proj(z)
            alpha = torch.full(
                (B, self.K), 1.0 / self.K, device=x.device, dtype=x.dtype
            )
            return z, alpha

        cond_cat = torch.cat(c_projected, dim=-1)  # (B, cond_hidden * (K-1))
        gamma_beta = self.film_gen(cond_cat)  # (B, 2*output_dim)
        gamma_pre, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + gamma_pre
        if self.gamma_min is not None or self.gamma_max is not None:
            gamma = gamma.clamp(min=self.gamma_min, max=self.gamma_max)

        z = gamma * x + beta
        if self.residual:
            z = z + x

        if self.output_proj is not None:
            z = self.output_proj(z)

        # alpha sidecar (진단용 uniform — FiLM 은 attention 아님).
        alpha = torch.full((B, self.K), 1.0 / self.K, device=x.device, dtype=x.dtype)
        return z, alpha
