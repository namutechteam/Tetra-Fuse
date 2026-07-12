"""RobustAttentionFusion — anti-collapse modality fusion (tau annealing + dropout + residual).

설계:
    1. K 개 모달리티를 shared projection space 로 매핑.
    2. gate 네트워크로 softmax alpha 학습.
    3. modality dropout (training only): 최소 1개 active 보장.
    4. temperature annealing (tau): 학습 진행에 따라 softmax 연화(sharp) → 경화.
    5. residual: weighted sum + mean(projected) 잔차 → information washout 방지.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("robust_attention")
class RobustAttentionFusion(nn.Module):
    """Anti-collapse modality fusion with temperature annealing, dropout, residual.

    Args:
        input_dims: 각 모달리티 embedding 차원 (길이 K).
        output_dim: 공통 projection 차원.
        tau_init: 초기 softmax temperature (높을수록 균등 분포).
        tau_min: annealing 하한.
        anneal_epochs: tau 가 tau_min 에 도달하는 epoch 수.
        modality_dropout: training 중 각 modality zero-out 확률 (최소 1개 active 보장).
        residual: weighted sum 에 mean(projected) 잔차 추가 여부.
        warmup_epochs: warmup 기간 (epoch 수). > 0 이면 처음 N epoch 동안 gate 를
            freeze + alpha=1/K 균등 hard-coded → backbone 들이 사전학습 feature 를
            task 에 맞춰 적응할 시간 확보. lit_module 의 ``on_train_epoch_start``
            가 ``set_warmup_active`` 로 토글한다.
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
        self.tau = tau_init  # 현재 temperature
        self._warmup_active = False  # lit_module 에서 토글

        if self.K == 0:
            raise ValueError("RobustAttentionFusion 은 최소 1 개 input 이 필요합니다.")

        # pre-LayerNorm: 각 modality 입력의 norm 차이 흡수.
        # CheMeleon(2048) vs MoLFormer/UniMol2(768) 처럼 scale 비대칭 케이스에 권장.
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
        """현재 epoch 의 temperature 설정."""
        self.tau = tau

    def set_warmup_active(self, active: bool) -> None:
        """Warmup 모드 토글. active=True 면 gate freeze + alpha=1/K 강제 사용.

        Why: 학습 초기에 gate (lr=1e-3) 가 backbone (lr_lora=5e-5) 보다 압도적으로
        빠르게 적응 → 사전학습 feature 강한 modality 로 alpha collapse. warmup 기간
        동안 gate 출력을 무시하면 backbone 들이 task 에 맞춰 충분히 적응한 후 gate
        가 fusion 결정.
        """
        self._warmup_active = active
        if hasattr(self, "gate"):
            for p in self.gate.parameters():
                p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """다음 forward 한 번에만 적용되는 per-sample modality presence mask.

        Args:
            mask: ``(B, K)`` bool — True 면 그 sample 의 해당 modality 사용, False
                면 fusion alpha 차단(softmax logit -inf). 호출자(LitModule.encode)
                가 forward 직전에 호출. forward 가 끝나면 자동으로 None 으로 reset.
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        """K 개 embedding 을 학습된 alpha 로 fusion.

        Args:
            embeddings: 길이 K 의 tensor 리스트, 각 (B, d_k).

        Returns:
            z_fused: (B, output_dim)
            alpha: (B, K) softmax weights
        """
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} 개 embedding 필요, 받은 수: {len(embeddings)}")

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]
        projected = [proj(e) for proj, e in zip(self.projs, embeddings)]

        if self.K == 1:
            alpha = torch.ones(projected[0].size(0), 1, device=projected[0].device)
            return projected[0], alpha

        # stack: (B, K, D)
        stacked = torch.stack(projected, dim=1)
        B, K, D = stacked.shape

        # warmup: gate 우회 → alpha=1/K 균등 (backbone 만 학습되도록).
        if self._warmup_active:
            alpha = torch.full((B, K), 1.0 / K, device=stacked.device, dtype=stacked.dtype)
            z_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
            if self.residual:
                z_fused = z_fused + stacked.mean(dim=1)
            return z_fused, alpha

        # gate logits
        cat = torch.cat(projected, dim=-1)  # (B, K*D)
        logits = self.gate(cat)  # (B, K)

        # modality dropout: training 중 각 sample 마다 독립적으로 modality drop
        if self.training and self.modality_dropout > 0:
            mask = torch.rand(B, K, device=logits.device) > self.modality_dropout
            # 전부 drop 된 경우 → 전부 활성화 (최소 1개 modality 보장)
            any_active = mask.any(dim=-1, keepdim=True)
            mask = mask | ~any_active
            logits = logits.masked_fill(~mask, -1e9)

        # per-sample presence mask: 누락된 modality 는 alpha 0 (softmax logit -inf).
        # 한 sample 의 모든 modality 가 missing 인 경우는 collate 가 사전 제거 했지만,
        # 방어적으로 전부 missing row 는 균등 분포로 fallback (NaN 방지).
        presence = getattr(self, "_presence_mask", None)
        if presence is not None:
            presence = presence.to(device=logits.device)
            any_present = presence.any(dim=-1, keepdim=True)
            safe_presence = presence | ~any_present  # 전부 False 인 row 만 전부 True
            logits = logits.masked_fill(~safe_presence, -1e9)
            self._presence_mask = None  # 다음 forward 까지 reset

        # softmax with current temperature
        alpha = torch.softmax(logits / self.tau, dim=-1)  # (B, K)

        # weighted sum
        z_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)  # (B, D)

        # residual: mean(projected) 추가 → information washout 방지
        if self.residual:
            z_fused = z_fused + stacked.mean(dim=1)

        return z_fused, alpha
