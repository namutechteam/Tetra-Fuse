"""FiLM fusion — affine modulation of primary modality by auxiliary modalities.

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.

Design:
    - ``embeddings[0]`` = primary (assumes CheMeleon), ``embeddings[1:]`` = auxiliary conditions.
    - project primary to ``output_dim`` → ``x``.
    - project each of the K-1 auxiliary conditions to ``cond_hidden`` → concat → MLP → ``(γ_pre, β)``.
    - ``γ = 1 + γ_pre`` (identity init: γ≈1, β≈0 → preserves primary early in training).
    - (optional) clamp ``γ`` to ``[gamma_min, gamma_max]`` — combined with residual,
      the effective coefficient of x is ``γ+1``, so the lower bound structurally blocks primary washout
      (coefficient 0 at ``γ→-1``). Same philosophy as the GradNorm ``w_k`` clamp.
    - ``z = γ ⊙ x + β`` (+ optional residual ``+x``).
    - (optional) ``z = output_proj(z)`` — post-fusion operating directly on the fused z
      Linear. GradNorm shared_layer candidate (passed through by every one of the 22 tasks' forward
      and a single W tensor that directly sees the fused multimodal signal). Always trainable
      (unaffected by warmup). When activated with identity-init (weight=I, bias=0),
      preserves existing FiLM forward behavior.

Anti-collapse mechanisms:
    - condition dropout: zero-out each auxiliary condition per-sample during training.
    - presence mask: for a missing auxiliary modality (e.g. a sample where UniMol2 ETKDG failed),
      zero out its contribution.
    - warmup: freeze film_gen + (γ=1, β=0) for the first N epochs → train primary alone.
    - identity-init: last Linear weight=0 + bias=0 → modulation is inert at the start of training.

Note:
    for fusion signature consistency, alpha is returned as uniform ``(B, K)`` (FiLM is not attention
    so its diagnostic meaning is weak). It is natural to leave entropy_weight at 0.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("film")
class FilmFusion(nn.Module):
    """Fusion that performs affine modulation of primary modality by auxiliary modalities.

    Args:
        input_dims: length K. ``input_dims[0]`` = primary, the rest = auxiliary conditions.
            K ≥ 2 required.
        output_dim: primary projection / modulation dimension (head input dimension).
        cond_hidden: projection dimension for each auxiliary condition (before concat).
        condition_dropout: sample-wise zero-out probability for each condition during training.
        residual: adds a primary residual in the form ``z = γ⊙x + β + x`` (prevents information washout).
        pre_norm: input LayerNorm (absorbs scale asymmetry across modalities).
        film_init_identity: initialize the last Linear to 0 → start at γ≈1, β≈0.
        warmup_epochs: freeze film_gen + identity modulation for the first N epochs.
            toggled by lit_module ``on_train_epoch_start`` via ``set_warmup_active``.
        use_output_proj: if ``True``, append ``nn.Linear(output_dim, output_dim)``
            to project the fused ``z`` once more. identity-init (weight=I,
            bias=0), so the forward output at activation time is identical to when deactivated.
            usable as a GradNorm shared_layer candidate — a single W operating directly
            on the fused signal (passed through by every one of the 22 tasks' forward, always trainable).
        gamma_min: when set, the lower bound of ``γ``. Combined with residual, the x coefficient is
            ``γ+1``, so ``gamma_min > -1`` blocks primary washout. Recommended 0.3.
        gamma_max: when set, the upper bound of ``γ`` (suppresses modulation blowup). Recommended 3.0.
            if both ``gamma_min``/``gamma_max`` are ``None`` (default), clamp is not applied —
            preserves existing recipe behavior; opt in via recipe when used.
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
                "FilmFusion requires primary + ≥1 auxiliary modality "
                f"(K={self.K}). If using a single modality, leave fusion itself empty."
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

        # condition projections: each d_aux_i → cond_hidden
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

        # Optional post-fusion projection. a single Linear operating directly on the fused z
        # — GradNorm shared_layer candidate (passed through by every task's forward, always trainable).
        # with identity-init, forward output starts identical to the deactivated case.
        self.output_proj: nn.Linear | None
        if use_output_proj:
            self.output_proj = nn.Linear(output_dim, output_dim)
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = None

    def set_warmup_active(self, active: bool) -> None:
        """Toggle warmup mode. active=True → film_gen freeze + (γ=1, β=0) hard-code."""
        self._warmup_active = active
        for p in self.film_gen.parameters():
            p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """Per-sample modality presence mask (B, K) applied only to the next single forward.

        ``mask[:, 0]`` (primary) is ignored — primary always exists.
        for samples where ``mask[:, 1:]`` is False, zero out that auxiliary condition's projected value.
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")

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
                # presence[:, 0] = primary, presence[:, i+1] = i-th auxiliary
                p = presence[:, i + 1].to(dtype=c_projected[i].dtype).unsqueeze(-1)
                c_projected[i] = c_projected[i] * p
            self._presence_mask = None  # reset until next forward

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

        # alpha sidecar (diagnostic uniform — FiLM is not attention).
        alpha = torch.full((B, self.K), 1.0 / self.K, device=x.device, dtype=x.dtype)
        return z, alpha
