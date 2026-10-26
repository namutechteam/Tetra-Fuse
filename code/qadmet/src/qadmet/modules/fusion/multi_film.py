"""Multi-primary FiLM + cross-attention late fusion (symmetric FiLM).

Why:
    Single-primary ``film`` (CheMeleon main + auxiliary concat → γ/β) has primary selection
    dominate the outcome, and auxiliary modality is not a first-class citizen. ``multi_film`` treats each of K
    modalities as primary exactly once (self-anchor residual + affine modulation from remaining
    modalities) to produce K branch representations, then via modality-as-token
    cross-attention mixes them **late**, once.

    ⚠ Fully symmetric (mutual conditioning at every layer) fusion is known in molecular representation literature for modality collapse ·
    unstable optimization (MuMo, arXiv:2510.23640, etc. deliberately choose asymmetric
    design). This module is designed with "weak symmetry" to reduce that risk:
      - Each branch anchors its own primary via self-residual (``+x_i``) (prevents washout).
      - Interaction is late-fused at a single final cross-attention (not mutual
        conditioning at every layer).
      - identity-init + γ clamp + condition dropout + branch dropout to
        block modulation blowup · branch collapse.

Design:
    1. branch i (each modality as primary):
         x_i = primary_proj_i(norm_i(e_i))                         ∈ R^D
         for each conditioner j≠i, (γ_ij, β_ij) = film_gen_ij(cond_proj_ij(e_j))
         γ_ij = clamp(1 + γ_pre_ij, [gamma_min, gamma_max])
         delta_i = mean_{j≠i}(γ_ij ⊙ x_i + β_ij)        ← parallel summation (mean: stable scale)
         z_i = x_i + delta_i                              ← self-anchor residual
    2. stack(z_0..z_{K-1}) → (B, K, D)
    3. cross-attention mixer: TransformerEncoderLayer × N (norm_first/gelu,
       modality-as-token self-attn). Same pattern as GR-MF (group_routed).
       presence/branch-dropout is blocked via ``src_key_padding_mask``.
    4. masked mean pool over branch tokens → (B, D).
    5. (optional) output_proj (Linear D→D, identity-init).

Anti-collapse mechanisms:
    - condition dropout: zero-out each conditioner per-sample during training.
    - branch dropout: mask mixer input branch tokens per-sample during training
      (guarantees at least 1 active) — defends against attention collapse onto a single branch.
    - presence mask: missing modality → its conditioner contribution becomes 0 + that modality's
      branch token mask.
    - warmup: freeze film_gen/cond_proj for the first N epochs + modulation identity →
      mixer only mixes raw primaries (secures backbone adaptation time).
    - identity-init: film_gen final Linear=0 → starts at γ≈1, β≈0.

Note:
    for forward signature uniformity, alpha returns a uniform ``(B, K)`` (cross-attention
    attn weight is not directly exposed by nn.TransformerEncoder — weak diagnostic meaning).
    it is natural to leave entropy_weight at 0.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .registry import FUSIONS


@FUSIONS.register("multi_film")
class MultiFilmFusion(nn.Module):
    """Multi-primary FiLM branches + cross-attention late fusion.

    Args:
        input_dims: length K. Input dimension of each modality. Each of K modalities becomes
            primary once, producing K branches. K ≥ 2 required.
        output_dim: branch projection / modulation dimension (= head input dimension).
        cond_hidden: each conditioner projection dimension (film_gen input).
        condition_dropout: sample-wise zero-out probability for each conditioner during training.
        film_init_identity: film_gen final Linear 0-init → starts at γ≈1, β≈0.
        pre_norm: per-modality input LayerNorm (absorbs modality scale asymmetry).
        gamma_min / gamma_max: γ clamp range. Combined with residual, guarantees a lower bound on the x coefficient →
            blocks primary washout. Recommended 0.3 / 3.0. If both are None, clamp is not applied.
        warmup_epochs: freeze film_gen/cond_proj for the first N epochs + modulation identity.
            lit_module ``on_train_epoch_start`` toggles ``set_warmup_active``.
        cross_modal_layers: number of mixer TransformerEncoderLayers. If 0, without mixer
            branch average pool.
        cross_modal_heads: mixer nhead. ``output_dim % heads == 0`` required.
        cross_modal_ffn_mult: mixer FFN hidden = output_dim * mult.
        cross_modal_dropout: mixer dropout.
        branch_dropout: sample-wise mask probability for mixer input branch tokens during training
            (guarantees at least 1 active). Defends against branch collapse.
        use_output_proj: one final Linear(D, D, identity-init) stage. GradNorm
            shared_layer candidate (passed through by all task forwards, always trainable).
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
                f"MultiFilmFusion requires K>=2 modality (each modality as primary once), "
                f"K={self.K}."
            )
        if cross_modal_layers > 0 and output_dim % cross_modal_heads != 0:
            raise ValueError(
                f"output_dim({output_dim}) must be divisible by cross_modal_heads({cross_modal_heads}) "
                "but is not."
            )
        self.output_dim = output_dim
        self.condition_dropout = condition_dropout
        self.branch_dropout = branch_dropout
        self.warmup_epochs = warmup_epochs
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self._warmup_active = False
        self._presence_mask: Tensor | None = None

        # conditioner index for each branch i (excluding itself).
        self._cond_idx: list[list[int]] = [
            [j for j in range(self.K) if j != i] for i in range(self.K)
        ]

        if pre_norm:
            self.input_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.LayerNorm(d) for d in input_dims]
            )
        else:
            self.input_norms = None

        # primary projection for branch i: d_i → output_dim
        self.primary_projs = nn.ModuleList(
            [nn.Linear(input_dims[i], output_dim) for i in range(self.K)]
        )

        # (cond_proj, film_gen) per conditioner j of branch i.
        # nested ModuleList: cond_projs[i][local_j], film_gens[i][local_j].
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

        # cross-attention mixer (modality-as-token). Same pattern as GR-MF.
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
        """Warmup toggle. active=True → freeze all film_gen/cond_proj + modulation identity."""
        self._warmup_active = active
        for mlist in (self.film_gens, self.cond_projs):
            for branch in mlist:
                for m in branch:
                    for p in m.parameters():
                        p.requires_grad = not active

    def set_presence_mask(self, mask: Tensor) -> None:
        """Apply modality presence (B, K) bool for the next forward call only.

        ``mask[:, i]`` False → modality i is missing for that sample. branch i's
        token as primary is masked in the mixer, and as conditioner for other branches
        the contribution becomes 0.
        """
        self._presence_mask = mask

    def forward(self, embeddings: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(embeddings) != self.K:
            raise ValueError(f"K={self.K} embeddings required, received: {len(embeddings)}")

        if self.input_norms is not None:
            embeddings = [norm(e) for norm, e in zip(self.input_norms, embeddings)]

        x = [self.primary_projs[i](embeddings[i]) for i in range(self.K)]  # each (B, D)
        B, D = x[0].shape

        presence = self._presence_mask
        if presence is not None:
            presence = presence.to(device=x[0].device, dtype=torch.bool)
            # row where all are missing → fallback to all present (prevents NaN).
            any_present = presence.any(dim=-1, keepdim=True)
            presence = presence | ~any_present
            self._presence_mask = None

        branch_out: list[Tensor] = []
        for i in range(self.K):
            xi = x[i]
            if self._warmup_active:
                branch_out.append(xi + xi)  # modulation identity + residual (matches film.py)
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
            delta = torch.stack(deltas, dim=0).mean(dim=0)  # parallel summation (mean)
            branch_out.append(xi + delta)  # self-anchor residual

        stacked = torch.stack(branch_out, dim=1)  # (B, K, D)

        # mixer key_padding_mask: True = ignore that branch token.
        kpm: Tensor | None = None
        if presence is not None:
            kpm = ~presence  # (B, K)
        if self.training and self.branch_dropout > 0:
            keep = torch.rand(B, self.K, device=stacked.device) > self.branch_dropout
            any_keep = keep.any(dim=-1, keepdim=True)
            keep = keep | ~any_keep  # row where all are dropped → keep all
            drop = ~keep
            kpm = drop if kpm is None else (kpm | drop)
        if kpm is not None:
            # guard against rows that are fully padded (presence + dropout simultaneously) → unmask all.
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

        # alpha sidecar (diagnostic uniform — cross-attn weight not exposed).
        alpha = torch.full((B, self.K), 1.0 / self.K, device=z.device, dtype=z.dtype)
        return z, alpha
