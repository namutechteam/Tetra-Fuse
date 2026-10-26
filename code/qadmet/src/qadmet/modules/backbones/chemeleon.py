"""CheMeleonBackbone — pretrained CheMeleon GNN(frozen) + AdapterGNN/AdapterGIN(trainable).

Design:
    - CheMeleon BondMessagePassing + MeanAggregation weights are always frozen.
    - chemprop ``BondMessagePassing`` internally performs message passing ``depth`` times
      in a closed-loop. Each loop iteration is regarded as one "MP layer".

Adapter mode (``adapter_kind``):
    - ``"paper"`` (default) — faithful implementation of AdapterGNN (Li et al., AAAI'24).
        * Inlines chemprop forward, and for each depth iteration, paper Eq. (8)
          sums the two adapters via parallel wiring (bond-level, d=2048):

            H_l = update(message(H_{l-1}, bmg), H_0)
                  + s1 · A1(H_{l-1})
                  + s2 · A2(message(H_{l-1}, bmg))

          where ``update(M, H_0) = tau(H_0 + W_h(M)) + dropout`` (chemprop frozen path)
          corresponds to the paper's ``BN(MLP(MP(x_l)))`` slot. chemprop itself has no internal layer BN,
          so the paper's outer BN is replaced by the adapter's internal BN.
        * Each adapter A: down → ReLU → up → BN (Eq. 7), scaled by an external scalar s.
        * s1, s2 are each adapter's ``gamma`` (learnable, init=0). At the start of training,
          forward is exactly equivalent to the frozen-only baseline.
        * depth=3 (chemprop default) → 2 pre + 2 post = 4 adapter blocks.
        * Since BondMP does not expose the atom-level d_h intermediate state, the adapter is also bond-level.
    - ``"lora_gin"`` (legacy) — outer-only AdapterGINBlock stack.
        * ``n_adapter_layers`` atom-only GIN stacked on top of the full frozen MP output (N_atoms, 2048).

Then frozen ``agg`` (MeanAggregation) → graph-level → trainable ``self.proj``
(Linear + ReLU [+ Dropout]) → ``(B, d_h)``.

Lazy import:
    chemprop is imported only inside ``__init__`` (heavy package).

Backward-compat:
    - The legacy ``adaptor_dropout`` kwarg is an alias for ``adapter_dropout`` (deprecation).
    - ``adapter_kind`` default changed to "paper" — incompatible with existing trained weights, retraining required.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
from torch import Tensor, nn

from ._adapter_gin import AdapterGINBlock
from ._adapter_gnn import AdapterGNNBlock
from .registry import BACKBONES


@BACKBONES.register("chemeleon")
class CheMeleonBackbone(nn.Module):
    """CheMeleon pretrained GNN(frozen) + AdapterGNN/AdapterGIN(trainable) wrapper.

    Args:
        d_h: ``self.proj`` output dimension (= ``output_dim``).
        checkpoint_path: CheMeleon weight file path (if None, random-init smoke test).
        d_xd: additional descriptor concat dimension (default 0 — descriptor unused).
        adapter_kind: ``"paper"`` (default, AdapterGNN paper-faithful) or
            ``"lora_gin"`` (legacy outer-only ``AdapterGINBlock``).
        adapter_bottleneck: adapter bottleneck dim ``r``.
        adapter_dropout: adapter residual branch dropout.
        n_adapter_layers: only meaningful for ``"lora_gin"`` — number of outer AdapterGINBlocks.
            In ``"paper"`` mode, determined automatically as ``depth-1`` (chemprop depth).
        proj_dropout: dropout for ``self.proj`` (0 if none).
        adaptor_dropout: **Deprecated** — alias for ``adapter_dropout``.

    Attributes:
        CHEMELEON_HIDDEN: BondMessagePassing output dimension (always 2048).
        output_dim: d_h.
        adapter_kind: the adapter mode used.

    Note:
        - frozen mp/agg are fixed with ``requires_grad=False`` + ``eval()``.
        - frozen modules remain in eval even when ``train(mode)`` is called.
    """

    CHEMELEON_HIDDEN = 2048  # BondMessagePassing output dim

    def __init__(
        self,
        d_h: int = 300,
        checkpoint_path: str | None = None,
        d_xd: int = 0,
        adapter_kind: Literal["paper", "lora_gin"] = "paper",
        adapter_bottleneck: int = 256,
        adapter_dropout: float = 0.0,
        n_adapter_layers: int = 2,
        proj_dropout: float = 0.0,
        adaptor_dropout: float | None = None,
    ) -> None:
        super().__init__()
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, MeanAggregation
        from chemprop.nn.predictors import RegressionFFN

        if adapter_kind not in ("paper", "lora_gin"):
            raise ValueError(
                f"adapter_kind must be 'paper' or 'lora_gin' (received: {adapter_kind!r})."
            )

        if adaptor_dropout is not None:
            warnings.warn(
                "CheMeleonBackbone: `adaptor_dropout` is deprecated — use `adapter_dropout`. "
                "Mapped as an alias this time.",
                DeprecationWarning,
                stacklevel=2,
            )
            adapter_dropout = adaptor_dropout

        if adapter_kind == "lora_gin" and n_adapter_layers < 1:
            raise ValueError(
                f"n_adapter_layers must be >= 1 for adapter_kind='lora_gin' "
                f"(received: {n_adapter_layers})."
            )

        # Load from checkpoint or random init
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, weights_only=True)
            mp = BondMessagePassing(**ckpt["hyper_parameters"])
            mp.load_state_dict(ckpt["state_dict"])
        else:
            # smoke test: random-init (same d_h=2048 as notebook hyper_parameters)
            mp = BondMessagePassing(d_h=self.CHEMELEON_HIDDEN)

        agg = MeanAggregation()

        # dummy predictor for MPNN constructor compatibility (not actually used — only message_passing/agg are called)
        dummy_predictor = RegressionFFN(
            n_tasks=1,
            input_dim=self.CHEMELEON_HIDDEN + d_xd,
            hidden_dim=4,
            n_layers=1,
        )
        self._mpnn = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=dummy_predictor,
            batch_norm=False,
        )

        # ---- frozen mp/agg/bn/X_d_transform ----
        self._mpnn.message_passing.requires_grad_(False)
        self._mpnn.agg.requires_grad_(False)
        self._mpnn.message_passing.eval()
        self._mpnn.agg.eval()
        # bn is likely Identity due to batch_norm=False, but freeze it to be safe
        self._mpnn.bn.requires_grad_(False)
        self._mpnn.bn.eval()
        if hasattr(self._mpnn, "X_d_transform"):
            self._mpnn.X_d_transform.requires_grad_(False)
            self._mpnn.X_d_transform.eval()

        self.adapter_kind = adapter_kind

        if adapter_kind == "paper":
            # ---- paper-faithful: pre/post pair for each chemprop MP iteration (bond-level) ----
            depth = self._mpnn.message_passing.depth
            if depth < 2:
                raise ValueError(
                    f"adapter_kind='paper' requires chemprop depth>=2 (current: {depth}). "
                    "If depth=1, there is no message-passing iteration, so there is no place to insert an adapter."
                )
            n_pairs = depth - 1  # chemprop loop runs (depth-1) iterations
            self.pre_adapters = nn.ModuleList(
                [
                    AdapterGNNBlock(
                        d=self.CHEMELEON_HIDDEN,
                        r=adapter_bottleneck,
                        dropout=adapter_dropout,
                    )
                    for _ in range(n_pairs)
                ]
            )
            self.post_adapters = nn.ModuleList(
                [
                    AdapterGNNBlock(
                        d=self.CHEMELEON_HIDDEN,
                        r=adapter_bottleneck,
                        dropout=adapter_dropout,
                    )
                    for _ in range(n_pairs)
                ]
            )
            self.n_adapter_pairs = n_pairs
            # clear legacy attr (no external dependency — new training)
            self.adapter_blocks = nn.ModuleList()
        else:
            # ---- legacy lora_gin: outer-only AdapterGINBlock stack ----
            self.adapter_blocks = nn.ModuleList(
                [
                    AdapterGINBlock(
                        d=self.CHEMELEON_HIDDEN,
                        r=adapter_bottleneck,
                        with_edge_attr=False,
                        dropout=adapter_dropout,
                    )
                    for _ in range(n_adapter_layers)
                ]
            )
            self.n_adapter_layers = n_adapter_layers

        # ---- trainable post-pool projection ----
        proj_layers: list[nn.Module] = [
            nn.Linear(self.CHEMELEON_HIDDEN + d_xd, d_h),
            nn.ReLU(),
        ]
        if proj_dropout > 0:
            proj_layers.append(nn.Dropout(proj_dropout))
        self.proj = nn.Sequential(*proj_layers)

        self.d_h = d_h
        self.d_xd = d_xd
        self.output_dim = d_h

    def train(self, mode: bool = True):
        """frozen submodules always remain in eval (blocks Lightning train() propagation).

        Args:
            mode: toggles train/eval only for trainable submodules (adapters / proj).

        Returns:
            self
        """
        super().train(mode)
        # force frozen modules to eval
        self._mpnn.message_passing.eval()
        self._mpnn.agg.eval()
        self._mpnn.bn.eval()
        if hasattr(self._mpnn, "X_d_transform"):
            self._mpnn.X_d_transform.eval()
        return self

    def _frozen_mp_forward_with_paper_adapters(
        self, bmg, V_d: Tensor | None = None
    ) -> Tensor:
        """Inlines chemprop ``BondMessagePassing.forward`` + paper Eq.(8) parallel adapter.

        Reproduces the structure of chemprop 2.2.3's ``_MessagePassingBase.forward`` + bond-level
        AdapterGNNBlock's two branches (A1: H_in, A2: M=message(H_in, bmg)) are element-wise summed to the frozen update
        result. chemprop internals (``W_i/W_h/W_o``, ``message``,
        ``update``, ``tau``, ``depth``, ``undirected``, ``initialize``, ``finalize``,
        ``graph_transform``, ``V_d_transform``) are called as-is.

        Args:
            bmg: chemprop ``BatchMolGraph``.
            V_d: atom feature augmentation (optional).

        Returns:
            atom-level ``(N_atoms, d_h)`` (finalize output).
        """
        mp = self._mpnn.message_passing
        bmg = mp.graph_transform(bmg)
        H_0 = mp.initialize(bmg)  # bond-level (V_bonds, d_h)
        H = mp.tau(H_0)

        for idx in range(mp.depth - 1):
            if mp.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2
            # frozen branch: MP(x_l)=M, then update — = BN(MLP(MP)) slot
            M = mp.message(H, bmg)
            H_frozen = mp.update(M, H_0)
            # parallel adapter branches (s · A(·), no residual inside)
            a1 = self.pre_adapters[idx](H)
            a2 = self.post_adapters[idx](M)
            H = H_frozen + a1 + a2

        # atom-level scatter (chemprop base.py:208-211 as-is)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(
            len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device
        ).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
        return mp.finalize(M, bmg.V, V_d)

    def forward(
        self, bmg, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """CheMeleon mp + adapter → agg → proj.

        Args:
            bmg: batch molecule graph (chemprop BatchMolGraph).
            V_d: atom feature augmentation (optional).
            X_d: descriptor concat dimension (optional, used only when d_xd > 0).

        Returns:
            h: (B, d_h) projection output.
        """
        if self.adapter_kind == "paper":
            # MP forward with bond-level pre/post adapter inlined
            H_v = self._frozen_mp_forward_with_paper_adapters(bmg, V_d)
        else:
            # frozen mp → atom-level (N, 2048)
            H_v = self._mpnn.message_passing(bmg, V_d)
            # trainable atom-only adapter GINs (residual, LoRA-zero-init)
            edge_index = bmg.edge_index
            for adapter in self.adapter_blocks:
                H_v = adapter(H_v, edge_index, edge_attr=None)

        # frozen agg → (B, 2048)
        graph_h = self._mpnn.agg(H_v, bmg.batch)
        graph_h = self._mpnn.bn(graph_h)

        # X_d concat (same sequence as chemprop fingerprint)
        if X_d is not None:
            graph_h = torch.cat((graph_h, self._mpnn.X_d_transform(X_d)), dim=1)

        return self.proj(graph_h)
