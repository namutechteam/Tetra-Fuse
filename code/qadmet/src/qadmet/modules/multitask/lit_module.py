"""MultitaskLitModule — Spec based multi-task LightningModule.

Assembly flow (Step B design, 2026-05-01):
    1. ``BackboneSpec`` → BACKBONES.build → primary representation ``H (B, D_main)``.
       For DMPNN, ``batch["X_d"]`` (descriptor stack) inside chemprop ``MPNN.fingerprint``
       is concatenated after the backbone output internally, becoming ``D_main = d_h + d_xd``.
    2. In ``FeatureSpec.embeddings`` order (excluding consumed keys), receive modality vectors and
       construct the ``[H_1, ..., H_K, *embeddings]`` list. **descriptor is not included in fusion
       ** — already absorbed via the backbone's X_d concat.
    3. If ``FusionSpec`` is present, create the fusion module via FUSIONS.build; otherwise
       use the backbone output as-is (the case of using only a single backbone + descriptors).
    4. ``HeadSpec`` → HEADS.build → ``preds (B, n_tasks)``.
    5. ``LossSpec`` → LOSSES.build → compute training/validation loss.

Legacy ``GroupedTaskUncertaintyMPNN._encode`` (pipeline_mtl.py:1056-1129)'s 4 branches
(LoRA / frozen MoLFormer / frozen UniMol2 / legacy concat) into spec + adapter
generalized form. It handles combinations of a single trainable backbone + frozen auxiliary modalities.

Batch dict contract:
    backbone-specific keys + ``modalities`` (= embeddings only) + ``targets`` + (optional)
    ``sample_weights``. Per-backbone input is dispatched by ``_INPUT_ADAPTERS``.

    e.g.) DMPNN backbone (descriptors=[rdkit2d, xtb], embeddings=[molformer])::

        {
            "bmg": <chemprop.BatchMolGraph>,
            "V_d": <Tensor> | None,
            "X_d": (B, 204),  # rdkit2d(200) + xtb(4) — fingerprint concatenated after H.
            "modalities": {"molformer": (B, 768)},  # only included in fusion.
            "targets": (B, n_tasks),
            "sample_weights": (B,),
        }

    e.g.) MoLFormer-LoRA backbone::

        {
            "token_ids": (B, L),
            "attention_mask": (B, L),
            "modalities": {},
            "targets": (B, n_tasks),
        }

Lazy import:
    ``lightning`` is imported at this module's top level (LightningModule inheritance required).
    However, ``qadmet.modules`` ``__init__`` does not import this sub-package, so
    importing ``qadmet.modules`` itself does not pull in lightning.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import lightning as L
import torch
from torch import Tensor, nn

from qadmet.core.specs import AuxLossSpec, BackboneSpec, FusionSpec, HeadSpec, LossSpec
from qadmet.modules.backbones import BACKBONES
from qadmet.modules.backbones._adapter_gnn import AdapterGNNBlock
from qadmet.modules.fusion import FUSIONS
from qadmet.modules.heads import HEADS
from qadmet.modules.losses import LOSSES

# ---------------------------------------------------------------------------- input adapter

#: backbone forward call adapter: ``(backbone_module, batch_dict) -> Tensor (B, D)``.
BackboneAdapter = Callable[[nn.Module, Mapping[str, Any]], Tensor]

_INPUT_ADAPTERS: dict[str, BackboneAdapter] = {}


def register_input_adapter(backbone_type: str) -> Callable[[BackboneAdapter], BackboneAdapter]:
    """backbone type → adapter registration decorator.

    Why:
        forward signatures differ per backbone (DMPNN: bmg/V_d/X_d, MoLFormer-LoRA:
        token_ids/attention_mask, UniMol2-LoRA: vendored batch dict). In the LitModule body
        branching logic is not placed here; it is separated so it can be registered externally.

    Returns:
        Registration decorator. Re-registering the same key raises ``KeyError``.
    """

    def deco(fn: BackboneAdapter) -> BackboneAdapter:
        if backbone_type in _INPUT_ADAPTERS:
            raise KeyError(f"input adapter already registered: {backbone_type!r}")
        _INPUT_ADAPTERS[backbone_type] = fn
        return fn

    return deco


@register_input_adapter("dmpnn")
def _dmpnn_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    return backbone(batch["bmg"], V_d=batch.get("V_d"), X_d=batch.get("X_d"))


@register_input_adapter("molformer_lora")
def _molformer_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """MoLFormer-LoRA input — for STL it's the batch top-level key, for multi-backbone MTL it's the consumed slot."""
    if "token_ids" in batch:
        return backbone(batch["token_ids"], batch["attention_mask"])
    consumed: Mapping[str, Tensor] = batch.get("__consumed__") or {}
    if "molformer_tokens" in consumed:
        mt = consumed["molformer_tokens"]
        # mt: (B, 2L) float — first L = token_ids, last L = attention_mask. backbone expects long.
        L = mt.shape[1] // 2
        return backbone(mt[:, :L].long(), mt[:, L:].long())
    raise KeyError(
        "molformer_lora adapter: either batch['token_ids'] or "
        "consumed['molformer_tokens'] is required."
    )


@register_input_adapter("unimol2_lora")
def _unimol2_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """STL: ``batch['unimol2_batch']``. MTL multi-backbone: retrieve dict from the ``__consumed__`` slot."""
    if "unimol2_batch" in batch:
        return backbone(**batch["unimol2_batch"])
    consumed: Mapping[str, Any] = batch.get("__consumed__") or {}
    for k, payload in consumed.items():
        if isinstance(payload, dict):
            return backbone(**payload)
    raise KeyError(
        "unimol2_lora adapter: either batch['unimol2_batch'] or "
        "consumed[<unimol2_lmdb-like>] dict is required."
    )


@register_input_adapter("chemeleon")
def _chemeleon_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    return backbone(batch["bmg"], V_d=batch.get("V_d"), X_d=batch.get("X_d"))


@register_input_adapter("graphmvp")
def _graphmvp_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """STL: ``batch['graphmvp_data']``. MTL multi-backbone: retrieve from the ``__consumed__`` slot.

    payload is ``torch_geometric.data.Batch`` (collate converts SMILES → PyG Data → Batch).
    Lightning ``move_data_to_device`` only recognizes dict/Tensor, so PyG Batch is, in this
    adapter, moved directly to the backbone parameter device.
    """
    pyg_batch = None
    if "graphmvp_data" in batch:
        pyg_batch = batch["graphmvp_data"]
    else:
        consumed: Mapping[str, Any] = batch.get("__consumed__") or {}
        if "graphmvp_data" in consumed:
            pyg_batch = consumed["graphmvp_data"]
    if pyg_batch is None:
        raise KeyError(
            "graphmvp adapter: either batch['graphmvp_data'] or "
            "consumed['graphmvp_data'] is required."
        )
    try:
        device = next(p.device for p in backbone.parameters() if p.requires_grad)
    except StopIteration:
        device = next(backbone.parameters()).device
    pyg_batch = pyg_batch.to(device)
    return backbone(pyg_batch)


@register_input_adapter("mlp_embedding")
def _mlp_embedding_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """Precomputed embedding passthrough — for MTL multi-backbone it's the ``__consumed__`` slot,
    for STL, retrieve the single modality tensor from the top level.

    consumed payload is a ``(B, d)`` float Tensor (the embedding missing from the modalities dict).
    same ``__consumed__`` handling as the molformer_lora adapter, but a plain tensor rather than a token.
    """
    consumed: Mapping[str, Any] = batch.get("__consumed__") or {}
    for payload in consumed.values():
        if isinstance(payload, Tensor):
            return backbone(payload)
    # STL fallback: the case where modalities contains only a single tensor modality.
    modalities: Mapping[str, Any] = batch.get("modalities") or {}
    tensors = [v for v in modalities.values() if isinstance(v, Tensor)]
    if len(tensors) == 1:
        return backbone(tensors[0])
    raise KeyError(
        "mlp_embedding adapter: either consumed[<embedding>] tensor or a single modality "
        "tensor is required."
    )


# ---------------------------------------------------------------------------- LitModule


class MultitaskLitModule(L.LightningModule):
    """Spec-based multi-task LightningModule.

    Args:
        backbone_spec: BackboneSpec — primary trainable trunk.
        head_spec: HeadSpec — ``linear`` | ``grouped`` etc.
        loss_spec: LossSpec — ``mixed_task_uncertainty`` | ``kendall_uncertainty``.
        n_tasks: number of output tasks.
        task_types: ``"regression"`` / ``"classification"`` labels of length ``n_tasks``.
        fusion_spec: FusionSpec | None. If None, without auxiliary modalities/extra backbone,
            use the primary backbone output directly.
        extra_backbone_specs: additional trainable trunks. Each BackboneSpec's
            if ``consumes`` is not empty, the corresponding modality keys are fusion modality
            Excluded from the stack and handed to the backbone adapter's ``batch['__consumed__']``,
            it is passed along (formerly phase1.4a — D-MPNN + LoRA-MoLFormer).
        ordered_tasks: HeadSpec.type=="grouped" or task name order used for lineage metadata.
        modality_dims: auxiliary modality name → dimension. Only **embeddings keys** need be included
            (descriptors flow via backbone X_d, so they are not included in the modalities dict).
            Consumed keys must also be present in embeddings (molformer_tokens for LoRA-MoLFormer).
        descriptors / embeddings: receives the two fields of FeatureSpec as-is.
            descriptors flows directly into the backbone (DMPNN)'s X_d and is excluded from the fusion input.
            Only embeddings join the fusion input; consumed keys are routed to an extra backbone instead.
        task_to_group / groups: additional arguments for GroupedHead.
        task_weights: (n_tasks,) inverse-freq weight tensor (optional).
        fusion_output_dim: fallback when FusionSpec.kwargs has no ``output_dim``.
        lr: max_lr of the Noam schedule (head/fusion/non-LoRA params).
        init_lr: starting lr of Noam warmup.
        final_lr: ending lr of Noam cooldown.
        warmup_epochs: duration of Noam warmup (number of epochs).
        lr_lora: if not None, only parameters whose name contains ``"lora_"`` are grouped
            Grouped and ``lr_lora`` applied (equivalent to the former ``run_stl_*_lora.configure_optimizers``).
            If None, a single lr is used.

    Note:
        This class is independent of the training data pipeline (Step 3.6) and is only responsible
        for **module assembly**. The forward input is assumed to be a dict built by the collate function,
        dispatched by the adapter for each backbone type.
    """

    def __init__(
        self,
        *,
        backbone_spec: BackboneSpec,
        head_spec: HeadSpec,
        loss_spec: LossSpec,
        n_tasks: int,
        task_types: list[str],
        fusion_spec: FusionSpec | None = None,
        extra_backbone_specs: tuple[BackboneSpec, ...] = (),
        ordered_tasks: list[str] | None = None,
        modality_dims: Mapping[str, int] | None = None,
        descriptors: tuple[str, ...] = (),
        embeddings: tuple[str, ...] = (),
        task_to_group: Mapping[str, str] | None = None,
        groups: list[str] | None = None,
        task_weights: Tensor | None = None,
        fusion_output_dim: int = 300,
        lr: float = 1e-3,
        init_lr: float = 1e-4,
        final_lr: float = 1e-4,
        warmup_epochs: int = 2,
        lr_lora: float | None = None,
        lr_fusion: float | None = None,
        aux_loss_spec: AuxLossSpec | None = None,
        head_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()

        if len(task_types) != n_tasks:
            raise ValueError(
                f"len(task_types)={len(task_types)} != n_tasks={n_tasks}"
            )

        self.lr = lr
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.lr_lora = lr_lora
        self.lr_fusion = lr_fusion
        self.head_warmup_epochs = int(head_warmup_epochs)
        # Plain bool cache so that once head freeze has been enabled at least once, unfreeze trainable state
        # doesn't need to be set every epoch. Even if the head is frozen at the point configure_optimizers is
        # called, the optimizer still holds the params, so after unfreezing it will train automatically
        # (only requires_grad is toggled).
        self._head_warmup_state: bool | None = None
        self.n_tasks = n_tasks
        # For diagnostics: accumulate per-task average loss per epoch
        self._task_loss_epoch_means: list[list[float]] = []
        self._task_loss_batch_sum: list[float] | None = None
        self._task_loss_batch_n: int = 0
        self.task_types: list[str] = list(task_types)
        self.ordered_tasks: list[str] | None = (
            list(ordered_tasks) if ordered_tasks is not None else None
        )
        self.descriptors: tuple[str, ...] = tuple(descriptors)
        self.embeddings: tuple[str, ...] = tuple(embeddings)
        # descriptors flow directly into the backbone X_d, so they are not included in the modality dict.
        # Both consumes validation / fusion modality input only target embeddings.
        self._modality_keys: list[str] = list(self.embeddings)
        # ---- backbones (primary + extras) ----
        all_specs: tuple[BackboneSpec, ...] = (backbone_spec, *extra_backbone_specs)
        consumed_keys: list[str] = []
        for s in all_specs:
            if s.type not in _INPUT_ADAPTERS:
                raise KeyError(
                    f"No input adapter registered for backbone type {s.type!r}. "
                    f"Registered keys: {sorted(_INPUT_ADAPTERS)}"
                )
            for k in s.consumes:
                if k in consumed_keys:
                    raise ValueError(
                        f"modality {k!r} is consumed by two backbones simultaneously — "
                        f"One modality can only consume one backbone."
                    )
                consumed_keys.append(k)
        self._backbone_types: list[str] = [s.type for s in all_specs]
        self._backbone_consumes: list[tuple[str, ...]] = [tuple(s.consumes) for s in all_specs]
        self._consumed_keys: tuple[str, ...] = tuple(consumed_keys)

        backbones: list[nn.Module] = []
        backbone_out_dims: list[int] = []
        for s in all_specs:
            bb = BACKBONES.build(s)
            d = getattr(bb, "output_dim", None)
            if not isinstance(d, int) or d <= 0:
                raise AttributeError(
                    f"backbone {s.type!r} must expose ``output_dim: int`` "
                    f"(got: {d!r})"
                )
            backbones.append(bb)
            backbone_out_dims.append(d)
        # For single-backbone compatibility, self.backbone is exposed as primary + also holds a ModuleList.
        self.backbones: nn.ModuleList = nn.ModuleList(backbones)
        self.backbone: nn.Module = backbones[0]

        # ---- AdapterGNN PEFT diagnostic hook ----
        # Only targets the paper-faithful ``AdapterGNNBlock``. The legacy ``AdapterGINBlock``
        # (lora_gin kind) is excluded because it includes an internal residual, making delta separation ambiguous.
        # Key convention: f"adapter/{bb_type}_{bb_idx}/{container}_{j}".
        self._adapter_blocks_meta: list[tuple[str, AdapterGNNBlock]] = []
        self._adapter_stats: dict[str, dict[str, list[Tensor]]] = {}
        for bb_idx, (bb_type, bb) in enumerate(zip(self._backbone_types, self.backbones)):
            for container_name in ("pre_adapters", "post_adapters", "adapter_blocks"):
                mlist = getattr(bb, container_name, None)
                if mlist is None:
                    continue
                for j, blk in enumerate(mlist):
                    if not isinstance(blk, AdapterGNNBlock):
                        continue
                    key = f"adapter/{bb_type}_{bb_idx}/{container_name}_{j}"
                    self._adapter_blocks_meta.append((key, blk))
                    blk.register_forward_hook(self._make_adapter_hook(key))

        # ---- LoRA PEFT diagnostics (weight-space ΔW) ----
        # peft's ``lora.Linear`` holds lora_A/lora_B ModuleDict — detected via duck-typing
        # (to preserve the backbone's lazy peft import design: peft is not imported
        # here). Unlike AdapterGNN's γ gate, LoRA has B zero-initialized, so ΔW=scaling·(B@A)
        # growing from 0 is itself the γ counterpart → measured directly from weights
        # in on_train_epoch_end (no forward hook needed, 1 GPU sync per epoch). molformer_lora / unimol2_lora
        # are both automatically captured.
        self._lora_modules_meta: list[tuple[str, nn.Module]] = []
        for bb_idx, (bb_type, bb) in enumerate(zip(self._backbone_types, self.backbones)):
            for mod_name, m in bb.named_modules():
                if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                    key = f"lora/{bb_type}_{bb_idx}/{mod_name}"
                    self._lora_modules_meta.append((key, m))

        # ---- fusion ----
        modality_dims_d: dict[str, int] = dict(modality_dims or {})
        if self._modality_keys:
            missing = [k for k in self._modality_keys if k not in modality_dims_d]
            if missing:
                raise ValueError(f"missing modality_dims: {missing}")

        # Validate consumed keys — must be present in embeddings (descriptors flow via backbone
        # X_d, so they are not a target for consumes).
        for k in self._consumed_keys:
            if k not in self._modality_keys:
                raise ValueError(
                    f"backbone consumes={k!r} not found in features.embeddings. "
                    f"Add the corresponding modality to features.embeddings in the recipe, or "
                    f"correct consumes. (descriptors flow via backbone X_d, so they are "
                    f"not a target for consumes.)"
                )

        # fusion modality input = modality keys excluding consumed.
        self._fusion_modality_keys: list[str] = [
            k for k in self._modality_keys if k not in self._consumed_keys
        ]
        n_fusion_inputs = len(backbone_out_dims) + len(self._fusion_modality_keys)
        if self._fusion_modality_keys and fusion_spec is None:
            raise ValueError(
                "There are auxiliary modalities but FusionSpec is None — need a spec for how to merge them"
            )
        if len(extra_backbone_specs) > 0 and fusion_spec is None:
            raise ValueError("FusionSpec is required if extra_backbone_specs is present")

        if fusion_spec is not None and n_fusion_inputs >= 2:
            input_dims = list(backbone_out_dims) + [
                modality_dims_d[k] for k in self._fusion_modality_keys
            ]
            fkwargs: dict[str, Any] = dict(fusion_spec.kwargs)
            fkwargs.setdefault("output_dim", fusion_output_dim)
            # group_routed fusion requires the same groups list as the head — auto-injected.
            # head's groups and fusion's groups axis must correspond 1:1 for group_idx to match.
            if fusion_spec.type == "group_routed":
                if groups is None:
                    raise ValueError(
                        "fusion.type='group_routed' requires groups based on head.routing. "
                        "Set head.type to either 'grouped' or 'grouped_routed'."
                    )
                fkwargs.setdefault("groups", list(groups))
            fusion_cls = FUSIONS.get(fusion_spec.type)
            self.fusion: nn.Module | None = fusion_cls(input_dims=input_dims, **fkwargs)
            head_input_dim = self.fusion.output_dim
        else:
            # Single backbone single input: bypass fusion.
            self.fusion = None
            head_input_dim = backbone_out_dims[0]
        self._fusion_fallback_type: str | None = (
            fusion_spec.fallback if fusion_spec is not None else None
        )
        self.fusion_spec: FusionSpec | None = fusion_spec
        self._last_alpha: Tensor | None = None  # sidecar: store alpha after encode()

        # ---- head ----
        head_cls = HEADS.get(head_spec.type)
        head_kwargs: dict[str, Any] = dict(head_spec.kwargs)
        if head_spec.type in ("grouped", "grouped_routed"):
            if not self.ordered_tasks or task_to_group is None or not groups:
                raise ValueError(
                    f"head_spec.type={head_spec.type!r} "
                    "requires all of ordered_tasks/task_to_group/groups"
                )
            self.head = head_cls(
                input_dim=head_input_dim,
                ordered_tasks=self.ordered_tasks,
                task_to_group=dict(task_to_group),
                groups=list(groups),
                **head_kwargs,
            )
        elif head_spec.type == "linear":
            self.head = head_cls(input_dim=head_input_dim, n_tasks=n_tasks, **head_kwargs)
        else:
            # Head to be registered later: assume generic signature (input_dim/n_tasks).
            self.head = head_cls(input_dim=head_input_dim, n_tasks=n_tasks, **head_kwargs)

        # ---- loss ----
        loss_cls = LOSSES.get(loss_spec.type)
        loss_kwargs: dict[str, Any] = dict(loss_spec.kwargs)
        self._use_gradnorm: bool = loss_spec.type == "mixed_task_gradnorm"
        self._use_pcgrad: bool = loss_spec.type == "mixed_task_uncertainty_pcgrad"
        if loss_spec.type == "mixed_task_uncertainty_pcgrad":
            # Kendall UW + PCGrad (manual_optimization). This branch takes priority — calls base subclass.
            self.loss_module = loss_cls(
                n_tasks=n_tasks,
                task_types=list(task_types),
                task_weights=task_weights,
                classification=loss_spec.classification,
                regression=loss_spec.regression,
                **loss_kwargs,
            )
            self.automatic_optimization = False
            # PCGrad shared params cache (filled in setup_pcgrad_params after configure_optimizers).
            self._pcgrad_shared_params_cache: list[nn.Parameter] | None = None
            self._pcgrad_shared_shapes_cache: list[torch.Size] | None = None
            if aux_loss_spec is not None and aux_loss_spec.weight > 0:
                raise NotImplementedError(
                    "mixed_task_uncertainty_pcgrad + aux_loss used simultaneously is not supported."
                )
        elif loss_spec.type == "mixed_task_uncertainty":
            self.loss_module = loss_cls(
                n_tasks=n_tasks,
                task_types=list(task_types),
                task_weights=task_weights,
                classification=loss_spec.classification,
                regression=loss_spec.regression,
                **loss_kwargs,
            )
        elif loss_spec.type == "mixed_task_gradnorm":
            self.loss_module = loss_cls(
                n_tasks=n_tasks,
                task_types=list(task_types),
                task_weights=task_weights,
                classification=loss_spec.classification,
                regression=loss_spec.regression,
                **loss_kwargs,
            )
            # The GradNorm algorithm separates main backward and w-aux backward into two stages,
            # so it cannot be expressed with Lightning automatic optimization.
            self.automatic_optimization = False
            # Initial L̄_k(0) — lazily filled on the first batch. To avoid host sync on clone,
            # registered as a buffer (DDP compatible).
            self.register_buffer(
                "gradnorm_L_init", torch.ones(n_tasks), persistent=True
            )
            # On resume, even if gradnorm_L_init is restored from the checkpoint, the set flag
            # being a python attribute caused it to reset to False, leading to a bug where it was overwritten on the first step.
            # Promoted to a BoolTensor buffer to persist together with state_dict.
            self.register_buffer(
                "_gradnorm_L_init_set",
                torch.zeros((), dtype=torch.bool),
                persistent=True,
            )
            # shared-layer parameters are resolved in setup() (after the backbone is fully constructed).
            self._gradnorm_shared_params_cache: list[Tensor] | None = None
            if aux_loss_spec is not None and aux_loss_spec.weight > 0:
                raise NotImplementedError(
                    "mixed_task_gradnorm + aux_loss used simultaneously is not supported."
                )
        elif loss_spec.type == "single_task":
            if n_tasks != 1 or len(task_types) != 1:
                raise ValueError(
                    f"single_task only supports n_tasks=1 (n_tasks={n_tasks}, "
                    f"task_types={task_types!r})"
                )
            self.loss_module = loss_cls(
                task_type=task_types[0],
                classification=loss_spec.classification,
                regression=loss_spec.regression,
                **loss_kwargs,
            )
        else:
            raise NotImplementedError(
                f"loss_spec.type={loss_spec.type!r} — Step 3.5/3.6 not supported"
            )

        # ---- per-modality aux loss (optional) ----
        # Attach a separate head + loss instance to each backbone's pre-fusion output H_i.
        # Diagnostic signal: if aux_loss/i is small, the task signal remains alive with modality i alone.
        # Disabled: aux_loss_spec is None or weight == 0.
        self.aux_loss_spec: AuxLossSpec | None = aux_loss_spec
        self.aux_heads: nn.ModuleList | None = None
        self.aux_loss_modules: nn.ModuleList | None = None
        self._last_H_list: list[Tensor] | None = None  # sidecar after encode()
        if aux_loss_spec is not None and aux_loss_spec.weight > 0:
            if aux_loss_spec.head_type != "linear":
                raise NotImplementedError(
                    f"aux_loss.head_type={aux_loss_spec.head_type!r} not supported "
                    f"(currently only 'linear')"
                )
            self.aux_heads = nn.ModuleList(
                [nn.Linear(d, n_tasks) for d in backbone_out_dims]
            )
            aux_loss_type = aux_loss_spec.loss_type or loss_spec.type
            if aux_loss_type != "mixed_task_uncertainty":
                raise NotImplementedError(
                    f"aux_loss.loss_type={aux_loss_type!r} not supported "
                    f"(currently only 'mixed_task_uncertainty')"
                )
            aux_loss_cls = LOSSES.get(aux_loss_type)
            self.aux_loss_modules = nn.ModuleList([
                aux_loss_cls(
                    n_tasks=n_tasks,
                    task_types=list(task_types),
                    task_weights=task_weights,
                    classification=loss_spec.classification,
                    regression=loss_spec.regression,
                )
                for _ in backbones
            ])

    # ---- core forward ----

    def encode(self, batch: Mapping[str, Any]) -> Tensor:
        """Batch dict → fused/raw representation Z.

        Flow:
            1. Remove consumed modality keys from modalities and move them to the ``__consumed__`` slot
               to prepare the augmented batch. Each backbone adapter reads its own consumes
               keys and converts them into input.
            2. Collect ``H_i`` in the order of primary + extra backbones.
            3. If fusion exists, combine ``[H_1, ..., H_K, *non_consumed_modalities]``.
            4. If there is no fusion, return the single backbone output.

        Args:
            batch: backbone input keys + (if present) ``modalities`` dict.

        Returns:
            (B, head_input_dim) tensor.
        """
        modalities: Mapping[str, Tensor] = batch.get("modalities") or {}
        if self._modality_keys:
            missing = [k for k in self._modality_keys if k not in modalities]
            if missing:
                raise KeyError(f"Missing keys in batch['modalities']: {missing}")

        # Inject the consumed slot into batch (adapter reads it as ``batch['__consumed__']``).
        if self._consumed_keys:
            consumed_payload = {k: modalities[k] for k in self._consumed_keys}
            aug_batch = dict(batch)
            existing = aug_batch.get("__consumed__")
            aug_batch["__consumed__"] = (
                {**existing, **consumed_payload} if existing else consumed_payload
            )
        else:
            aug_batch = batch

        H_list: list[Tensor] = []
        for bb_type, bb in zip(self._backbone_types, self.backbones):
            adapter = _INPUT_ADAPTERS[bb_type]
            H_list.append(adapter(bb, aug_batch))

        # aux loss / diagnostic sidecar — preserve per-modality output before fusion call.
        if self.aux_loss_modules is not None:
            self._last_H_list = list(H_list)

        if self.fusion is None:
            return H_list[0]

        # ---- per-sample modality presence mask (B, K) — block fusion alpha for missing modality ----
        # collate injects ``{modality_key: (B,) bool}`` into ``batch['modality_present']``.
        # presence of a backbone slot = AND of the modalities consumed by that backbone.
        # presence of a fusion modality slot = the present mask of that modality (all True if absent).
        modality_present: Mapping[str, Tensor] = batch.get("modality_present") or {}
        if modality_present and hasattr(self.fusion, "set_presence_mask"):
            B = H_list[0].size(0)
            device = H_list[0].device
            slot_presence: list[Tensor] = []
            for consumes in self._backbone_consumes:
                if not consumes:
                    slot_presence.append(torch.ones(B, dtype=torch.bool, device=device))
                else:
                    p = torch.ones(B, dtype=torch.bool, device=device)
                    for k in consumes:
                        if k in modality_present:
                            p = p & modality_present[k].to(device=device, dtype=torch.bool)
                    slot_presence.append(p)
            for k in self._fusion_modality_keys:
                if k in modality_present:
                    slot_presence.append(modality_present[k].to(device=device, dtype=torch.bool))
                else:
                    slot_presence.append(torch.ones(B, dtype=torch.bool, device=device))
            presence_mask = torch.stack(slot_presence, dim=1)  # (B, K)
            self.fusion.set_presence_mask(presence_mask)

        embeds = list(H_list) + [modalities[k] for k in self._fusion_modality_keys]
        z, alpha = self.fusion(embeds)
        # Store alpha sidecar (used in entropy reg / sigma coupling)
        if self.training and alpha is not None:
            self._last_alpha = alpha
        # alpha distribution is for training diagnostics. Future implementations where fusion does not return alpha
        # are fine since the forward signature has been unified.
        if self.training and alpha is not None and alpha.dim() == 2 and alpha.size(1) > 1:
            for i in range(alpha.size(1)):
                self.log(
                    f"alpha_{i}",
                    alpha[:, i].mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=alpha.size(0),
                )
            # Strengthen alpha diagnostic logging
            alpha_max_mean = alpha.max(dim=1)[0].mean()
            alpha_min_mean = alpha.min(dim=1)[0].mean()
            self.log(
                "alpha_max_mean",
                alpha_max_mean,
                on_step=False,
                on_epoch=True,
                batch_size=alpha.size(0),
            )
            self.log(
                "alpha_min_mean",
                alpha_min_mean,
                on_step=False,
                on_epoch=True,
                batch_size=alpha.size(0),
            )

        # GR-MF: log per-(group, modality) alpha. Applies only to fusion where group_alpha is exposed.
        # Key signal for checking whether robust_attention's collapse structurally disappears in GR-MF.
        group_alpha = getattr(self.fusion, "last_group_alpha", None)
        if (
            self.training
            and group_alpha is not None
            and group_alpha.dim() == 3
        ):
            G, K = group_alpha.size(1), group_alpha.size(2)
            groups_names = getattr(self.fusion, "groups", None)
            B_alpha = group_alpha.size(0)
            for g in range(G):
                gname = (
                    groups_names[g] if groups_names and g < len(groups_names) else f"g{g}"
                )
                row = group_alpha[:, g, :].mean(dim=0)  # (K,)
                for k in range(K):
                    self.log(
                        f"group_alpha/{gname}/m{k}",
                        row[k],
                        on_step=False,
                        on_epoch=True,
                        batch_size=B_alpha,
                    )
                self.log(
                    f"group_alpha_max/{gname}",
                    row.max(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=B_alpha,
                )
                self.log(
                    f"group_alpha_min/{gname}",
                    row.min(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=B_alpha,
                )
        return z

    def forward(self, batch: Mapping[str, Any]) -> Tensor:  # type: ignore[override]
        Z = self.encode(batch)
        return self.head(Z)

    # ---- loss dispatch ----

    def _compute_loss(
        self,
        preds: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        return self.loss_module(preds, targets, sample_weights)

    # ---- Lightning hooks ----

    def on_train_epoch_start(self) -> None:
        """Fusion temperature annealing + warmup toggle (RobustAttention) + head freeze."""
        # head warmup: freeze head for the first N epochs of training → ckpt selection
        # starts from the point where gate/adapter has been trained to some extent. backward flows through the head output
        # so the gradient reaches the fusion gate (only the head's own weight update is stopped).
        if self.head_warmup_epochs > 0:
            should_freeze = self.current_epoch < self.head_warmup_epochs
            if self._head_warmup_state != should_freeze:
                for p in self.head.parameters():
                    p.requires_grad = not should_freeze
                self._head_warmup_state = should_freeze
            self.log("train/head_frozen", float(should_freeze), on_epoch=True)
        if hasattr(self.fusion, "set_tau") and self.fusion_spec is not None:
            tau_init = self.fusion_spec.kwargs.get("tau_init", 2.0)
            tau_min = self.fusion_spec.kwargs.get("tau_min", 0.5)
            anneal_epochs = self.fusion_spec.kwargs.get("anneal_epochs", 50)
            # progress: 0 (epoch 0) → 1 (epoch >= anneal_epochs)
            progress = min(1.0, self.current_epoch / max(1, anneal_epochs))
            tau = tau_init - (tau_init - tau_min) * progress
            self.fusion.set_tau(tau)
            self.log("train/fusion_tau", tau)
        if hasattr(self.fusion, "set_warmup_active") and self.fusion_spec is not None:
            warmup_epochs = self.fusion_spec.kwargs.get("warmup_epochs", 0)
            active = self.current_epoch < warmup_epochs
            self.fusion.set_warmup_active(active)
            self.log("train/fusion_warmup", float(active))

    def _gate_kl_to_uniform(self) -> Tensor | None:
        """KL(α || uniform_K) regularization term for GR-MF group_alpha.

        Purpose:
            Since the static gate (group_routed) has no per-sample input dependency, the per-sample
            RGR dynamics are absent, but gradient-based concentration can sharpen α toward one-hot
            direction (as in Switch-Transformer / Aux-Loss-Free MoE's
            standard result). Via KL-to-uniform, the group-level mean α is weakly pulled near uniform
            so that it stays there.

        Return value is a scalar (B, n_groups averaged). If gate_kl_weight is absent or group_alpha is
        not exposed, then None.
        """
        spec = getattr(self, "fusion_spec", None)
        if spec is None:
            return None
        weight = float(getattr(spec, "gate_kl_weight", 0.0) or 0.0)
        if weight <= 0:
            return None
        alpha = getattr(self.fusion, "_last_group_alpha", None)
        if alpha is None or alpha.dim() != 3:
            return None
        K = alpha.size(-1)
        # KL(p || uniform_K) = Σ p · log(p · K) = -H(p) + log K.
        log_K = float(torch.tensor(float(K)).log())
        kl = (alpha * (alpha.clamp_min(1e-12).log() + log_K)).sum(dim=-1).mean()
        self.log("train/gate_kl", kl.detach(), on_step=False, on_epoch=True)
        return weight * kl

    def training_step(self, batch: Mapping[str, Any], batch_idx: int = 0) -> Tensor:
        if self._use_gradnorm:
            return self._gradnorm_training_step(batch, batch_idx)
        if self._use_pcgrad:
            return self._pcgrad_training_step(batch, batch_idx)
        preds = self.forward(batch)
        loss = self._compute_loss(
            preds, batch["targets"], batch.get("sample_weights")
        )

        # GR-MF gate load-balance: KL(group_alpha || uniform_K).
        kl_term = self._gate_kl_to_uniform()
        if kl_term is not None:
            loss = loss + kl_term

        # Entropy regularization (RobustAttention/UncertaintyAttention)
        if (
            getattr(self, "fusion_spec", None) is not None
            and self.fusion_spec.entropy_weight > 0
            and getattr(self, "_last_alpha", None) is not None
        ):
            alpha = self._last_alpha
            # H(alpha) = -sum(alpha * log(alpha))
            H = -(alpha * (alpha + 1e-8).log()).sum(dim=-1).mean()
            loss = loss - self.fusion_spec.entropy_weight * H  # maximize H
            self.log("train/fusion_entropy", H.detach())

        # Sigma coupling (UncertaintyAttention): loss_module.log_vars → set_sigma_signal
        if (
            hasattr(self.fusion, "set_sigma_signal")
            and hasattr(self.loss_module, "log_vars")
        ):
            sigma_mean = (
                self.loss_module.log_vars.detach().exp().sqrt().mean().item()
            )
            self.fusion.set_sigma_signal(sigma_mean)

        # Per-modality aux loss — diagnoses each backbone's standalone learning capacity + regularization.
        if self.aux_loss_modules is not None and self._last_H_list is not None:
            targets = batch["targets"]
            sw = batch.get("sample_weights")
            aux_total: Tensor | None = None
            for i, (aux_head, aux_mod, H_i) in enumerate(
                zip(self.aux_heads, self.aux_loss_modules, self._last_H_list)
            ):
                aux_pred = aux_head(H_i)
                aux_loss_i = aux_mod(aux_pred, targets, sw)
                self.log(
                    f"aux_loss/{i}",
                    aux_loss_i.detach(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=preds.size(0),
                )
                aux_total = aux_loss_i if aux_total is None else aux_total + aux_loss_i
            if aux_total is not None:
                loss = loss + self.aux_loss_spec.weight * aux_total
                self.log(
                    "aux_loss/total",
                    aux_total.detach(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=preds.size(0),
                )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=preds.size(0),
        )
        if hasattr(self.loss_module, "_last_task_loss"):
            tl = self.loss_module._last_task_loss.cpu().tolist()
            if self._task_loss_batch_sum is None:
                self._task_loss_batch_sum = [0.0] * self.n_tasks
            for i, v in enumerate(tl):
                self._task_loss_batch_sum[i] += v
            self._task_loss_batch_n += 1
        return loss

    def on_train_epoch_end(self) -> None:
        if self._task_loss_batch_sum is not None and self._task_loss_batch_n > 0:
            means = [s / self._task_loss_batch_n for s in self._task_loss_batch_sum]
            self._task_loss_epoch_means.append(means)
            self._task_loss_batch_sum = None
            self._task_loss_batch_n = 0
        if hasattr(self.loss_module, "log_vars") and self.ordered_tasks:
            sigmas = (self.loss_module.log_vars / 2).exp().detach().cpu()
            for i, name in enumerate(self.ordered_tasks):
                self.log(f"sigma/{name}", sigmas[i].item(), on_epoch=True)
        # GradNorm: logs per-task w (same interface as Kendall σ).
        if self._use_gradnorm and self.ordered_tasks:
            w_cpu = self.loss_module.w.detach().cpu()
            for i, name in enumerate(self.ordered_tasks):
                self.log(f"gradnorm_w/{name}", w_cpu[i].item(), on_epoch=True)
        # AdapterGNN PEFT diagnostics: (a) per-block γ + (b) delta RMS / rel-delta.
        #   gamma: signed scalar (paper init=0, should grow slowly to be normal).
        #   gamma_abs: allows tracking magnitude even if γ learns both negative/positive.
        #   delta_rms: ||s·A(x)||_RMS — the amount the adapter actually perturbs the backbone distribution.
        #   rel_delta: ||delta||_RMS / ||input||_RMS — perturbation ratio relative to input
        #     (standard PEFT diagnostic in LoRA literature — viewed together since γ absolute value alone is insensitive).
        for key, blk in self._adapter_blocks_meta:
            g = blk.gamma.detach().item()
            self.log(f"{key}/gamma", g, on_epoch=True)
            self.log(f"{key}/gamma_abs", abs(g), on_epoch=True)
            s = self._adapter_stats.get(key)
            if s and s["d_rms"]:
                d_mean = torch.stack(s["d_rms"]).mean().item()
                r_mean = torch.stack(s["rel"]).mean().item()
                self.log(f"{key}/delta_rms", d_mean, on_epoch=True)
                self.log(f"{key}/rel_delta", r_mean, on_epoch=True)
                s["d_rms"].clear()
                s["rel"].clear()
        # LoRA PEFT diagnostics: ΔW = scaling·(B@A). Since B is 0-init, the amount growing from 0 is
        #   the AdapterGNN γ counterpart.
        #   dW_norm: ‖ΔW‖_F — absolute magnitude by which LoRA has moved away from base.
        #   rel_dW:  ‖ΔW‖_F / ‖W_base‖_F — ratio relative to base weights (standard PEFT diagnostic).
        for key, m in self._lora_modules_meta:
            adapters = list(getattr(m, "active_adapters", None) or m.lora_A.keys())
            for a in adapters:
                A = m.lora_A[a].weight.detach()          # (r, in)
                B = m.lora_B[a].weight.detach()           # (out, r)
                scaling = float(m.scaling[a])
                dW = (B @ A) * scaling                     # (out, in)
                dW_norm = dW.norm().item()
                suffix = "" if a == "default" else f"_{a}"
                self.log(f"{key}/dW_norm{suffix}", dW_norm, on_epoch=True)
                base = getattr(getattr(m, "base_layer", m), "weight", None)
                if base is not None:
                    rel = dW_norm / (base.detach().norm().item() + 1e-12)
                    self.log(f"{key}/rel_dW{suffix}", rel, on_epoch=True)

    def _make_adapter_hook(self, key: str):
        """AdapterGNNBlock forward hook factory.

        During training (``self.training``), the hook accumulates the RMS of input/output every batch into a 0-d
        tensor. The ``.item()`` call is deferred until epoch end, limiting GPU sync to once per epoch
        (avoiding per-batch sync).
        """

        def hook(_mod: nn.Module, inputs: tuple, output: Tensor) -> None:
            if not self.training:
                return
            with torch.no_grad():
                x = inputs[0].detach()
                d = output.detach()
                x_rms = x.pow(2).mean().sqrt()
                d_rms = d.pow(2).mean().sqrt()
                s = self._adapter_stats.setdefault(key, {"d_rms": [], "rel": []})
                s["d_rms"].append(d_rms)
                s["rel"].append(d_rms / (x_rms + 1e-12))

        return hook

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int = 0) -> Tensor:
        preds = self.forward(batch)
        loss = self._compute_loss(
            preds, batch["targets"], batch.get("sample_weights")
        )
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=preds.size(0),
        )
        # Aux loss is logged for diagnostics only — not included in val_loss / early-stop signal.
        # Generalization of the actual training is determined by the main task head. aux tracks each backbone's standalone learning capacity.
        if self.aux_loss_modules is not None and self._last_H_list is not None:
            targets = batch["targets"]
            sw = batch.get("sample_weights")
            with torch.no_grad():
                for i, (aux_head, aux_mod, H_i) in enumerate(
                    zip(self.aux_heads, self.aux_loss_modules, self._last_H_list)
                ):
                    aux_pred = aux_head(H_i)
                    aux_val_i = aux_mod(aux_pred, targets, sw)
                    self.log(
                        f"aux_val_loss/{i}",
                        aux_val_i.detach(),
                        on_step=False,
                        on_epoch=True,
                        batch_size=preds.size(0),
                    )
        return loss

    # ---- GradNorm path ----

    def _resolve_gradnorm_shared_params(self) -> list[Tensor]:
        """Resolves the parameter target for GradNorm gradient norm measurement.

        Resolution rules:
            1) If ``loss_module.shared_layer_name`` is specified, via dotted path
               search the LitModule attribute. If the result is ``nn.Module``, use that module's
               ``weight`` only; if ``nn.Parameter``, use it as-is.
            2) If None, use the ``weight`` of the primary backbone's last ``nn.Linear``
               for auto-detect (general trunk → layer immediately before head, e.g. CheMeleon/AdapterGIN).

        Returns:
            list[Tensor] (requires_grad=True only). ``ValueError`` if all are length 0.
        """
        if self._gradnorm_shared_params_cache is not None:
            return self._gradnorm_shared_params_cache

        name: str | None = getattr(self.loss_module, "shared_layer_name", None)
        params: list[Tensor] = []

        if name is not None and name.strip():
            obj: Any = self
            for part in name.split("."):
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            if isinstance(obj, nn.Parameter):
                if obj.requires_grad:
                    params.append(obj)
            elif isinstance(obj, nn.Module):
                w = getattr(obj, "weight", None)
                if isinstance(w, nn.Parameter) and w.requires_grad:
                    params.append(w)
                else:
                    # If it is an arbitrary nn.Module — all trainable parameters within it.
                    params.extend(p for p in obj.parameters() if p.requires_grad)
            else:
                raise TypeError(
                    f"shared_layer={name!r} is neither nn.Module nor nn.Parameter: {type(obj)}"
                )
        else:
            # auto-detect: backbone's last nn.Linear.weight.
            last_linear: nn.Linear | None = None
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear) and m.weight.requires_grad:
                    last_linear = m
            if last_linear is None:
                raise ValueError(
                    "GradNorm shared_layer auto-detect failed: backbone has no trainable "
                    "nn.Linear. Please specify explicitly via loss.kwargs.shared_layer."
                )
            params = [last_linear.weight]

        if not params:
            raise ValueError(
                f"GradNorm shared_layer failed to expose a trainable parameter (name={name!r})."
            )
        self._gradnorm_shared_params_cache = params
        return params

    def _gradnorm_training_step(
        self, batch: Mapping[str, Any], batch_idx: int = 0
    ) -> Tensor:
        """Manual optimization: GradNorm two-stage backward + LR scheduler step.

        Lightning automatic_optimization=False mode. ``self.optimizers()`` is a
        (main_op
        """
        opt_main, opt_w = self.optimizers()
        sched = self.lr_schedulers()

        preds = self.forward(batch)
        targets = batch["targets"]
        sw = batch.get("sample_weights")

        # 1) per-task mean loss (preserve graph)
        L_k = self.loss_module.per_task_losses(preds, targets, sw)  # (K,)

        # 2) L̄_k(0): lazily filled at first step. Tasks with 0 are replaced with 1 (safe division).
        #    The set flag is a buffer, so if True on resume it is not reinitialized.
        if not bool(self._gradnorm_L_init_set):
            L_init_local = L_k.detach().clone()
            L_init_local = torch.where(
                L_init_local > 1e-8,
                L_init_local,
                torch.ones_like(L_init_local),
            )
            self.gradnorm_L_init.copy_(L_init_local)
            self._gradnorm_L_init_set.fill_(True)

        # 3) ||∇_W L̄_k||_2 (scalar constant). create_graph=False → blocks W gradient flow.
        shared_params = self._resolve_gradnorm_shared_params()
        K = self.n_tasks
        G_const_parts: list[Tensor] = []
        for k in range(K):
            grads = torch.autograd.grad(
                L_k[k],
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            sq_sum = torch.zeros((), device=L_k.device, dtype=L_k.dtype)
            for g in grads:
                if g is not None:
                    sq_sum = sq_sum + (g.detach() ** 2).sum()
            G_const_parts.append(sq_sum.clamp_min(1e-12).sqrt())
        G_const = torch.stack(G_const_parts)  # (K,) detached

        # 4) G_W^(k) = w_k · G_const_k. Differentiable only with respect to w.
        w = self.loss_module.w
        G_w = w * G_const  # (K,)

        # 5) target_k = mean_k G_w · r_k^α (no grad, no L_grad backward into w via target).
        with torch.no_grad():
            L_tilde = (L_k.detach() / self.gradnorm_L_init).clamp_min(1e-8)
            r = L_tilde / L_tilde.mean().clamp_min(1e-8)
            target = (G_w.detach().mean() * r.pow(self.loss_module.alpha)).detach()

        # 6) L_grad → updates only w.
        L_grad = (G_w - target).abs().sum()

        # 7) Main loss: remove w's main-backward contribution via w.detach().
        L_main = (w.detach() * L_k).sum()
        # GR-MF gate load-balance: KL(group_alpha || uniform_K). Summed only into L_main
        # (separated since L_grad only updates w). The KL graph is the fusion alpha from forward
        # connected as a node — backpropagated together in manual_backward(L_main).
        kl_term = self._gate_kl_to_uniform()
        if kl_term is not None:
            L_main = L_main + kl_term

        opt_main.zero_grad(set_to_none=True)
        opt_w.zero_grad(set_to_none=True)
        # Since L_main's backward consumes the L_k graph, L_grad goes first (retain_graph).
        # Since G_const is already detached, L_grad backward does not flow through to W.
        self.manual_backward(L_grad, retain_graph=True)
        self.manual_backward(L_main)
        opt_main.step()
        opt_w.step()

        # 8) w renormalize → sum=K + individual values hard-bounded to [0.3, 3.0].
        # Why bound: if one task monopolizes w (e.g. 4.5×), the LoRA delta overfits to that task
        # alone, and training on the remaining K-1 tasks stalls. The original Chen 2018 paper
        # has no bound, but divergence was observed with the LoRA + 22-task ADMET combination.
        # Order: clamp → renormalize → final clamp. Since renormalize can slightly
        # break the bound, clamp twice (sum=K is soft, individual bound is hard).
        with torch.no_grad():
            w.data.clamp_(min=0.3, max=3.0)
            w.data.mul_(K / w.data.sum().clamp_min(1e-8))
            w.data.clamp_(min=0.3, max=3.0)

        # 9) per-step LR scheduler (manual, since automatic_optimization=False).
        if sched is not None:
            if isinstance(sched, list):
                for s in sched:
                    s.step()
            else:
                sched.step()

        # 10) logging + diagnostic sidecar (compatible with the existing mixed_task_uncertainty path).
        bs = preds.size(0)
        self.log("train_loss", L_main.detach(), prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("gradnorm/aux_loss", L_grad.detach(), on_epoch=True, batch_size=bs)
        self.log("gradnorm/w_mean", w.detach().mean(), on_epoch=True, batch_size=bs)
        self.log("gradnorm/w_max", w.detach().max(), on_epoch=True, batch_size=bs)
        self.log("gradnorm/w_min", w.detach().min(), on_epoch=True, batch_size=bs)
        self.log("gradnorm/G_mean", G_const.detach().mean(), on_epoch=True, batch_size=bs)

        if hasattr(self.loss_module, "_last_task_loss"):
            tl = self.loss_module._last_task_loss.cpu().tolist()
            if self._task_loss_batch_sum is None:
                self._task_loss_batch_sum = [0.0] * self.n_tasks
            for i, v in enumerate(tl):
                self._task_loss_batch_sum[i] += v
            self._task_loss_batch_n += 1

        return L_main.detach()

    # ---- PCGrad path ----

    def _resolve_pcgrad_shared_params(self) -> tuple[list[nn.Parameter], list[torch.Size]]:
        """PCGrad projection target parameters + shape cache.

        scope:
            If ``loss_module.pcgrad_scope`` is ``"trunk"`` (default), the backbone +
            fusion + extra_backbones trainable parameters; if ``"all"``, includes up to
            the head (loss_module.log_vars is excluded — task-specific, so PCGrad is meaningless).

        Returns:
            (params, shapes) — for autograd.grad calls + PCGrad flatten/unflatten.
        """
        if self._pcgrad_shared_params_cache is not None:
            assert self._pcgrad_shared_shapes_cache is not None
            return self._pcgrad_shared_params_cache, self._pcgrad_shared_shapes_cache

        scope: str = getattr(self.loss_module, "pcgrad_scope", "trunk")
        params: list[nn.Parameter] = []

        if scope == "trunk":
            sources: list[nn.Module] = list(self.backbones)
            if self.fusion is not None:
                sources.append(self.fusion)
            for mod in sources:
                for p in mod.parameters():
                    if p.requires_grad:
                        params.append(p)
        elif scope == "all":
            # Includes head — excludes log_vars (task-specific).
            log_var_id = id(self.loss_module.log_vars)
            for p in self.parameters():
                if p.requires_grad and id(p) != log_var_id:
                    params.append(p)
        else:
            raise ValueError(f"pcgrad_scope={scope!r} — only 'trunk'|'all' supported")

        if not params:
            raise ValueError(
                f"PCGrad shared_params set is empty (scope={scope!r}). "
                "Check the requires_grad parameters of backbone/fusion."
            )

        shapes: list[torch.Size] = [p.shape for p in params]
        self._pcgrad_shared_params_cache = params
        self._pcgrad_shared_shapes_cache = shapes
        return params, shapes

    def _pcgrad_training_step(
        self, batch: Mapping[str, Any], batch_idx: int = 0
    ) -> Tensor:
        """Kendall UW + PCGrad manual_optimization.

        Flow:
            1) forward → preds.
            2) loss_module.weighted_per_task_losses → (K,) Kendall-weighted task loss
               + scalar log_var regularizer + (K,) task_has_data mask.
            3) For each task k, call autograd.grad(weighted_L_k, shared_params,
               retain_graph=True). Skip dead tasks.
            4) PCGrad projection → per-shared-param gradients.
            5) autograd.grad(total_loss, non_shared) → fills head / log_vars only.
               (Since head is downstream of trunk, it does not backpropagate into trunk →
               saves 1 trunk backward compared to a full backward. If DDP, manually
               all_reduce the non_shared gradients too.)
            6) Overwrite shared params .grad with the PCGrad result.
            7) opt.step() + scheduler.step().
        """
        from qadmet.modules.losses.pcgrad import (
            project_conflicting_gradients,
            cosine_conflict_rate,
        )

        opt = self.optimizers()
        sched = self.lr_schedulers()

        preds = self.forward(batch)
        targets = batch["targets"]
        sw = batch.get("sample_weights")

        # Kendall-weighted per-task loss + log_var reg (preserve graph).
        weighted_L_k, log_var_reg, task_has_data = self.loss_module.weighted_per_task_losses(
            preds, targets, sw
        )
        K = self.n_tasks

        shared_params, shapes = self._resolve_pcgrad_shared_params()
        eps = getattr(self.loss_module, "pcgrad_eps", 1e-12)

        # Per-task gradients — fill dead tasks with a 0 list.
        grads_per_task: list[list[Tensor | None]] = []
        valid_task: list[bool] = []
        has_data_cpu = task_has_data.detach().cpu().tolist()
        for k in range(K):
            if not has_data_cpu[k]:
                grads_per_task.append([None] * len(shared_params))
                valid_task.append(False)
                continue
            # autograd.grad: shared params only — head / log_vars not included.
            # Keep retain_graph=True for K calls. create_graph=False (2nd order not needed).
            grads_k = torch.autograd.grad(
                weighted_L_k[k],
                shared_params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            grads_per_task.append(list(grads_k))
            valid_task.append(True)

        # PCGrad projection (Yu et al. 2020).
        projected = project_conflicting_gradients(
            grads_per_task, shapes, valid_task=valid_task, eps=eps,
        )

        # DDP sync — since autograd.grad bypasses the DDP reducer, each rank has a
        # different PCGrad gradient (DistributedSampler provides a different batch per
        # rank). Synchronize via all_reduce(SUM)/world → MEAN to restore the DDP
        # assumption of opt.step() (all ranks have identical gradients). No-op in non-DDP.
        import torch.distributed as dist
        ddp_active = dist.is_available() and dist.is_initialized()
        world = float(dist.get_world_size()) if ddp_active else 1.0
        if ddp_active:
            for g in projected:
                if g is None:
                    continue
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g.div_(world)

        # Compute head / log_vars gradients only. Previously, manual_backward(total_loss)
        # backpropagated through the entire graph, filling even shared(trunk)'s .grad,
        # which was then overwritten with the PCGrad result → an entire trunk backward pass was discarded. autograd.grad(total_loss,
        # non_shared) only backpropagates up to head and does not enter trunk (since head is
        # downstream of trunk), so trunk backward is skipped entirely — per step trunk backward
        # goes from K+1 → K. Since autograd.grad bypasses the DDP reducer, synchronize the
        # non_shared gradients across ranks via manual all_reduce too, same as projected.
        opt.zero_grad(set_to_none=True)
        total_loss = weighted_L_k.sum() + log_var_reg
        shared_id_set = {id(p) for p in shared_params}
        non_shared = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in shared_id_set
        ]
        if non_shared:
            non_shared_grads = torch.autograd.grad(
                total_loss, non_shared, allow_unused=True, retain_graph=False,
            )
            if ddp_active:
                for g in non_shared_grads:
                    if g is None:
                        continue
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)
                    g.div_(world)
            for p, g in zip(non_shared, non_shared_grads):
                if g is not None:
                    p.grad = g.detach()

        # Overwrite shared params .grad with the PCGrad result.
        for p, g_new in zip(shared_params, projected):
            if g_new is None:
                continue
            # If the PCGrad gradient is 0, leave it as 0 — if left as None, the optimizer
            # will not update that parameter (the intent of set_to_none=True).
            p.grad = g_new.detach()

        opt.step()

        # per-step LR scheduler (manual, since automatic_optimization=False).
        if sched is not None:
            if isinstance(sched, list):
                for s in sched:
                    s.step()
            else:
                sched.step()

        # Diagnostic logging — cosine conflict rate / mean cosine. Signal of PCGrad working.
        bs = preds.size(0)
        try:
            conflict_rate, mean_cos = cosine_conflict_rate(
                grads_per_task, shapes, valid_task=valid_task, eps=eps,
            )
            self.log("pcgrad/conflict_rate", float(conflict_rate), on_epoch=True, batch_size=bs)
            self.log("pcgrad/mean_cosine", float(mean_cos), on_epoch=True, batch_size=bs)
        except Exception:
            # Diagnostic failures do not block training.
            pass

        self.log("train_loss", total_loss.detach(), prog_bar=True, on_epoch=True, batch_size=bs)

        if hasattr(self.loss_module, "_last_task_loss"):
            tl = self.loss_module._last_task_loss.cpu().tolist()
            if self._task_loss_batch_sum is None:
                self._task_loss_batch_sum = [0.0] * self.n_tasks
            for i, v in enumerate(tl):
                self._task_loss_batch_sum[i] += v
            self._task_loss_batch_n += 1

        return total_loss.detach()

    def configure_optimizers(self):  # type: ignore[override]
        """Adam + Noam warmup schedule (equivalent to the old chemprop MPNN.configure_optimizers).

        Non-LoRA path: Adam(init_lr) + NoamLike LR scheduler (per-step).
            Linear rise from init_lr → lr(max_lr) over warmup_epochs,
            then linear decay from lr → final_lr.

        LoRA path (``lr_lora`` set): AdamW(wd=0) + differential lr, no schedule.
            Equivalent to the old ``run_stl_molformer_lora.configure_optimizers``.
            The "lora_" match assumes PEFT standard naming (``lora_A``, ``lora_B``).

        Note:
            PyTorch automatically ignores parameters with requires_grad=False.
        """
        from chemprop.schedulers import build_NoamLike_LRSched

        # GradNorm: main opt (everything else) + aux opt (loss_module.w only).
        # In Lightning manual_optimization mode, self.optimizers() returns two.
        if self._use_gradnorm:
            w_param: nn.Parameter = self.loss_module.w
            w_id = id(w_param)
            main_params: list[nn.Parameter] = [
                p for p in self.parameters()
                if p.requires_grad and id(p) != w_id
            ]
            main_opt = torch.optim.Adam(main_params, lr=self.init_lr)
            # GradNorm w's learning rate is the same as main (paper recommendation; renormalize absorbs the scale).
            w_opt = torch.optim.Adam([w_param], lr=self.lr)
            steps_per_epoch = self.trainer.num_training_batches
            warmup_steps = self.warmup_epochs * steps_per_epoch
            cooldown_steps = (self.trainer.max_epochs - self.warmup_epochs) * steps_per_epoch
            main_sched = build_NoamLike_LRSched(
                main_opt, warmup_steps, cooldown_steps,
                self.init_lr, self.lr, self.final_lr,
            )
            return (
                [main_opt, w_opt],
                [{"scheduler": main_sched, "interval": "step"}],
            )

        if self.lr_lora is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
            steps_per_epoch = self.trainer.num_training_batches
            warmup_steps = self.warmup_epochs * steps_per_epoch
            cooldown_steps = (self.trainer.max_epochs - self.warmup_epochs) * steps_per_epoch
            lr_sched = build_NoamLike_LRSched(
                optimizer, warmup_steps, cooldown_steps,
                self.init_lr, self.lr, self.final_lr,
            )
            if self._use_pcgrad:
                # manual_optimization mode: scheduler is returned as a list → self.lr_schedulers()
                # returns a single object → .step() is called manually in _pcgrad_training_step.
                return ([optimizer], [{"scheduler": lr_sched, "interval": "step"}])
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

        # Toggle for baseline comparison — if `QADMET_LEGACY_WD=1`, restores the old path (wd=0 for all groups).
        # Enabled for baseline training when validating the 4-modality overfit defense package.
        import os
        if os.environ.get("QADMET_LEGACY_WD") == "1":
            lora_params: list = []
            gate_params: list = []
            fusion_params: list = []
            other_params: list = []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if "lora_" in name:
                    lora_params.append(p)
                elif name == "fusion.group_gate":
                    gate_params.append(p)
                elif self.lr_fusion is not None and name.startswith("fusion."):
                    fusion_params.append(p)
                else:
                    other_params.append(p)
            if not lora_params:
                raise ValueError(
                    "lr_lora was set but no 'lora_' parameters were found."
                )
            legacy_groups: list[dict] = [
                {"params": lora_params, "lr": self.lr_lora, "weight_decay": 0.0},
                {"params": other_params, "lr": self.lr, "weight_decay": 0.0},
            ]
            if gate_params:
                legacy_groups.append(
                    {"params": gate_params, "lr": self.lr, "weight_decay": 0.0}
                )
            if self.lr_fusion is not None:
                if not fusion_params:
                    raise ValueError(
                        "lr_fusion was set but no 'fusion.*' parameters were found."
                    )
                legacy_groups.append(
                    {"params": fusion_params, "lr": self.lr_fusion, "weight_decay": 0.0}
                )
            return torch.optim.AdamW(legacy_groups)

        # Norm-type parameters use wd=0 (standard convention). Identified by id (avoids prefix-matching instability).
        nodecay_ids: set[int] = set()
        for m in self.modules():
            if isinstance(
                m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
            ):
                for p in m.parameters(recurse=False):
                    nodecay_ids.add(id(p))

        lora_params: list = []
        head_params: list = []
        gate_params: list = []
        fusion_params: list = []
        nodecay_params: list = []
        other_params: list = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(p)
                continue
            is_nodecay = (
                id(p) in nodecay_ids
                or name.endswith(".bias")
                or name.endswith("log_vars")
                or name.endswith("log_precision")
            )
            if is_nodecay:
                nodecay_params.append(p)
            elif name == "fusion.group_gate":
                # GR-MF's group-routing gate (24 scalars) only, as a separate group.
                # lr_fusion (conservative value for modality_projs / cross_modal / output_proj)
                # training is too slow this way (α spread 0.06pp over 7 epochs in v2).
                # head lr equal to (self.lr) for acceleration, wd=0 (24 scalars don't need decay).
                gate_params.append(p)
            elif name.startswith("head."):
                head_params.append(p)
            elif self.lr_fusion is not None and name.startswith("fusion."):
                fusion_params.append(p)
            else:
                other_params.append(p)
        if not lora_params:
            raise ValueError(
                "lr_lora is set but could not find 'lora_' parameter. "
                "Check whether the backbone (molformer_lora / unimol2_lora) has LoRA enabled."
            )
        # WD policy (overfit prevention, based on 4-modality recipe):
        #   head=0.05 (prevent overfit on small-data tasks)
        #   fusion=0.01 / general backbone(adaptor etc.)=0.01
        #   LoRA=0 (PEFT standard), Norm/bias/log_vars/log_precision=0
        groups: list[dict] = [
            {"params": lora_params, "lr": self.lr_lora, "weight_decay": 0.0},
        ]
        if head_params:
            groups.append({"params": head_params, "lr": self.lr, "weight_decay": 0.10})
        if gate_params:
            groups.append({"params": gate_params, "lr": self.lr, "weight_decay": 0.0})
        if other_params:
            groups.append({"params": other_params, "lr": self.lr, "weight_decay": 0.01})
        if nodecay_params:
            groups.append({"params": nodecay_params, "lr": self.lr, "weight_decay": 0.0})
        if self.lr_fusion is not None:
            if not fusion_params:
                raise ValueError(
                    "lr_fusion is set but could not find 'fusion.*' parameter."
                )
            groups.append(
                {"params": fusion_params, "lr": self.lr_fusion, "weight_decay": 0.01}
            )
        elif fusion_params:
            # if lr_fusion is not set, fusion is also merged into the general lr (wd=0.01).
            groups.append({"params": fusion_params, "lr": self.lr, "weight_decay": 0.01})
        return torch.optim.AdamW(groups)
