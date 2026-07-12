"""MultitaskLitModule — Spec 기반 다중-task LightningModule.

조립 흐름 (Step B 디자인, 2026-05-01):
    1. ``BackboneSpec`` → BACKBONES.build → 1차 표현 ``H (B, D_main)``.
       DMPNN 의 경우 ``batch["X_d"]`` (descriptor stack) 가 chemprop ``MPNN.fingerprint``
       내부에서 backbone 출력 뒤에 concat 되어 ``D_main = d_h + d_xd`` 가 된다.
    2. ``FeatureSpec.embeddings`` 순서로 (consumed 키 제외) modality 벡터를 받아
       ``[H_1, ..., H_K, *embeddings]`` list 를 구성한다. **descriptor 는 fusion 에
       들어가지 않는다** — 이미 backbone 의 X_d concat 으로 흡수됨.
    3. ``FusionSpec`` 이 있으면 FUSIONS.build 로 fusion module 생성, 없으면
       backbone 출력만 그대로 사용 (단일 backbone + descriptors 만 쓰는 case).
    4. ``HeadSpec`` → HEADS.build → ``preds (B, n_tasks)``.
    5. ``LossSpec`` → LOSSES.build → 학습/검증 loss 계산.

옛 ``GroupedTaskUncertaintyMPNN._encode`` (pipeline_mtl.py:1056-1129) 의 4 분기
(LoRA / frozen MoLFormer / frozen UniMol2 / legacy concat) 를 spec + adapter 로
일반화한 형태이다. 단일 trainable backbone + frozen 보조 모달리티 조합을 다룬다.

배치 dict 계약:
    backbone-specific 키 + ``modalities`` (= embeddings 만) + ``targets`` + (옵션)
    ``sample_weights``. backbone 별 입력은 ``_INPUT_ADAPTERS`` 가 dispatch 한다.

    예) DMPNN backbone (descriptors=[rdkit2d, xtb], embeddings=[molformer])::

        {
            "bmg": <chemprop.BatchMolGraph>,
            "V_d": <Tensor> | None,
            "X_d": (B, 204),  # rdkit2d(200) + xtb(4) — fingerprint 가 H 뒤에 concat.
            "modalities": {"molformer": (B, 768)},  # fusion 에만 들어감.
            "targets": (B, n_tasks),
            "sample_weights": (B,),
        }

    예) MoLFormer-LoRA backbone::

        {
            "token_ids": (B, L),
            "attention_mask": (B, L),
            "modalities": {},
            "targets": (B, n_tasks),
        }

Lazy import:
    ``lightning`` 은 본 모듈 top-level 에서 import 한다 (LightningModule 상속 필수).
    그러나 ``qadmet.modules`` ``__init__`` 은 본 sub-package 를 import 하지 않으므로
    ``qadmet.modules`` 자체 import 는 lightning 을 끌어들이지 않는다.
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

# ---------------------------------------------------------------------------- 입력 adapter

#: backbone forward 호출 어댑터: ``(backbone_module, batch_dict) -> Tensor (B, D)``.
BackboneAdapter = Callable[[nn.Module, Mapping[str, Any]], Tensor]

_INPUT_ADAPTERS: dict[str, BackboneAdapter] = {}


def register_input_adapter(backbone_type: str) -> Callable[[BackboneAdapter], BackboneAdapter]:
    """backbone type → adapter 등록 데코레이터.

    Why:
        backbone 별 forward 시그니처가 다르다 (DMPNN: bmg/V_d/X_d, MoLFormer-LoRA:
        token_ids/attention_mask, UniMol2-LoRA: vendored batch dict). LitModule 본체에
        분기 로직을 두지 않고 외부에서 등록 가능하게 분리한다.

    Returns:
        등록 데코레이터. 동일 키 재등록은 ``KeyError``.
    """

    def deco(fn: BackboneAdapter) -> BackboneAdapter:
        if backbone_type in _INPUT_ADAPTERS:
            raise KeyError(f"input adapter 가 이미 등록됨: {backbone_type!r}")
        _INPUT_ADAPTERS[backbone_type] = fn
        return fn

    return deco


@register_input_adapter("dmpnn")
def _dmpnn_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    return backbone(batch["bmg"], V_d=batch.get("V_d"), X_d=batch.get("X_d"))


@register_input_adapter("molformer_lora")
def _molformer_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """MoLFormer-LoRA 입력 — STL 은 batch 최상위 키, multi-backbone MTL 은 consumed slot."""
    if "token_ids" in batch:
        return backbone(batch["token_ids"], batch["attention_mask"])
    consumed: Mapping[str, Tensor] = batch.get("__consumed__") or {}
    if "molformer_tokens" in consumed:
        mt = consumed["molformer_tokens"]
        # mt: (B, 2L) float — 앞 L = token_ids, 뒤 L = attention_mask. backbone 은 long 기대.
        L = mt.shape[1] // 2
        return backbone(mt[:, :L].long(), mt[:, L:].long())
    raise KeyError(
        "molformer_lora adapter: batch['token_ids'] 또는 "
        "consumed['molformer_tokens'] 중 하나가 필요합니다."
    )


@register_input_adapter("unimol2_lora")
def _unimol2_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """STL: ``batch['unimol2_batch']``. MTL multi-backbone: ``__consumed__`` 슬롯에서 dict 회수."""
    if "unimol2_batch" in batch:
        return backbone(**batch["unimol2_batch"])
    consumed: Mapping[str, Any] = batch.get("__consumed__") or {}
    for k, payload in consumed.items():
        if isinstance(payload, dict):
            return backbone(**payload)
    raise KeyError(
        "unimol2_lora adapter: batch['unimol2_batch'] 또는 "
        "consumed[<unimol2_lmdb-like>] dict 중 하나가 필요합니다."
    )


@register_input_adapter("chemeleon")
def _chemeleon_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    return backbone(batch["bmg"], V_d=batch.get("V_d"), X_d=batch.get("X_d"))


@register_input_adapter("graphmvp")
def _graphmvp_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """STL: ``batch['graphmvp_data']``. MTL multi-backbone: ``__consumed__`` 슬롯에서 회수.

    payload 는 ``torch_geometric.data.Batch`` (collate 가 SMILES → PyG Data → Batch).
    Lightning ``move_data_to_device`` 는 dict/Tensor 만 인식하므로 PyG Batch 는 본
    어댑터에서 backbone 파라미터 device 로 직접 이동.
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
            "graphmvp adapter: batch['graphmvp_data'] 또는 "
            "consumed['graphmvp_data'] 중 하나가 필요합니다."
        )
    try:
        device = next(p.device for p in backbone.parameters() if p.requires_grad)
    except StopIteration:
        device = next(backbone.parameters()).device
    pyg_batch = pyg_batch.to(device)
    return backbone(pyg_batch)


@register_input_adapter("mlp_embedding")
def _mlp_embedding_adapter(backbone: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """사전계산 embedding passthrough — MTL multi-backbone 은 ``__consumed__`` 슬롯,
    STL 은 단일 modality 텐서를 최상위에서 회수.

    consumed payload 는 ``(B, d)`` float Tensor (modalities dict 에서 빠진 embedding).
    molformer_lora 어댑터와 같은 ``__consumed__`` 처리지만 token 이 아닌 plain tensor.
    """
    consumed: Mapping[str, Any] = batch.get("__consumed__") or {}
    for payload in consumed.values():
        if isinstance(payload, Tensor):
            return backbone(payload)
    # STL fallback: modalities 에 단일 텐서 modality 만 있는 경우.
    modalities: Mapping[str, Any] = batch.get("modalities") or {}
    tensors = [v for v in modalities.values() if isinstance(v, Tensor)]
    if len(tensors) == 1:
        return backbone(tensors[0])
    raise KeyError(
        "mlp_embedding adapter: consumed[<embedding>] tensor 또는 단일 modality "
        "텐서가 필요합니다."
    )


# ---------------------------------------------------------------------------- LitModule


class MultitaskLitModule(L.LightningModule):
    """Spec 기반 multi-task LightningModule.

    Args:
        backbone_spec: BackboneSpec — primary trainable trunk.
        head_spec: HeadSpec — ``linear`` | ``grouped`` 등.
        loss_spec: LossSpec — ``mixed_task_uncertainty`` | ``kendall_uncertainty``.
        n_tasks: 출력 task 수.
        task_types: 길이 ``n_tasks`` 의 ``"regression"`` / ``"classification"`` 라벨.
        fusion_spec: FusionSpec | None. None 이면 보조 모달리티/extra backbone 없이
            primary backbone 출력 직접 사용.
        extra_backbone_specs: 추가 trainable trunk 들. 각 BackboneSpec 의
            ``consumes`` 가 비어있지 않으면 해당 modality 키들은 fusion modality
            stack 에서 제외되고 backbone 어댑터의 ``batch['__consumed__']`` 로
            전달된다 (옛 phase1.4a — D-MPNN + LoRA-MoLFormer).
        ordered_tasks: HeadSpec.type=="grouped" 또는 lineage 메타에 사용할 task 이름 순서.
        modality_dims: 보조 모달리티 이름 → 차원. **embeddings 키만** 포함하면 된다
            (descriptors 는 backbone X_d 로 흐르므로 modalities dict 에 들어오지 않음).
            consumed 키도 embeddings 안에 있어야 한다 (LoRA-MoLFormer 의 molformer_tokens).
        descriptors / embeddings: FeatureSpec 의 두 필드를 그대로 받는다.
            descriptors 는 backbone (DMPNN) 의 X_d 로 직접 흐르고 fusion 입력에는 빠진다.
            embeddings 만 fusion 입력에 합류하며 consumed 키는 추가 backbone 으로 우회된다.
        task_to_group / groups: GroupedHead 추가 인자.
        task_weights: (n_tasks,) inverse-freq weight tensor (옵션).
        fusion_output_dim: FusionSpec.kwargs 에 ``output_dim`` 이 없을 때 fallback.
        lr: Noam schedule 의 max_lr (head/fusion/non-LoRA params).
        init_lr: Noam warmup 시작 lr.
        final_lr: Noam cooldown 종료 lr.
        warmup_epochs: Noam warmup 기간 (epoch 수).
        lr_lora: 비-None 이면 ``"lora_"`` 가 이름에 포함된 파라미터만 별도 그룹으로
            묶어 ``lr_lora`` 적용 (옛 ``run_stl_*_lora.configure_optimizers`` 동등).
            None 이면 단일 lr 사용.

    Note:
        본 클래스는 학습 데이터 파이프라인(Step 3.6) 와 무관하게 **모듈 조립**만
        담당한다. forward 입력은 collate 함수가 만든 dict 라고 가정하며 backbone
        type 별 adapter 가 dispatch 한다.
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
        # head freeze 가 한 번이라도 활성화되면 unfreeze 후 trainable 상태를
        # epoch 마다 set 하지 않아도 되도록 plain bool 캐시. configure_optimizers 가
        # 호출되는 시점에 head 가 freeze 되어 있어도 optimizer 가 param 을 잡고
        # 있으므로 unfreeze 후 자동으로 학습됨 (requires_grad 만 토글).
        self._head_warmup_state: bool | None = None
        self.n_tasks = n_tasks
        # 진단용: 에폭별 per-task 평균 loss 누적
        self._task_loss_epoch_means: list[list[float]] = []
        self._task_loss_batch_sum: list[float] | None = None
        self._task_loss_batch_n: int = 0
        self.task_types: list[str] = list(task_types)
        self.ordered_tasks: list[str] | None = (
            list(ordered_tasks) if ordered_tasks is not None else None
        )
        self.descriptors: tuple[str, ...] = tuple(descriptors)
        self.embeddings: tuple[str, ...] = tuple(embeddings)
        # descriptors 는 backbone X_d 로 직접 흐르므로 modality dict 에 들어오지 않는다.
        # consumes 검증 / fusion modality 입력은 모두 embeddings 만 대상.
        self._modality_keys: list[str] = list(self.embeddings)
        # ---- backbones (primary + extras) ----
        all_specs: tuple[BackboneSpec, ...] = (backbone_spec, *extra_backbone_specs)
        consumed_keys: list[str] = []
        for s in all_specs:
            if s.type not in _INPUT_ADAPTERS:
                raise KeyError(
                    f"backbone type {s.type!r} 에 대한 input adapter 미등록. "
                    f"등록된 키: {sorted(_INPUT_ADAPTERS)}"
                )
            for k in s.consumes:
                if k in consumed_keys:
                    raise ValueError(
                        f"modality {k!r} 가 두 backbone 에서 동시에 consumed — "
                        f"한 modality 는 하나의 backbone 만 소비 가능."
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
                    f"backbone {s.type!r} 가 ``output_dim: int`` 를 노출해야 함 "
                    f"(받음: {d!r})"
                )
            backbones.append(bb)
            backbone_out_dims.append(d)
        # 단일 backbone 호환을 위해 self.backbone 은 primary 로 노출 + ModuleList 도 보유.
        self.backbones: nn.ModuleList = nn.ModuleList(backbones)
        self.backbone: nn.Module = backbones[0]

        # ---- AdapterGNN PEFT 진단 hook ----
        # paper-faithful ``AdapterGNNBlock`` 만 대상. legacy ``AdapterGINBlock``
        # (lora_gin kind) 는 internal residual 을 포함해서 delta 분리가 모호하므로 제외.
        # key 규약: f"adapter/{bb_type}_{bb_idx}/{container}_{j}".
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

        # ---- LoRA PEFT 진단 (weight-space ΔW) ----
        # peft ``lora.Linear`` 은 lora_A/lora_B ModuleDict 를 보유 — duck-typing 으로
        # 발견한다 (backbone 의 lazy peft import 설계 유지: 여기서 peft 를 import 하지
        # 않음). AdapterGNN 의 γ 게이트와 달리 LoRA 는 B 가 0-init 이라 ΔW=scaling·(B@A)
        # 가 0 에서 자라는 양 자체가 γ 대응물 → on_train_epoch_end 에서 weight 로 직접
        # 측정 (forward hook 불필요, GPU sync epoch 당 1회). molformer_lora / unimol2_lora
        # 모두 자동 포착.
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
                raise ValueError(f"modality_dims 누락: {missing}")

        # consumed 키 검증 — 반드시 embeddings 안에 있어야 함 (descriptors 는 backbone
        # X_d 로 흐르므로 consumes 대상이 아님).
        for k in self._consumed_keys:
            if k not in self._modality_keys:
                raise ValueError(
                    f"backbone consumes={k!r} 가 features.embeddings 에 없음. "
                    f"recipe 의 features.embeddings 에 해당 modality 를 추가하거나 "
                    f"consumes 를 정정하세요. (descriptors 는 backbone X_d 로 흐르므로 "
                    f"consumes 대상이 아닙니다.)"
                )

        # fusion modality 입력 = consumed 제외한 modality 키들.
        self._fusion_modality_keys: list[str] = [
            k for k in self._modality_keys if k not in self._consumed_keys
        ]
        n_fusion_inputs = len(backbone_out_dims) + len(self._fusion_modality_keys)
        if self._fusion_modality_keys and fusion_spec is None:
            raise ValueError(
                "보조 모달리티가 있는데 FusionSpec 이 None — 어떻게 합칠지 명세 필요"
            )
        if len(extra_backbone_specs) > 0 and fusion_spec is None:
            raise ValueError("extra_backbone_specs 가 있으면 FusionSpec 필수")

        if fusion_spec is not None and n_fusion_inputs >= 2:
            input_dims = list(backbone_out_dims) + [
                modality_dims_d[k] for k in self._fusion_modality_keys
            ]
            fkwargs: dict[str, Any] = dict(fusion_spec.kwargs)
            fkwargs.setdefault("output_dim", fusion_output_dim)
            # group_routed fusion 은 head 와 동일한 groups list 가 필요 — 자동 주입.
            # head 의 groups 와 fusion 의 groups axis 가 1:1 대응해야 group_idx 일치.
            if fusion_spec.type == "group_routed":
                if groups is None:
                    raise ValueError(
                        "fusion.type='group_routed' 는 head.routing 기반 groups 필수. "
                        "head.type 을 'grouped' 또는 'grouped_routed' 로 두세요."
                    )
                fkwargs.setdefault("groups", list(groups))
            fusion_cls = FUSIONS.get(fusion_spec.type)
            self.fusion: nn.Module | None = fusion_cls(input_dims=input_dims, **fkwargs)
            head_input_dim = self.fusion.output_dim
        else:
            # 단일 backbone 단일 입력: fusion 우회.
            self.fusion = None
            head_input_dim = backbone_out_dims[0]
        self._fusion_fallback_type: str | None = (
            fusion_spec.fallback if fusion_spec is not None else None
        )
        self.fusion_spec: FusionSpec | None = fusion_spec
        self._last_alpha: Tensor | None = None  # sidecar: encode() 후 alpha 저장

        # ---- head ----
        head_cls = HEADS.get(head_spec.type)
        head_kwargs: dict[str, Any] = dict(head_spec.kwargs)
        if head_spec.type in ("grouped", "grouped_routed"):
            if not self.ordered_tasks or task_to_group is None or not groups:
                raise ValueError(
                    f"head_spec.type={head_spec.type!r} 는 "
                    "ordered_tasks/task_to_group/groups 모두 필요"
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
            # 추후 등록될 head: 일반 시그니처(input_dim/n_tasks) 가정.
            self.head = head_cls(input_dim=head_input_dim, n_tasks=n_tasks, **head_kwargs)

        # ---- loss ----
        loss_cls = LOSSES.get(loss_spec.type)
        loss_kwargs: dict[str, Any] = dict(loss_spec.kwargs)
        self._use_gradnorm: bool = loss_spec.type == "mixed_task_gradnorm"
        self._use_pcgrad: bool = loss_spec.type == "mixed_task_uncertainty_pcgrad"
        if loss_spec.type == "mixed_task_uncertainty_pcgrad":
            # Kendall UW + PCGrad (manual_optimization). 본 분기 우선 — base subclass 호출.
            self.loss_module = loss_cls(
                n_tasks=n_tasks,
                task_types=list(task_types),
                task_weights=task_weights,
                classification=loss_spec.classification,
                regression=loss_spec.regression,
                **loss_kwargs,
            )
            self.automatic_optimization = False
            # PCGrad shared params 캐시 (configure_optimizers 후 setup_pcgrad_params 에서 채움).
            self._pcgrad_shared_params_cache: list[nn.Parameter] | None = None
            self._pcgrad_shared_shapes_cache: list[torch.Size] | None = None
            if aux_loss_spec is not None and aux_loss_spec.weight > 0:
                raise NotImplementedError(
                    "mixed_task_uncertainty_pcgrad + aux_loss 동시 사용 미지원."
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
            # GradNorm 알고리즘은 main backward 와 w-aux backward 두 단계가 분리되어
            # 있으므로 Lightning automatic optimization 으로는 표현이 불가능.
            self.automatic_optimization = False
            # 초기 L̄_k(0) — 첫 batch 에서 lazy 채움. clone 시 host 동기화를 피하기 위해
            # buffer 로 등록 (DDP 호환).
            self.register_buffer(
                "gradnorm_L_init", torch.ones(n_tasks), persistent=True
            )
            # resume 시 gradnorm_L_init 이 checkpoint 에서 복원되어도 set 플래그가
            # python attribute 면 False 로 초기화되어 첫 step 에 덮어쓰이는 버그가 있었음.
            # BoolTensor buffer 로 승격하여 state_dict 와 함께 영속화.
            self.register_buffer(
                "_gradnorm_L_init_set",
                torch.zeros((), dtype=torch.bool),
                persistent=True,
            )
            # shared-layer 파라미터는 setup() 시 resolve (backbone 이 다 만들어진 뒤).
            self._gradnorm_shared_params_cache: list[Tensor] | None = None
            if aux_loss_spec is not None and aux_loss_spec.weight > 0:
                raise NotImplementedError(
                    "mixed_task_gradnorm + aux_loss 동시 사용 미지원."
                )
        elif loss_spec.type == "single_task":
            if n_tasks != 1 or len(task_types) != 1:
                raise ValueError(
                    f"single_task 는 n_tasks=1 만 지원 (n_tasks={n_tasks}, "
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
                f"loss_spec.type={loss_spec.type!r} — Step 3.5/3.6 미지원"
            )

        # ---- per-modality aux loss (선택) ----
        # 각 backbone 의 pre-fusion 출력 H_i 에 별도 head + loss 인스턴스 부착.
        # 진단 신호: aux_loss/i 가 작으면 modality i 단독으로 task 신호 살아있음.
        # 비활성: aux_loss_spec is None or weight == 0.
        self.aux_loss_spec: AuxLossSpec | None = aux_loss_spec
        self.aux_heads: nn.ModuleList | None = None
        self.aux_loss_modules: nn.ModuleList | None = None
        self._last_H_list: list[Tensor] | None = None  # encode() 후 sidecar
        if aux_loss_spec is not None and aux_loss_spec.weight > 0:
            if aux_loss_spec.head_type != "linear":
                raise NotImplementedError(
                    f"aux_loss.head_type={aux_loss_spec.head_type!r} 미지원 "
                    f"(현재 'linear' 만)"
                )
            self.aux_heads = nn.ModuleList(
                [nn.Linear(d, n_tasks) for d in backbone_out_dims]
            )
            aux_loss_type = aux_loss_spec.loss_type or loss_spec.type
            if aux_loss_type != "mixed_task_uncertainty":
                raise NotImplementedError(
                    f"aux_loss.loss_type={aux_loss_type!r} 미지원 "
                    f"(현재 'mixed_task_uncertainty' 만)"
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
        """배치 dict → fused/raw representation Z.

        흐름:
            1. consumed modality 키들을 modalities 에서 빼서 ``__consumed__`` slot
               으로 옮긴 augmented batch 준비. 각 backbone 어댑터가 자기 consumes
               키를 읽어 입력으로 변환한다.
            2. primary + extra backbone 들 순서로 ``H_i`` 수집.
            3. fusion 이 있으면 ``[H_1, ..., H_K, *non_consumed_modalities]`` 결합.
            4. fusion 없으면 단일 backbone 출력 반환.

        Args:
            batch: backbone 입력 키 + (있다면) ``modalities`` dict.

        Returns:
            (B, head_input_dim) tensor.
        """
        modalities: Mapping[str, Tensor] = batch.get("modalities") or {}
        if self._modality_keys:
            missing = [k for k in self._modality_keys if k not in modalities]
            if missing:
                raise KeyError(f"batch['modalities'] 에 누락된 키: {missing}")

        # consumed slot 을 batch 에 주입 (어댑터가 ``batch['__consumed__']`` 로 읽음).
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

        # aux loss / 진단용 sidecar — fusion 호출 전 per-modality 출력 보존.
        if self.aux_loss_modules is not None:
            self._last_H_list = list(H_list)

        if self.fusion is None:
            return H_list[0]

        # ---- per-sample modality presence mask (B, K) — 누락 modality 의 fusion alpha 차단 ----
        # collate 가 ``batch['modality_present']`` 에 ``{modality_key: (B,) bool}`` 로 주입.
        # backbone slot 의 presence = 그 backbone 이 consumes 한 modality 들의 AND.
        # fusion modality slot 의 presence = 그 modality 의 present 마스크 (없으면 모두 True).
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
        # alpha sidecar 저장 (entropy reg / sigma coupling 에서 사용)
        if self.training and alpha is not None:
            self._last_alpha = alpha
        # alpha 분포는 학습 진단용. fusion 이 alpha 를 반환하지 않는 미래 구현은
        # forward 시그니처를 통일했으므로 상관없음.
        if self.training and alpha is not None and alpha.dim() == 2 and alpha.size(1) > 1:
            for i in range(alpha.size(1)):
                self.log(
                    f"alpha_{i}",
                    alpha[:, i].mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=alpha.size(0),
                )
            # alpha 진단 로깅 강화
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

        # GR-MF: per-(group, modality) alpha 로깅. group_alpha 가 노출된 fusion 만 대상.
        # robust_attention 의 collapse 가 GR-MF 에서 구조적으로 사라지는지 확인하는 핵심 신호.
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
        """Fusion temperature annealing + warmup 토글 (RobustAttention) + head freeze."""
        # head warmup: 학습 초기 N epoch 동안 head 동결 → ckpt selection 이
        # gate/adapter 가 어느 정도 학습된 시점부터 시작. backward 는 head 출력으로
        # 흘러가 fusion gate 에 gradient 가 도달함 (head 자체 weight 만 업데이트 중단).
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
        """GR-MF group_alpha 의 KL(α || uniform_K) 정규화 항.

        목적:
            정적 gate (group_routed) 는 sample-별 input 의존성이 없어 per-sample
            RGR 다이내믹은 없지만, gradient-based concentration 으로 α 가 one-hot
            방향으로 sharpen 될 수 있다 (Switch-Transformer / Aux-Loss-Free MoE 의
            표준 결과). KL-to-uniform 으로 group-level 평균 α 가 uniform 근처로
            유지되도록 약하게 끌어준다.

        반환값은 scalar (B, n_groups 평균). gate_kl_weight 없거나 group_alpha 가
        노출되지 않으면 None.
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
            loss = loss - self.fusion_spec.entropy_weight * H  # H 최대화
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

        # Per-modality aux loss — 각 backbone 단독 학습력 진단 + regularization.
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
        # GradNorm: per-task w 로깅 (Kendall σ 와 동일 인터페이스).
        if self._use_gradnorm and self.ordered_tasks:
            w_cpu = self.loss_module.w.detach().cpu()
            for i, name in enumerate(self.ordered_tasks):
                self.log(f"gradnorm_w/{name}", w_cpu[i].item(), on_epoch=True)
        # AdapterGNN PEFT 진단: (a) per-block γ + (b) delta RMS / rel-delta.
        #   gamma: 부호 있는 scalar (paper init=0, 천천히 자라야 정상).
        #   gamma_abs: γ 가 음/양 둘 다 학습되어도 크기 추적 가능.
        #   delta_rms: ||s·A(x)||_RMS — adapter 가 실제로 backbone 분포를 흔드는 양.
        #   rel_delta: ||delta||_RMS / ||input||_RMS — 입력 대비 perturbation 비율
        #     (LoRA literature 의 표준 PEFT 진단 — γ 절댓값만 보면 둔감하므로 같이 봄).
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
        # LoRA PEFT 진단: ΔW = scaling·(B@A). B 가 0-init → 0 에서 자라는 양이
        #   AdapterGNN γ 대응물.
        #   dW_norm: ‖ΔW‖_F — LoRA 가 base 에서 멀어진 절대 크기.
        #   rel_dW:  ‖ΔW‖_F / ‖W_base‖_F — base 가중치 대비 비율 (PEFT 표준 진단).
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

        Hook 은 학습 중 (``self.training``) 매 배치마다 입력/출력의 RMS 를 0-d
        텐서로 누적. ``.item()`` 호출은 epoch end 까지 미뤄 GPU sync 1회/epoch 로
        제한 (배치당 sync 회피).
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
        # Aux loss 는 진단용으로만 로깅 — val_loss / early-stop 신호에는 미포함.
        # 본 학습 일반화는 main task head 가 결정. aux 는 backbone 별 단독 학습력 추적.
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
        """GradNorm 그라디언트 norm 측정 대상 파라미터 해석.

        해석 규칙:
            1) ``loss_module.shared_layer_name`` 가 지정되어 있으면 dotted path 로
               LitModule attribute 탐색. 결과가 ``nn.Module`` 이면 그 모듈의
               ``weight`` 만, ``nn.Parameter`` 면 그 자체를 사용.
            2) None 이면 primary backbone 의 마지막 ``nn.Linear`` 의 ``weight`` 로
               auto-detect (CheMeleon/AdapterGIN 등 일반 trunk → head 직전 layer).

        반환:
            list[Tensor] (requires_grad=True 만). 모두 0 length 면 ``ValueError``.
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
                    # 임의 nn.Module 인 경우 — 그 안의 모든 trainable 파라미터.
                    params.extend(p for p in obj.parameters() if p.requires_grad)
            else:
                raise TypeError(
                    f"shared_layer={name!r} 가 nn.Module 또는 nn.Parameter 가 아님: {type(obj)}"
                )
        else:
            # auto-detect: backbone 의 마지막 nn.Linear.weight.
            last_linear: nn.Linear | None = None
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear) and m.weight.requires_grad:
                    last_linear = m
            if last_linear is None:
                raise ValueError(
                    "GradNorm shared_layer auto-detect 실패: backbone 에 trainable "
                    "nn.Linear 가 없음. loss.kwargs.shared_layer 로 명시하세요."
                )
            params = [last_linear.weight]

        if not params:
            raise ValueError(
                f"GradNorm shared_layer 가 trainable 파라미터를 노출하지 못함 (name={name!r})."
            )
        self._gradnorm_shared_params_cache = params
        return params

    def _gradnorm_training_step(
        self, batch: Mapping[str, Any], batch_idx: int = 0
    ) -> Tensor:
        """Manual optimization: GradNorm 두 단계 backward + LR scheduler step.

        Lightning automatic_optimization=False 모드. ``self.optimizers()`` 는
        (main_opt, w_opt) 튜플; ``self.lr_schedulers()`` 는 main scheduler 만.
        """
        opt_main, opt_w = self.optimizers()
        sched = self.lr_schedulers()

        preds = self.forward(batch)
        targets = batch["targets"]
        sw = batch.get("sample_weights")

        # 1) per-task mean loss (graph 보존)
        L_k = self.loss_module.per_task_losses(preds, targets, sw)  # (K,)

        # 2) L̄_k(0): 첫 step 에서 lazy 채움. 0 인 task 는 1 로 치환 (division 안전).
        #    set 플래그는 buffer 라 resume 시 True 면 재초기화되지 않음.
        if not bool(self._gradnorm_L_init_set):
            L_init_local = L_k.detach().clone()
            L_init_local = torch.where(
                L_init_local > 1e-8,
                L_init_local,
                torch.ones_like(L_init_local),
            )
            self.gradnorm_L_init.copy_(L_init_local)
            self._gradnorm_L_init_set.fill_(True)

        # 3) ||∇_W L̄_k||_2 (스칼라 상수). create_graph=False → W gradient 흐름 차단.
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

        # 4) G_W^(k) = w_k · G_const_k. w 에 대해서만 미분가능.
        w = self.loss_module.w
        G_w = w * G_const  # (K,)

        # 5) target_k = mean_k G_w · r_k^α (no grad, no L_grad backward into w via target).
        with torch.no_grad():
            L_tilde = (L_k.detach() / self.gradnorm_L_init).clamp_min(1e-8)
            r = L_tilde / L_tilde.mean().clamp_min(1e-8)
            target = (G_w.detach().mean() * r.pow(self.loss_module.alpha)).detach()

        # 6) L_grad → w 만 업데이트.
        L_grad = (G_w - target).abs().sum()

        # 7) Main loss: w.detach() 로 w 의 main-backward 기여 제거.
        L_main = (w.detach() * L_k).sum()
        # GR-MF gate load-balance: KL(group_alpha || uniform_K). L_main 에만 합산
        # (L_grad 는 w 만 업데이트하므로 분리). KL 그래프는 forward 의 fusion alpha
        # 노드로 연결됨 — manual_backward(L_main) 에서 함께 backprop.
        kl_term = self._gate_kl_to_uniform()
        if kl_term is not None:
            L_main = L_main + kl_term

        opt_main.zero_grad(set_to_none=True)
        opt_w.zero_grad(set_to_none=True)
        # L_main 의 backward 가 L_k 그래프를 소비하므로 L_grad 먼저 (retain_graph).
        # G_const 가 이미 detach 되어 있으므로 L_grad backward 는 W 까지 흐르지 않는다.
        self.manual_backward(L_grad, retain_graph=True)
        self.manual_backward(L_main)
        opt_main.step()
        opt_w.step()

        # 8) w renormalize → sum=K + 개별 값 [0.3, 3.0] hard bound.
        # Why bound: 한 task 가 w 를 독점 (예 4.5×) 하면 LoRA delta 가 그 task
        # 한 개에 overfit 하고 나머지 K-1 task 학습이 멈춤. Chen 2018 원논문은
        # bound 없지만 LoRA + 22-task ADMET 조합에서 발산 관찰됨.
        # 순서: clamp → renormalize → final clamp. renormalize 가 bound 를
        # 살짝 깰 수 있어 두 번 clamp (sum=K 는 soft, individual bound 는 hard).
        with torch.no_grad():
            w.data.clamp_(min=0.3, max=3.0)
            w.data.mul_(K / w.data.sum().clamp_min(1e-8))
            w.data.clamp_(min=0.3, max=3.0)

        # 9) per-step LR scheduler (automatic_optimization=False 이므로 수동).
        if sched is not None:
            if isinstance(sched, list):
                for s in sched:
                    s.step()
            else:
                sched.step()

        # 10) 로깅 + 진단 sidecar (기존 mixed_task_uncertainty path 와 호환).
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
        """PCGrad projection 대상 파라미터 + shape 캐시.

        scope:
            ``loss_module.pcgrad_scope`` 가 ``"trunk"`` (default) 이면 backbone +
            fusion + extra_backbones 의 trainable 파라미터, ``"all"`` 이면 head 까지
            포함 (loss_module.log_vars 는 제외 — task-specific 이라 PCGrad 무의미).

        Returns:
            (params, shapes) — autograd.grad 호출 + PCGrad flatten/unflatten 용.
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
            # head 포함 — log_vars 제외 (task-specific).
            log_var_id = id(self.loss_module.log_vars)
            for p in self.parameters():
                if p.requires_grad and id(p) != log_var_id:
                    params.append(p)
        else:
            raise ValueError(f"pcgrad_scope={scope!r} — 'trunk'|'all' 만 지원")

        if not params:
            raise ValueError(
                f"PCGrad shared_params 집합이 비었습니다 (scope={scope!r}). "
                "backbone/fusion 의 requires_grad 파라미터를 확인하세요."
            )

        shapes: list[torch.Size] = [p.shape for p in params]
        self._pcgrad_shared_params_cache = params
        self._pcgrad_shared_shapes_cache = shapes
        return params, shapes

    def _pcgrad_training_step(
        self, batch: Mapping[str, Any], batch_idx: int = 0
    ) -> Tensor:
        """Kendall UW + PCGrad manual_optimization.

        흐름:
            1) forward → preds.
            2) loss_module.weighted_per_task_losses → (K,) Kendall 가중 task loss
               + scalar log_var regularizer + (K,) task_has_data mask.
            3) 각 task k 에 대해 autograd.grad(weighted_L_k, shared_params,
               retain_graph=True). dead task 는 스킵.
            4) PCGrad projection → shared params 별 그라디언트.
            5) autograd.grad(total_loss, non_shared) → head / log_vars 만 채움.
               (head 는 trunk 의 downstream 이라 trunk 로 역전파하지 않음 →
               전체 backward 대비 trunk backward 1 회 절약. DDP 면 non_shared
               그라디언트도 수동 all_reduce.)
            6) shared params .grad 를 PCGrad 결과로 덮어쓰기.
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

        # Kendall 가중 per-task loss + log_var reg (그래프 보존).
        weighted_L_k, log_var_reg, task_has_data = self.loss_module.weighted_per_task_losses(
            preds, targets, sw
        )
        K = self.n_tasks

        shared_params, shapes = self._resolve_pcgrad_shared_params()
        eps = getattr(self.loss_module, "pcgrad_eps", 1e-12)

        # task 별 그라디언트 — dead task 는 0 list 로 채움.
        grads_per_task: list[list[Tensor | None]] = []
        valid_task: list[bool] = []
        has_data_cpu = task_has_data.detach().cpu().tolist()
        for k in range(K):
            if not has_data_cpu[k]:
                grads_per_task.append([None] * len(shared_params))
                valid_task.append(False)
                continue
            # autograd.grad: shared params 만 — head / log_vars 미포함.
            # retain_graph=True 로 K 번 호출 유지. create_graph=False (2nd order 불필요).
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

        # DDP 동기화 — autograd.grad 는 DDP reducer 를 우회하므로 rank 마다
        # PCGrad gradient 가 다르다 (DistributedSampler 가 rank 별로 다른 batch
        # 제공). all_reduce(SUM)/world → MEAN 으로 동기화해 opt.step() 의 DDP
        # 가정 (모든 rank 동일 그라디언트) 회복. 비-DDP 에서는 no-op.
        import torch.distributed as dist
        ddp_active = dist.is_available() and dist.is_initialized()
        world = float(dist.get_world_size()) if ddp_active else 1.0
        if ddp_active:
            for g in projected:
                if g is None:
                    continue
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g.div_(world)

        # head / log_vars 그라디언트만 계산. 과거엔 manual_backward(total_loss) 로
        # 전체 그래프를 역전파해 shared(trunk) 의 .grad 까지 채운 뒤 PCGrad 결과로
        # 덮어썼다 → trunk backward 1 회분이 통째로 폐기됐다. autograd.grad(total_loss,
        # non_shared) 는 head 까지만 역전파하고 trunk 로 진입하지 않으므로 (head 는
        # trunk 의 downstream) trunk backward 를 아예 생략한다 — step 당 trunk backward
        # K+1 → K. autograd.grad 는 DDP reducer 를 우회하므로 projected 와 동일하게
        # non_shared 그라디언트도 수동 all_reduce 로 rank 동기화한다.
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

        # shared params .grad 를 PCGrad 결과로 덮어쓰기.
        for p, g_new in zip(shared_params, projected):
            if g_new is None:
                continue
            # PCGrad 그라디언트가 0 이면 그대로 0 으로 — None 으로 두면 optimizer 가
            # 해당 파라미터를 업데이트하지 않음 (set_to_none=True 의 의도).
            p.grad = g_new.detach()

        opt.step()

        # per-step LR scheduler (automatic_optimization=False 이므로 수동).
        if sched is not None:
            if isinstance(sched, list):
                for s in sched:
                    s.step()
            else:
                sched.step()

        # 진단 로깅 — cosine conflict rate / 평균 cosine. PCGrad 의 작동 신호.
        bs = preds.size(0)
        try:
            conflict_rate, mean_cos = cosine_conflict_rate(
                grads_per_task, shapes, valid_task=valid_task, eps=eps,
            )
            self.log("pcgrad/conflict_rate", float(conflict_rate), on_epoch=True, batch_size=bs)
            self.log("pcgrad/mean_cosine", float(mean_cos), on_epoch=True, batch_size=bs)
        except Exception:
            # 진단 실패는 학습을 막지 않는다.
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
        """Adam + Noam warmup schedule (옛 chemprop MPNN.configure_optimizers 동등).

        비-LoRA path: Adam(init_lr) + NoamLike LR scheduler (per-step).
            warmup_epochs 동안 init_lr → lr(max_lr) 선형 상승,
            이후 lr → final_lr 선형 감소.

        LoRA path (``lr_lora`` set): AdamW(wd=0) + 차등 lr, 스케줄 없음.
            옛 ``run_stl_molformer_lora.configure_optimizers`` 와 동등.
            "lora_" 매칭은 PEFT 표준 네이밍 (``lora_A``, ``lora_B``) 가정.

        Note:
            requires_grad=False 파라미터는 PyTorch 가 자동으로 무시한다.
        """
        from chemprop.schedulers import build_NoamLike_LRSched

        # GradNorm: main opt(나머지 전체) + aux opt(loss_module.w 만).
        # Lightning manual_optimization 모드에서 self.optimizers() 가 두 개를 반환.
        if self._use_gradnorm:
            w_param: nn.Parameter = self.loss_module.w
            w_id = id(w_param)
            main_params: list[nn.Parameter] = [
                p for p in self.parameters()
                if p.requires_grad and id(p) != w_id
            ]
            main_opt = torch.optim.Adam(main_params, lr=self.init_lr)
            # GradNorm w 의 학습률은 main 과 동일 (논문 권장; renormalize 가 scale 흡수).
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
                # manual_optimization 모드: scheduler 는 list 로 반환 → self.lr_schedulers()
                # 가 단일 객체 반환 → _pcgrad_training_step 에서 .step() 수동 호출.
                return ([optimizer], [{"scheduler": lr_sched, "interval": "step"}])
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

        # 베이스라인 비교용 토글 — `QADMET_LEGACY_WD=1` 이면 옛 path (모든 그룹 wd=0) 복원.
        # 4-modality overfit 방어 패키지 검증 시 baseline 학습에서 활성화.
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
                    "lr_lora 가 설정됐지만 'lora_' 파라미터를 찾지 못함."
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
                        "lr_fusion 가 설정됐지만 'fusion.*' 파라미터를 찾지 못함."
                    )
                legacy_groups.append(
                    {"params": fusion_params, "lr": self.lr_fusion, "weight_decay": 0.0}
                )
            return torch.optim.AdamW(legacy_groups)

        # Norm 계열 파라미터는 wd=0 (표준 관례). id 기반으로 식별 (prefix 매칭 불안정 회피).
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
                # GR-MF 의 group-routing gate (24 스칼라) 만 별도 group.
                # lr_fusion (modality_projs / cross_modal / output_proj 용 보수적 값)
                # 으로는 학습이 너무 느림 (v2 에서 7 epoch 동안 α spread 0.06pp).
                # head lr 동등 (self.lr) 으로 가속, wd=0 (24 스칼라는 decay 불필요).
                gate_params.append(p)
            elif name.startswith("head."):
                head_params.append(p)
            elif self.lr_fusion is not None and name.startswith("fusion."):
                fusion_params.append(p)
            else:
                other_params.append(p)
        if not lora_params:
            raise ValueError(
                "lr_lora 가 설정됐지만 'lora_' 파라미터를 찾지 못함. "
                "LoRA 가 활성화된 backbone (molformer_lora / unimol2_lora) 인지 확인."
            )
        # WD 정책 (overfit 방어, 4-modality recipe 기준):
        #   head=0.05 (작은 데이터 task overfit 차단)
        #   fusion=0.01 / 일반 backbone(adaptor 등)=0.01
        #   LoRA=0 (PEFT 표준), Norm/bias/log_vars/log_precision=0
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
                    "lr_fusion 가 설정됐지만 'fusion.*' 파라미터를 찾지 못함."
                )
            groups.append(
                {"params": fusion_params, "lr": self.lr_fusion, "weight_decay": 0.01}
            )
        elif fusion_params:
            # lr_fusion 미설정이면 fusion 도 일반 lr 로 통합 (wd=0.01).
            groups.append({"params": fusion_params, "lr": self.lr, "weight_decay": 0.01})
        return torch.optim.AdamW(groups)
