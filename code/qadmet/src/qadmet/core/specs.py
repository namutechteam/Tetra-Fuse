"""ExperimentSpec and subordinate *Spec dataclasses.

Design principles:
- One spec, one responsibility. Orthogonal decomposition into backbone / fusion / head / loss / trainer / ensemble.
- Uses frozen dataclass + immutable containers → friendly to recipe hashing, caching, lineage.
- recipe(YAML) ↔ spec conversion is handled by ``qadmet.core.recipe``. This module defines pure data only.

Validation responsibilities:
- ``__post_init__`` only checks self-consistency (type, range).
- Cross-domain consistency (e.g., if kind=train then model is required) is checked in ``ExperimentSpec.validate()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping

# ---------------------------------------------------------------------------- Utilities


def _freeze_kwargs(value: Any) -> Mapping[str, Any]:
    """Normalizes kwargs into a frozen mapping (protects dataclass frozen semantics)."""
    if value is None:
        return MappingProxyType({})
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, dict):
        return MappingProxyType(dict(value))
    raise TypeError(f"kwargs must be dict or None: {type(value)}")


def _freeze_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, set)):
        return tuple(value)
    raise TypeError(f"tuple/list is required: {type(value)}")


# ---------------------------------------------------------------------------- Modality/Model


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset spec used for training/evaluation."""

    name: str  # registry key (e.g., "tdc_admet_22")
    split: str = "scaffold_5seed"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    # Temporarily overrides the routing group instead of TaskSet.initial_grouping (for group exploration).
    # {group_name: [task, ...]} — if None, uses TaskSet's default grouping.
    # CLI example: -o dataset.grouping_override='{"clf": ["HIA_Hou", ...], "reg": [...]}'
    grouping_override: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _freeze_tuple(self.seeds))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))
        if self.grouping_override is not None:
            frozen = MappingProxyType(
                {str(g): tuple(ts) for g, ts in self.grouping_override.items()}
            )
            object.__setattr__(self, "grouping_override", frozen)


@dataclass(frozen=True)
class FeatureSpec:
    """Feature combination used for training (builder key per modality)."""

    graph: str | None = None  # e.g., "dmpnn"
    descriptors: tuple[str, ...] = ()  # e.g., ("rdkit2d", "xtb")
    embeddings: tuple[str, ...] = ()  # e.g., ("molformer", "unimol2")
    missing_policy: str = "mask_token"  # MissingPolicy registry key

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", _freeze_tuple(self.descriptors))
        object.__setattr__(self, "embeddings", _freeze_tuple(self.embeddings))


@dataclass(frozen=True)
class BackboneSpec:
    """nn module backbone spec (registry key + constructor kwargs).

    Attributes:
        consumes: modality keys that this backbone reads.
            Used e.g. when LoRA-MoLFormer, in multi-backbone training, consumes the ``molformer_tokens`` modality
            as a token input. If empty, it is not separated from the fusion modality stack
            and the backbone input adapter uses only the batch's top-level keys.
            Keys listed here are excluded from the fusion modality input and are passed
            to the backbone adapter via ``batch['__consumed__']``.
    """

    type: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    consumes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))
        object.__setattr__(self, "consumes", _freeze_tuple(self.consumes))


@dataclass(frozen=True)
class FusionSpec:
    """Modality fusion spec.

    Attributes:
        entropy_weight: alpha entropy regularization strength. Default 0.0 (disabled).
            If > 0, encourages modality diversity via loss -= entropy_weight * H(alpha).

    Note:
        ``fallback`` is the substitute key used when the primary fusion fails, e.g. due to alpha-collapse.
    """

    type: str
    fallback: str | None = None
    entropy_weight: float = 0.0
    gate_kl_weight: float = 0.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


@dataclass(frozen=True)
class HeadSpec:
    """Task head spec (per-task or grouped)."""

    type: str = "linear"
    routing: str | None = None  # "taskset_grouping" | None (legacy alias: "admet_7_group")
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


@dataclass(frozen=True)
class LossSpec:
    """Training loss spec.

    Why:
        Allows classification/regression loss and task weighting strategy to be registered separately.
        Alternatives such as GradNorm, besides Kendall uncertainty, can be swapped in via the task_weighting key.
    """

    type: str  # "mixed_task_uncertainty" | "single_task"
    classification: str = "bce_with_logits"
    regression: str = "masked_mae"
    task_weighting: str | None = "kendall_uncertainty"
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


@dataclass(frozen=True)
class AuxLossSpec:
    """Per-modality aux loss spec (backbone standalone learning capacity diagnosis + regularization).

    Attaches a separate head + loss instance to each backbone's pre-fusion output H_i
    to directly measure "can this modality alone solve the 22 tasks".

        L_total = L_main + weight * Σ_i L_aux_i

    Each aux loss has its own ``log_vars`` (Kendall σ), so per-modality calibration
    is independent from the main loss. Diagnostic signals:
        - if aux_loss/i decreases, task signal is alive in modality i's backbone output
        - aux_loss/i on a high plateau → issue with modality i's backbone (or LoRA hp)

    Attributes:
        weight: λ_aux (>=0). No effect if 0.
        head_type: only "linear" is supported (Linear(D_i, n_tasks)) — simplified for diagnostics.
        loss_type: LOSSES key for the aux loss. If None, reuses the main loss_spec.type.
        head_kwargs: extra arguments per head_type (currently unused for linear).
    """

    weight: float = 0.0
    head_type: str = "linear"
    loss_type: str | None = None
    head_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_kwargs", _freeze_kwargs(self.head_kwargs))
        if self.weight < 0:
            raise ValueError(f"aux_loss.weight >= 0 required: {self.weight}")


@dataclass(frozen=True)
class ModelSpec:
    """One model instance: primary backbone (+ optional extras + fusion) + head.

    Attributes:
        backbone: the primary trainable trunk. Single-backbone training uses only this one.
        extra_backbones: additional trainable trunks (old phase1.4a — D-MPNN +
            LoRA-MoLFormer joint training pattern). Enters the fusion input in order
            after the primary output. If an entry's ``consumes`` is non-empty, the corresponding
            modality keys are excluded from the fusion modality stack.
        fusion: operates with K=1+len(extra_backbones)+(number of non-consumed modalities) inputs.
            If extra_backbones is present, fusion is effectively required (inputs ≥ 2).
        head: per-task or grouped head.
        aux_loss: per-modality aux loss spec. Disabled if None (default).
    """

    type: str  # "multitask" | "singletask"
    backbone: BackboneSpec
    fusion: FusionSpec | None = None
    head: HeadSpec = field(default_factory=lambda: HeadSpec(type="linear"))
    extra_backbones: tuple[BackboneSpec, ...] = ()
    aux_loss: AuxLossSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_backbones", _freeze_tuple(self.extra_backbones))
        if self.extra_backbones and self.fusion is None:
            raise ValueError(
                "FusionSpec is required when extra_backbones is present — primary + extras "
                "specify a fusion module to merge the outputs."
            )

    @property
    def all_backbones(self) -> tuple[BackboneSpec, ...]:
        """Combined tuple of primary + extras (for building LitModule)."""
        return (self.backbone, *self.extra_backbones)


@dataclass(frozen=True)
class TrainerSpec:
    """Training loop hyperparameters."""

    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_epochs: int = 100
    patience: int = 20
    precision: str = "32-true"
    early_stopping: bool = True
    batch_size: int = 64
    lr: float = 1e-3  # max_lr for the Noam schedule.
    init_lr: float = 1e-4  # Noam warmup starting lr.
    final_lr: float = 1e-4  # Noam cooldown ending lr.
    warmup_epochs: int = 2  # number of Noam warmup epochs.
    lr_lora: float | None = None  # if non-None, applies a separate lr to "lora_" parameters (head/fusion use lr).
    lr_fusion: float | None = None  # if non-None, applies a separate lr to fusion module parameters (gate slowdown → gives backbone time to catch up).
    num_workers: int = 4  # number of DataLoader workers. If 0, synchronous loading on the main process (useful for smoke tests).
    head_warmup_epochs: int = 0  # freezes head parameters while epoch < this (gate-first warmup).
    predict_ckpt: str = "best"  # "best" | "last" — selects the ckpt to use in the predict stage.
    extra_callbacks: tuple[str, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)  # for extra lightning.Trainer arguments only (gradient_clip_val, etc.).

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _freeze_tuple(self.seeds))
        object.__setattr__(self, "extra_callbacks", _freeze_tuple(self.extra_callbacks))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs > 0 required: {self.max_epochs}")
        if self.patience < 0:
            raise ValueError(f"patience >= 0 required: {self.patience}")
        if self.lr <= 0:
            raise ValueError(f"lr(max_lr) must be positive: {self.lr}")
        if self.init_lr <= 0:
            raise ValueError(f"init_lr must be positive: {self.init_lr}")
        if self.final_lr <= 0:
            raise ValueError(f"final_lr must be positive: {self.final_lr}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs >= 0 required: {self.warmup_epochs}")
        if self.lr_lora is not None and self.lr_lora <= 0:
            raise ValueError(f"lr_lora must be positive: {self.lr_lora}")
        if self.lr_fusion is not None and self.lr_fusion <= 0:
            raise ValueError(f"lr_fusion must be positive: {self.lr_fusion}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers >= 0 required: {self.num_workers}")
        if self.head_warmup_epochs < 0:
            raise ValueError(f"head_warmup_epochs >= 0 required: {self.head_warmup_epochs}")
        if self.predict_ckpt not in ("best", "last"):
            raise ValueError(
                f"predict_ckpt only allows 'best' or 'last': {self.predict_ckpt!r}"
            )


@dataclass(frozen=True)
class EnsembleSpec:
    """Ensemble (stacking) pipeline spec.

    bases is a list of PredictionSet artifact names. Each name points to an artifact created by the ``train`` or ``predict``
    runner.
    """

    strategy: str  # MetaLearner registry key
    bases: tuple[str, ...] = ()
    meta_model: str | None = None  # "xgboost" | "ridge" | "logreg" | None
    calibrator: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bases", _freeze_tuple(self.bases))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation/comparison stage spec."""

    benchmark: str = "tdc_admet"
    aggregate: str = "mean"  # "mean" | "median"
    compare_to: tuple[str, ...] = ()  # names of artifacts to compare against
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compare_to", _freeze_tuple(self.compare_to))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


# ---------------------------------------------------------------------------- Top-level


VALID_KINDS = ("train", "precompute", "predict", "stack", "evaluate")


@dataclass(frozen=True)
class ExperimentSpec:
    """Top-level spec for a single experiment — the form into which Recipe(YAML) is serialized.

    Args:
        name: experiment name (becomes the artifact directory name).
        kind: run type. One of ``VALID_KINDS``.
        dataset / features / model / loss / trainer / ensemble / evaluation:
            Depending on kind, only some fields are meaningful (see validate()).
        notes: free-form memo. Exposed in indexing/search/lineage metadata.
    """

    name: str
    kind: str
    dataset: DatasetSpec | None = None
    features: FeatureSpec | None = None
    model: ModelSpec | None = None
    loss: LossSpec | None = None
    trainer: TrainerSpec | None = None
    ensemble: EnsembleSpec | None = None
    evaluation: EvaluationSpec | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}: {self.kind!r}")
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"name must be a non-empty string: {self.name!r}")

    def validate(self) -> None:
        """Checks required-field consistency per kind. Raises ValueError on failure."""
        required: dict[str, tuple[str, ...]] = {
            "train": ("dataset", "features", "model", "loss", "trainer"),
            "precompute": ("features",),
            "predict": ("dataset",),
            "stack": ("ensemble",),
            "evaluate": ("evaluation",),
        }
        missing = [name for name in required[self.kind] if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"ExperimentSpec(kind={self.kind!r}) missing required fields: {missing}"
            )


# ---------------------------------------------------------------------------- Serialization helpers


def spec_to_dict(spec: Any) -> Any:
    """Recursively converts dataclass(spec) or containers into plain dict / list / scalar.

    Why:
        Used for recipe lineage hashing and YAML saving. Unlike ``dataclasses.asdict``,
        handles both MappingProxyType and nested dataclasses correctly.
    """
    if spec is None or isinstance(spec, (str, int, float, bool)):
        return spec
    if isinstance(spec, (tuple, list)):
        return [spec_to_dict(x) for x in spec]
    if isinstance(spec, (dict, MappingProxyType)):
        return {k: spec_to_dict(v) for k, v in spec.items()}
    if is_dataclass(spec):
        return {f.name: spec_to_dict(getattr(spec, f.name)) for f in fields(spec)}
    raise TypeError(f"non-serializable type: {type(spec)}")
