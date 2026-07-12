"""ExperimentSpec 및 하위 *Spec dataclass.

설계 원칙:
- 한 spec 한 책임. backbone / fusion / head / loss / trainer / ensemble 을 직교 분해.
- frozen dataclass + immutable 컨테이너 사용 → recipe 해시화·캐싱·lineage 친화적.
- recipe(YAML) ↔ spec 변환은 ``qadmet.core.recipe`` 가 담당. 본 모듈은 순수 데이터 정의.

검증 책임:
- ``__post_init__`` 에서 자체 정합성만 확인 (타입·범위).
- 도메인 간 정합성(예: kind=train 이면 model 필수)은 ``ExperimentSpec.validate()`` 에서.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping

# ---------------------------------------------------------------------------- 유틸


def _freeze_kwargs(value: Any) -> Mapping[str, Any]:
    """kwargs 를 frozen mapping 으로 정규화한다 (dataclass frozen 보호)."""
    if value is None:
        return MappingProxyType({})
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, dict):
        return MappingProxyType(dict(value))
    raise TypeError(f"kwargs 는 dict 또는 None 이어야 합니다: {type(value)}")


def _freeze_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, set)):
        return tuple(value)
    raise TypeError(f"tuple/list 가 필요합니다: {type(value)}")


# ---------------------------------------------------------------------------- 모달리티/모델


@dataclass(frozen=True)
class DatasetSpec:
    """학습/평가에 사용할 데이터셋 명세."""

    name: str  # registry 키 (예: "tdc_admet_22")
    split: str = "scaffold_5seed"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    # routing group 을 TaskSet.initial_grouping 대신 임시로 덮어쓴다 (그룹 탐색용).
    # {group_name: [task, ...]} — None 이면 TaskSet 기본 grouping 사용.
    # CLI 예: -o dataset.grouping_override='{"clf": ["HIA_Hou", ...], "reg": [...]}'
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
    """학습에 사용할 feature 조합 (모달리티별 builder 키)."""

    graph: str | None = None  # 예: "dmpnn"
    descriptors: tuple[str, ...] = ()  # 예: ("rdkit2d", "xtb")
    embeddings: tuple[str, ...] = ()  # 예: ("molformer", "unimol2")
    missing_policy: str = "mask_token"  # MissingPolicy registry 키

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", _freeze_tuple(self.descriptors))
        object.__setattr__(self, "embeddings", _freeze_tuple(self.embeddings))


@dataclass(frozen=True)
class BackboneSpec:
    """nn 모듈 backbone 명세 (registry 의 키 + 생성자 kwargs).

    Attributes:
        consumes: 본 backbone 이 읽어들이는 modality 키들.
            multi-backbone 학습에서 LoRA-MoLFormer 가 ``molformer_tokens`` modality 를
            토큰 입력으로 소비하는 경우 등에 사용. 비어있으면 fusion 모달리티 stack
            에서 분리되지 않으며 backbone 입력 어댑터가 batch 최상위 키만 사용한다.
            여기에 적힌 키는 fusion modality 입력에서 제외되고 backbone 어댑터의
            ``batch['__consumed__']`` 로 전달된다.
    """

    type: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    consumes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))
        object.__setattr__(self, "consumes", _freeze_tuple(self.consumes))


@dataclass(frozen=True)
class FusionSpec:
    """모달리티 fusion 명세.

    Attributes:
        entropy_weight: alpha entropy 정규화 강도. 기본값 0.0 (미사용).
            > 0 이면 loss -= entropy_weight * H(alpha) 로 modality 다양성 장려.

    Note:
        ``fallback`` 은 alpha-collapse 등으로 주 fusion 이 실패할 때 대체 키.
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
    """task head 명세 (per-task 또는 grouped)."""

    type: str = "linear"
    routing: str | None = None  # "taskset_grouping" | None (레거시 별칭: "admet_7_group")
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


@dataclass(frozen=True)
class LossSpec:
    """학습 loss 명세.

    Why:
        분류·회귀 loss 와 task weighting 전략을 분리해서 등록 가능하게 한다.
        Kendall uncertainty 외 GradNorm 등 대안을 task_weighting 키로 교체할 수 있다.
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
    """Per-modality aux loss 명세 (backbone 단독 학습력 진단 + regularization).

    각 backbone 의 pre-fusion 출력 H_i 에 별도 head + loss 인스턴스를 달아
    "이 modality 단독으로 22 task 를 풀 수 있는가" 를 직접 측정한다.

        L_total = L_main + weight * Σ_i L_aux_i

    각 aux loss 는 자체 ``log_vars`` (Kendall σ) 를 가져 modality 별 calibration
    이 main loss 와 독립적이다. 진단 신호:
        - aux_loss/i 가 작아지면 modality i 의 backbone 출력에 task 신호 살아있음
        - aux_loss/i 가 high plateau → modality i 의 backbone (또는 LoRA hp) 문제

    Attributes:
        weight: λ_aux (>=0). 0 이면 효과 없음.
        head_type: "linear" 만 지원 (Linear(D_i, n_tasks)) — 진단용 단순화.
        loss_type: aux loss 의 LOSSES 키. None 이면 main loss_spec.type 재사용.
        head_kwargs: head_type 별 추가 인자 (현재 linear 는 사용 안 함).
    """

    weight: float = 0.0
    head_type: str = "linear"
    loss_type: str | None = None
    head_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_kwargs", _freeze_kwargs(self.head_kwargs))
        if self.weight < 0:
            raise ValueError(f"aux_loss.weight >= 0 필요: {self.weight}")


@dataclass(frozen=True)
class ModelSpec:
    """모델 한 그루: primary backbone (+ optional extras + fusion) + head.

    Attributes:
        backbone: 주 trainable trunk. 단일 backbone 학습은 이 한 개만 사용.
        extra_backbones: 추가 trainable trunk 들 (옛 phase1.4a — D-MPNN +
            LoRA-MoLFormer 동시 학습 패턴). primary 출력 다음 순서대로 fusion
            입력에 들어간다. 각 entry 의 ``consumes`` 가 비어있지 않으면 해당
            modality 키들은 fusion modality stack 에서 제외된다.
        fusion: K=1+len(extra_backbones)+(non-consumed modality 수) 입력으로 동작.
            extra_backbones 가 있으면 fusion 은 사실상 필수 (input 이 ≥ 2 개).
        head: per-task 또는 grouped head.
        aux_loss: per-modality aux loss spec. None 이면 비활성 (기본).
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
                "extra_backbones 가 있으면 FusionSpec 이 필수입니다 — primary + extras "
                "출력을 합칠 fusion 모듈을 명세하세요."
            )

    @property
    def all_backbones(self) -> tuple[BackboneSpec, ...]:
        """primary + extras 통합 튜플 (LitModule 빌드용)."""
        return (self.backbone, *self.extra_backbones)


@dataclass(frozen=True)
class TrainerSpec:
    """학습 루프 하이퍼파라미터."""

    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_epochs: int = 100
    patience: int = 20
    precision: str = "32-true"
    early_stopping: bool = True
    batch_size: int = 64
    lr: float = 1e-3  # Noam schedule 의 max_lr.
    init_lr: float = 1e-4  # Noam warmup 시작 lr.
    final_lr: float = 1e-4  # Noam cooldown 종료 lr.
    warmup_epochs: int = 2  # Noam warmup epoch 수.
    lr_lora: float | None = None  # 비-None 이면 "lora_" 파라미터에 별도 lr 적용 (head/fusion 은 lr 사용).
    lr_fusion: float | None = None  # 비-None 이면 fusion module 파라미터에 별도 lr 적용 (gate slowdown → backbone catch-up 시간 확보).
    num_workers: int = 4  # DataLoader 워커 수. 0 이면 메인 프로세스 동기 로딩 (smoke 테스트에 유용).
    head_warmup_epochs: int = 0  # epoch < this 동안 head 파라미터 freeze (gate-first warmup).
    predict_ckpt: str = "best"  # "best" | "last" — predict 단계에서 사용할 ckpt 선택.
    extra_callbacks: tuple[str, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)  # lightning.Trainer 추가 인자 전용 (gradient_clip_val 등).

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _freeze_tuple(self.seeds))
        object.__setattr__(self, "extra_callbacks", _freeze_tuple(self.extra_callbacks))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs > 0 필요: {self.max_epochs}")
        if self.patience < 0:
            raise ValueError(f"patience >= 0 필요: {self.patience}")
        if self.lr <= 0:
            raise ValueError(f"lr(max_lr) 는 양수여야 합니다: {self.lr}")
        if self.init_lr <= 0:
            raise ValueError(f"init_lr 는 양수여야 합니다: {self.init_lr}")
        if self.final_lr <= 0:
            raise ValueError(f"final_lr 는 양수여야 합니다: {self.final_lr}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs >= 0 필요: {self.warmup_epochs}")
        if self.lr_lora is not None and self.lr_lora <= 0:
            raise ValueError(f"lr_lora 는 양수여야 합니다: {self.lr_lora}")
        if self.lr_fusion is not None and self.lr_fusion <= 0:
            raise ValueError(f"lr_fusion 는 양수여야 합니다: {self.lr_fusion}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers >= 0 필요: {self.num_workers}")
        if self.head_warmup_epochs < 0:
            raise ValueError(f"head_warmup_epochs >= 0 필요: {self.head_warmup_epochs}")
        if self.predict_ckpt not in ("best", "last"):
            raise ValueError(
                f"predict_ckpt 는 'best' 또는 'last' 만 허용: {self.predict_ckpt!r}"
            )


@dataclass(frozen=True)
class EnsembleSpec:
    """ensemble (stacking) 파이프라인 명세.

    bases 는 PredictionSet artifact 의 이름 목록. 각 이름은 ``train`` 또는 ``predict``
    runner 가 만든 artifact 를 가리킨다.
    """

    strategy: str  # MetaLearner registry 키
    bases: tuple[str, ...] = ()
    meta_model: str | None = None  # "xgboost" | "ridge" | "logreg" | None
    calibrator: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bases", _freeze_tuple(self.bases))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


@dataclass(frozen=True)
class EvaluationSpec:
    """평가/비교 단계 명세."""

    benchmark: str = "tdc_admet"
    aggregate: str = "mean"  # "mean" | "median"
    compare_to: tuple[str, ...] = ()  # 비교 대상 artifact 이름들
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compare_to", _freeze_tuple(self.compare_to))
        object.__setattr__(self, "kwargs", _freeze_kwargs(self.kwargs))


# ---------------------------------------------------------------------------- 최상위


VALID_KINDS = ("train", "precompute", "predict", "stack", "evaluate")


@dataclass(frozen=True)
class ExperimentSpec:
    """실험 한 건의 최상위 명세 — Recipe(YAML) 가 직렬화되는 형태.

    Args:
        name: 실험 이름 (artifact 디렉토리 이름이 된다).
        kind: 실행 종류. ``VALID_KINDS`` 중 하나.
        dataset / features / model / loss / trainer / ensemble / evaluation:
            kind 에 따라 일부만 의미를 가진다 (validate() 참조).
        notes: 자유 메모. 인덱싱·검색·lineage 메타에 노출.
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
            raise ValueError(f"kind 는 {VALID_KINDS} 중 하나여야 합니다: {self.kind!r}")
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"name 은 비어있지 않은 문자열이어야 합니다: {self.name!r}")

    def validate(self) -> None:
        """kind 별 필수 필드 정합성 검사. 실패 시 ValueError."""
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
                f"ExperimentSpec(kind={self.kind!r}) 필수 필드 누락: {missing}"
            )


# ---------------------------------------------------------------------------- 직렬화 helpers


def spec_to_dict(spec: Any) -> Any:
    """dataclass(spec) 또는 컨테이너를 plain dict / list / scalar 로 재귀 변환.

    Why:
        recipe lineage 해싱·YAML 저장에 사용. ``dataclasses.asdict`` 와 다르게
        MappingProxyType 와 nested dataclass 모두 정확히 처리한다.
    """
    if spec is None or isinstance(spec, (str, int, float, bool)):
        return spec
    if isinstance(spec, (tuple, list)):
        return [spec_to_dict(x) for x in spec]
    if isinstance(spec, (dict, MappingProxyType)):
        return {k: spec_to_dict(v) for k, v in spec.items()}
    if is_dataclass(spec):
        return {f.name: spec_to_dict(getattr(spec, f.name)) for f in fields(spec)}
    raise TypeError(f"직렬화 불가 타입: {type(spec)}")
