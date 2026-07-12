"""Recipe: YAML ↔ ExperimentSpec 변환.

기능:
- ``!include path`` 태그로 다른 YAML 을 인라인 (DRY 컴포지션).
- CLI override (dot-notation) 을 spec dict 에 머지.
- dict → 중첩 dataclass(spec) 빌드.

비-목표:
- 환경변수 치환·동적 표현식. Recipe 는 정적 선언으로 유지한다.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from qadmet.core.paths import recipes_dir
from qadmet.core.specs import (
    AuxLossSpec,
    BackboneSpec,
    DatasetSpec,
    EnsembleSpec,
    EvaluationSpec,
    ExperimentSpec,
    FeatureSpec,
    FusionSpec,
    HeadSpec,
    LossSpec,
    ModelSpec,
    TrainerSpec,
)


# ---------------------------------------------------------------------------- !include

class _IncludeLoader(yaml.SafeLoader):
    """!include 태그를 지원하는 SafeLoader."""

    base_dir: Path

    def __init__(self, stream: Any) -> None:
        try:
            self._self_dir = Path(stream.name).resolve().parent
        except AttributeError:
            self._self_dir = self.base_dir
        super().__init__(stream)


def _construct_include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    raw = loader.construct_scalar(node)  # type: ignore[arg-type]
    candidate = Path(raw)
    # 절대경로 / recipes 루트 상대경로 / 현재 yaml 위치 상대경로 순으로 탐색.
    search = [
        candidate if candidate.is_absolute() else None,
        recipes_dir() / candidate,
        loader._self_dir / candidate,
    ]
    for p in search:
        if p is not None and p.exists():
            return load_yaml(p)
    raise FileNotFoundError(f"!include 대상 없음: {raw} (탐색: {[str(s) for s in search if s]})")


_IncludeLoader.add_constructor("!include", _construct_include)
_IncludeLoader.base_dir = recipes_dir()


def load_yaml(path: str | Path) -> Any:
    """YAML 파일을 dict 로 로드 (!include 처리 포함)."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"recipe 파일 없음: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_IncludeLoader)


# ---------------------------------------------------------------------------- override


def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    """CLI override 를 dict 에 머지한다.

    형식: ``["trainer.max_epochs=2", "trainer.seeds=[0,1]"]``.
    값은 yaml.safe_load 로 파싱 → 정수/리스트/문자열 자동 추론.

    Why:
        같은 recipe 를 smoke test, sweep 등 다양한 컨텍스트에서 재사용할 수 있도록
        파일을 매번 복제하지 않고 in-place 덮어쓴다.
    """
    if not overrides:
        return data
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override 형식이 'key=value' 가 아닙니다: {item!r}")
        key, raw_val = item.split("=", 1)
        keys = key.strip().split(".")
        try:
            value = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            value = raw_val
        cursor = out
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
            if not isinstance(cursor, dict):
                raise TypeError(f"override 경로 충돌: {key!r} 중간 값이 dict 가 아님")
        cursor[keys[-1]] = value
    return out


# ---------------------------------------------------------------------------- dict → spec


def _build_optional(cls, value):
    return cls(**value) if value is not None else None


def _build_backbone(d: dict | None) -> BackboneSpec | None:
    return BackboneSpec(**d) if d is not None else None


def _build_fusion(d: dict | None) -> FusionSpec | None:
    return FusionSpec(**d) if d is not None else None


def _build_head(d: dict | None) -> HeadSpec:
    return HeadSpec(**d) if d is not None else HeadSpec(type="linear")


def _build_aux_loss(d: dict | None) -> AuxLossSpec | None:
    return AuxLossSpec(**d) if d is not None else None


def _build_model(d: dict | None) -> ModelSpec | None:
    if d is None:
        return None
    backbone = _build_backbone(d.get("backbone"))
    if backbone is None:
        raise ValueError("model.backbone 은 필수입니다.")
    raw_extras = d.get("extra_backbones") or ()
    extras = tuple(BackboneSpec(**e) for e in raw_extras)
    return ModelSpec(
        type=d["type"],
        backbone=backbone,
        fusion=_build_fusion(d.get("fusion")),
        head=_build_head(d.get("head")),
        extra_backbones=extras,
        aux_loss=_build_aux_loss(d.get("aux_loss")),
    )


def dict_to_spec(data: dict) -> ExperimentSpec:
    """recipe dict → ExperimentSpec 변환."""
    if not isinstance(data, dict):
        raise TypeError(f"recipe 최상위는 dict 여야 합니다: {type(data)}")
    spec = ExperimentSpec(
        name=data["name"],
        kind=data["kind"],
        dataset=_build_optional(DatasetSpec, data.get("dataset")),
        features=_build_optional(FeatureSpec, data.get("features")),
        model=_build_model(data.get("model")),
        loss=_build_optional(LossSpec, data.get("loss")),
        trainer=_build_optional(TrainerSpec, data.get("trainer")),
        ensemble=_build_optional(EnsembleSpec, data.get("ensemble")),
        evaluation=_build_optional(EvaluationSpec, data.get("evaluation")),
        notes=data.get("notes", ""),
    )
    spec.validate()
    return spec


def load_recipe(path: str | Path, *, overrides: list[str] | None = None) -> ExperimentSpec:
    """YAML 경로 → ExperimentSpec (with override 적용)."""
    data = load_yaml(path)
    data = apply_overrides(data, overrides)
    return dict_to_spec(data)
