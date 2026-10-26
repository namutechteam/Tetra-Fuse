"""Recipe: YAML ↔ ExperimentSpec conversion.

Features:
- Inline other YAML via the ``!include path`` tag (DRY composition).
- Merge CLI override (dot-notation) into spec dict.
- Build nested dataclass(spec) from dict.

Non-goals:
- Environment variable substitution / dynamic expressions. Recipe stays as a static declaration.
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
    """SafeLoader supporting the !include tag."""

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
    # Search in order: absolute path / path relative to recipes root / path relative to current yaml location.
    search = [
        candidate if candidate.is_absolute() else None,
        recipes_dir() / candidate,
        loader._self_dir / candidate,
    ]
    for p in search:
        if p is not None and p.exists():
            return load_yaml(p)
    raise FileNotFoundError(f"!include target not found: {raw} (searched: {[str(s) for s in search if s]})")


_IncludeLoader.add_constructor("!include", _construct_include)
_IncludeLoader.base_dir = recipes_dir()


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file as a dict (including !include handling)."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"recipe file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_IncludeLoader)


# ---------------------------------------------------------------------------- override


def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    """Merge CLI overrides into a dict.

    Format: ``["trainer.max_epochs=2", "trainer.seeds=[0,1]"]``.
    Values are parsed via yaml.safe_load → int/list/string auto-inferred.

    Why:
        So that the same recipe can be reused across different contexts such as smoke test, sweep, etc.,
        overwrite in-place instead of duplicating the file each time.
    """
    if not overrides:
        return data
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override format is not 'key=value': {item!r}")
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
                raise TypeError(f"override path conflict: intermediate value of {key!r} is not a dict")
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
        raise ValueError("model.backbone is required.")
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
    """Convert recipe dict → ExperimentSpec."""
    if not isinstance(data, dict):
        raise TypeError(f"recipe top level must be a dict: {type(data)}")
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
    """YAML path → ExperimentSpec (with override applied)."""
    data = load_yaml(path)
    data = apply_overrides(data, overrides)
    return dict_to_spec(data)
