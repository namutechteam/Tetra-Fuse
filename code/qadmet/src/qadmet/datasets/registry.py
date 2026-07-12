"""DATASETS registry: ``DatasetSpec.name`` 에 등록된 로더를 조회."""

from __future__ import annotations

from qadmet.core.registry import Registry

DATASETS: Registry = Registry("datasets")
