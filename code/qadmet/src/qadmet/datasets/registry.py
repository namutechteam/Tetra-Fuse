"""DATASETS registry: Look up the loader registered under ``DatasetSpec.name``."""

from __future__ import annotations

from qadmet.core.registry import Registry

DATASETS: Registry = Registry("datasets")
