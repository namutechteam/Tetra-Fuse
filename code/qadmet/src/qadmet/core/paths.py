"""Single source of truth for filesystem paths.

Supports environment variable override to change location per user/CI.

Default layout:
    QADMET_ROOT (= /nas/tk_ahn/ADMET-MTL/qadmet by default)
        ├── data/         raw data + precomputed parquet
        ├── recipes/      YAML recipe
        ├── artifacts/    runner output (kind/name/run_id/...)
        ├── checkpoints*/ legacy training artifacts (parity baseline, do not touch)
        └── logs/
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root estimation: 4 levels above this file's location (src/qadmet/core/paths.py → repo).
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def root() -> Path:
    """qadmet repo root. ``QADMET_ROOT`` environment variable takes precedence."""
    env = os.environ.get("QADMET_ROOT")
    return Path(env).resolve() if env else _DEFAULT_ROOT


def data_dir() -> Path:
    return Path(os.environ.get("QADMET_DATA_DIR", root() / "data")).resolve()


def recipes_dir() -> Path:
    return Path(os.environ.get("QADMET_RECIPES_DIR", root() / "recipes")).resolve()


def artifacts_dir() -> Path:
    return Path(os.environ.get("QADMET_ARTIFACTS_DIR", root() / "artifacts")).resolve()


def logs_dir() -> Path:
    return Path(os.environ.get("QADMET_LOGS_DIR", root() / "logs")).resolve()


def ensure(path: Path) -> Path:
    """Create the directory if it does not exist, then return the absolute path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
