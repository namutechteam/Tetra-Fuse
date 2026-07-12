"""파일시스템 경로 단일 진실원.

환경변수 override 를 지원하여 사용자별/CI 별로 위치를 바꿀 수 있다.

기본 레이아웃:
    QADMET_ROOT (= /nas/tk_ahn/ADMET-MTL/qadmet by default)
        ├── data/         원본 데이터 + 사전 계산 parquet
        ├── recipes/      YAML recipe
        ├── artifacts/    runner 산출물 (kind/name/run_id/...)
        ├── checkpoints*/ 옛 학습 산출물 (parity 기준, 손대지 않음)
        └── logs/
"""

from __future__ import annotations

import os
from pathlib import Path

# repo 루트 추정: 본 파일 위치의 4 단계 위 (src/qadmet/core/paths.py → repo).
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def root() -> Path:
    """qadmet repo 루트. ``QADMET_ROOT`` 환경변수가 우선."""
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
    """디렉토리가 없으면 만든 뒤 절대경로 반환."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
