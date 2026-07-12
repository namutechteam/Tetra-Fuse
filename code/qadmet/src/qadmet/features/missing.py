"""MissingPolicy: 모달리티 결측 처리 전략.

CLAUDE.md 의 개선점 — UniMol2 3D 변환 실패 분자에 대한 graceful 처리 — 을
정식 API 로 노출한다. 옛 코드에서는 ``np.nan_to_num(..., nan=0.0)`` + 별도
mask 채널이 흩어져 있었다.

전략:
- ``zero``: NaN → 0 으로 치환 (옛 default 동작).
- ``mask_token``: NaN → 0 으로 치환 + 별도 mask 채널을 concat. 모델은 mask 채널을
  참조하여 결측 여부를 학습한다.
- ``drop_row``: NaN 이 하나라도 있는 row 를 제외 (학습용; 추론 시에는 부적합).
"""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from qadmet.core.registry import Registry


class MissingPolicy(Protocol):
    """전략 인터페이스: array → (cleaned_array, mask_array_or_None)."""

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]: ...


MISSING_POLICIES: Registry[MissingPolicy] = Registry("missing_policies")


@MISSING_POLICIES.register("zero")
def _zero(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), None


@MISSING_POLICIES.register("mask_token")
def _mask_token(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    """결측 채널별 0/1 mask 를 동반 반환.

    Note:
        반환된 mask 는 호출자가 ``np.concatenate([cleaned, mask], axis=1)`` 등으로
        사용. 모델 입력 차원이 늘어나므로 BackboneSpec 의 input dim 산출 시 반영해야 한다.
    """
    mask = np.isnan(x).astype(np.float32)
    cleaned = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return cleaned, mask


@MISSING_POLICIES.register("drop_row")
def _drop_row(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    keep = ~np.isnan(x).any(axis=1)
    return x[keep].astype(np.float32), keep.astype(bool)


def apply_policy(name: str, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    """편의 함수: registry lookup + 적용."""
    return MISSING_POLICIES.get(name)(x)
