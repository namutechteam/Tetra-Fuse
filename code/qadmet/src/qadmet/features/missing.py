"""MissingPolicy: Modality missing-value handling strategy.

The improvement noted in CLAUDE.md — graceful handling for molecules that fail UniMol2 3D conversion —
is exposed as a formal API. In the old code, ``np.nan_to_num(..., nan=0.0)`` + a separate
mask channel were scattered around.

Strategies:
- ``zero``: Replace NaN → 0 (old default behavior).
- ``mask_token``: Replace NaN → 0 + concat a separate mask channel. The model
  references the mask channel to learn whether values are missing.
- ``drop_row``: Exclude any row containing even one NaN (for training; unsuitable at inference time).
"""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from qadmet.core.registry import Registry


class MissingPolicy(Protocol):
    """Strategy interface: array → (cleaned_array, mask_array_or_None)."""

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]: ...


MISSING_POLICIES: Registry[MissingPolicy] = Registry("missing_policies")


@MISSING_POLICIES.register("zero")
def _zero(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), None


@MISSING_POLICIES.register("mask_token")
def _mask_token(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    """Returns accompanied by a 0/1 mask per missing channel.

    Note:
        The returned mask is used by the caller via e.g. ``np.concatenate([cleaned, mask], axis=1)``
        Since this increases the model input dimension, it must be reflected when computing input dim in BackboneSpec.
    """
    mask = np.isnan(x).astype(np.float32)
    cleaned = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return cleaned, mask


@MISSING_POLICIES.register("drop_row")
def _drop_row(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    keep = ~np.isnan(x).any(axis=1)
    return x[keep].astype(np.float32), keep.astype(bool)


def apply_policy(name: str, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    """Convenience function: registry lookup + apply."""
    return MISSING_POLICIES.get(name)(x)
