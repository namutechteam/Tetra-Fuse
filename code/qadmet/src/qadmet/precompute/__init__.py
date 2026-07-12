"""feature 사전 계산 베이스. builder 등록은 ``qadmet.features.builders`` 가 담당."""

from qadmet.precompute.base import PrecomputeBase  # noqa: F401
import qadmet.features  # noqa: F401  — builder 등록 트리거 (idempotent)
