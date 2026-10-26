"""feature precompute base. builder registration is handled by ``qadmet.features.builders``."""

from qadmet.precompute.base import PrecomputeBase  # noqa: F401
import qadmet.features  # noqa: F401  — builder registration trigger (idempotent)
