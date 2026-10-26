"""Registry instances for the ensembles domain.

Three kinds:
    - ``STRATEGIES``: registers ``MetaLearner`` subclasses (legacy stacking v1~v6).
    - ``META_MODELS``: wrappers for scikit-learn-style estimators (XGBoost / Ridge / LogReg).
    - ``CALIBRATORS``: Platt / isotonic calibrators (optional, used in v6 hybrid etc.).

Each Registry is looked up by the ``EnsembleSpec.strategy / meta_model / calibrator`` string.
"""

from __future__ import annotations

from qadmet.core.registry import Registry

STRATEGIES: Registry = Registry("strategies")
META_MODELS: Registry = Registry("meta_models")
CALIBRATORS: Registry = Registry("calibrators")

__all__ = ["STRATEGIES", "META_MODELS", "CALIBRATORS"]
