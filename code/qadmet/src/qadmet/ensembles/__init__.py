"""ensembles: PredictionSet / MetaDataset / MetaLearner / Calibrator pipeline.

Legacy ``scripts-old/run_stacking_v{1..6}.py`` 's 6 stacking strategies into ``MetaLearner``
subclasses, integrating what the ``train`` runner produces, ``predictions.parquet`` (long format)
into ``PredictionSet`` as a single entry point and reinterprets it.

Just by import, ``STRATEGIES`` gets populated (the strategies module triggers decorator registration).
"""

from qadmet.ensembles.registry import (
    CALIBRATORS,
    META_MODELS,
    STRATEGIES,
)
from qadmet.ensembles import strategies as _strategies  # noqa: F401  — strategy registration trigger
from qadmet.ensembles import calibrators as _calibrators  # noqa: F401  — calibrator registration trigger
from qadmet.ensembles import meta_models as _meta_models  # noqa: F401  — meta_model registration trigger

__all__ = [
    "STRATEGIES",
    "META_MODELS",
    "CALIBRATORS",
]
