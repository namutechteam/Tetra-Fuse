"""feature abstraction: builder registry, parquet/LMDB cache, scaler, missing policy.

``qadmet.features`` import alone registers the 5 builders (``rdkit2d`` / ``xtb`` / ``molformer``
/ ``molformer_tokens`` / ``unimol2``) in ``FEATURE_BUILDERS``.
``runners.train`` etc. builder lookup becomes possible from data paths.
"""

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.features.scalers import BlockScaler, fit_block_scalers
from qadmet.features.stores import ParquetStore
from qadmet.features.missing import MISSING_POLICIES, MissingPolicy
from qadmet.features import builders as _builders  # noqa: F401  — triggers builder registration

__all__ = [
    "FEATURE_BUILDERS",
    "BlockScaler",
    "fit_block_scalers",
    "ParquetStore",
    "MISSING_POLICIES",
    "MissingPolicy",
]
