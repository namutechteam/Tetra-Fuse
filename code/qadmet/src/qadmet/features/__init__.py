"""feature 추상화: builder registry, parquet/LMDB 캐시, scaler, missing policy.

``qadmet.features`` import 만으로 5 개 builder (``rdkit2d`` / ``xtb`` / ``molformer``
/ ``molformer_tokens`` / ``unimol2``) 가 ``FEATURE_BUILDERS`` 에 등록된다.
``runners.train`` 등 데이터 경로에서 builder lookup 이 가능해진다.
"""

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.features.scalers import BlockScaler, fit_block_scalers
from qadmet.features.stores import ParquetStore
from qadmet.features.missing import MISSING_POLICIES, MissingPolicy
from qadmet.features import builders as _builders  # noqa: F401  — builder 등록 트리거

__all__ = [
    "FEATURE_BUILDERS",
    "BlockScaler",
    "fit_block_scalers",
    "ParquetStore",
    "MISSING_POLICIES",
    "MissingPolicy",
]
