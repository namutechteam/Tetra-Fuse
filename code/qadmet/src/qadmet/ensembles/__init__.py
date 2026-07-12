"""ensembles: PredictionSet / MetaDataset / MetaLearner / Calibrator 파이프라인.

옛 ``scripts-old/run_stacking_v{1..6}.py`` 의 6 가지 stacking 전략을 ``MetaLearner``
서브클래스로 통합하고, ``train`` runner 가 만든 ``predictions.parquet`` (long 포맷)
를 ``PredictionSet`` 단일 진입점으로 재해석한다.

import 만으로 ``STRATEGIES`` 가 채워진다 (strategies 모듈이 데코레이터 등록 트리거).
"""

from qadmet.ensembles.registry import (
    CALIBRATORS,
    META_MODELS,
    STRATEGIES,
)
from qadmet.ensembles import strategies as _strategies  # noqa: F401  — strategy 등록 트리거
from qadmet.ensembles import calibrators as _calibrators  # noqa: F401  — calibrator 등록 트리거
from qadmet.ensembles import meta_models as _meta_models  # noqa: F401  — meta_model 등록 트리거

__all__ = [
    "STRATEGIES",
    "META_MODELS",
    "CALIBRATORS",
]
