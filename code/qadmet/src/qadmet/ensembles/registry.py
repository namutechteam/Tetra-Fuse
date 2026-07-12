"""ensembles 도메인의 Registry 인스턴스.

세 종류:
    - ``STRATEGIES``: ``MetaLearner`` 서브클래스 등록 (옛 stacking v1~v6).
    - ``META_MODELS``: scikit-learn 류 추정기 래퍼 (XGBoost / Ridge / LogReg).
    - ``CALIBRATORS``: Platt / isotonic 보정기 (선택, v6 hybrid 등에서 사용).

각 Registry 는 ``EnsembleSpec.strategy / meta_model / calibrator`` 문자열로 lookup.
"""

from __future__ import annotations

from qadmet.core.registry import Registry

STRATEGIES: Registry = Registry("strategies")
META_MODELS: Registry = Registry("meta_models")
CALIBRATORS: Registry = Registry("calibrators")

__all__ = ["STRATEGIES", "META_MODELS", "CALIBRATORS"]
