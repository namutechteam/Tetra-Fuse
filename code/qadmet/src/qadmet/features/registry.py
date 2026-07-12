"""FeatureBuilderRegistry: 모달리티별 feature 계산기 등록.

각 builder 는 ``compute(smiles_list) -> dict[str, np.ndarray]`` 인터페이스만 만족하면 된다.
구체 builder 는 ``qadmet.features.builders.*`` 에 위치한다.
"""

from __future__ import annotations

from qadmet.core.registry import Registry

FEATURE_BUILDERS: Registry = Registry("feature_builders")
