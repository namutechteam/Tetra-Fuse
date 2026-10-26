"""FeatureBuilderRegistry: Registers feature calculators per modality.

Each builder only needs to satisfy the ``compute(smiles_list) -> dict[str, np.ndarray]`` interface.
Concrete builders are located in ``qadmet.features.builders.*``.
"""

from __future__ import annotations

from qadmet.core.registry import Registry

FEATURE_BUILDERS: Registry = Registry("feature_builders")
