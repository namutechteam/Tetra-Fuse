"""Original dataset loader + splitter + transform.

This package does not directly depend on chemprop objects (decoupled from training code). Training
pipeline receives this module's output (pd.DataFrame) from ``qadmet.training`` and
converts it to a chemprop datapoint.
"""

from qadmet.datasets.registry import DATASETS
from qadmet.datasets import tdc_admet  # noqa: F401  — import-time registration
from qadmet.datasets import pxr_activity  # noqa: F401  — PXR challenge

__all__ = ["DATASETS"]
