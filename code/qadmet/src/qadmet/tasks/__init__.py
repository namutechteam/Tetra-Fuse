"""TDC ADMET task definition — single source of truth.

When ``qadmet.tasks`` is imported, ``TASK_REGISTRY`` and ``TASK_SETS`` are automatically initialized.
Other domain modules must obtain all task name, metric, and group information only from this package.
"""

from qadmet.tasks.registry import TASK_REGISTRY, TASK_SETS, Task, TaskSet
from qadmet.tasks import tdc_admet  # noqa: F401  — import-time registration
from qadmet.tasks import pxr  # noqa: F401  — PXR challenge 4-task
from qadmet.tasks import polaris_fang  # noqa: F401  — Polaris ADME-Fang 6-task

__all__ = ["TASK_REGISTRY", "TASK_SETS", "Task", "TaskSet"]
