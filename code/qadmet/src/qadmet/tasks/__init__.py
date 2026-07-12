"""TDC ADMET task 정의 — 단일 진실원.

``qadmet.tasks`` 를 import 하면 ``TASK_REGISTRY`` 와 ``TASK_SETS`` 가 자동 초기화된다.
다른 도메인 모듈은 task 이름·메트릭·그룹 정보를 모두 이 패키지에서만 가져와야 한다.
"""

from qadmet.tasks.registry import TASK_REGISTRY, TASK_SETS, Task, TaskSet
from qadmet.tasks import tdc_admet  # noqa: F401  — import-time 등록
from qadmet.tasks import pxr  # noqa: F401  — PXR challenge 4-task
from qadmet.tasks import polaris_fang  # noqa: F401  — Polaris ADME-Fang 6-task

__all__ = ["TASK_REGISTRY", "TASK_SETS", "Task", "TaskSet"]
