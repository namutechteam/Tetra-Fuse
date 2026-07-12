"""TaskRegistry: task 한 건의 메타데이터(타입·메트릭·로그변환 여부 등) 보관.

TaskSet 은 task 들의 묶음 + grouping 정의. recipe 에서 ``dataset.name`` 으로
참조한다 (예: ``tdc_admet_22``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from qadmet.core.registry import Registry


@dataclass(frozen=True)
class Task:
    """ADMET task 한 건의 메타데이터.

    Attributes:
        name: TDC 데이터셋 이름 (예: "Caco2_Wang"). leaderboard·체크포인트 키.
        type: "classification" | "regression".
        metric: TDC group.evaluate_many() 에 넘길 metric key (예: "mae", "auroc").
        log_transform: True 면 학습 전 ``np.log1p`` 적용 (heavy-tail 회귀에 사용).
        notes: 자유 메모.
    """

    name: str
    type: str
    metric: str
    log_transform: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("classification", "regression"):
            raise ValueError(f"Task.type 은 classification|regression 이어야 합니다: {self.type!r}")


@dataclass(frozen=True)
class TaskSet:
    """task 들의 묶음 + (선택) grouping 정의.

    grouping 은 ``{group_name: [task_name, ...]}`` 매핑. learned_grouping 등
    동적 그룹 전략은 ``initial_grouping`` 으로 사용된다.
    """

    name: str
    tasks: tuple[str, ...]
    initial_grouping: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    notes: str = ""

    def get_tasks(self) -> tuple[Task, ...]:
        return tuple(TASK_REGISTRY.get(t) for t in self.tasks)

    def task_types(self) -> list[str]:
        return [TASK_REGISTRY.get(t).type for t in self.tasks]

    def classification_tasks(self) -> list[str]:
        return [t for t in self.tasks if TASK_REGISTRY.get(t).type == "classification"]

    def regression_tasks(self) -> list[str]:
        return [t for t in self.tasks if TASK_REGISTRY.get(t).type == "regression"]

    def log_transform_tasks(self) -> list[str]:
        return [t for t in self.tasks if TASK_REGISTRY.get(t).log_transform]

    def metric_for(self, task_name: str) -> str:
        return TASK_REGISTRY.get(task_name).metric


TASK_REGISTRY: Registry[Task] = Registry("tasks")
TASK_SETS: Registry[TaskSet] = Registry("task_sets")


def register_task(task: Task) -> Task:
    """Task 인스턴스를 등록하고 반환."""
    TASK_REGISTRY.register(task.name)(task)  # type: ignore[arg-type]
    return task


def register_task_set(ts: TaskSet) -> TaskSet:
    TASK_SETS.register(ts.name)(ts)  # type: ignore[arg-type]
    return ts


def resolve_task_types(task_names: Sequence[str]) -> list[str]:
    """이름 목록 → 타입 리스트 (registry 미등록 시 ValueError)."""
    return [TASK_REGISTRY.get(t).type for t in task_names]
