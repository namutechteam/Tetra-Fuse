"""TaskRegistry: stores metadata for a single task (type, metric, log-transform flag, etc.).

TaskSet is a bundle of tasks + grouping definition. Referenced via ``dataset.name`` in the recipe
(e.g., ``tdc_admet_22``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from qadmet.core.registry import Registry


@dataclass(frozen=True)
class Task:
    """Metadata for a single ADMET task.

    Attributes:
        name: TDC dataset name (e.g., "Caco2_Wang"). Key for leaderboard/checkpoint.
        type: "classification" | "regression".
        metric: metric key to pass to TDC group.evaluate_many() (e.g., "mae", "auroc").
        log_transform: if True, applies ``np.log1p`` before training (used for heavy-tail regression).
        notes: free-form notes.
    """

    name: str
    type: str
    metric: str
    log_transform: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("classification", "regression"):
            raise ValueError(f"Task.type must be classification|regression: {self.type!r}")


@dataclass(frozen=True)
class TaskSet:
    """A bundle of tasks + (optional) grouping definition.

    grouping is a ``{group_name: [task_name, ...]}`` mapping. learned_grouping and other
    dynamic grouping strategies are used as ``initial_grouping``.
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
    """Registers and returns a Task instance."""
    TASK_REGISTRY.register(task.name)(task)  # type: ignore[arg-type]
    return task


def register_task_set(ts: TaskSet) -> TaskSet:
    TASK_SETS.register(ts.name)(ts)  # type: ignore[arg-type]
    return ts


def resolve_task_types(task_names: Sequence[str]) -> list[str]:
    """Name list → type list (raises ValueError if not registered in registry)."""
    return [TASK_REGISTRY.get(t).type for t in task_names]
