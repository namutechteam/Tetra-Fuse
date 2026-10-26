"""``recipes/groupings/*.yaml`` → TaskSet registration.

Externalizes grouping (which task belongs to which group) into YAML instead of hardcoding it in code.
Task metadata (type/metric/log_transform) and canonical task order (``ALL_TASKS`` etc.) are
still single-sourced from ``tdc_admet.py`` — YAML only holds the grouping.

YAML format::

    name: tdc_admet_22_causal6        # TaskSet name to register
    tasks: tdc_admet_22               # source of task list/order (task_lists key) or "auto"
    notes: "..."
    groups:
      oral_uptake: [Solubility_AqSolDB, Caco2_Wang, ...]
      ...

If ``tasks`` is ``"auto"`` (or omitted), the task list/order is derived from the flattened order of groups
(for subset tasksets, e.g. 5_sanity). If a key is specified, that canonical order is enforced — the 22-variant
keeps the same task order regardless of grouping order, preserving checkpoint/leaderboard parity.
"""

from __future__ import annotations

from typing import Mapping

import yaml

from qadmet.core.paths import recipes_dir
from qadmet.tasks.registry import TaskSet, register_task_set


def _validate_partition(
    name: str, tasks: tuple[str, ...], grouping: Mapping[str, tuple[str, ...]]
) -> None:
    """Validate that grouping is a partition covering tasks exactly once."""
    covered = [t for ts in grouping.values() for t in ts]
    dupes = sorted({t for t in covered if covered.count(t) > 1})
    if dupes:
        raise ValueError(f"grouping YAML({name}): task duplicated across multiple groups = {dupes}")
    if set(covered) != set(tasks):
        missing = sorted(set(tasks) - set(covered))
        extra = sorted(set(covered) - set(tasks))
        raise ValueError(
            f"grouping YAML({name}) does not exactly cover the task list — "
            f"missing={missing}, extra={extra}"
        )


def load_grouping_task_sets(task_lists: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Read all ``recipes/groupings/*.yaml`` files and register them as TaskSets.

    Args:
        task_lists: mapping of canonical task lists that the ``tasks`` field can reference
            (e.g. ``{"tdc_admet_22": ALL_TASKS}``).

    Returns:
        List of registered TaskSet names (in registration order).

    Raises:
        FileNotFoundError: if the groupings directory does not exist.
        KeyError: if ``tasks`` is an unknown task_lists key.
        ValueError: if grouping is not a partition of the task list.
    """
    gdir = recipes_dir() / "groupings"
    if not gdir.is_dir():
        raise FileNotFoundError(f"grouping directory not found: {gdir}")

    registered: list[str] = []
    for path in sorted(gdir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = data["name"]
        grouping = {g: tuple(members) for g, members in data["groups"].items()}

        tasks_ref = data.get("tasks", "auto")
        if tasks_ref in (None, "auto"):
            # Derive task list from flattened order of groups (subset taskset).
            tasks = tuple(t for ts in grouping.values() for t in ts)
        elif tasks_ref in task_lists:
            tasks = tuple(task_lists[tasks_ref])
        else:
            raise KeyError(
                f"grouping YAML({name}).tasks={tasks_ref!r} not registered — "
                f"known keys: {sorted(task_lists)} (or 'auto')"
            )

        _validate_partition(name, tasks, grouping)
        register_task_set(
            TaskSet(
                name=name,
                tasks=tasks,
                initial_grouping=grouping,
                notes=data.get("notes", ""),
            )
        )
        registered.append(name)
    return registered
