"""``recipes/groupings/*.yaml`` → TaskSet 등록.

grouping(어떤 task 가 어떤 그룹에 속하는지)을 코드 하드코딩 대신 YAML 로 외부화한다.
Task 메타데이터(type/metric/log_transform)와 canonical task 순서(``ALL_TASKS`` 등)는
여전히 ``tdc_admet.py`` 가 단일 진실원 — YAML 은 grouping 만 담는다.

YAML 형식::

    name: tdc_admet_22_causal6        # 등록될 TaskSet 이름
    tasks: tdc_admet_22               # task 목록·순서 출처 (task_lists 키) 또는 "auto"
    notes: "..."
    groups:
      oral_uptake: [Solubility_AqSolDB, Caco2_Wang, ...]
      ...

``tasks`` 가 ``"auto"`` (또는 생략) 면 task 목록·순서를 groups 펼친 순서에서 도출한다
(subset taskset 용, 예: 5_sanity). 명시된 키면 그 canonical 순서를 강제한다 — 22-variant
는 grouping 순서와 무관하게 동일 task 순서를 유지해 체크포인트/leaderboard parity 를 보존한다.
"""

from __future__ import annotations

from typing import Mapping

import yaml

from qadmet.core.paths import recipes_dir
from qadmet.tasks.registry import TaskSet, register_task_set


def _validate_partition(
    name: str, tasks: tuple[str, ...], grouping: Mapping[str, tuple[str, ...]]
) -> None:
    """grouping 이 tasks 를 정확히 한 번씩 덮는 partition 인지 검증."""
    covered = [t for ts in grouping.values() for t in ts]
    dupes = sorted({t for t in covered if covered.count(t) > 1})
    if dupes:
        raise ValueError(f"grouping YAML({name}): task 가 여러 그룹에 중복 = {dupes}")
    if set(covered) != set(tasks):
        missing = sorted(set(tasks) - set(covered))
        extra = sorted(set(covered) - set(tasks))
        raise ValueError(
            f"grouping YAML({name}) 가 task 목록을 정확히 덮지 못함 — "
            f"missing={missing}, extra={extra}"
        )


def load_grouping_task_sets(task_lists: Mapping[str, tuple[str, ...]]) -> list[str]:
    """``recipes/groupings/*.yaml`` 을 전부 읽어 TaskSet 으로 등록.

    Args:
        task_lists: ``tasks`` 필드가 참조할 수 있는 canonical task 목록 매핑
            (예: ``{"tdc_admet_22": ALL_TASKS}``).

    Returns:
        등록한 TaskSet 이름 리스트 (등록 순서).

    Raises:
        FileNotFoundError: groupings 디렉토리가 없으면.
        KeyError: ``tasks`` 가 알 수 없는 task_lists 키면.
        ValueError: grouping 이 task 목록의 partition 이 아니면.
    """
    gdir = recipes_dir() / "groupings"
    if not gdir.is_dir():
        raise FileNotFoundError(f"grouping 디렉토리 없음: {gdir}")

    registered: list[str] = []
    for path in sorted(gdir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = data["name"]
        grouping = {g: tuple(members) for g, members in data["groups"].items()}

        tasks_ref = data.get("tasks", "auto")
        if tasks_ref in (None, "auto"):
            # groups 펼친 순서에서 task 목록 도출 (subset taskset).
            tasks = tuple(t for ts in grouping.values() for t in ts)
        elif tasks_ref in task_lists:
            tasks = tuple(task_lists[tasks_ref])
        else:
            raise KeyError(
                f"grouping YAML({name}).tasks={tasks_ref!r} 미등록 — "
                f"알려진 키: {sorted(task_lists)} (또는 'auto')"
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
