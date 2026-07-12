"""task grouping 헬퍼.

grouping 표현 형식::

    grouping = {"transporter": ("HIA_Hou", "Pgp_Broccatelli", ...), ...}

학습 코드(GroupedHead 등)가 사용하기 쉬운 형태로 변환하는 유틸을 제공한다.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def task_to_group_index(
    grouping: Mapping[str, Sequence[str]], task_order: Sequence[str]
) -> list[int]:
    """task_order 의 각 task 가 속한 그룹 인덱스(0..G-1) 리스트.

    Why:
        GroupedHead 가 task → group routing 시 사용. 같은 인덱스 매핑을 학습/평가
        양쪽에서 일관되게 쓸 수 있도록 헬퍼로 분리.

    Raises:
        ValueError: task_order 의 task 가 어느 그룹에도 없으면.
    """
    name_to_idx: dict[str, int] = {}
    for gi, (gname, members) in enumerate(grouping.items()):
        for t in members:
            name_to_idx[t] = gi
    out: list[int] = []
    missing: list[str] = []
    for t in task_order:
        if t not in name_to_idx:
            missing.append(t)
        else:
            out.append(name_to_idx[t])
    if missing:
        raise ValueError(f"grouping 에 누락된 task: {missing}")
    return out


def n_groups(grouping: Mapping[str, Sequence[str]]) -> int:
    return len(grouping)


def group_names(grouping: Mapping[str, Sequence[str]]) -> list[str]:
    return list(grouping.keys())
