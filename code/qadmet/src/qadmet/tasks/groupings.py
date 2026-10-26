"""task grouping helper.

grouping representation format::

    grouping = {"transporter": ("HIA_Hou", "Pgp_Broccatelli", ...), ...}

Provides a utility to convert into a form that is easy for training code (GroupedHead, etc.) to use.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def task_to_group_index(
    grouping: Mapping[str, Sequence[str]], task_order: Sequence[str]
) -> list[int]:
    """List of group indices (0..G-1) each task in task_order belongs to.

    Why:
        GroupedHead uses this for task → group routing. The same index mapping across
        both training/evaluation sides is separated into a helper for consistent use.

    Raises:
        ValueError: if a task in task_order does not belong to any group.
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
        raise ValueError(f"task missing from grouping: {missing}")
    return out


def n_groups(grouping: Mapping[str, Sequence[str]]) -> int:
    return len(grouping)


def group_names(grouping: Mapping[str, Sequence[str]]) -> list[str]:
    return list(grouping.keys())
