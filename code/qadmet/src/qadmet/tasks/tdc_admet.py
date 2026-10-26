"""TDC ADMET 22 task registration + grouping TaskSet loading.

Original: scripts-old/pipeline_mtl.py:52-127.
Task metadata (type/metric/log_transform) and the task list/order must not have their values changed
— the single source of truth for leaderboard score comparison.

grouping (which task belongs to which group) is externalized to ``recipes/groupings/*.yaml``,
and on import :func:`qadmet.tasks.grouping_loader.load_grouping_task_sets` reads and registers it.
The g1/clfreg/pertask comparison groups are mechanically derived from ``ALL_TASKS``, so they are kept in code.
"""

from __future__ import annotations

from qadmet.tasks.grouping_loader import load_grouping_task_sets
from qadmet.tasks.registry import Task, TaskSet, register_task, register_task_set

# ---------------------------------------------------------------------------- classification tasks

_CLASSIFICATION = [
    Task("HIA_Hou", "classification", "auroc"),
    Task("Pgp_Broccatelli", "classification", "auroc"),
    Task("Bioavailability_Ma", "classification", "auroc"),
    Task("BBB_Martins", "classification", "auroc"),
    Task("CYP2C9_Veith", "classification", "auprc"),
    Task("CYP2D6_Veith", "classification", "auprc"),
    Task("CYP3A4_Veith", "classification", "auprc"),
    Task("CYP2C9_Substrate_CarbonMangels", "classification", "auprc"),
    Task("CYP2D6_Substrate_CarbonMangels", "classification", "auprc"),
    Task("CYP3A4_Substrate_CarbonMangels", "classification", "auroc"),
    Task("hERG", "classification", "auroc"),
    Task("AMES", "classification", "auroc"),
    Task("DILI", "classification", "auroc"),
]

# ---------------------------------------------------------------------------- regression tasks

_REGRESSION = [
    Task("Caco2_Wang", "regression", "mae"),
    Task("Lipophilicity_AstraZeneca", "regression", "mae"),
    Task("Solubility_AqSolDB", "regression", "mae"),
    Task("PPBR_AZ", "regression", "mae"),
    # The 4 PK parameters below have heavy-tail distributions → log1p transform.
    Task("VDss_Lombardo", "regression", "spearman", log_transform=True),
    Task("Half_Life_Obach", "regression", "spearman", log_transform=True),
    Task("Clearance_Hepatocyte_AZ", "regression", "spearman", log_transform=True),
    Task("Clearance_Microsome_AZ", "regression", "spearman", log_transform=True),
    Task("LD50_Zhu", "regression", "mae"),
]

for _t in _CLASSIFICATION + _REGRESSION:
    register_task(_t)


# ---------------------------------------------------------------------------- canonical task list

# Single source of truth for the task list/order. ``tasks: tdc_admet_22`` in the grouping YAML references this order,
# maintaining the same task order regardless of grouping configuration (checkpoint/leaderboard parity).
ALL_TASKS: tuple[str, ...] = tuple(t.name for t in _CLASSIFICATION + _REGRESSION)

# Canonical task list that the ``tasks`` field in the grouping YAML can reference.
TASK_LISTS: dict[str, tuple[str, ...]] = {
    "tdc_admet_22": ALL_TASKS,
}

# ---------------------------------------------------------------------------- TDC22 + PXR3 = 25-task
# plan_details.md §2.2. Append PXR 3-task after TDC 22 (fixed order = parity).
# pxr.py only depends on the registry → no circular import. Must be registered before grouping_loader is called
# so that ``tasks: tdc_pxr_25`` in the grouping YAML resolves.
from qadmet.tasks.pxr import PXR_3  # noqa: E402

TDC_PXR_25: tuple[str, ...] = ALL_TASKS + PXR_3
TASK_LISTS["tdc_pxr_25"] = TDC_PXR_25

# For STL-CheMeleon — all 25 tasks are singleton groups (per-task, no grouping).
register_task_set(
    TaskSet(
        name="tdc_pxr_25_pertask",
        tasks=TDC_PXR_25,
        initial_grouping={t: (t,) for t in TDC_PXR_25},
        notes="TDC22+PXR3 25-task, singleton group per task (STL-CheMeleon, no grouping).",
    )
)


# ---------------------------------------------------------------------------- grouping TaskSet loading
# recipes/groupings/*.yaml → register_task_set:
#   - tdc_admet_22          (empirical 7-group, baseline)
#   - tdc_admet_22_mech     (mechanism-aligned 7-group)
#   - tdc_admet_22_causal6  (causal-chain 6-group)
#   - tdc_admet_5_sanity    (5-task subset for 4-modality overfit diagnosis)
load_grouping_task_sets(TASK_LISTS)


# ---------------------------------------------------------------------------- reference TaskSet for group exploration
# Task list is identical to tdc_admet_22 — only the grouping differs (comparison group for routing group experiments).
# All 22 tasks share the same data/eval set → objectives can be compared directly.
# The 3 below are baselines mechanically derived from ALL_TASKS / task type, so instead of externalizing to YAML
# they are generated in code (externalizing would risk duplication/drift with ALL_TASKS).
_CLS_NAMES: tuple[str, ...] = tuple(t.name for t in _CLASSIFICATION)
_REG_NAMES: tuple[str, ...] = tuple(t.name for t in _REGRESSION)

# G=1 — hard-share across all tasks (single trunk).
register_task_set(
    TaskSet(
        name="tdc_admet_22_g1",
        tasks=ALL_TASKS,
        initial_grouping={"all": ALL_TASKS},
        notes="TDC ADMET 22-task, single group (fully hard-share) — lower-bound comparison group for grouping experiments.",
    )
)

# G=2 — classification/regression split.
register_task_set(
    TaskSet(
        name="tdc_admet_22_clfreg",
        tasks=ALL_TASKS,
        initial_grouping={"classification": _CLS_NAMES, "regression": _REG_NAMES},
        notes="TDC ADMET 22-task, clf(13)/reg(9) 2-group — comparison group for grouping experiments.",
    )
)

# G=22 — independent trunk per task (no sharing).
register_task_set(
    TaskSet(
        name="tdc_admet_22_pertask",
        tasks=ALL_TASKS,
        initial_grouping={t: (t,) for t in ALL_TASKS},
        notes="TDC ADMET 22-task, singleton group per task (no sharing) — comparison group for grouping experiments.",
    )
)
