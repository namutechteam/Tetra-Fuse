"""OpenADMET PXR Blind Challenge — PXR-specific TaskSet.

4 regression tasks configured (primary + 3 aux):
  - PXR_pEC50         : Challenge evaluation target (RAE)
  - PXR_Emax          : log2FC vs baseline, aux
  - PXR_selectivity   : PXR pEC50 - null pEC50 (batch-free differential signal), aux
  - PXR_null_pEC50    : PXR-null counter assay pEC50, aux

Note:
    PXR_pEC50 uses the value, pre-centered in datasets/pxr_activity.py with batch-004 reference frame, as the
    training target (corrects for OCNT batch effect in the cell).
"""

from __future__ import annotations

from qadmet.tasks.registry import Task, TaskSet, register_task, register_task_set


_PXR_REGRESSION = [
    Task("PXR_pEC50",       "regression", "rae",
         notes="batch-centered to batch-004 frame; OpenADMET primary metric"),
    Task("PXR_Emax",        "regression", "mae",
         notes="log2FC vs baseline; aux"),
    Task("PXR_selectivity", "regression", "mae",
         notes="PXR pEC50 − null pEC50 (batch-free same-plate diff); aux"),
    Task("PXR_null_pEC50",  "regression", "mae",
         notes="counter-assay (PXR-null line) pEC50; aux"),
]

for _t in _PXR_REGRESSION:
    register_task(_t)

ALL_PXR_TASKS: tuple[str, ...] = tuple(t.name for t in _PXR_REGRESSION)


# ---------------------------------------------------------------------------- 25-task(TDC22+PXR3) design
# plan_details.md's new design: source-dataset=task. PXR_pEC50/PXR_null_pEC50 are reused from above,
# screen(single-conc 8.3µM log2FC) only is new. selectivity/Emax/crude are not turned into tasks.
register_task(
    Task("PXR_screen_log2fc", "regression", "spearman",
         notes="single-conc 8.251µM log2_fc_estimate (Library, SMILES-mean); aux, drc pEC50 and ρ0.73")
)

# tdc_pxr_25 grouping/loader references PXR 3-task (order fixed).
PXR_3: tuple[str, ...] = ("PXR_pEC50", "PXR_null_pEC50", "PXR_screen_log2fc")


# Single-group routing — all PXR task share the same head trunk (4 task are all
# different perspectives of the molecule's PXR-induction signal, so grouping splitting has no benefit).
PXR_1_GROUP: dict[str, tuple[str, ...]] = {
    "pxr_induction": ALL_PXR_TASKS,
}


register_task_set(
    TaskSet(
        name="pxr_4task",
        tasks=ALL_PXR_TASKS,
        initial_grouping=PXR_1_GROUP,
        notes=(
            "PXR challenge 4-task MTL (1 primary + 3 aux). scaffold_5seed split."
        ),
    )
)

register_task_set(
    TaskSet(
        name="pxr_4task_active",
        tasks=ALL_PXR_TASKS,
        initial_grouping=PXR_1_GROUP,
        notes=(
            "PXR challenge 4-task MTL — active-cluster CV split (test ≈ analog set)."
        ),
    )
)

# STL variant (primary only)
register_task_set(
    TaskSet(
        name="pxr_pec50_only",
        tasks=("PXR_pEC50",),
        initial_grouping={"pxr_induction": ("PXR_pEC50",)},
        notes="PXR primary task STL (scaffold split).",
    )
)

register_task_set(
    TaskSet(
        name="pxr_pec50_only_active",
        tasks=("PXR_pEC50",),
        initial_grouping={"pxr_induction": ("PXR_pEC50",)},
        notes="PXR primary task STL (active-cluster split).",
    )
)
