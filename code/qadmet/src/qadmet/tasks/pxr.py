"""OpenADMET PXR Blind Challenge — PXR-specific TaskSet.

4 회귀 task로 구성 (primary + 3 aux):
  - PXR_pEC50         : 챌린지 평가 대상 (RAE)
  - PXR_Emax          : log2FC vs baseline, aux
  - PXR_selectivity   : PXR pEC50 - null pEC50 (배치-free 차분 신호), aux
  - PXR_null_pEC50    : PXR-null 카운터 어세이 pEC50, aux

Note:
    PXR_pEC50 는 datasets/pxr_activity.py 에서 batch-004 reference frame 으로 미리
    중심화된 값을 학습 target 으로 쓴다 (셀의 OCNT batch 효과 보정).
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


# ---------------------------------------------------------------------------- 25-task(TDC22+PXR3) 디자인
# plan_details.md 의 새 설계: source-dataset=task. PXR_pEC50/PXR_null_pEC50 는 위에서 재사용,
# screen(single-conc 8.3µM log2FC) 만 신규. selectivity/Emax/crude 는 task 화하지 않음.
register_task(
    Task("PXR_screen_log2fc", "regression", "spearman",
         notes="single-conc 8.251µM log2_fc_estimate (Library, SMILES-mean); aux, drc pEC50 와 ρ0.73")
)

# tdc_pxr_25 grouping/loader 가 참조하는 PXR 3-task (순서 고정).
PXR_3: tuple[str, ...] = ("PXR_pEC50", "PXR_null_pEC50", "PXR_screen_log2fc")


# Single-group routing — 모든 PXR task 가 같은 head trunk 를 공유 (4 task 가 모두
# 분자의 PXR-induction 신호의 다른 관점이므로 grouping splitting 의 이득 없음).
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

# STL 변종 (primary 만)
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
