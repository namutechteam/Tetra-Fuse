"""TDC ADMET 22 task 등록 + grouping TaskSet 로드.

원본: scripts-old/pipeline_mtl.py:52-127.
Task 메타데이터(type/metric/log_transform)와 task 목록·순서는 값을 바꾸지 말 것
— leaderboard 점수 비교의 단일 진실원.

grouping(어떤 task 가 어떤 그룹인지)은 ``recipes/groupings/*.yaml`` 로 외부화되어
import 시 :func:`qadmet.tasks.grouping_loader.load_grouping_task_sets` 가 읽어 등록한다.
g1/clfreg/pertask 비교군은 ``ALL_TASKS`` 에서 기계적으로 도출되므로 코드에 둔다.
"""

from __future__ import annotations

from qadmet.tasks.grouping_loader import load_grouping_task_sets
from qadmet.tasks.registry import Task, TaskSet, register_task, register_task_set

# ---------------------------------------------------------------------------- 분류 task

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

# ---------------------------------------------------------------------------- 회귀 task

_REGRESSION = [
    Task("Caco2_Wang", "regression", "mae"),
    Task("Lipophilicity_AstraZeneca", "regression", "mae"),
    Task("Solubility_AqSolDB", "regression", "mae"),
    Task("PPBR_AZ", "regression", "mae"),
    # 아래 4 개 PK 파라미터는 heavy-tail 분포 → log1p 변환.
    Task("VDss_Lombardo", "regression", "spearman", log_transform=True),
    Task("Half_Life_Obach", "regression", "spearman", log_transform=True),
    Task("Clearance_Hepatocyte_AZ", "regression", "spearman", log_transform=True),
    Task("Clearance_Microsome_AZ", "regression", "spearman", log_transform=True),
    Task("LD50_Zhu", "regression", "mae"),
]

for _t in _CLASSIFICATION + _REGRESSION:
    register_task(_t)


# ---------------------------------------------------------------------------- canonical task 목록

# task 목록·순서의 단일 진실원. grouping YAML 의 ``tasks: tdc_admet_22`` 가 이 순서를 참조해
# grouping 구성과 무관하게 동일 task 순서를 유지한다 (체크포인트/leaderboard parity).
ALL_TASKS: tuple[str, ...] = tuple(t.name for t in _CLASSIFICATION + _REGRESSION)

# grouping YAML 의 ``tasks`` 필드가 참조할 수 있는 canonical task 목록.
TASK_LISTS: dict[str, tuple[str, ...]] = {
    "tdc_admet_22": ALL_TASKS,
}

# ---------------------------------------------------------------------------- TDC22 + PXR3 = 25-task
# plan_details.md §2.2. PXR 3-task 를 TDC 22 뒤에 append (순서 고정 = parity).
# pxr.py 는 registry 만 의존 → 순환 import 없음. grouping_loader 호출 전에 등록해야
# grouping YAML 의 ``tasks: tdc_pxr_25`` 가 resolve 된다.
from qadmet.tasks.pxr import PXR_3  # noqa: E402

TDC_PXR_25: tuple[str, ...] = ALL_TASKS + PXR_3
TASK_LISTS["tdc_pxr_25"] = TDC_PXR_25

# STL-CheMeleon 용 — 25-task 전부 singleton group (grouping 없이 per-task).
register_task_set(
    TaskSet(
        name="tdc_pxr_25_pertask",
        tasks=TDC_PXR_25,
        initial_grouping={t: (t,) for t in TDC_PXR_25},
        notes="TDC22+PXR3 25-task, task 당 singleton group (STL-CheMeleon, grouping 없음).",
    )
)


# ---------------------------------------------------------------------------- grouping TaskSet 로드
# recipes/groupings/*.yaml → register_task_set:
#   - tdc_admet_22          (경험적 7-group, baseline)
#   - tdc_admet_22_mech     (mechanism-aligned 7-group)
#   - tdc_admet_22_causal6  (causal-chain 6-group)
#   - tdc_admet_5_sanity    (4-modality overfit 진단 5-task subset)
load_grouping_task_sets(TASK_LISTS)


# ---------------------------------------------------------------------------- 그룹 탐색용 reference TaskSet
# tdc_admet_22 와 task list 는 동일 — grouping 만 다른 비교군 (routing group 실험).
# 모두 22 task 동일 데이터/평가셋 → objective 직접 비교 가능.
# 아래 3 개는 ALL_TASKS / task type 에서 기계적으로 도출되는 baseline 이라 YAML 외부화 대신
# 코드에서 생성한다 (외부화하면 ALL_TASKS 와 중복·drift 위험).
_CLS_NAMES: tuple[str, ...] = tuple(t.name for t in _CLASSIFICATION)
_REG_NAMES: tuple[str, ...] = tuple(t.name for t in _REGRESSION)

# G=1 — 전 task hard-share (단일 trunk).
register_task_set(
    TaskSet(
        name="tdc_admet_22_g1",
        tasks=ALL_TASKS,
        initial_grouping={"all": ALL_TASKS},
        notes="TDC ADMET 22-task, 단일 그룹 (완전 hard-share) — grouping 실험 하한 비교군.",
    )
)

# G=2 — 분류/회귀 분리.
register_task_set(
    TaskSet(
        name="tdc_admet_22_clfreg",
        tasks=ALL_TASKS,
        initial_grouping={"classification": _CLS_NAMES, "regression": _REG_NAMES},
        notes="TDC ADMET 22-task, clf(13)/reg(9) 2-group — grouping 실험 비교군.",
    )
)

# G=22 — task 당 독립 trunk (공유 없음).
register_task_set(
    TaskSet(
        name="tdc_admet_22_pertask",
        tasks=ALL_TASKS,
        initial_grouping={t: (t,) for t in ALL_TASKS},
        notes="TDC ADMET 22-task, task 당 singleton group (공유 없음) — grouping 실험 비교군.",
    )
)
