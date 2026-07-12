"""Polaris ADME-Fang external benchmark — 6-task regression TaskSet.

Fang et al. (2023) DMPK dataset via Polaris hub (biogen/adme-fang-*-reg-v1).
6 회귀 endpoint, 동일 화합물군에서 측정된 multitask 구조 (union 3,104 compounds,
85%가 >=2 task에 등장). TDC 22-task와 **합치지 않고** 독립 외부 MTL 블록으로 학습.

Data: data/polaris_adme_fang/<short>_{train_val,test}.csv (fetch_polaris.py 산출).
  train_val: [Drug, Y] (Y visible)
  test:      [Drug]     (Polaris가 test target을 숨김 → 진짜 held-out; 제출로 평가)

Metric: Polaris ADME-Fang 회귀는 Pearson r 기준 → task.metric="pearson".
Split: train_val을 seed별 random 분할(val_frac=0.125), PXR loader와 동일 방식.
"""

from __future__ import annotations

from qadmet.tasks.registry import Task, TaskSet, register_task, register_task_set


# short → (TDC-스타일 task name, Polaris target col). task name은 checkpoint/leaderboard 키.
_FANG_REGRESSION = [
    Task("Fang_HLM_CLint",  "regression", "pearson",
         notes="Polaris biogen/adme-fang-hclint-reg-v1; LOG_HLM_CLint (human liver microsome clearance)"),
    Task("Fang_RLM_CLint",  "regression", "pearson",
         notes="Polaris biogen/adme-fang-rclint-reg-v1; LOG_RLM_CLint (rat liver microsome clearance)"),
    Task("Fang_MDR1_ER",    "regression", "pearson",
         notes="Polaris biogen/adme-fang-perm-reg-v1; LOG_MDR1-MDCK_ER (efflux ratio / permeability)"),
    Task("Fang_Solubility", "regression", "pearson",
         notes="Polaris biogen/adme-fang-solu-reg-v1; LOG_SOLUBILITY"),
    Task("Fang_HPPB",       "regression", "pearson",
         notes="Polaris biogen/adme-fang-hppb-reg-v1; LOG_HPPB (human plasma protein binding)"),
    Task("Fang_RPPB",       "regression", "pearson",
         notes="Polaris biogen/adme-fang-rppb-reg-v1; LOG_RPPB (rat plasma protein binding)"),
]

for _t in _FANG_REGRESSION:
    register_task(_t)

FANG_6: tuple[str, ...] = tuple(t.name for t in _FANG_REGRESSION)

# short-code → CSV 파일 stem 매핑 (loader가 참조).
FANG_CSV_STEM: dict[str, str] = {
    "Fang_HLM_CLint":  "hclint",
    "Fang_RLM_CLint":  "rclint",
    "Fang_MDR1_ER":    "perm",
    "Fang_Solubility": "solu",
    "Fang_HPPB":       "hppb",
    "Fang_RPPB":       "rppb",
}

# Single-group routing — 6 endpoint 모두 같은 head trunk 공유. 전부 ADME 물성이라
# grouping splitting의 이득이 뚜렷하지 않으므로 우선 1-group으로 시작(추후 비교 가능).
FANG_1_GROUP: dict[str, tuple[str, ...]] = {"adme_fang": FANG_6}

register_task_set(
    TaskSet(
        name="polaris_fang_6",
        tasks=FANG_6,
        initial_grouping=FANG_1_GROUP,
        notes=(
            "Polaris ADME-Fang external benchmark, 6-task regression MTL "
            "(independent block, NOT merged with TDC-22). Pearson metric, "
            "random val split (val_frac=0.125)."
        ),
    )
)

# STL 변종 — 각 endpoint 단독 (per-task baseline).
register_task_set(
    TaskSet(
        name="polaris_fang_6_pertask",
        tasks=FANG_6,
        initial_grouping={"adme_fang": FANG_6},
        notes="Polaris ADME-Fang 6-task, per-task STL baseline (same data, STL routing).",
    )
)
