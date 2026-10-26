"""Polaris ADME-Fang external benchmark — 6-task regression TaskSet.

Fang et al. (2023) DMPK dataset via Polaris hub (biogen/adme-fang-*-reg-v1).
6 regression endpoints, multitask structure measured within the same compound set (union 3,104 compounds,
85% appear in >=2 tasks). Trained as an independent external MTL block, **not merged** with TDC 22-task.

Data: data/polaris_adme_fang/<short>_{train_val,test}.csv (output of fetch_polaris.py).
  train_val: [Drug, Y] (Y visible)
  test:      [Drug]     (Polaris hides the test target → true held-out; evaluated via submission)

Metric: Polaris ADME-Fang regression uses Pearson r as the criterion → task.metric="pearson".
Split: train_val split randomly per seed (val_frac=0.125), same approach as the PXR loader.
"""

from __future__ import annotations

from qadmet.tasks.registry import Task, TaskSet, register_task, register_task_set


# short → (TDC-style task name, Polaris target col). task name is the checkpoint/leaderboard key.
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

# short-code → CSV file stem mapping (referenced by the loader).
FANG_CSV_STEM: dict[str, str] = {
    "Fang_HLM_CLint":  "hclint",
    "Fang_RLM_CLint":  "rclint",
    "Fang_MDR1_ER":    "perm",
    "Fang_Solubility": "solu",
    "Fang_HPPB":       "hppb",
    "Fang_RPPB":       "rppb",
}

# Single-group routing — all 6 endpoints share the same head trunk. Since they are all ADME properties,
# the benefit of grouping splitting is not clear, so start with 1-group first (comparison possible later).
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

# STL variant — each endpoint standalone (per-task baseline).
register_task_set(
    TaskSet(
        name="polaris_fang_6_pertask",
        tasks=FANG_6,
        initial_grouping={"adme_fang": FANG_6},
        notes="Polaris ADME-Fang 6-task, per-task STL baseline (same data, STL routing).",
    )
)
