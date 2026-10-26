#!/usr/bin/env python
"""Export the regression target scalers needed to un-normalize Tetra-Fuse predictions.

WHY THIS SCRIPT EXISTS
----------------------
`qadmet` normalizes regression targets during training and un-normalizes the
predictions afterwards:

    y_pred = raw_output * scaler.scale_[j] + scaler.mean_[j]     # always
    y_pred = expm1(y_pred)                                       # log1p tasks only

The `StandardScaler` is fitted on the *training fold* labels
(`datasets/loaders.py`, `StandardScaler().fit(train_dset._Y[:, reg_idx])`) and is
**not** stored in the checkpoint or the run artifact. Without it, the exported
PEFT weights produce predictions on a normalized scale — unusable for the ten
regression endpoints. The twelve classification endpoints need only a sigmoid
and are unaffected.

This script recomputes those scalers through the *same* code path the training
runner used (`load_fold` -> optional `log1p` -> `StandardScaler.fit`), so the
values are the ones the models were actually trained against, and writes them
next to the exported weights.

SEED SEMANTICS (verified against the runners, not assumed)
---------------------------------------------------------
`runners/train.py` and `runners/train_stl.py` loop over `spec.trainer.seeds`
(= [0,1,2,3,4]) and pass that same integer straight into
`load_fold(..., seed=seed)`, while naming the checkpoint directory `seed_<n>`.
So `seed_0/` really is TDC fold seed 0. The `dataset.seeds: [1..5]` field that
also appears in the manifests is vestigial and does NOT drive the split.

STL vs MTL
----------
* STL runs fit the scaler on the single-task fold  (`filter_fold_to_task`),
  so each (task, seed) has its own mean/scale.
* MTL runs fit one scaler over all regression columns of the 22-task fold, so
  the mean/scale for a task is shared across that seed's multitask model.
Both cases are emitted per (member, task, seed) so the inference code never has
to re-derive which regime applies.

OUTPUT
------
    <out>/target_scalers.json
      {"<member>": {"<task>": {"<seed>": {"mean": float, "scale": float,
                                          "log1p": bool}}}}

USAGE
-----
    python code/export_scalers.py --out weights            # all members
    python code/export_scalers.py --members stl_chemeleon   # subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# `export_weights.py` owns the member -> (recipe, run_id, multitask) table;
# reuse it so the two exports can never disagree about what a member is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_weights import MEMBERS  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="weights", help="output root (same as export_weights.py)")
    ap.add_argument("--members", nargs="*", default=list(MEMBERS))
    ap.add_argument("--repo-root", default="./qadmet",
                    help="qadmet repo root (holds the cached TDC data/ directory)")
    args = ap.parse_args()

    # qadmet resolves the TDC cache relative to the process CWD, so run from there.
    os.chdir(args.repo_root)

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    import qadmet.modules  # noqa: F401  — registry registration side effect
    from qadmet.datasets.loaders import LOG_TRANSFORM_TASKS
    from qadmet.datasets.tdc_admet import load_fold
    from qadmet.tasks.registry import TASK_SETS

    out: dict = {}
    for member in args.members:
        cfg = MEMBERS[member]
        # Every member trained on the same 22-task TDC dataset; STL just slices it.
        dataset_name = "tdc_admet_22"
        # Task names/types come from the TaskSet registry, exactly as the runners
        # do (`TASK_SETS.get(spec.dataset.name)` -> .tasks / .task_types()).
        task_set = TASK_SETS.get(dataset_name)
        task_names = list(task_set.tasks)
        task_types = list(task_set.task_types())
        out[member] = {}
        for seed in SEEDS:
            fold = load_fold(dataset_name, seed=seed)
            train_df = fold.train_df.copy()

            # Mirror the loader: log1p BEFORE fitting the scaler.
            log_tasks = set()
            for t in task_names:
                idx = task_names.index(t)
                if t in LOG_TRANSFORM_TASKS and task_types[idx] == "regression" \
                        and t in train_df.columns:
                    train_df[t] = np.log1p(train_df[t])
                    log_tasks.add(t)

            reg_tasks = [t for t, ty in zip(task_names, task_types) if ty == "regression"]
            # chemprop stores its label matrix as float32 (`train_dset._Y`), and the
            # scaler in training is fitted on THAT array. Casting to float32 here
            # reproduces the fitted values bit-for-bit; fitting in float64 leaves a
            # ~1e-9 relative offset in `scale_`.
            if cfg["multitask"]:
                # One scaler fitted jointly over all regression columns. sklearn
                # computes per-column statistics and ignores NaN, so the sparse
                # multitask label matrix gives the same per-task values as STL.
                Y = train_df[reg_tasks].to_numpy(dtype=np.float32)
                sc = StandardScaler().fit(Y)
                for j, t in enumerate(reg_tasks):
                    out[member].setdefault(t, {})[str(seed)] = dict(
                        mean=float(sc.mean_[j]), scale=float(sc.scale_[j]),
                        log1p=t in log_tasks)
            else:
                # STL: the fold is filtered to one task, so the scaler is 1-column
                # and fitted only on rows where that label is present.
                for t in reg_tasks:
                    y = train_df[t].to_numpy(dtype=np.float32)
                    y = y[~np.isnan(y)].reshape(-1, 1)
                    sc = StandardScaler().fit(y)
                    out[member].setdefault(t, {})[str(seed)] = dict(
                        mean=float(sc.mean_[0]), scale=float(sc.scale_[0]),
                        log1p=t in log_tasks)
            print(f"{member} seed {seed}: {len(reg_tasks)} regression tasks", flush=True)

    dest = os.path.join(args.out, "target_scalers.json")
    if not os.path.isabs(dest):
        dest = os.path.join(os.getcwd(), dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    n = sum(len(s) for m in out.values() for s in m.values())
    print(f"\nwrote {dest}  ({len(out)} members, {n} (task,seed) scaler entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
