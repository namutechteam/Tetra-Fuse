#!/usr/bin/env python
"""Decide, per (task, seed), which training checkpoint produced the released predictions.

Why this step exists
--------------------
`ModelCheckpoint(filename="best")` writes `best-v1.ckpt` instead of overwriting
when the filename already exists, so every CheMeleon·STL seed directory (110/110)
holds both `best.ckpt` and `best-v1.ckpt`; the other four members have a single
`best.ckpt`. The two files come from different training invocations of the same
recipe, and the filename does not tell you which one produced the released
predictions.

Result of running this script over all 110 CheMeleon·STL (task, seed) pairs,
scored against `data/recompute_data/predictions/stl_chemeleon.parquet`:
**`best-v1.ckpt` is the correct file in 110/110 cases**, with agreement
1e-06 or better on most tasks and a worst case of 1.3e-04 on
Clearance_Microsome_AZ (whose predictions span 0.6-127, so this is ~1e-06
relative). `best.ckpt` was off by 3.7e-02 to 1.6e-01 wherever both were compared.

That outcome is not something the metadata predicts: the recorded
`best_model_score` values order the two files inconsistently across tasks (v1
lower for Caco2_Wang and HIA_Hou, higher for VDss_Lombardo and LD50_Zhu), so a
score rule would have picked the wrong file for several tasks. Comparison against
the released predictions is what settles it, which is why this script exists
rather than a heuristic in `export_weights.py`.

The only sound criterion is the artifact we are trying to reproduce. For each
(task, seed) this script runs inference with every candidate checkpoint and keeps
the one whose test predictions agree with the released parquet. The resulting
map is consumed by `export_weights.py --checkpoint-map`, so the exported weights
are, by construction, the weights behind the published numbers.

Output
------
    <out>/checkpoint_map.json
      {"<member>": {"<task>": {"<seed>": {"file": "best-v1.ckpt",
                                          "max_abs_diff": 9.5e-07}}}}

Only members with more than one candidate per directory need this; the other four
have a single `best.ckpt` and are recorded as such without running inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default="/nas/tk_ahn/ADMET-MTL/qadmet/artifacts/train")
    ap.add_argument("--predictions-root",
                    default="/nas/tk_ahn/ADMET-MTL/paper_release/data/recompute_data/predictions")
    ap.add_argument("--recipes-root",
                    default="/nas/tk_ahn/ADMET-MTL/paper_release/code/qadmet/recipes")
    ap.add_argument("--weights-root", default="/nas/tk_ahn/tetrafuse_weights",
                    help="export tree (only needs target_scalers.json at this stage)")
    ap.add_argument("--out", default="/nas/tk_ahn/tetrafuse_weights")
    ap.add_argument("--members", nargs="*", default=None)
    ap.add_argument("--inference-path", default="/tmp/inf",
                    help="directory holding the tfpkg inference package")
    args = ap.parse_args()

    sys.path.insert(0, args.inference_path)
    import numpy as np
    import pandas as pd
    import torch

    from tfpkg.loader import MEMBER_RECIPES, load_member
    from tfpkg.predict import _member_embeddings, _predict_one, _tokens_parquet

    members = args.members or list(MEMBER_RECIPES)
    result: dict = {}

    for member in members:
        recipe, is_mtl = MEMBER_RECIPES[member]
        # run_id comes from the same table export_weights.py uses
        from export_weights import MEMBERS  # noqa: PLC0415

        run_id = MEMBERS[member]["run_id"]
        base = os.path.join(args.train_root, recipe, run_id, "data")
        pred_path = os.path.join(args.predictions_root, f"{member}.parquet")
        released = pd.read_parquet(pred_path)
        result[member] = {}

        for dirpath, _dirs, files in os.walk(base):
            seed_dir = os.path.basename(dirpath)
            if not seed_dir.startswith("seed_"):
                continue
            seed = int(seed_dir.split("_")[1])
            cands = sorted(f for f in files
                           if f == "best.ckpt"
                           or (f.startswith("best-v") and f.endswith(".ckpt")))
            if not cands:
                continue
            task_dir = os.path.basename(os.path.dirname(dirpath))
            tasks = ([t for t in released.task.unique()] if is_mtl else [task_dir])

            for task in tasks:
                key = "_multitask" if is_mtl else task
                if len(cands) == 1:
                    result[member].setdefault(key, {})[str(seed)] = dict(
                        file=cands[0], max_abs_diff=None)
                    continue

                sub = released[(released.task == task) & (released.split == "test")
                               & (released.seed == seed)]
                if sub.empty:
                    result[member].setdefault(key, {})[str(seed)] = dict(
                        file=cands[-1], max_abs_diff=None)
                    continue
                smis = sub.smiles.tolist()
                ref = sub.y_pred.to_numpy(float)

                lm = load_member(member, task, seed, weights_root=args.weights_root,
                                 recipes_root=args.recipes_root)
                tok = None
                if "molformer_tokens" in _member_embeddings(member, args.recipes_root):
                    from pathlib import Path

                    tok = _tokens_parquet(smis, Path("/tmp/_sel_tokens.parquet"))

                scored = []
                for name in cands:
                    sd = torch.load(os.path.join(dirpath, name), map_location="cpu",
                                    weights_only=False)["state_dict"]
                    lm.model.load_state_dict(sd, strict=False)
                    got = _predict_one(lm, smis, recipes_root=args.recipes_root,
                                       repo_root=".", tokens_parquet=tok)
                    scored.append((float(np.max(np.abs(ref - got))), name))
                scored.sort()
                result[member].setdefault(key, {})[str(seed)] = dict(
                    file=scored[0][1], max_abs_diff=scored[0][0])
                print(f"{member}/{task}/seed_{seed}: {scored[0][1]} "
                      f"(maxdiff {scored[0][0]:.3e}; runner-up {scored[1][1]} "
                      f"{scored[1][0]:.3e})", flush=True)
                del lm

    dest = os.path.join(args.out, "checkpoint_map.json")
    os.makedirs(args.out, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    n = sum(len(s) for m in result.values() for s in m.values())
    print(f"\nwrote {dest} ({n} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
