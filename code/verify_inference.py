#!/usr/bin/env python
"""Check that the inference path reproduces the released predictions.

This is the acceptance test for the model-usage axis. For each (member, task,
seed) it rebuilds the model from the released PEFT weights, predicts the TDC test
split, and compares against `data/recompute_data/predictions/<member>.parquet` —
the same parquets the paper's analysis notebooks consume. If these agree, then
predictions made on new molecules are produced by the same models that generated
the published numbers.

    python code/verify_inference.py --out verification/inference_parity.csv

OUTPUT: one row per (member, task, seed) with n, max/mean absolute difference,
the difference relative to the prediction range, and Pearson r.

Expected result (see verification/inference_parity.md): four members agree to
~1e-06; MAP·STL agrees to ~1e-02 absolute / ~1e-02 relative with r > 0.9999,
a documented residual whose cause is recorded in that note.
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MEMBERS = ("mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map")


def main() -> int:
    here = Path(__file__).resolve().parent
    release = here.parent
    src = here / "qadmet" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-root", default=str(release / "weights"))
    ap.add_argument("--recipes-root", default=str(here / "qadmet" / "recipes"))
    ap.add_argument("--repo-root", default=str(here / "qadmet"))
    ap.add_argument("--predictions-root",
                    default=str(release / "data" / "recompute_data" / "predictions"))
    ap.add_argument("--out", default=str(release / "verification" / "inference_parity.csv"))
    ap.add_argument("--members", nargs="*", default=list(MEMBERS))
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="default: a 4-endpoint spread (2 classification, 2 regression)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0],
                    help="default: seed 0 only (pass 0 1 2 3 4 for the full check)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    from qadmet.inference.loader import load_member
    from qadmet.inference.predict import (
        _member_embeddings, _predict_one, _tokens_parquet,
    )

    # A spread rather than all 22: two classification and two regression
    # endpoints, including a log1p-transformed target (VDss) so the scaler
    # inversion is exercised.
    tasks = args.tasks or ["HIA_Hou", "BBB_Martins", "Caco2_Wang", "VDss_Lombardo"]
    rows = []
    for member in args.members:
        released = pd.read_parquet(Path(args.predictions_root) / f"{member}.parquet")
        wants_tokens = "molformer_tokens" in _member_embeddings(member, args.recipes_root)
        for task in tasks:
            for seed in args.seeds:
                sub = released[(released.task == task) & (released.split == "test")
                               & (released.seed == seed)]
                if sub.empty:
                    print(f"skip {member}/{task}/seed_{seed}: no released rows",
                          file=sys.stderr)
                    continue
                smiles = sub.smiles.tolist()
                ref = sub.y_pred.to_numpy(float)

                lm = load_member(member, task, seed, weights_root=args.weights_root,
                                 recipes_root=args.recipes_root,
                                 repo_root=args.repo_root, device=args.device)
                tokens = (_tokens_parquet(smiles, Path("/tmp/_verify_tokens.parquet"))
                          if wants_tokens else None)
                got = _predict_one(lm, smiles, recipes_root=Path(args.recipes_root),
                                   repo_root=Path(args.repo_root),
                                   tokens_parquet=tokens)
                del lm

                diff = np.abs(ref - got)
                span = float(np.ptp(ref)) or 1.0
                rows.append(dict(
                    member=member, task=task, seed=seed, n=len(diff),
                    max_abs_diff=f"{diff.max():.3e}",
                    mean_abs_diff=f"{diff.mean():.3e}",
                    max_rel_to_range=f"{diff.max() / span:.3e}",
                    pearson_r=f"{float(np.corrcoef(ref, got)[0, 1]):.7f}",
                ))
                print(f"{member:14} {task:16} seed{seed} n={len(diff):5} "
                      f"max={diff.max():.3e} r={rows[-1]['pearson_r']}", flush=True)

    if not rows:
        print("nothing verified", file=sys.stderr)
        return 1
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = max(float(r["max_abs_diff"]) for r in rows)
    print(f"\nwrote {dest} ({len(rows)} rows); worst max_abs_diff {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
