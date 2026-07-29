#!/usr/bin/env python
"""
Generate TDC ADMET Benchmark Group submission from frozen base-model predictions.

Builds the k=5 Tetra-fuse ensemble (uniform mean of D-MPNN.MTL, MoLFormer.MTL,
CheMeleon.STL, GraphMVP.STL, MAP.STL), aligns each benchmark's held-out TEST
predictions to the official TDC test-set SMILES order, assembles the per-seed
predictions_list, and runs admet_group.evaluate_many() -> mean +/- std per
benchmark (the exact numbers submitted to the TDC leaderboard).

This is the matched-seed protocol: 5 seeds -> 5 dicts -> evaluate_many averages.
Raw metrics (MAE/AUROC/AUPRC/Spearman) are computed by TDC itself, identically
to every other leaderboard entry.

Usage:
    python make_tdc_submission.py \
        --baselines /nas/tk_ahn/ADMET-MTL/qadmet_public/artifacts/baselines \
        --tdc-path  /nas/tk_ahn/.tdc_data \
        --out-dir   ./tdc_submission
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

# k=5 Tetra-fuse members (uniform-mean late fusion)
K5 = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map"]
N_SEEDS = 5

# our task label -> TDC benchmark name is a simple lowercasing; map explicitly
def task_to_benchmark(task: str) -> str:
    return task.lower()

def load_members(baselines_dir):
    dfs = {}
    for m in K5:
        p = os.path.join(baselines_dir, m, "baseline", "predictions.parquet")
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing predictions for member {m}: {p}")
        dfs[m] = pd.read_parquet(p)
    return dfs

def ensemble_test_pred(dfs, task, seed):
    """Uniform mean over K5 members for (task, test split, seed), keyed by SMILES."""
    maps = []
    for m in K5:
        s = dfs[m]
        sub = s[(s.task == task) & (s.split == "test") & (s.seed == seed)]
        if len(sub) == 0:
            return None
        maps.append(dict(zip(sub.smiles, sub.y_pred)))
    common = set(maps[0])
    for mm in maps[1:]:
        common &= set(mm)
    return {sm: float(np.mean([mm[sm] for mm in maps])) for sm in common}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="/nas/tk_ahn/ADMET-MTL/qadmet_public/artifacts/baselines")
    ap.add_argument("--tdc-path",  default="/nas/tk_ahn/.tdc_data")
    ap.add_argument("--out-dir",   default="./tdc_submission")
    ap.add_argument("--include-bioavailability", action="store_true",
                    help="include Bioavailability_Ma (excluded by default: unreliable leaderboard rank)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = admet_group(path=args.tdc_path)
    dfs = load_members(args.baselines)

    all_tasks = sorted(dfs["mtl_dmpnn"].task.unique())
    if not args.include_bioavailability:
        all_tasks = [t for t in all_tasks if t != "Bioavailability_Ma"]
    print(f"[info] building submission for {len(all_tasks)} benchmarks x {N_SEEDS} seeds "
          f"(ensemble = uniform mean of {K5})")

    # TDC expects: predictions_list = [ {benchmark_name: [preds aligned to test SMILES]}, ... ]  (one dict per seed)
    predictions_list = []
    coverage = []
    for seed in range(N_SEEDS):
        seed_dict = {}
        for task in all_tasks:
            bench = task_to_benchmark(task)
            tdc_test = g.get(bench)["test"]
            tdc_smiles = list(tdc_test["Drug"])
            ep = ensemble_test_pred(dfs, task, seed)
            if ep is None:
                sys.exit(f"[FATAL] no ensemble prediction for {task} seed {seed}")
            missing = [s for s in tdc_smiles if s not in ep]
            if missing:
                sys.exit(f"[FATAL] {bench} seed {seed}: {len(missing)} TDC test SMILES "
                         f"not covered by our predictions (need exact test-set match)")
            arr = np.array([ep[s] for s in tdc_smiles], dtype=float)
            seed_dict[bench] = arr
            if seed == 0:
                coverage.append((bench, len(tdc_smiles)))
        predictions_list.append(seed_dict)
        print(f"[info] seed {seed}: assembled {len(seed_dict)} benchmarks")

    # TDC computes raw metric per seed, then mean/std across seeds -> leaderboard numbers
    results = g.evaluate_many(predictions_list)

    # results: {benchmark: [mean, std]}
    rows = []
    for bench, val in sorted(results.items()):
        mean_v, std_v = float(val[0]), float(val[1])
        rows.append({"benchmark": bench, "mean": round(mean_v, 4), "std": round(std_v, 4)})
    res_df = pd.DataFrame(rows)

    csv_path = os.path.join(args.out_dir, "tdc_submission_scores.csv")
    json_path = os.path.join(args.out_dir, "tdc_submission_scores.json")
    res_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({b: [r["mean"], r["std"]] for b, r in zip(res_df["benchmark"], rows)}, f, indent=2)

    print("\n=== TDC evaluate_many() results (copy these into the submission form) ===")
    print(res_df.to_string(index=False))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {json_path}")
    print("\nSubmit via the current form linked on https://tdcommons.ai/benchmark/overview/ "
          "(results + a short model summary; no code/paper release required).")

if __name__ == "__main__":
    main()
