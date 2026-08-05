#!/usr/bin/env python
"""Export the fine-tuned (PEFT) weights of the five Tetra-Fuse ensemble members.

WHY THIS SCRIPT EXISTS
----------------------
Each training checkpoint (`best.ckpt`) contains *both* the frozen pretrained
backbone weights (CheMeleon / GraphMVP / MoLFormer-XL — distributed under their
own licenses, so we do not ship them) *and* the weights we actually trained
(AdapterGNN adapters, LoRA matrices, projection layers, task heads, MAP
fusion). This script keeps only the latter, so the released archive is
license-clean and about half the size of the raw checkpoints.

A tensor is treated as FROZEN (and therefore dropped) only when all three hold:
  1. the member is known to load that pretrained backbone,
  2. the checkpoint key maps onto a key of that pretrained file, AND
  3. the shapes match exactly.
Rule 3 is load-bearing: D-MPNN is trained from scratch and its message-passing
tensors are named `W_i/W_h/W_o` just like CheMeleon's, but with different shapes
(300-d vs 2048-d). Without the shape check they would be wrongly dropped and
inference would silently run with randomly initialized message passing.

Everything else is exported. Erring toward exporting is deliberate: a tensor
kept unnecessarily costs disk, a tensor dropped wrongly breaks inference.

OUTPUT LAYOUT
-------------
    weights/tetrafuse/<member>/<task>/seed_<n>/peft.pt       # STL members
    weights/tetrafuse/<member>/_multitask/seed_<n>/peft.pt   # MTL members
    weights/weights_manifest.csv                             # sha256 + sizes

Each `peft.pt` is a dict:
    {"state_dict": {...}, "meta": {member, task, seed, recipe, run_id,
                                   source_ckpt, source_ckpt_sha256,
                                   n_tensors, n_params, dtype}}

Every released file is named `peft.pt` regardless of which training checkpoint it
came from, so the tree is uniform; `meta["source_ckpt"]` (and the matching column
in `weights_manifest.csv`) records whether that was `best.ckpt` or
`best-v1.ckpt`.

Pass `--checkpoint-map <checkpoint_map.json>` (produced by
`code/select_checkpoints.py`) to pick the checkpoint that reproduces the released
predictions for each (task, seed); without it, `pick_final_checkpoint` falls back
to a score heuristic, which is NOT reliable for CheMeleon·STL.

USAGE
-----
    python code/export_weights.py --train-root /path/to/qadmet/artifacts/train \
                                  --weights-root /path/to/qadmet/data/weights \
                                  --out weights
    # --fp16 halves the archive size; omit it for bit-exact fp32 parity
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import torch

# --- the five ensemble members: run directory, run_id, and which pretrained
#     backbones each one loads (frozen) ----------------------------------------
MEMBERS = {
    "mtl_dmpnn": dict(
        recipe="mtl_dmpnn_robust_causal6_kendall", run_id="63cabbaf3772",
        multitask=True, pretrained=[],                      # trained from scratch
    ),
    "mtl_molformer": dict(
        recipe="mtl_molformer_robust_causal6_kendall", run_id="f6b3e3db0740",
        multitask=True, pretrained=["molformer"],
    ),
    "stl_chemeleon": dict(
        recipe="stl_chemeleon_adaptivegnn_paper", run_id="641681c81003",
        multitask=False, pretrained=["chemeleon"],
    ),
    "stl_graphmvp": dict(
        recipe="stl_graphmvp_adaptivegnn_paper", run_id="898dfd67a49b",
        multitask=False, pretrained=["graphmvp"],
    ),
    "stl_map": dict(
        recipe="stl_4modality_robust_nounimol2", run_id="a57008843b9b",
        multitask=False, pretrained=["chemeleon", "molformer", "graphmvp"],
    ),
}

BACKBONE_PREFIXES = ("backbones.0.", "backbones.1.", "backbones.2.", "backbone.")


def load_pretrained_keys(weights_root: str) -> dict:
    """key -> shape maps for each pretrained backbone we might need to drop."""
    out = {}
    chem = os.path.join(weights_root, "chemeleon_mp.pt")
    if os.path.exists(chem):
        sd = torch.load(chem, map_location="cpu", weights_only=False)["state_dict"]
        out["chemeleon"] = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
    gm = os.path.join(weights_root, "graphmvp", "graphmvp_hybrid.pth")
    if os.path.exists(gm):
        sd = torch.load(gm, map_location="cpu", weights_only=False)["model"]
        out["graphmvp"] = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
    return out


def is_frozen(key: str, shape: tuple, pretrained: list, pre: dict) -> bool:
    """True only if this tensor is restorable from a pretrained file we don't ship."""
    # MoLFormer-XL base weights come from HuggingFace at the pinned revision;
    # LoRA matrices sit inside the same subtree and MUST be kept.
    if "molformer" in pretrained and "molformer" in key.lower() and "lora" not in key.lower():
        return True
    for prefix in BACKBONE_PREFIXES:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if "chemeleon" in pretrained and "chemeleon" in pre:
            cand = suffix.replace("_mpnn.message_passing.", "")
            if pre["chemeleon"].get(cand) == shape:      # shape check is load-bearing
                return True
        if "graphmvp" in pretrained and "graphmvp" in pre:
            cand = suffix.replace("gin.", "")
            if pre["graphmvp"].get(cand) == shape:
                return True
    return False


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def export_one(ckpt_path: str, dest: str, member: str, task: str, seed: int,
               cfg: dict, pre: dict, fp16: bool):
    """Write the PEFT subset of one checkpoint.

    Returns (n_tensors, n_params, bytes, source_ckpt_basename).
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    keys = [k for k in sd if hasattr(sd[k], "shape")]
    # `backbone.` and `backbones.0.` are aliases of the same tensors in some
    # checkpoints; keep the canonical `backbones.` form only.
    has_b = any(k.startswith("backbone.") for k in keys)
    has_bs = any(k.startswith("backbones.") for k in keys)
    keys = [k for k in keys if not (has_b and has_bs and k.startswith("backbone."))]

    keep = {}
    for k in keys:
        if is_frozen(k, tuple(sd[k].shape), cfg["pretrained"], pre):
            continue
        t = sd[k]
        keep[k] = t.half() if (fp16 and t.is_floating_point()) else t

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    payload = {
        "state_dict": keep,
        # `source_ckpt` keeps the provenance readable: every released file is named
        # `peft.pt`, so recording which training checkpoint it came from
        # (`best.ckpt` or a collision-suffixed `best-v1.ckpt`) is the only way to
        # trace it back. Only CheMeleon·STL has more than one candidate; see
        # pick_final_checkpoint for the per-member counts.
        "meta": dict(member=member, task=task, seed=seed, recipe=cfg["recipe"],
                     run_id=cfg["run_id"], pretrained=cfg["pretrained"],
                     source_ckpt=os.path.basename(ckpt_path),
                     source_ckpt_sha256=sha256(ckpt_path),
                     n_tensors=len(keep),
                     n_params=int(sum(v.numel() for v in keep.values())),
                     dtype="fp16" if fp16 else "fp32"),
    }
    torch.save(payload, dest)
    return (len(keep), payload["meta"]["n_params"], os.path.getsize(dest),
            payload["meta"]["source_ckpt"])


def pick_final_checkpoint(dirpath: str, filenames) -> str:
    """Return the checkpoint that `trainer.predict(ckpt_path="best")` would have used.

    Why this is not simply `best.ckpt`
    ----------------------------------
    When a ModelCheckpoint filename collides, Lightning writes `best-v1.ckpt`
    (then `-v2`, ...) instead of overwriting. Counting `best-v*.ckpt` under each
    member's canonical run directory gives: mtl_dmpnn 0, mtl_molformer 0,
    stl_chemeleon 110, stl_graphmvp 0, stl_map 0 (out of 5 / 5 / 110 / 110 / 110
    `best.ckpt` files respectively) — so only CheMeleon·STL has a second
    candidate, and only there does the choice matter.

    Exporting the wrong one is silent: the weights load cleanly and predictions
    look plausible, but they are not the predictions in the paper. Verified on
    CheMeleon·STL / Caco2_Wang / seed 0 against the released `predictions.parquet`
    — `best.ckpt` is off by 8.4e-02 while `best-v1.ckpt` matches to 9.5e-07.

    Prefer `--checkpoint-map`: this score heuristic is only a fallback, and it is
    known to pick wrong. Scoring both candidates against the released predictions
    (see code/select_checkpoints.py) shows `best-v1.ckpt` is correct for all 110
    CheMeleon·STL (task, seed) pairs, whereas the recorded `best_model_score`
    orders the two files inconsistently across tasks — v1 lower for Caco2_Wang and
    HIA_Hou, higher for VDss_Lombardo and LD50_Zhu — so this rule would select
    `best.ckpt` for the latter and silently ship weights that do not reproduce the
    paper.

    Selection rule: each candidate records the `best_model_score` its
    ModelCheckpoint callback was tracking (a monitored validation loss, so lower
    is better). We take the candidate with the lowest recorded score — that is
    the state the callback considered best, which is what `ckpt_path="best"`
    restores. `best_model_path` alone cannot decide it: every file points at
    itself. Ties, or checkpoints without a score, fall back to the highest `-vN`
    suffix, which Lightning wrote last.
    """
    cands = sorted(f for f in filenames
                   if f == "best.ckpt" or (f.startswith("best-v") and f.endswith(".ckpt")))
    if len(cands) == 1:
        return os.path.join(dirpath, cands[0])

    if not os.environ.get("EXPORT_WEIGHTS_ALLOW_SCORE_HEURISTIC"):
        raise RuntimeError(
            f"{dirpath} holds {len(cands)} candidate checkpoints ({', '.join(cands)}) "
            f"and no --checkpoint-map entry covers it. The score heuristic picks the "
            f"wrong file for some tasks, so refusing rather than exporting weights "
            f"that may not reproduce the paper.\n"
            f"  Fix: python code/select_checkpoints.py   (writes checkpoint_map.json)\n"
            f"  then: python code/export_weights.py --checkpoint-map <path>\n"
            f"  Override (not recommended): EXPORT_WEIGHTS_ALLOW_SCORE_HEURISTIC=1")

    scored: list[tuple[float, str]] = []
    for name in cands:
        path = os.path.join(dirpath, name)
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        for value in (blob.get("callbacks") or {}).values():
            if isinstance(value, dict) and value.get("best_model_score") is not None:
                scored.append((float(value["best_model_score"]), path))
                break
    if scored:
        scored.sort(key=lambda t: t[0])
        return scored[0][1]
    return os.path.join(dirpath, cands[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default="/nas/tk_ahn/ADMET-MTL/qadmet/artifacts/train",
                    help="qadmet artifacts/train root holding the training runs")
    ap.add_argument("--weights-root", default="/nas/tk_ahn/ADMET-MTL/qadmet/data/weights",
                    help="directory with the pretrained backbone files")
    ap.add_argument("--out", default="weights", help="output root (weights/)")
    ap.add_argument("--fp16", action="store_true", help="store as float16 (half the size)")
    ap.add_argument("--members", nargs="*", default=list(MEMBERS),
                    help="subset of members to export")
    ap.add_argument("--checkpoint-map", default=None,
                    help="checkpoint_map.json from code/select_checkpoints.py — names "
                         "the checkpoint that reproduces the released predictions for "
                         "each (member, task, seed). Strongly recommended: without it "
                         "CheMeleon·STL falls back to a score heuristic that picks the "
                         "wrong file for some tasks.")
    args = ap.parse_args()

    ckpt_map = {}
    if args.checkpoint_map:
        with open(args.checkpoint_map) as fh:
            ckpt_map = json.load(fh)

    pre = load_pretrained_keys(args.weights_root)
    missing = [b for m in args.members for b in MEMBERS[m]["pretrained"]
               if b in ("chemeleon", "graphmvp") and b not in pre]
    if missing:
        print(f"ERROR: pretrained file(s) for {sorted(set(missing))} not found under "
              f"{args.weights_root}; cannot tell frozen weights apart from trained ones.",
              file=sys.stderr)
        return 1

    rows = []
    for member in args.members:
        cfg = MEMBERS[member]
        # (see pick_final_checkpoint below for the best.ckpt vs best-vN.ckpt rule)
        base = os.path.join(args.train_root, cfg["recipe"], cfg["run_id"], "data")
        if not os.path.isdir(base):
            print(f"skip {member}: {base} not found", file=sys.stderr)
            continue
        # MTL: data/seed_<n>/best.ckpt      STL: data/<task>/seed_<n>/best.ckpt
        found = 0
        for dirpath, _dirnames, filenames in os.walk(base):
            if "best.ckpt" not in filenames:
                continue
            rel = os.path.relpath(dirpath, base).split(os.sep)
            seed_dir = rel[-1]
            if not seed_dir.startswith("seed_"):
                continue
            seed = int(seed_dir.split("_")[1])
            task = "_multitask" if cfg["multitask"] else (rel[-2] if len(rel) > 1 else "_multitask")
            dest = os.path.join(args.out, "tetrafuse", member, task, seed_dir, "peft.pt")
            src = None
            if ckpt_map:
                entry = (ckpt_map.get(member, {}).get(task, {}) or {}).get(str(seed))
                if entry and entry.get("file"):
                    src = os.path.join(dirpath, entry["file"])
                    if not os.path.exists(src):
                        raise FileNotFoundError(
                            f"checkpoint map names {entry['file']} for "
                            f"{member}/{task}/seed_{seed} but it is not in {dirpath}")
            if src is None:
                src = pick_final_checkpoint(dirpath, filenames)
            nt, npar, nbytes, src_name = export_one(src, dest,
                                                    member, task, seed, cfg, pre,
                                                    args.fp16)
            rows.append(dict(member=member, task=task, seed=seed,
                             path=os.path.relpath(dest, args.out),
                             source_ckpt=src_name,
                             n_tensors=nt, n_params=npar, bytes=nbytes,
                             sha256=sha256(dest)))
            found += 1
        print(f"{member}: exported {found} checkpoints", flush=True)

    if not rows:
        print("nothing exported", file=sys.stderr)
        return 1
    man = os.path.join(args.out, "weights_manifest.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["bytes"] for r in rows)
    print(f"\n{len(rows)} files, {total/1e9:.2f} GB -> {args.out}/")
    print(f"manifest: {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
