#!/usr/bin/env python
"""Predict 22 ADMET endpoints for new molecules with the released Tetra-Fuse weights.

This is a thin standalone wrapper around `qadmet.inference.predict_smiles`, for
people who want to run predictions without installing the package as a CLI. The
`qadmet predict` subcommand does exactly the same thing.

QUICK START
-----------
    # 1. install the environment (see README.md) and download the backbones
    #    into code/qadmet/data/weights/  (see weights/README.md)
    # 2. put the released PEFT weights under weights/tetrafuse/
    # 3. predict
    python code/predict_smiles.py --smiles "CCO" "c1ccccc1O"
    python code/predict_smiles.py --input my_molecules.csv --output preds.csv

OUTPUT
------
One row per (molecule, endpoint):

    smiles,task,task_type,tetrafuse,mtl_dmpnn,mtl_molformer,stl_chemeleon,stl_graphmvp,stl_map

`tetrafuse` is the k=5 uniform-mean ensemble — the model reported in the paper.
The five member columns are each member's own seed-averaged prediction, kept so
an unexpected ensemble value can be traced to the member behind it.

Regression endpoints are returned in their original units (the target scaler and
any log1p transform are inverted); classification endpoints are probabilities.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `qadmet` importable when this script is run straight from a git clone
# without `pip install -e`.
_SRC = Path(__file__).resolve().parent / "qadmet" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _read_smiles(args) -> list[str]:
    """Collect SMILES from --smiles, --input (.txt or .csv), or stdin."""
    smiles: list[str] = list(args.smiles or [])
    if args.input:
        path = Path(args.input)
        lines = path.read_text().splitlines()
        if path.suffix.lower() == ".csv":
            import csv

            reader = csv.DictReader(lines)
            if reader.fieldnames and args.smiles_column not in reader.fieldnames:
                raise SystemExit(
                    f"ERROR: column {args.smiles_column!r} not in {path} "
                    f"(has {reader.fieldnames}); pass --smiles-column")
            smiles += [r[args.smiles_column].strip() for r in reader
                       if r.get(args.smiles_column)]
        else:
            smiles += [ln.strip() for ln in lines if ln.strip()]
    if not smiles and not sys.stdin.isatty():
        smiles += [ln.strip() for ln in sys.stdin if ln.strip()]
    if not smiles:
        raise SystemExit("ERROR: no SMILES given (use --smiles, --input, or stdin)")
    return smiles


def main() -> int:
    here = Path(__file__).resolve().parent           # code/
    release = here.parent                            # repository root

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", nargs="*", help="SMILES strings")
    ap.add_argument("--input", help="SMILES file (.txt one per line, or .csv)")
    ap.add_argument("--smiles-column", default="smiles",
                    help="column to read when --input is a CSV (default: smiles)")
    ap.add_argument("--output", help="write CSV here (default: stdout)")
    ap.add_argument("--weights-root", default=str(release / "weights"),
                    help="released PEFT weight tree (see weights/README.md)")
    ap.add_argument("--recipes-root", default=str(here / "qadmet" / "recipes"))
    ap.add_argument("--repo-root", default=str(here / "qadmet"),
                    help="directory holding data/weights/ with the pretrained backbones")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of endpoints (default: all 22)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4],
                    help="seeds to average (default 0-4, the paper ensemble)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    smiles = _read_smiles(args)

    from qadmet.inference import predict_smiles

    df = predict_smiles(smiles, weights_root=args.weights_root,
                        recipes_root=args.recipes_root, repo_root=args.repo_root,
                        tasks=args.tasks, seeds=tuple(args.seeds),
                        device=args.device, batch_size=args.batch_size)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output}  ({len(df)} rows, {df.smiles.nunique()} molecules "
              f"x {df.task.nunique()} endpoints)")
    else:
        df.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
