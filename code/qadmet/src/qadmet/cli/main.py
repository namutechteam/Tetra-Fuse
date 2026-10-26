"""qadmet CLI entry — single ``qadmet`` command + subcommands.

Usage example::

    qadmet train --recipe recipes/train/mtl_dmpnn_grouped.yaml
    qadmet precompute --recipe recipes/precompute/rdkit2d.yaml
    qadmet stack --recipe recipes/ensemble/stack_hybrid_blend.yaml
    qadmet evaluate --recipe recipes/evaluate/leaderboard_diff.yaml
    qadmet lineage show <kind> <name>
    qadmet artifacts list [--kind train]
    qadmet recipe validate <path...>

Note:
    A separate ``predict`` subcommand is not provided — the ``train`` runner
    saves ``predictions.parquet`` for both val + test splits immediately after training
    as an artifact (since Step 4.2), so that result is itself the PredictionSet
    . If inference on external data without training becomes necessary later, then
    add ``runners/predict.py`` + recipe at that time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qadmet.cli._argparse_yaml import add_recipe_args, add_runtime_args


def _run_with_recipe(args: argparse.Namespace, expected_kind: str) -> int:
    """Common: load recipe → ExperimentRunner.run."""
    # Lazy import — minimize heavy dependencies (torch, etc.).
    from qadmet.core.experiment import ExperimentRunner
    from qadmet.core.recipe import load_recipe
    import qadmet.runners  # noqa: F401  — triggers runner registration

    spec = load_recipe(args.recipe, overrides=args.override or [])
    if spec.kind != expected_kind:
        sys.stderr.write(
            f"recipe.kind={spec.kind!r} does not match expected value ({expected_kind!r}).\n"
        )
        return 2

    runtime: dict | None = None
    if expected_kind == "train":
        from qadmet.core.distributed import runtime_from_args

        runtime = runtime_from_args(args)

    runner = ExperimentRunner(runtime=runtime, force=bool(getattr(args, "force", False)))
    artifact = runner.run(spec)
    print(f"[{spec.kind}] {spec.name}  run_id={artifact.run_id}")
    print(f"  → {artifact.root}")
    return 0


def _cmd_lineage_show(args: argparse.Namespace) -> int:
    from qadmet.core.artifacts import ArtifactStore
    from qadmet.core.lineage import to_dot, trace_dependencies

    store = ArtifactStore()
    art = store.latest(args.kind, args.name)
    if art is None:
        sys.stderr.write(f"artifact not found: kind={args.kind} name={args.name}\n")
        return 1
    chain = trace_dependencies(store, art)
    print(to_dot(chain))
    return 0


def _cmd_artifacts_list(args: argparse.Namespace) -> int:
    from qadmet.core.artifacts import ArtifactStore

    store = ArtifactStore()
    base = store.root if args.kind is None else store.root / args.kind
    if not base.exists():
        return 0
    for kind_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for name_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in name_dir.iterdir() if (p / "manifest.yaml").exists()):
                print(f"{kind_dir.name}/{name_dir.name}/{run_dir.name}")
    return 0


def _cmd_recipe_validate(args: argparse.Namespace) -> int:
    from qadmet.core.recipe import load_recipe

    failures: list[tuple[str, str]] = []
    for raw in args.paths:
        for path in sorted(Path().glob(raw)) if any(c in raw for c in "*?[") else [Path(raw)]:
            try:
                load_recipe(path)
                print(f"OK   {path}")
            except Exception as e:  # noqa: BLE001
                failures.append((str(path), repr(e)))
                print(f"FAIL {path}: {e}")
    return 1 if failures else 0



def _cmd_predict(args) -> int:
    """SMILES -> 22 ADMET endpoint predictions with the released Tetra-Fuse weights.

    Reads SMILES from --smiles, a file (--input, one per line or a CSV column),
    or stdin; writes a long table (one row per molecule x endpoint) to --output
    or stdout.
    """
    import sys

    from qadmet.inference import predict_smiles

    smiles: list[str] = []
    if args.smiles:
        smiles.extend(args.smiles)
    if args.input:
        import csv as _csv
        from pathlib import Path as _Path

        path = _Path(args.input)
        text = path.read_text().splitlines()
        if path.suffix.lower() == ".csv":
            reader = _csv.DictReader(text)
            col = args.smiles_column
            if reader.fieldnames and col not in reader.fieldnames:
                print(f"ERROR: column '{col}' not in {path} "
                      f"(has {reader.fieldnames}); pass --smiles-column",
                      file=sys.stderr)
                return 1
            smiles.extend(row[col].strip() for row in reader if row.get(col))
        else:
            smiles.extend(line.strip() for line in text if line.strip())
    if not smiles and not sys.stdin.isatty():
        smiles.extend(line.strip() for line in sys.stdin if line.strip())
    if not smiles:
        print("ERROR: no SMILES given (use --smiles, --input, or pipe to stdin)",
              file=sys.stderr)
        return 1

    df = predict_smiles(
        smiles,
        weights_root=args.weights_root,
        recipes_root=args.recipes_root,
        repo_root=args.repo_root,
        tasks=args.tasks,
        seeds=tuple(args.seeds),
        device=args.device,
        batch_size=args.batch_size,
    )
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output}  ({len(df)} rows, "
              f"{df.smiles.nunique()} molecules x {df.task.nunique()} endpoints)")
    else:
        df.to_csv(sys.stdout, index=False)
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qadmet",
        description="qadmet — TDC ADMET multi-task research platform",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train / precompute / stack / evaluate
    for kind in ("train", "precompute", "stack", "evaluate"):
        p = sub.add_parser(kind, help=f"run {kind} recipe")
        add_recipe_args(p)
        if kind == "train":
            add_runtime_args(p)
        p.set_defaults(_cmd=lambda a, k=kind: _run_with_recipe(a, k))

    # lineage show <kind> <name>
    p_lineage = sub.add_parser("lineage", help="lineage subcommand")
    sub_lineage = p_lineage.add_subparsers(dest="lineage_cmd", required=True)
    p_show = sub_lineage.add_parser("show", help="print artifact lineage graph (graphviz dot)")
    p_show.add_argument("kind", choices=("train", "precompute", "stack", "evaluate"))
    p_show.add_argument("name")
    p_show.set_defaults(_cmd=_cmd_lineage_show)

    # artifacts list
    p_arts = sub.add_parser("artifacts", help="artifacts subcommand")
    sub_arts = p_arts.add_subparsers(dest="artifacts_cmd", required=True)
    p_list = sub_arts.add_parser("list", help="list saved artifacts")
    p_list.add_argument("--kind", default=None)
    p_list.set_defaults(_cmd=_cmd_artifacts_list)


    # predict — SMILES -> ADMET endpoints using the released weights
    p_pred = sub.add_parser(
        "predict",
        help="predict 22 ADMET endpoints from SMILES (uses deployed weights)")
    p_pred.add_argument("--smiles", nargs="*", default=None,
                        help="SMILES string (multiple allowed)")
    p_pred.add_argument("--input", default=None,
                        help="SMILES file (.txt = one per line, .csv = --smiles-column)")
    p_pred.add_argument("--smiles-column", default="smiles",
                        help="column name to read when --input is CSV (default smiles)")
    p_pred.add_argument("--output", default=None,
                        help="output CSV path (stdout if omitted)")
    p_pred.add_argument("--weights-root", default="weights",
                        help="PEFT weights tree (see weights/README.md)")
    p_pred.add_argument("--recipes-root", default="code/qadmet/recipes",
                        help="recipe directory")
    p_pred.add_argument("--repo-root", default="code/qadmet",
                        help="base directory for backbone checkpoint path (data/weights/...)")
    p_pred.add_argument("--tasks", nargs="*", default=None,
                        help="subset of endpoints to predict (default: all 22)")
    p_pred.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4],
                        help="seeds to average (default 0-4 = paper ensemble)")
    p_pred.add_argument("--device", default="cpu", help="cpu or cuda")
    p_pred.add_argument("--batch-size", type=int, default=32)
    p_pred.set_defaults(_cmd=_cmd_predict)

    # recipe validate
    p_rec = sub.add_parser("recipe", help="recipe subcommand")
    sub_rec = p_rec.add_subparsers(dest="recipe_cmd", required=True)
    p_val = sub_rec.add_parser("validate", help="validate that recipe YAMLs pass schema")
    p_val.add_argument("paths", nargs="+")
    p_val.set_defaults(_cmd=_cmd_recipe_validate)

    args = parser.parse_args(argv)
    return args._cmd(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
