"""qadmet CLI entry — 단일 ``qadmet`` 명령 + 서브커맨드.

사용 예::

    qadmet train --recipe recipes/train/mtl_dmpnn_grouped.yaml
    qadmet precompute --recipe recipes/precompute/rdkit2d.yaml
    qadmet stack --recipe recipes/ensemble/stack_hybrid_blend.yaml
    qadmet evaluate --recipe recipes/evaluate/leaderboard_diff.yaml
    qadmet lineage show <kind> <name>
    qadmet artifacts list [--kind train]
    qadmet recipe validate <path...>

Note:
    별도의 ``predict`` 서브커맨드는 두지 않는다 — ``train`` runner 가
    학습 직후 val + test split 모두에 대한 ``predictions.parquet`` 를
    artifact 로 저장하므로 (Step 4.2 부터), 그 결과가 곧 PredictionSet
    이다. 추후 학습 없이 외부 데이터에 대한 추론이 필요해지면 그때
    ``runners/predict.py`` + recipe 를 추가한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qadmet.cli._argparse_yaml import add_recipe_args, add_runtime_args


def _run_with_recipe(args: argparse.Namespace, expected_kind: str) -> int:
    """공통: recipe 로드 → ExperimentRunner.run."""
    # 지연 import — 무거운 의존(torch 등) 최소화.
    from qadmet.core.experiment import ExperimentRunner
    from qadmet.core.recipe import load_recipe
    import qadmet.runners  # noqa: F401  — runner 등록 트리거

    spec = load_recipe(args.recipe, overrides=args.override or [])
    if spec.kind != expected_kind:
        sys.stderr.write(
            f"recipe.kind={spec.kind!r} 가 기대값({expected_kind!r}) 과 다릅니다.\n"
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
        sys.stderr.write(f"artifact 없음: kind={args.kind} name={args.name}\n")
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
        description="qadmet — TDC ADMET 멀티태스크 연구 플랫폼",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train / precompute / stack / evaluate
    for kind in ("train", "precompute", "stack", "evaluate"):
        p = sub.add_parser(kind, help=f"{kind} recipe 실행")
        add_recipe_args(p)
        if kind == "train":
            add_runtime_args(p)
        p.set_defaults(_cmd=lambda a, k=kind: _run_with_recipe(a, k))

    # lineage show <kind> <name>
    p_lineage = sub.add_parser("lineage", help="lineage 서브커맨드")
    sub_lineage = p_lineage.add_subparsers(dest="lineage_cmd", required=True)
    p_show = sub_lineage.add_parser("show", help="artifact lineage 그래프 출력 (graphviz dot)")
    p_show.add_argument("kind", choices=("train", "precompute", "stack", "evaluate"))
    p_show.add_argument("name")
    p_show.set_defaults(_cmd=_cmd_lineage_show)

    # artifacts list
    p_arts = sub.add_parser("artifacts", help="artifacts 서브커맨드")
    sub_arts = p_arts.add_subparsers(dest="artifacts_cmd", required=True)
    p_list = sub_arts.add_parser("list", help="저장된 artifact 목록")
    p_list.add_argument("--kind", default=None)
    p_list.set_defaults(_cmd=_cmd_artifacts_list)


    # predict — SMILES -> ADMET endpoints using the released weights
    p_pred = sub.add_parser(
        "predict",
        help="SMILES 로부터 22 ADMET endpoint 예측 (배포 가중치 사용)")
    p_pred.add_argument("--smiles", nargs="*", default=None,
                        help="SMILES 문자열 (여러 개 가능)")
    p_pred.add_argument("--input", default=None,
                        help="SMILES 파일 (.txt = 한 줄에 하나, .csv = --smiles-column)")
    p_pred.add_argument("--smiles-column", default="smiles",
                        help="--input 이 CSV 일 때 읽을 열 이름 (기본 smiles)")
    p_pred.add_argument("--output", default=None,
                        help="결과 CSV 경로 (생략 시 stdout)")
    p_pred.add_argument("--weights-root", default="weights",
                        help="PEFT 가중치 트리 (weights/README.md 참고)")
    p_pred.add_argument("--recipes-root", default="code/qadmet/recipes",
                        help="recipe 디렉토리")
    p_pred.add_argument("--repo-root", default="code/qadmet",
                        help="백본 체크포인트 경로(data/weights/...)의 기준 디렉토리")
    p_pred.add_argument("--tasks", nargs="*", default=None,
                        help="예측할 endpoint 부분집합 (기본: 전체 22개)")
    p_pred.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4],
                        help="평균낼 seed (기본 0-4 = 논문 앙상블)")
    p_pred.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    p_pred.add_argument("--batch-size", type=int, default=32)
    p_pred.set_defaults(_cmd=_cmd_predict)

    # recipe validate
    p_rec = sub.add_parser("recipe", help="recipe 서브커맨드")
    sub_rec = p_rec.add_subparsers(dest="recipe_cmd", required=True)
    p_val = sub_rec.add_parser("validate", help="recipe YAML 들이 schema 통과하는지 검증")
    p_val.add_argument("paths", nargs="+")
    p_val.set_defaults(_cmd=_cmd_recipe_validate)

    args = parser.parse_args(argv)
    return args._cmd(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
