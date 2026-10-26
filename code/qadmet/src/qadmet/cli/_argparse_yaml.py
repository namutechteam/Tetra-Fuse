"""argparse + YAML override merge helper."""

from __future__ import annotations

import argparse


def add_recipe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipe", required=True, help="Path to recipe YAML to run (recipes/<kind>/<name>.yaml)",
    )
    parser.add_argument(
        "--override", "-o", action="append", default=[],
        help="Override spec fields via dot-notation (can be used repeatedly). "
             "e.g.: --override trainer.max_epochs=2 --override trainer.seeds='[0,1]'",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if existing artifact is present (deletes and recreates directory if run_id matches)",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    """DDP / infra execution options. Independent of recipe (no effect on run_id hash).

    Exposed only on train subcommand. Example::

        qadmet train --recipe ... --devices 4 --strategy ddp
        torchrun --nproc_per_node=4 -m qadmet.cli.main train \\
            --recipe ... --devices 4 --strategy ddp
    """
    parser.add_argument(
        "--devices", type=int, default=1,
        help="Number of devices to use (default: 1). Must match torchrun --nproc_per_node when using DDP.",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Lightning strategy: 'ddp' or None (default → Lightning auto). "
             "Omit for single-device training.",
    )
    parser.add_argument(
        "--num-nodes", type=int, default=1,
        help="Number of distributed nodes (default: 1).",
    )
    parser.add_argument(
        "--accelerator", type=str, default=None,
        help="'gpu' / 'cpu' / None (default → auto-detect CUDA).",
    )
    parser.add_argument(
        "--find-unused-parameters", action="store_true",
        help="DDPStrategy(find_unused_parameters=True) — safety net. For when the forward path "
             "touches only some parameters per batch; enable only then (usually OFF).",
    )
    parser.add_argument(
        "--no-ddp-static-graph", dest="ddp_static_graph",
        action="store_false", default=True,
        help="Disable DDPStrategy(static_graph=True). Default is True — "
             "used to avoid NCCL hang when combined with find_unused_parameters=True. "
             "Set to False only for models whose forward path changes every iteration.",
    )
    parser.add_argument(
        "--predict-only", dest="predict_only", action="store_true",
        help="Skip fit and only regenerate predictions.parquet from the existing seed_N/best.ckpt "
             "Must be run as a single process (non-DDP) (torchrun not required). "
             "For recovery when ckpt has gone stale due to an interrupted re-run.",
    )
