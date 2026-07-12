"""argparse + YAML override 머지 헬퍼."""

from __future__ import annotations

import argparse


def add_recipe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipe", required=True, help="실행할 recipe YAML 경로 (recipes/<kind>/<name>.yaml)",
    )
    parser.add_argument(
        "--override", "-o", action="append", default=[],
        help="dot-notation 으로 spec 필드 덮어쓰기 (반복 사용 가능). "
             "예: --override trainer.max_epochs=2 --override trainer.seeds='[0,1]'",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="기존 artifact 가 있어도 강제 재실행 (run_id 동일 시 디렉토리 삭제 후 재생성)",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    """DDP / 인프라 실행 옵션. recipe 와 무관 (run_id 해시 영향 없음).

    train subcommand 에만 노출. 예::

        qadmet train --recipe ... --devices 4 --strategy ddp
        torchrun --nproc_per_node=4 -m qadmet.cli.main train \\
            --recipe ... --devices 4 --strategy ddp
    """
    parser.add_argument(
        "--devices", type=int, default=1,
        help="사용 device 수 (default: 1). DDP 시 torchrun --nproc_per_node 와 일치시킬 것.",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Lightning strategy: 'ddp' 또는 None (default → Lightning auto). "
             "single-device 학습이면 생략.",
    )
    parser.add_argument(
        "--num-nodes", type=int, default=1,
        help="분산 노드 수 (default: 1).",
    )
    parser.add_argument(
        "--accelerator", type=str, default=None,
        help="'gpu' / 'cpu' / None (default → CUDA 감지 자동).",
    )
    parser.add_argument(
        "--find-unused-parameters", action="store_true",
        help="DDPStrategy(find_unused_parameters=True) — 안전판. forward path 가 "
             "batch 마다 일부 파라미터만 건드릴 때만 활성화 (보통 OFF).",
    )
    parser.add_argument(
        "--no-ddp-static-graph", dest="ddp_static_graph",
        action="store_false", default=True,
        help="DDPStrategy(static_graph=True) 비활성화. 기본은 True — "
             "find_unused_parameters=True 와 함께 쓸 때 NCCL hang 회피용. "
             "forward path 가 매 iter 마다 달라지는 모델만 False 로.",
    )
    parser.add_argument(
        "--predict-only", dest="predict_only", action="store_true",
        help="fit 을 생략하고 기존 seed_N/best.ckpt 로부터 predictions.parquet 만 "
             "재생성. 단일 프로세스(non-DDP)로 실행할 것 (torchrun 불필요). "
             "중단된 재실행으로 ckpt 가 stale 해진 경우 복구용.",
    )
