"""DDP rank-aware helper.

torchrun 또는 Lightning auto-launch 로 띄운 멀티 프로세스 환경에서 본 모듈이
runner 의 rank/world 판단·동기화·정리 동작을 통일한다. ``trainer`` 객체 없이도
호출 가능해야 하므로 (artifact 생성 등 fit 이전 단계 포함) 환경 변수 기반으로
rank 를 판단한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any


def is_rank_zero() -> bool:
    """현재 프로세스가 global rank 0 인지.

    torchrun 은 ``RANK`` / ``LOCAL_RANK`` 를, Lightning auto-launch 는 spawn 후
    동일 키를 세팅한다. 미설정 (= 단일 프로세스) 이면 항상 True.
    """
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def world_size() -> int:
    """등록된 world size. 미설정이면 1."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_process_group_if_torchrun() -> bool:
    """torchrun (``WORLD_SIZE>1``) 환경이면 process group 을 미리 init.

    Why:
        Lightning ``DDPStrategy`` 는 ``Trainer.fit()`` 시점에 group 을 만들지만, runner
        가 fit 이전에 rank-0 만 artifact 디렉토리를 생성하고 rank-1+ 가 ``store.load``
        로 회수하는 패턴이라 fit 이전에도 cross-rank barrier 가 필요하다. 본 함수는
        rank-aware 디스크 race 회피용으로 group 을 일찍 초기화한다. Lightning 은
        ``is_initialized()`` 가 True 면 재사용하므로 충돌 없음.

    Returns:
        실제 초기화 여부 (이미 초기화돼 있거나 단일 프로세스면 False).
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    try:
        import torch
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return False
    if not dist.is_available() or dist.is_initialized():
        return False
    # NCCL 는 init 전에 current device 가 설정돼 있어야 한다 (rank → local GPU 매핑).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True


def barrier_if_distributed() -> None:
    """torch.distributed 가 초기화돼 있을 때만 barrier. 그 외에는 no-op.

    ``init_process_group_if_torchrun()`` 으로 fit 이전에도 group 을 미리 띄워두면
    이 barrier 가 실제로 동작한다.
    """
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_process_group() -> None:
    """시드 루프 종료 후 process group 명시적 해제. 프로세스 종료 시 자동이지만
    fit + predict 가 별도 Trainer 로 갈리는 비표준 패턴이라 안전판."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def runtime_from_args(parsed: argparse.Namespace) -> dict[str, Any]:
    """CLI parse 결과 + 환경 변수 → manifest.runtime 으로 직렬화할 dict.

    해시 영향 없는 (run_id 와 독립) 기록 전용. 사용자가 CLI 안 줘도 항상 채워서
    "GPU 1개 single-device 모드" 도 manifest 에 명시적으로 남기게 한다.
    """
    return {
        "cli_args": list(sys.argv[1:]),
        "devices": getattr(parsed, "devices", 1),
        "strategy": getattr(parsed, "strategy", None),
        "num_nodes": getattr(parsed, "num_nodes", 1),
        "accelerator": getattr(parsed, "accelerator", None),
        "find_unused_parameters": getattr(parsed, "find_unused_parameters", False),
        "ddp_static_graph": getattr(parsed, "ddp_static_graph", True),
        "predict_only": getattr(parsed, "predict_only", False),
        "world_size": world_size(),
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
