"""DDP rank-aware helper.

torchrun or Lightning auto-launch launches this module in a multi-process environment,
unifying the runner's rank/world determination, synchronization, and cleanup behavior. Even without a ``trainer`` object
it must be callable (including pre-fit stages such as artifact creation), so based on environment variables
rank is determined.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any


def is_rank_zero() -> bool:
    """Whether the current process is global rank 0.

    torchrun sets ``RANK`` / ``LOCAL_RANK``, and Lightning auto-launch after spawn
    sets the same keys. If unset (= single process), always True.
    """
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def world_size() -> int:
    """Registered world size. 1 if unset."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_process_group_if_torchrun() -> bool:
    """In a torchrun (``WORLD_SIZE>1``) environment, pre-init the process group.

    Why:
        Lightning ``DDPStrategy`` creates the group at ``Trainer.fit()`` time, but the runner
        creates the artifact directory with rank-0 only before fit, and rank-1+ retrieves it via ``store.load``,
        so a cross-rank barrier is needed even before fit. This function
        initializes the group early to avoid rank-aware disk races. Lightning
        reuses it if ``is_initialized()`` is True, so there is no conflict.

    Returns:
        Whether actual initialization occurred (False if already initialized or single process).
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
    # NCCL requires the current device to be set before init (rank → local GPU mapping).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True


def barrier_if_distributed() -> None:
    """Barrier only when torch.distributed is initialized. No-op otherwise.

    If the group is pre-launched before fit via ``init_process_group_if_torchrun()``,
    this barrier actually takes effect.
    """
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_process_group() -> None:
    """Explicitly destroy the process group after the seed loop ends. This is automatic on process exit, but
    acts as a safeguard since fit + predict split into separate Trainer instances is a non-standard pattern."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def runtime_from_args(parsed: argparse.Namespace) -> dict[str, Any]:
    """Dict to serialize from CLI parse results + environment variables into manifest.runtime.

    Record-only, with no effect on the hash (independent of run_id). Always filled in even if the user doesn't supply CLI args,
    so that "single-device mode with 1 GPU" is also explicitly recorded in the manifest.
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
