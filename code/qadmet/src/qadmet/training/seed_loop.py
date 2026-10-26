"""Multi-seed training loop.

Reorganizes the seed loop from the old ``run_mtl_*`` (scripts-old/pipeline_mtl.py:1502-1839) based on spec
—per seed: (a) seed_everything (b) build model (c) Trainer.fit
(d) performs test prediction, and accumulates and returns ``SeedRun`` results.

This module does not contain the training/prediction logic itself; it accepts callback functions (build/train/predict)
as externally supplied responsibilities. The actual invocation is done by ``qadmet.runners.train``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from qadmet.core.logging import get_logger
from qadmet.core.seed import seed_everything

logger = get_logger("qadmet.training.seed_loop")


@dataclass
class SeedRun:
    """A bundle of single-seed training results."""

    seed: int
    ckpt_path: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    val_predictions: Any = None  # Free-form (DataFrame / dict, etc.). Defined by the runner.
    test_predictions: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def run_seed_loop(
    *,
    seeds: Sequence[int],
    fit_one: Callable[[int, Path], SeedRun],
    ckpt_root: Path,
) -> list[SeedRun]:
    """Calls ``fit_one(seed, ckpt_dir)`` for each seed and collects the results.

    Args:
        seeds: List of seeds to train.
        fit_one: Callback responsible for training+prediction of a single seed.
            Args (seed, seed_ckpt_dir) → returns ``SeedRun``.
        ckpt_root: Parent of per-seed checkpoint directories. Automatically creates ``seed_{N}`` sub-
            directories.

    Returns:
        List of ``SeedRun`` (in input order).

    Note:
        ``seed_everything`` is only exposed to be called inside fit_one — besides training,
        multiple stages are affected by the seed, so fit_one decides when to call it.
    """
    runs: list[SeedRun] = []
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_dir = ckpt_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        logger.info("seed start", extra={"payload": {"seed": seed, "ckpt_dir": str(seed_dir)}})
        run = fit_one(seed, seed_dir)
        runs.append(run)
        logger.info(
            "seed done",
            extra={
                "payload": {
                    "seed": seed,
                    "ckpt": str(run.ckpt_path) if run.ckpt_path else None,
                    "metrics": run.metrics,
                }
            },
        )
        # Memory release between seeds: CUDA reserved memory left by the previous seed's model/optimizer
        # is freed to prevent fragmentation-accumulated OOM when building the next seed. Recipes close to the
        # GPU ceiling, such as 4-modality, OOM at seed 1+ without this (observed).
        _release_between_seeds()
    return runs


def _release_between_seeds() -> None:
    """Releases Python GC + CUDA cache at seed boundaries (if available)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
