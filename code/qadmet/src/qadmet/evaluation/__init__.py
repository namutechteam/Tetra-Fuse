"""evaluation: metric computation/comparison tools for new artifacts.

Composed of three modules:

- ``tdc_benchmark``: ``predictions.parquet`` → same 7-column as old phase result CSV
  metric table (`load_predictions`, `per_task_metric_df`, `evaluate_artifact`).
  self-contained based on sklearn, with no TDC internet dependency.
- ``compare``: merges new artifact + old phase CSV into a wide pivot and W/L/T
  summary computation. Used by ``runners.evaluate``.
- ``leaderboard``: TDC web leaderboard scrape + cache (optional). Old
  ``compare_leaderboard.py`` absorbed. requests/bs4 lazy import.

Why:
    parity directory is a one-time regression prevention tool for old phase results ↔ new artifacts.
    evaluation package is user-facing(向) — called from recipe to create artifacts and
    leaves comparison results in the lineage graph. ``parity._metrics`` is this module's
    compatibility wrapper that directly reuses the ``tdc_benchmark`` function.

Note:
    This package does not use the registry, so there are no import-time side effects.
    Users ``from qadmet.evaluation.tdc_benchmark import evaluate_artifact``
    call directly in this form.
"""

from qadmet.evaluation.tdc_benchmark import (
    evaluate_artifact,
    load_predictions,
    metric_direction,
    per_task_metric_df,
)

__all__ = [
    "evaluate_artifact",
    "load_predictions",
    "metric_direction",
    "per_task_metric_df",
]
