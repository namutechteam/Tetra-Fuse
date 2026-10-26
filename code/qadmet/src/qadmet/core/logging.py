"""Structured logging.

Simultaneously outputs console (human) + jsonl (machine) streams. Training/evaluation metrics, step progress
are recorded in a consistent format for easy reading during lineage analysis.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Flattens arbitrary dict passed via logger.info(..., extra={"payload": {...}}).
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "qadmet", *, jsonl_path: Path | None = None) -> logging.Logger:
    """Reuses existing logger if name matches. Prevents duplicate handler registration."""
    logger = logging.getLogger(name)
    if getattr(logger, "_qadmet_initialized", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(jsonl_path, encoding="utf-8")
        file_handler.setFormatter(_JsonLineFormatter())
        logger.addHandler(file_handler)

    logger._qadmet_initialized = True  # type: ignore[attr-defined]
    return logger
