"""Split-related utility. Currently a placeholder since TDC scaffold split is used as-is."""

from __future__ import annotations


def split_signature(name: str, seeds: tuple[int, ...]) -> str:
    """Deterministic string for stabilizing recipe lineage hash."""
    return f"{name}::seeds={list(seeds)}"
