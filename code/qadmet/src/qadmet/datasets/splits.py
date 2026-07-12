"""split 관련 유틸. 현재는 TDC scaffold split 을 그대로 사용하므로 placeholder."""

from __future__ import annotations


def split_signature(name: str, seeds: tuple[int, ...]) -> str:
    """recipe lineage hash 안정화용 결정적 문자열."""
    return f"{name}::seeds={list(seeds)}"
