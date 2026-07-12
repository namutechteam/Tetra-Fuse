"""artifact 의존성 그래프 추적.

각 artifact 의 manifest 에는 ``inputs: [{kind, name, run_id}, ...]`` 가 있다.
``trace_dependencies`` 는 그 그래프를 BFS 로 펼쳐 모든 조상 artifact 를 반환한다.
``to_dot`` 은 graphviz dot 문자열을 생성해 사람이 시각화할 수 있게 한다.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from qadmet.core.artifacts import Artifact, ArtifactStore


def trace_dependencies(
    store: ArtifactStore,
    artifact: Artifact,
    *,
    max_depth: int = 32,
) -> list[Artifact]:
    """artifact 의 모든 조상(자기 자신 포함, BFS 순서)을 반환한다."""
    seen: set[tuple[str, str, str]] = set()
    order: list[Artifact] = []
    queue: deque[tuple[Artifact, int]] = deque([(artifact, 0)])
    while queue:
        node, depth = queue.popleft()
        key = (node.kind, node.name, node.run_id)
        if key in seen:
            continue
        seen.add(key)
        order.append(node)
        if depth >= max_depth:
            continue
        for inp in node.manifest.get("inputs", []) or []:
            try:
                child = store.load(inp["kind"], inp["name"], inp["run_id"])
            except FileNotFoundError:
                continue
            queue.append((child, depth + 1))
    return order


def to_dot(nodes: Iterable[Artifact]) -> str:
    """artifact 목록을 graphviz dot 그래프로 출력."""
    nodes = list(nodes)
    lines = ["digraph qadmet_lineage {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    seen: set[str] = set()
    for n in nodes:
        nid = f"{n.kind}__{n.name}__{n.run_id}"
        if nid in seen:
            continue
        seen.add(nid)
        label = f"{n.kind}\\n{n.name}\\n{n.run_id}"
        lines.append(f'  "{nid}" [label="{label}"];')
    for n in nodes:
        nid = f"{n.kind}__{n.name}__{n.run_id}"
        for inp in n.manifest.get("inputs", []) or []:
            iid = f"{inp['kind']}__{inp['name']}__{inp['run_id']}"
            lines.append(f'  "{iid}" -> "{nid}";')
    lines.append("}")
    return "\n".join(lines)
